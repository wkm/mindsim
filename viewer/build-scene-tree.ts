/**
 * buildSceneTree — constructs a SceneTree from a viewer manifest JSON.
 *
 * Hierarchy:
 *   Robot (root)
 *     Assembly? (if manifest.assemblies exists)
 *       Body
 *         Component (mounted parts: non-servo, non-horn, non-fastener)
 *           SubPart (fasteners grouped: "6x M3 SHC")
 *         Component (servo per joint)
 *           SubPart (horns)
 *           SubPart (fastener groups)
 *         Joint (linking parent body -> child body)
 *           [child Body subtree]
 *         Layer (design layers from joint.design_layers[], hidden by default)
 *         Wire group (hidden by default)
 *
 * meshIds are empty at build time — populated when meshes are loaded.
 */

import { NodeKind, SceneTree } from './design-scene.ts';

// ---------------------------------------------------------------------------
// Manifest types (minimal — just what buildSceneTree reads)
// ---------------------------------------------------------------------------

interface ManifestMount {
  label: string;
  component_name: string;
  component_type: string;
}

interface ManifestBody {
  name: string;
  mesh: string;
  kind: string;
  parent: string | null;
  mounts?: ManifestMount[];
  joint?: string;
}

interface ManifestDesignLayer {
  kind: string;
  mesh: string;
  parent_body: string;
}

interface ManifestJoint {
  name: string;
  parent_body: string;
  child_body: string;
  servo: string;
  design_layers?: ManifestDesignLayer[];
}

interface ManifestPart {
  id: string;
  name: string;
  category: string;
  parent_body: string;
  joint?: string;
  mount_label?: string;
}

interface ManifestAssembly {
  name: string;
  path: string;
  bodies: string[];
  sub_assemblies: ManifestAssembly[];
}

interface ViewerManifest {
  bot_name: string;
  assemblies?: ManifestAssembly[];
  bodies: ManifestBody[];
  joints: ManifestJoint[];
  parts?: ManifestPart[];
}

// ---------------------------------------------------------------------------
// Fastener grouping helper
// ---------------------------------------------------------------------------

interface FastenerGroup {
  key: string;
  name: string;
  label: string;
  count: number;
  items: ManifestPart[];
}

function groupFasteners(fasteners: ManifestPart[]): FastenerGroup[] {
  const groups: Record<string, FastenerGroup> = {};
  for (const f of fasteners) {
    const key = f.name;
    if (!groups[key]) {
      groups[key] = { key, name: f.name, label: f.name, count: 0, items: [] };
    }
    groups[key].count++;
    groups[key].items.push(f);
  }
  return Object.values(groups).map((g) => ({
    ...g,
    label: g.count > 1 ? `${g.count}\u00d7 ${g.name}` : g.name,
  }));
}

// ---------------------------------------------------------------------------
// buildSceneTree
// ---------------------------------------------------------------------------

export function buildSceneTree(manifest: ViewerManifest): SceneTree {
  const tree = new SceneTree();
  const parts = manifest.parts || [];

  // Index helpers
  const bodiesByName: Record<string, ManifestBody> = {};
  for (const b of manifest.bodies) bodiesByName[b.name] = b;

  const childJointsOf: Record<string, ManifestJoint[]> = {};
  for (const j of manifest.joints) {
    if (!childJointsOf[j.parent_body]) childJointsOf[j.parent_body] = [];
    childJointsOf[j.parent_body].push(j);
  }

  const partsByBody: Record<string, ManifestPart[]> = {};
  const partsByJoint: Record<string, ManifestPart[]> = {};
  for (const p of parts) {
    if (!partsByBody[p.parent_body]) partsByBody[p.parent_body] = [];
    partsByBody[p.parent_body].push(p);
    if (p.joint) {
      if (!partsByJoint[p.joint]) partsByJoint[p.joint] = [];
      partsByJoint[p.joint].push(p);
    }
  }

  // Robot root
  const rootId = `robot:${manifest.bot_name}`;
  const rootChildren: string[] = [];

  tree.addNode({
    id: rootId,
    kind: NodeKind.Robot,
    label: manifest.bot_name,
    children: rootChildren,
    hidden: false,
    parentId: null,
    meshIds: [],
  });

  // Build body subtree, returns body node id
  function buildBody(body: ManifestBody, parentId: string): string {
    const bodyId = `body:${body.name}`;
    const bodyChildren: string[] = [];
    const bodyParts = partsByBody[body.name] || [];
    const joints = childJointsOf[body.name] || [];

    // 1. Mounted components (non-servo, non-horn, non-fastener, not joint-associated)
    const mountedComps = bodyParts.filter(
      (p) =>
        p.category !== 'servo' &&
        p.category !== 'horn' &&
        p.category !== 'fastener' &&
        p.category !== 'wire' &&
        !p.joint,
    );

    const mountFasteners = bodyParts.filter((p) => p.category === 'fastener' && !p.joint);

    for (const comp of mountedComps) {
      const compId = `part:${comp.id}`;
      const compChildren: string[] = [];

      // Fasteners belonging to this component's mount
      const compFasteners = mountFasteners.filter((f) => f.mount_label === comp.mount_label);
      const grouped = groupFasteners(compFasteners);
      for (const group of grouped) {
        const fId = `fastener-group:${comp.id}:${group.key}`;
        tree.addNode({
          id: fId,
          kind: NodeKind.SubPart,
          label: group.label,
          children: [],
          hidden: false,
          parentId: compId,
          meshIds: [],
        });
        compChildren.push(fId);
      }

      tree.addNode({
        id: compId,
        kind: NodeKind.Component,
        label: comp.name,
        children: compChildren,
        hidden: false,
        parentId: bodyId,
        meshIds: [],
      });
      bodyChildren.push(compId);
    }

    // 2. Servo components per joint + joint nodes
    for (const joint of joints) {
      const jointParts = partsByJoint[joint.name] || [];
      const servos = jointParts.filter((p) => p.category === 'servo');
      const horns = jointParts.filter((p) => p.category === 'horn');
      const fasteners = jointParts.filter((p) => p.category === 'fastener');

      for (const servo of servos) {
        const servoId = `part:${servo.id}`;
        const servoChildren: string[] = [];

        // Horn sub-parts
        for (const horn of horns) {
          const hornId = `part:${horn.id}`;
          tree.addNode({
            id: hornId,
            kind: NodeKind.SubPart,
            label: horn.name,
            children: [],
            hidden: false,
            parentId: servoId,
            meshIds: [],
          });
          servoChildren.push(hornId);
        }

        // Fastener sub-parts (grouped)
        const grouped = groupFasteners(fasteners);
        for (const group of grouped) {
          const fId = `fastener-group:${joint.name}:${group.key}`;
          tree.addNode({
            id: fId,
            kind: NodeKind.SubPart,
            label: group.label,
            children: [],
            hidden: false,
            parentId: servoId,
            meshIds: [],
          });
          servoChildren.push(fId);
        }

        tree.addNode({
          id: servoId,
          kind: NodeKind.Component,
          label: `${servo.name} @ ${joint.name}`,
          children: servoChildren,
          hidden: false,
          parentId: bodyId,
          meshIds: [],
        });
        bodyChildren.push(servoId);
      }

      // Joint node -> child body
      const jointId = `joint:${joint.name}`;
      const jointChildren: string[] = [];

      const childBody = bodiesByName[joint.child_body];
      if (childBody) {
        const childBodyId = buildBody(childBody, jointId);
        jointChildren.push(childBodyId);
      }

      tree.addNode({
        id: jointId,
        kind: NodeKind.Joint,
        label: joint.name,
        children: jointChildren,
        hidden: false,
        parentId: bodyId,
        meshIds: [],
      });
      bodyChildren.push(jointId);

      // Design layers (hidden by default)
      if (joint.design_layers) {
        for (const layer of joint.design_layers) {
          const layerId = `layer:${joint.name}:${layer.kind}`;
          tree.addNode({
            id: layerId,
            kind: NodeKind.Layer,
            label: `${layer.kind} (${joint.name})`,
            children: [],
            hidden: true,
            parentId: bodyId,
            meshIds: [],
          });
          bodyChildren.push(layerId);
        }
      }
    }

    // 3. Wire group (hidden by default)
    const wires = bodyParts.filter((p) => p.category === 'wire');
    if (wires.length > 0) {
      const wireGroupId = `wire-group:${body.name}`;
      tree.addNode({
        id: wireGroupId,
        kind: NodeKind.Component,
        label: `Wires (${wires.length})`,
        children: [],
        hidden: true,
        parentId: bodyId,
        meshIds: [],
      });
      bodyChildren.push(wireGroupId);
    }

    tree.addNode({
      id: bodyId,
      kind: NodeKind.Body,
      label: body.name,
      children: bodyChildren,
      hidden: false,
      parentId: parentId,
      meshIds: [],
    });

    return bodyId;
  }

  // Build tree structure based on assemblies or flat body list
  if (manifest.assemblies && manifest.assemblies.length > 0) {
    function buildAssembly(asm: ManifestAssembly, parentId: string): string {
      const asmId = `assembly:${asm.path || asm.name}`;
      const asmChildren: string[] = [];

      // Sub-assemblies
      for (const sub of asm.sub_assemblies) {
        const subId = buildAssembly(sub, asmId);
        asmChildren.push(subId);
      }

      // Bodies in this assembly — find root bodies (no parent or parent outside assembly)
      const asmBodySet = new Set(asm.bodies);
      const rootBodies = asm.bodies
        .map((name) => bodiesByName[name])
        .filter((b) => b && (b.parent === null || !asmBodySet.has(b.parent)));

      for (const body of rootBodies) {
        const bodyId = buildBody(body, asmId);
        asmChildren.push(bodyId);
      }

      tree.addNode({
        id: asmId,
        kind: NodeKind.Assembly,
        label: asm.name,
        children: asmChildren,
        hidden: false,
        parentId: parentId,
        meshIds: [],
      });

      return asmId;
    }

    for (const asm of manifest.assemblies) {
      const asmId = buildAssembly(asm, rootId);
      rootChildren.push(asmId);
    }
  } else {
    // No assemblies — find root body (parent === null)
    const rootBody = manifest.bodies.find((b) => b.parent === null);
    if (rootBody) {
      const bodyId = buildBody(rootBody, rootId);
      rootChildren.push(bodyId);
    }
  }

  return tree;
}
