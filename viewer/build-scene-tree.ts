/**
 * buildSceneTree — constructs a SceneTree from a viewer manifest JSON.
 *
 * Hierarchy:
 *   Robot (root)
 *     Assembly (from manifest.assemblies, or auto-generated from kinematic structure)
 *       Body
 *       Component (mounted parts: non-servo, non-horn, non-fastener)
 *         SubPart (fasteners grouped: "6x M3 SHC")
 *       Sub-Assembly (one per joint, containing servo + joint + child body)
 *         Component (servo)
 *           SubPart (horns)
 *           SubPart (fastener groups)
 *         Joint
 *         Body (child)
 *         Component (mounted on child body)
 *         [recursive sub-assemblies for child joints]
 *       Layer (design layers, hidden by default)
 *       Wire group (hidden by default)
 *
 * Bodies never contain sub-bodies. Assemblies are the containers.
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

  // Humanize a joint name: "left_wheel" → "Left Wheel"
  function humanizeJointName(name: string): string {
    return name
      .split('_')
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ');
  }

  // Build a flat body node (no joint/servo children — those go in sub-assemblies)
  function buildBodyNode(body: ManifestBody, parentId: string): string {
    const bodyId = `body:${body.name}`;
    tree.addNode({
      id: bodyId,
      kind: NodeKind.Body,
      label: body.name,
      children: [],
      hidden: false,
      parentId: parentId,
      meshIds: [],
    });
    return bodyId;
  }

  // Build mounted component nodes for a body, adding them as children of parentId
  function buildMountedComponents(body: ManifestBody, parentId: string, parentChildren: string[]): void {
    const bodyParts = partsByBody[body.name] || [];

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
        parentId: parentId,
        meshIds: [],
      });
      parentChildren.push(compId);
    }
  }

  // Build wire group for a body, adding it as a child of parentId
  function buildWireGroup(body: ManifestBody, parentId: string, parentChildren: string[]): void {
    const bodyParts = partsByBody[body.name] || [];
    const wires = bodyParts.filter((p) => p.category === 'wire');
    if (wires.length > 0) {
      const wireGroupId = `wire-group:${body.name}`;
      tree.addNode({
        id: wireGroupId,
        kind: NodeKind.Component,
        label: `Wires (${wires.length} segments)`,
        children: [],
        hidden: true,
        parentId: parentId,
        meshIds: [],
      });
      parentChildren.push(wireGroupId);
    }
  }

  // Build a sub-assembly from a joint: servo + joint node + child body + child components + recurse
  function buildJointAssembly(joint: ManifestJoint, parentId: string): string {
    const asmId = `assembly:joint:${joint.name}`;
    const asmChildren: string[] = [];

    // Servo component with horn/fastener sub-parts
    const jointParts = partsByJoint[joint.name] || [];
    const servos = jointParts.filter((p) => p.category === 'servo');
    const horns = jointParts.filter((p) => p.category === 'horn');
    const fasteners = jointParts.filter((p) => p.category === 'fastener');

    for (const servo of servos) {
      const servoId = `part:${servo.id}`;
      const servoChildren: string[] = [];

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
        parentId: asmId,
        meshIds: [],
      });
      asmChildren.push(servoId);
    }

    // Joint node
    const jointId = `joint:${joint.name}`;
    tree.addNode({
      id: jointId,
      kind: NodeKind.Joint,
      label: joint.name,
      children: [],
      hidden: false,
      parentId: asmId,
      meshIds: [],
    });
    asmChildren.push(jointId);

    // Child body
    const childBody = bodiesByName[joint.child_body];
    if (childBody) {
      const childBodyId = buildBodyNode(childBody, asmId);
      asmChildren.push(childBodyId);

      // Components mounted on child body
      buildMountedComponents(childBody, asmId, asmChildren);

      // Recurse: joints from the child body create nested sub-assemblies
      const childJoints = childJointsOf[childBody.name] || [];
      for (const childJoint of childJoints) {
        const subAsmId = buildJointAssembly(childJoint, asmId);
        asmChildren.push(subAsmId);
      }

      // Wire group for child body
      buildWireGroup(childBody, asmId, asmChildren);
    }

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
          parentId: asmId,
          meshIds: [],
        });
        asmChildren.push(layerId);
      }
    }

    tree.addNode({
      id: asmId,
      kind: NodeKind.Assembly,
      label: humanizeJointName(joint.name),
      children: asmChildren,
      hidden: false,
      parentId: parentId,
      meshIds: [],
    });

    return asmId;
  }

  // Build auto-generated assembly tree from kinematic structure
  function buildAutoAssemblyTree(rootBody: ManifestBody): string {
    const asmId = `assembly:${manifest.bot_name}`;
    const asmChildren: string[] = [];

    // Root body
    const bodyId = buildBodyNode(rootBody, asmId);
    asmChildren.push(bodyId);

    // Components mounted on root body
    buildMountedComponents(rootBody, asmId, asmChildren);

    // Sub-assemblies from joints on root body
    const joints = childJointsOf[rootBody.name] || [];
    for (const joint of joints) {
      const subAsmId = buildJointAssembly(joint, asmId);
      asmChildren.push(subAsmId);
    }

    // Wire group for root body
    buildWireGroup(rootBody, asmId, asmChildren);

    tree.addNode({
      id: asmId,
      kind: NodeKind.Assembly,
      label: `${humanizeJointName(manifest.bot_name)} Assembly`,
      children: asmChildren,
      hidden: false,
      parentId: rootId,
      meshIds: [],
    });

    return asmId;
  }

  // Build tree structure based on assemblies or auto-generate from kinematic structure
  if (manifest.assemblies && manifest.assemblies.length > 0) {
    // Legacy path for bots with explicit assembly definitions
    // (kept for backward compatibility but uses old body-recursive approach)

    function buildBodyLegacy(body: ManifestBody, parentId: string): string {
      const bodyId = `body:${body.name}`;
      const bodyChildren: string[] = [];
      const bodyParts = partsByBody[body.name] || [];
      const joints = childJointsOf[body.name] || [];

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

      for (const joint of joints) {
        const jointParts = partsByJoint[joint.name] || [];
        const servos = jointParts.filter((p) => p.category === 'servo');
        const horns = jointParts.filter((p) => p.category === 'horn');
        const fasteners = jointParts.filter((p) => p.category === 'fastener');

        for (const servo of servos) {
          const servoId = `part:${servo.id}`;
          const servoChildren: string[] = [];
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

        const jointId = `joint:${joint.name}`;
        const jointChildren: string[] = [];
        const childBody = bodiesByName[joint.child_body];
        if (childBody) {
          const childBodyId = buildBodyLegacy(childBody, jointId);
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
      }

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

    function buildAssembly(asm: ManifestAssembly, parentId: string): string {
      const asmId = `assembly:${asm.path || asm.name}`;
      const asmChildren: string[] = [];

      for (const sub of asm.sub_assemblies) {
        const subId = buildAssembly(sub, asmId);
        asmChildren.push(subId);
      }

      const asmBodySet = new Set(asm.bodies);
      const rootBodies = asm.bodies
        .map((name) => bodiesByName[name])
        .filter((b) => b && (b.parent === null || !asmBodySet.has(b.parent)));

      for (const body of rootBodies) {
        const bodyId = buildBodyLegacy(body, asmId);
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
    // Auto-generate assemblies from kinematic structure
    const rootBody = manifest.bodies.find((b) => b.parent === null);
    if (rootBody) {
      const asmId = buildAutoAssemblyTree(rootBody);
      rootChildren.push(asmId);
    }
  }

  return tree;
}
