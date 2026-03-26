/**
 * buildSceneTree -- constructs a SceneTree from a viewer manifest JSON.
 *
 * Hierarchy:
 *   Robot (root)
 *     Assembly (auto-generated from kinematic structure)
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
 * meshIds are empty at build time -- populated when meshes are loaded.
 */

import { NodeKind, SceneTree } from './design-scene.ts';
import type { ManifestBody, ManifestJoint, ViewerManifest } from './manifest-types.ts';
import { indexManifest } from './manifest-types.ts';
import { groupFasteners, humanizeJointName } from './utils.ts';

// ---------------------------------------------------------------------------
// buildSceneTree
// ---------------------------------------------------------------------------

export function buildSceneTree(manifest: ViewerManifest): SceneTree {
  const tree = new SceneTree();
  const idx = indexManifest(manifest);

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

  // Build a flat body node (no joint/servo children -- those go in sub-assemblies)
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

  // Build mounted component nodes for a body from mounts[], adding them as children of parentId
  function buildMountedComponents(body: ManifestBody, parentId: string, parentChildren: string[]): void {
    const bodyMounts = idx.mountsByBody[body.name] || [];
    const bodyParts = idx.partsByBody[body.name] || [];
    const mountFasteners = bodyParts.filter((p) => p.category === 'fastener' && !p.joint);

    for (const mount of bodyMounts) {
      const compId = `mount:${mount.body}:${mount.label}`;
      const compChildren: string[] = [];

      const compFasteners = mountFasteners.filter((f) => f.mount_label === mount.label);
      const grouped = groupFasteners(compFasteners);
      for (const group of grouped) {
        const fId = `fastener-group:mount:${mount.body}:${mount.label}:${group.key}`;
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
        label: mount.component,
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
    const bodyParts = idx.partsByBody[body.name] || [];
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

    // Servo and horn component bodies from bodies[] (role="component")
    const servos = idx.servosByJoint[joint.name] || [];
    const horns = idx.hornsByJoint[joint.name] || [];
    // Fasteners still come from parts[]
    const jointParts = idx.partsByJoint[joint.name] || [];
    const fasteners = jointParts.filter((p) => p.category === 'fastener');

    for (const servo of servos) {
      const servoId = `body:${servo.name}`;
      const servoChildren: string[] = [];

      for (const horn of horns) {
        const hornId = `body:${horn.name}`;
        tree.addNode({
          id: hornId,
          kind: NodeKind.SubPart,
          label: horn.component ?? 'Horn disc',
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
        label: `${servo.component ?? servo.name} @ ${joint.name}`,
        children: servoChildren,
        hidden: false,
        parentId: asmId,
        meshIds: [],
      });
      asmChildren.push(servoId);
    }

    // Child body
    const childBody = idx.bodiesByName[joint.child_body];
    if (childBody) {
      const childBodyId = buildBodyNode(childBody, asmId);
      asmChildren.push(childBodyId);

      // Components mounted on child body
      buildMountedComponents(childBody, asmId, asmChildren);

      // Recurse: joints from the child body create nested sub-assemblies
      const childJoints = idx.childJointsOf[childBody.name] || [];
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
    const joints = idx.childJointsOf[rootBody.name] || [];
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

  // Auto-generate assemblies from kinematic structure
  const rootBody = manifest.bodies.find((b) => b.parent === null && b.role === 'structure');
  if (rootBody) {
    const asmId = buildAutoAssemblyTree(rootBody);
    rootChildren.push(asmId);
  }

  return tree;
}
