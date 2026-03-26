/**
 * Unit tests for buildSceneTree -- validates that a manifest is correctly
 * transformed into a SceneTree hierarchy.
 *
 * Run with: pnpm exec tsx --test viewer/tests/build-scene-tree.test.ts
 */

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { buildSceneTree } from '../build-scene-tree.ts';
import { NodeKind } from '../design-scene.ts';
import type { ViewerManifest } from '../manifest-types.ts';

// ---------------------------------------------------------------------------
// Minimal test manifest
// ---------------------------------------------------------------------------

function makeTestManifest(): ViewerManifest {
  return {
    bot_name: 'test_bot',
    bodies: [
      // Root structural body
      {
        name: 'base',
        mesh: 'base.stl',
        role: 'structure',
        parent: null,
        pos: [0, 0, 0],
        quat: [1, 0, 0, 0],
      },
      // Child structural body (connected via joint)
      {
        name: 'arm',
        mesh: 'arm.stl',
        role: 'structure',
        parent: 'base',
        joint: 'shoulder',
        pos: [0, 0, 0.1],
        quat: [1, 0, 0, 0],
      },
      // Grandchild structural body
      {
        name: 'hand',
        mesh: 'hand.stl',
        role: 'structure',
        parent: 'arm',
        joint: 'wrist',
        pos: [0, 0, 0.2],
        quat: [1, 0, 0, 0],
      },
      // Servo component body for shoulder joint
      {
        name: 'servo_shoulder',
        mesh: 'servo_shoulder.stl',
        role: 'component',
        parent: 'base',
        category: 'servo',
        joint: 'shoulder',
        component: 'STS3215',
        pos: [0, 0, 0.05],
        quat: [1, 0, 0, 0],
      },
      // Horn component body
      {
        name: 'horn_shoulder',
        mesh: 'horn_shoulder.stl',
        role: 'component',
        parent: 'base',
        category: 'horn',
        joint: 'shoulder',
        component: 'Horn disc',
        pos: [0, 0, 0.06],
        quat: [1, 0, 0, 0],
      },
      // Servo for wrist
      {
        name: 'servo_wrist',
        mesh: 'servo_wrist.stl',
        role: 'component',
        parent: 'arm',
        category: 'servo',
        joint: 'wrist',
        component: 'STS3215',
        pos: [0, 0, 0.15],
        quat: [1, 0, 0, 0],
      },
    ],
    joints: [
      {
        name: 'shoulder',
        parent_body: 'base',
        child_body: 'arm',
        servo: 'STS3215',
        design_layers: [
          { kind: 'bracket', mesh: 'bracket_shoulder.stl', parent_body: 'base' },
          { kind: 'clearance', mesh: 'clearance_shoulder.stl', parent_body: 'base' },
        ],
      },
      {
        name: 'wrist',
        parent_body: 'arm',
        child_body: 'hand',
        servo: 'STS3215',
      },
    ],
    mounts: [
      {
        body: 'base',
        label: 'camera',
        component: 'OV5647',
        category: 'camera',
        mesh: 'comp_base_camera.stl',
        pos: [0, 0.02, 0.03],
        quat: [1, 0, 0, 0],
      },
      {
        body: 'base',
        label: 'battery',
        component: 'LiPo 2S',
        category: 'battery',
        mesh: 'comp_base_battery.stl',
        pos: [0, -0.02, 0],
        quat: [1, 0, 0, 0],
      },
    ],
    parts: [
      // Fasteners for shoulder joint
      {
        id: 'fastener_shoulder_ear_0',
        name: 'M2 SHC',
        category: 'fastener',
        parent_body: 'base',
        joint: 'shoulder',
        mesh: 'M2_SHC.stl',
        pos: [0.01, 0, 0.05],
        quat: [1, 0, 0, 0],
      },
      {
        id: 'fastener_shoulder_ear_1',
        name: 'M2 SHC',
        category: 'fastener',
        parent_body: 'base',
        joint: 'shoulder',
        mesh: 'M2_SHC.stl',
        pos: [-0.01, 0, 0.05],
        quat: [1, 0, 0, 0],
      },
      // Wire parts
      {
        id: 'wire_stub_base_camera_data',
        name: 'data',
        category: 'wire',
        wire_kind: 'stub',
        parent_body: 'base',
        mesh: 'wire_stub.stl',
        pos: [0, 0.02, 0.035],
        quat: [1, 0, 0, 0],
        color: [0.2, 0.6, 0.86, 1.0] as [number, number, number, number],
      },
      {
        id: 'wire_servo_bus_base_0',
        name: 'servo_bus',
        category: 'wire',
        parent_body: 'base',
        mesh: 'wire_servo_bus_base_0.stl',
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('buildSceneTree', () => {
  it('creates robot root node', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);
    const root = tree.getNode('robot:test_bot');
    assert.ok(root, 'root node should exist');
    assert.equal(root.kind, NodeKind.Robot);
    assert.equal(root.label, 'test_bot');
    assert.equal(root.parentId, null);
  });

  it('auto-generates assembly from kinematic structure', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);
    const asm = tree.getNode('assembly:test_bot');
    assert.ok(asm, 'top-level assembly should exist');
    assert.equal(asm.kind, NodeKind.Assembly);
    assert.equal(asm.parentId, 'robot:test_bot');
  });

  it('creates body nodes for structural bodies', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    const base = tree.getNode('body:base');
    assert.ok(base, 'base body node should exist');
    assert.equal(base.kind, NodeKind.Body);

    const arm = tree.getNode('body:arm');
    assert.ok(arm, 'arm body node should exist');
    assert.equal(arm.kind, NodeKind.Body);

    const hand = tree.getNode('body:hand');
    assert.ok(hand, 'hand body node should exist');
    assert.equal(hand.kind, NodeKind.Body);
  });

  it('creates component body nodes for servos', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    // Servo bodies become Component nodes inside joint assemblies
    const servoNode = tree.getNode('body:servo_shoulder');
    assert.ok(servoNode, 'servo_shoulder component node should exist');
    assert.equal(servoNode.kind, NodeKind.Component);
    // Should be parented under the joint assembly
    assert.equal(servoNode.parentId, 'assembly:joint:shoulder');
  });

  it('creates mount nodes for mounted components', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    const camera = tree.getNode('mount:base:camera');
    assert.ok(camera, 'camera mount node should exist');
    assert.equal(camera.kind, NodeKind.Component);
    assert.equal(camera.label, 'OV5647');

    const battery = tree.getNode('mount:base:battery');
    assert.ok(battery, 'battery mount node should exist');
    assert.equal(battery.kind, NodeKind.Component);
    assert.equal(battery.label, 'LiPo 2S');
  });

  it('creates wire group for wire parts', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    // base has 2 wire parts -> wire group should exist
    const wireGroup = tree.getNode('wire-group:base');
    assert.ok(wireGroup, 'wire group for base should exist');
    assert.equal(wireGroup.kind, NodeKind.Component);
    assert.ok(wireGroup.label.includes('Wires'));
  });

  it('creates fastener groups under joint assemblies', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    // Shoulder has 2x M2 SHC fasteners -> should be grouped
    const servoNode = tree.getNode('body:servo_shoulder');
    assert.ok(servoNode, 'servo node should exist');

    // Fastener group should be a child of the servo component
    const fastenerGroupId = servoNode.children.find((id) => id.startsWith('fastener-group:'));
    assert.ok(fastenerGroupId, 'servo should have a fastener group child');

    const fastenerGroup = tree.getNode(fastenerGroupId);
    assert.ok(fastenerGroup);
    assert.equal(fastenerGroup.kind, NodeKind.SubPart);
    // Label should indicate count: "2x M2 SHC"
    assert.ok(fastenerGroup.label.includes('M2 SHC'), `expected M2 SHC in label, got: ${fastenerGroup.label}`);
  });

  it('wire group starts visible', () => {
    // Regression test: wire groups should default to visible (hidden=false)
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    const wireGroup = tree.getNode('wire-group:base');
    assert.ok(wireGroup);
    assert.equal(wireGroup.hidden, false, 'wire group should start visible');
  });

  it('design layers are hidden by default', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    const bracketLayer = tree.getNode('layer:shoulder:bracket');
    assert.ok(bracketLayer, 'bracket layer should exist');
    assert.equal(bracketLayer.kind, NodeKind.Layer);
    assert.equal(bracketLayer.hidden, true, 'design layers should be hidden by default');
  });

  it('joint assemblies are named from joint', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    const shoulderAsm = tree.getNode('assembly:joint:shoulder');
    assert.ok(shoulderAsm, 'shoulder joint assembly should exist');
    assert.equal(shoulderAsm.kind, NodeKind.Assembly);
    // humanizeJointName("shoulder") -> "Shoulder"
    assert.equal(shoulderAsm.label, 'Shoulder');
  });

  it('horn bodies are SubPart children of servo Component', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    const hornNode = tree.getNode('body:horn_shoulder');
    assert.ok(hornNode, 'horn node should exist');
    assert.equal(hornNode.kind, NodeKind.SubPart);
    assert.equal(hornNode.parentId, 'body:servo_shoulder');
  });

  it('nested joints create nested assemblies', () => {
    const manifest = makeTestManifest();
    const tree = buildSceneTree(manifest);

    const wristAsm = tree.getNode('assembly:joint:wrist');
    assert.ok(wristAsm, 'wrist joint assembly should exist');
    // Wrist assembly should be nested under shoulder assembly
    assert.equal(wristAsm.parentId, 'assembly:joint:shoulder');
  });
});
