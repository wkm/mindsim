/**
 * Unit tests for DesignScene — the pure data model for the Design Viewer.
 *
 * Run with: pnpm exec tsx --test viewer/tests/design-scene.test.ts
 */

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { NodeKind, SceneTree } from '../design-scene.ts';

// ---------------------------------------------------------------------------
// Helper: build a small tree
//   root (Robot)
//     body (Body)
//       comp1 (Component)
//         layer1 (Layer, hidden by default)
//       comp2 (Component)
// ---------------------------------------------------------------------------

function buildTestTree(): SceneTree {
  const tree = new SceneTree();
  tree.addNode({
    id: 'root',
    kind: NodeKind.Robot,
    label: 'TestBot',
    children: ['body'],
    hidden: false,
    parentId: null,
    meshIds: [],
  });
  tree.addNode({
    id: 'body',
    kind: NodeKind.Body,
    label: 'base',
    children: ['comp1', 'comp2'],
    hidden: false,
    parentId: 'root',
    meshIds: ['mesh-body-1', 'mesh-body-2'],
  });
  tree.addNode({
    id: 'comp1',
    kind: NodeKind.Component,
    label: 'Camera',
    children: ['layer1'],
    hidden: false,
    parentId: 'body',
    meshIds: ['mesh-comp1'],
  });
  tree.addNode({
    id: 'comp2',
    kind: NodeKind.Component,
    label: 'Battery',
    children: [],
    hidden: false,
    parentId: 'body',
    meshIds: ['mesh-comp2'],
  });
  tree.addNode({
    id: 'layer1',
    kind: NodeKind.Layer,
    label: 'Bracket',
    children: [],
    hidden: true,
    parentId: 'comp1',
    meshIds: ['mesh-layer1'],
  });
  return tree;
}

// ---------------------------------------------------------------------------
// Basics
// ---------------------------------------------------------------------------

describe('SceneTree basics', () => {
  it('node is visible by default', () => {
    const tree = buildTestTree();
    assert.equal(tree.resolveVisibility('body'), true);
    assert.equal(tree.resolveVisibility('comp1'), true);
    assert.equal(tree.resolveVisibility('comp2'), true);
  });

  it('hidden node is not visible', () => {
    const tree = buildTestTree();
    // layer1 is hidden: true by construction
    assert.equal(tree.resolveVisibility('layer1'), false);
  });

  it('unknown node returns false', () => {
    const tree = buildTestTree();
    assert.equal(tree.resolveVisibility('nonexistent'), false);
  });
});

// ---------------------------------------------------------------------------
// Cascading hide
// ---------------------------------------------------------------------------

describe('SceneTree cascading hide', () => {
  it('hiding parent makes children invisible', () => {
    const tree = buildTestTree();
    tree.toggleHidden('body');
    assert.equal(tree.resolveVisibility('body'), false);
    assert.equal(tree.resolveVisibility('comp1'), false);
    assert.equal(tree.resolveVisibility('comp2'), false);
  });

  it('unhiding parent restores children', () => {
    const tree = buildTestTree();
    tree.toggleHidden('body'); // hide
    tree.toggleHidden('body'); // unhide
    assert.equal(tree.resolveVisibility('body'), true);
    assert.equal(tree.resolveVisibility('comp1'), true);
    assert.equal(tree.resolveVisibility('comp2'), true);
  });

  it('independently hidden child stays hidden after parent unhide', () => {
    const tree = buildTestTree();
    tree.toggleHidden('comp1'); // hide comp1
    tree.toggleHidden('body'); // hide body
    tree.toggleHidden('body'); // unhide body
    // comp1 was independently hidden, should stay hidden
    assert.equal(tree.resolveVisibility('comp1'), false);
    assert.equal(tree.resolveVisibility('comp2'), true);
  });
});

// ---------------------------------------------------------------------------
// Solo
// ---------------------------------------------------------------------------

describe('SceneTree solo', () => {
  it('solo shows only the soloed subtree', () => {
    const tree = buildTestTree();
    tree.solo('comp1');
    assert.equal(tree.resolveVisibility('comp1'), true);
    assert.equal(tree.resolveVisibility('comp2'), false);
  });

  it('solo shows ancestors on the path to root', () => {
    const tree = buildTestTree();
    tree.solo('comp1');
    assert.equal(tree.resolveVisibility('root'), true);
    assert.equal(tree.resolveVisibility('body'), true);
    assert.equal(tree.resolveVisibility('comp1'), true);
  });

  it('solo hides siblings of soloed node', () => {
    const tree = buildTestTree();
    tree.solo('comp1');
    assert.equal(tree.resolveVisibility('comp2'), false);
  });

  it('unsolo restores all visibility', () => {
    const tree = buildTestTree();
    tree.solo('comp1');
    tree.unsolo();
    assert.equal(tree.resolveVisibility('comp1'), true);
    assert.equal(tree.resolveVisibility('comp2'), true);
    assert.equal(tree.resolveVisibility('body'), true);
  });
});

// ---------------------------------------------------------------------------
// Combined
// ---------------------------------------------------------------------------

describe('SceneTree combined', () => {
  it('hidden node inside soloed subtree stays hidden', () => {
    const tree = buildTestTree();
    // layer1 is hidden by default, comp1 subtree includes layer1
    tree.solo('comp1');
    assert.equal(tree.resolveVisibility('comp1'), true);
    assert.equal(tree.resolveVisibility('layer1'), false); // still hidden
  });
});

// ---------------------------------------------------------------------------
// meshIds
// ---------------------------------------------------------------------------

describe('SceneTree meshIds', () => {
  it('resolveVisibility works with meshIds mapping', () => {
    const tree = buildTestTree();
    // Verify node has meshIds and visibility works for nodes with meshes
    const bodyNode = tree.getNode('body');
    assert.ok(bodyNode);
    assert.deepEqual(bodyNode.meshIds, ['mesh-body-1', 'mesh-body-2']);
    assert.equal(tree.resolveVisibility('body'), true);

    tree.toggleHidden('body');
    assert.equal(tree.resolveVisibility('body'), false);
    // meshIds are still on the node — visibility is resolved separately
    assert.deepEqual(bodyNode.meshIds, ['mesh-body-1', 'mesh-body-2']);
  });
});
