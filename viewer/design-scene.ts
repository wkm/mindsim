/**
 * DesignScene — pure data model for the Design Viewer's scene tree.
 *
 * No Three.js imports in SceneTree/SceneNode. DesignScene bridges
 * the tree to Three.js meshes via meshIds and syncVisibility().
 *
 * Key concepts:
 *   - SceneTree owns all nodes and provides visibility resolution
 *   - Visibility cascades: hidden parent hides all descendants
 *   - Solo: one subtree visible at a time (ancestors included)
 *   - meshIds: each node maps to zero or more Three.js mesh UUIDs
 */

import type * as THREE from 'three';

// ---------------------------------------------------------------------------
// NodeKind — the type of each tree node
// ---------------------------------------------------------------------------

export enum NodeKind {
  Robot = 'Robot',
  Assembly = 'Assembly',
  Body = 'Body',
  Component = 'Component',
  SubPart = 'SubPart',
  Joint = 'Joint',
  Layer = 'Layer',
}

// ---------------------------------------------------------------------------
// SceneNode — a single entry in the tree
// ---------------------------------------------------------------------------

export interface SceneNode {
  id: string;
  kind: NodeKind;
  label: string;
  children: string[];
  hidden: boolean;
  parentId: string | null;
  meshIds: string[];
}

// ---------------------------------------------------------------------------
// SceneTree — the full tree with visibility resolution
// ---------------------------------------------------------------------------

export class SceneTree {
  nodes: Map<string, SceneNode> = new Map();
  soloedId: string | null = null;

  addNode(node: SceneNode): void {
    this.nodes.set(node.id, node);
  }

  getNode(id: string): SceneNode | undefined {
    return this.nodes.get(id);
  }

  toggleHidden(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      node.hidden = !node.hidden;
    }
  }

  solo(nodeId: string): void {
    this.soloedId = nodeId;
  }

  unsolo(): void {
    this.soloedId = null;
  }

  /**
   * Resolve whether a node should be visible.
   *
   * Rules (evaluated in order):
   *   1. Unknown node -> false
   *   2. If the node or any ancestor is hidden -> false
   *   3. If solo is active:
   *      - The soloed node and its descendants are visible
   *      - Ancestors of the soloed node are visible (path to root)
   *      - Everything else is hidden
   *   4. Otherwise -> true
   */
  resolveVisibility(nodeId: string): boolean {
    const node = this.nodes.get(nodeId);
    if (!node) return false;

    // Check hidden cascade: walk up to root
    if (this._isHiddenByAncestor(nodeId)) return false;

    // Check solo
    if (this.soloedId !== null) {
      return this._isInSoloSubtree(nodeId);
    }

    return true;
  }

  /** Check if this node or any ancestor has hidden=true. */
  private _isHiddenByAncestor(nodeId: string): boolean {
    let current: string | null = nodeId;
    while (current !== null) {
      const node = this.nodes.get(current);
      if (!node) return true;
      if (node.hidden) return true;
      current = node.parentId;
    }
    return false;
  }

  /**
   * Check if nodeId is part of the solo subtree:
   *   - Is the soloed node itself
   *   - Is a descendant of the soloed node
   *   - Is an ancestor of the soloed node (path to root)
   */
  private _isInSoloSubtree(nodeId: string): boolean {
    const soloId = this.soloedId!;

    // Is this node an ancestor of the soloed node?
    let current: string | null = soloId;
    while (current !== null) {
      if (current === nodeId) return true;
      const node = this.nodes.get(current);
      if (!node) break;
      current = node.parentId;
    }

    // Is this node a descendant of the soloed node?
    return this._isDescendantOf(nodeId, soloId);
  }

  /** Check if nodeId is a descendant of ancestorId (BFS). */
  private _isDescendantOf(nodeId: string, ancestorId: string): boolean {
    const queue = [ancestorId];
    while (queue.length > 0) {
      const current = queue.shift()!;
      const node = this.nodes.get(current);
      if (!node) continue;
      for (const childId of node.children) {
        if (childId === nodeId) return true;
        queue.push(childId);
      }
    }
    return false;
  }
}

// ---------------------------------------------------------------------------
// DesignScene — bridges SceneTree to Three.js meshes
// ---------------------------------------------------------------------------

export class DesignScene {
  tree: SceneTree;
  meshes: Map<string, THREE.Mesh> = new Map();
  meshToNode: Map<string, string> = new Map();

  constructor(tree: SceneTree) {
    this.tree = tree;
  }

  /** Register a Three.js mesh under a tree node. */
  registerMesh(nodeId: string, mesh: THREE.Mesh): void {
    const uuid = mesh.uuid;
    this.meshes.set(uuid, mesh);
    this.meshToNode.set(uuid, nodeId);

    const node = this.tree.getNode(nodeId);
    if (node && !node.meshIds.includes(uuid)) {
      node.meshIds.push(uuid);
    }
  }

  /** Apply tree visibility to all registered meshes. */
  syncVisibility(): void {
    for (const [uuid, mesh] of this.meshes) {
      const nodeId = this.meshToNode.get(uuid);
      if (nodeId) {
        mesh.visible = this.tree.resolveVisibility(nodeId);
      } else {
        mesh.visible = false;
      }
    }
  }
}
