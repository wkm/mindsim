# Component Tree: Assemblies, Components & Visibility

**Date:** 2026-03-24
**Status:** Approved
**Workstream:** 2 of 3 (independent, parallel)

## Problem

The component tree mixes bodies, assemblies, servos, fasteners, and wires without clear semantic distinction. Users can't tell what's a designed assembly vs a leaf component. Hide doesn't cascade to children. Isolate UX is confusing — unclear what's isolated and why.

## Design

### Node Kinds (enum)

Every tree node has a kind from a closed enum:

```typescript
enum NodeKind {
    Robot,       // root node
    Assembly,    // group of bodies/components (no geometry)
    Body,        // fabricated/purchased physical part with mesh
    Component,   // leaf part mounted on a body (servo, camera, battery)
    SubPart,     // child of component (bracket, coupler, horn)
    Layer,       // overlay (wires, fasteners)
}
```

### Tree Structure

```
Wheeler Arm (Robot)
├── Base Assembly (Assembly)
│   ├── base (Body)
│   ├── Pi 5 (Component)
│   ├── Camera (Component)
│   └── Battery (Component)
├── Arm Assembly (Assembly)
│   ├── arm (Body)
│   ├── STS3215 @ arm_shoulder (Component)
│   │   ├── Bracket (SubPart)
│   │   ├── Coupler (SubPart)
│   │   └── Fasteners (SubPart)
│   └── Wheel Assembly (Assembly)
│       ├── wheel (Body)
│       ├── STS3215 @ arm_wrist (Component)
│       └── Pololu Wheel (Component)
├── Wires (Layer)
└── Fasteners (Layer)
```

### Visibility Model

Two independent toggles per node. Both cascade to the full subtree.

**Hide (eye icon):** Toggle this subtree invisible/visible.
- Hiding an assembly hides all its children
- Unhiding restores children to their previous state (not force-show)
- Stored as `hidden: boolean` per node

**Isolate (target icon):** Show only this subtree, dim/hide everything else.
- Additive: shift-click isolate on a second node adds it to the isolated set
- Click isolate on an already-isolated node removes it from the set
- When isolation set is empty, everything is shown
- Stored as a `Set<string>` of isolated node IDs on the scene

**Resolution:** A node is visible when:
1. It is not hidden, AND
2. None of its ancestors are hidden, AND
3. Either no isolation is active, OR it (or an ancestor) is in the isolated set

### Data Model

```typescript
interface SceneNode {
    id: string;
    kind: NodeKind;
    label: string;
    children: string[];
    hidden: boolean;       // user toggled hide
    parentId: string | null;
    bodyIds: number[];     // MuJoCo body IDs this node maps to (empty for Assembly)
}
```

`BotScene` changes:
- `nodes: Map<string, SceneNode>` — replaces flat `bodies` array for visibility
- `isolatedIds: Set<string>` — node IDs (not body IDs)
- `resolveVisibility(nodeId): boolean` — walks ancestors for hide + isolation
- Body-level opacity computation derives from node visibility

### Tree Interactions

| Action | Behavior |
|--------|----------|
| Click node | Select, show properties panel |
| Click eye | Toggle hide on this subtree |
| Click target | Toggle isolate (additive with shift) |
| Expand/collapse | Assembly and component nodes are collapsible |
| Search | Filter by name, highlights matching nodes |
| Category chips | Filter by NodeKind |

### Sync

`scene-sync.ts` iterates `SceneNode` tree instead of flat body list. For each node with `bodyIds`, computes effective visibility and applies to Three.js meshes (opacity, visible flag).

### Migration from Current Model

- Current `BotScene.bodies[]` visibility state moves to `SceneNode.hidden`
- Current `BotScene._isolatedIds` (body IDs) moves to `BotScene.isolatedIds` (node IDs)
- `ExploreMode` callbacks update nodes instead of bodies
- Manifest generation in Python adds `nodeKind` to each entry

## Files Changed

| File | Change |
|------|--------|
| `viewer/bot-scene.ts` | `SceneNode` model, `NodeKind` enum, node-based visibility |
| `viewer/component-tree.ts` | Render by NodeKind, cascading hide, additive isolate |
| `viewer/scene-sync.ts` | Node-tree traversal for visibility resolution |
| `viewer/explore-mode.ts` | Wire callbacks to node model |
| `botcad/emit/viewer.py` | Emit node kinds in manifest |

## Dependencies

None — fully independent. Workstream 3 uses Layer nodes from this tree model.
