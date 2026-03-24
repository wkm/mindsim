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
    Joint,       // kinematic joint connecting parent body to child body
    Layer,       // overlay (wires, fasteners) — used by workstream 3
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
│   ├── arm_shoulder (Joint) ──→ Wheel Assembly
│   └── Wheel Assembly (Assembly)
│       ├── wheel (Body)
│       ├── STS3215 @ arm_wrist (Component)
│       │   ├── Bracket (SubPart)
│       │   └── Fasteners (SubPart)
│       └── Pololu Wheel (Component)
```

Joint nodes appear in the tree to preserve kinematic chain visibility. They show which joint connects a parent assembly to a child assembly, and provide access to joint properties (range, servo specs) in the properties panel.

Layer nodes (Wires, Fasteners) are deferred to workstream 3, which will define their aggregation and interaction semantics.

### Visibility Model

Two independent toggles per node. Both cascade to the full subtree.

**Hide (eye icon):** Toggle this subtree invisible/visible.
- Hiding an assembly hides all its children
- `hidden` represents the user's explicit toggle on that specific node, not effective visibility
- Unhiding a parent does not force-show children that were independently hidden — each node's `hidden` flag is independent
- Effective visibility is computed by `resolveVisibility()` which checks both the node's own `hidden` flag and all ancestors

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
    hidden: boolean;          // user's explicit toggle on this node
    parentId: string | null;
    bodyNames: string[];      // body names from manifest (empty for Assembly/Joint)
}
```

Note: `bodyNames` stores manifest body names (strings), not MuJoCo numeric IDs. The name → ID mapping is built at runtime by `ExploreMode._buildBodyNameMap()` after MuJoCo WASM loads.

**Dual model — SceneNode + BodyState:**

`BodyState[]` persists for per-body rendering state (`ghosted`, `hovered`, `selected`, opacity). `SceneNode.hidden` feeds into visibility computation that ultimately sets `BodyState.visible`. The relationship:

- `SceneNode` — tree structure + user visibility intent (hidden, isolation)
- `BodyState` — per-body render state (ghosted, hovered, selected, computed opacity)
- `resolveVisibility(nodeId)` reads SceneNode tree → writes BodyState.visible

**Visibility resolution helper:** `resolveVisibility` lives on a `SceneTree` helper class (not on `BotScene` directly) to preserve `BotScene` as a pure data model without tree-walking logic.

```typescript
class SceneTree {
    nodes: Map<string, SceneNode>;
    isolatedIds: Set<string>;
    resolveVisibility(nodeId: string): boolean;
    resolveBodyVisibility(bodyName: string): boolean;
}
```

`BotScene` gains a `tree: SceneTree` field. The existing `BotScene.bodies: BodyState[]` remains for render state.

### Tree Interactions

| Action | Behavior |
|--------|----------|
| Click node | Select, show properties panel |
| Double-click node | Focus camera on node (existing behavior, preserved) |
| Click eye | Toggle `hidden` on this node |
| Shift-click target | Add/remove from isolation set |
| Click target (no shift) | Replace isolation set with this node |
| Expand/collapse | Assembly, Component, and Joint nodes are collapsible |
| Search | Filter by name, highlights matching nodes |
| Category chips | Filter by NodeKind |

### Sync Algorithm

`scene-sync.ts` sync function:
1. For each body name in the system, call `sceneTree.resolveBodyVisibility(bodyName)` to get effective visibility from the node tree
2. If effectively hidden → `bodyOpacity = 0`, `mesh.visible = false`
3. If visible, apply existing ghosted/hovered/selected logic from `BodyState`:
   - Ghosted → `opacity = GHOST_OPACITY`
   - Hovered → emissive highlight on structural meshes
   - Selected → emissive highlight (stronger)
4. Clone materials on first sync (existing behavior, preserved)

### Manifest Changes

The Python manifest emitter (`botcad/emit/viewer.py`) adds a `node_kind` field to each entry in the existing `bodies`, `joints`, `parts`, and `assemblies` arrays. Values map directly to `NodeKind` enum strings. The viewer builds the `SceneTree` from these existing arrays + kind annotations — no new top-level manifest structure needed.

### Migration from Current Model

- `BotScene.bodies[]` visibility state → `SceneNode.hidden` + `SceneTree.resolveVisibility()`
- `BotScene._isolatedIds` (body IDs) → `SceneTree.isolatedIds` (node IDs)
- `ExploreMode` callbacks update `SceneTree` nodes instead of `BodyState` directly
- `ComponentTree` renders nodes by `NodeKind` with appropriate icons and affordances

## Files Changed

| File | Change |
|------|--------|
| `viewer/bot-scene.ts` | `SceneNode`, `NodeKind` enum, `SceneTree` helper, `BotScene.tree` field |
| `viewer/component-tree.ts` | Render by NodeKind, cascading hide, additive isolate, Joint nodes |
| `viewer/scene-sync.ts` | Use `SceneTree.resolveBodyVisibility()` in sync loop |
| `viewer/explore-mode.ts` | Wire callbacks to SceneTree, preserve double-click focus |
| `botcad/emit/viewer.py` | Add `node_kind` to manifest entries |

## Dependencies

None — fully independent. Workstream 3 uses Layer nodes from this tree model.
