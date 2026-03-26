# Design Viewer — Three.js Bot Viewer with Per-Mesh Visibility

**Date:** 2026-03-26
**Status:** Draft
**Replaces:** Component tree workstream (exp/260324-component-tree), component browser

## Problem

The bot viewer uses MuJoCo WASM for rendering, which limits visibility to per-body (can't hide individual components/geoms). The component browser shows individual components with design layers but can't show the assembled bot. These are two halves of the same need.

## Design

### Two Viewer Tabs

| Tab | Renderer | Purpose |
|-----|----------|---------|
| **Design** (default) | Three.js, no MuJoCo | Component inspection, per-mesh visibility, materials, design layers, wiring, fasteners |
| **Sim** | MuJoCo WASM | Joint control, IK, physics, controller testing |

Both tabs share a single `Viewport3D` instance (scene, renderer, camera, controls). Tab switching swaps the scene content but preserves camera position/orientation. MuJoCo's animation loop pauses when Sim is not active.

The Design tab replaces both the current bot viewer's Explore mode AND the standalone component browser. When `?bot=X`, it shows the full bot. When `?component=X`, it shows a single component with its design layers.

### Manifest Enhancements

The manifest currently lacks positioning data for bodies and some parts. The Design Viewer needs complete transforms to place every mesh without MuJoCo.

**Add to manifest:**
- **Bodies:** `pos` and `quat` fields, derived from `body.world_pos` and `body.world_quat` (already computed by `bot.solve()`)
- **Component parts (`comp_*`):** Add `quat` (currently only `pos` is emitted)
- **Horn parts:** Add `pos` and `quat`
- **Fastener parts:** Add `pos` and `quat` derived from mounting point positions/axes in body-local frame
- **Design layers:** Bracket, coupler, clearance meshes are emitted in body-local frame (same as the body they attach to). Position using the parent body's world transform.

**New API endpoint:** `/api/bots/{bot}/viewer_manifest` — calls `emit_viewer_manifest()` on demand and returns JSON. No static file generation.

### Design Viewer Architecture

**Loading flow:**
1. Fetch manifest from `/api/bots/{bot}/viewer_manifest`
2. Build `SceneTree` from manifest (new code — see Data Model section)
3. For each body: fetch STL from `/api/bots/{bot}/meshes/{body_name}.stl`, position using body's `pos`/`quat`
4. For each component part: fetch component mesh(es), position using part's `pos`/`quat` relative to parent body
5. For multi-material components: fetch per-material STLs, position same as parent component
6. For servo joints: fetch servo + horn meshes, position using joint's `pos`/`quat`
7. Design layers (lazy): fetch bracket/coupler/clearance/insertion_channel STLs on demand when user toggles visible

**Each part = its own Three.js Mesh** with independent visibility. No body-level grouping constraint.

### Server Mesh Endpoints

The server's `_generate_bot_mesh()` already handles: body solids, `servo_*`, `horn_*`, `hardware_*`, `wire_*`, and multi-material `__` stems.

**Add handlers for design layers:**
- `bracket_{joint}` — bracket solid for a joint's servo
- `coupler_{joint}` — coupler solid
- `clearance_{joint}` — clearance envelope solid
- `insertion_{joint}` — insertion channel solid

Design layer meshes are generated in body-local frame (same coordinate system as the parent body mesh). The viewer positions them using the parent body's world transform — no additional offset needed.

### Data Model

**New code** (not yet on master — partially exists on component tree branch, will be rewritten):

```typescript
enum NodeKind {
    Robot, Assembly, Body, Component, SubPart, Joint, Layer
}

interface SceneNode {
    id: string;
    kind: NodeKind;
    label: string;
    children: string[];
    hidden: boolean;        // user's explicit toggle
    parentId: string | null;
    meshIds: string[];      // Three.js mesh UUIDs this node controls
}

class SceneTree {
    nodes: Map<string, SceneNode>;
    soloedId: string | null;  // right-click solo (replaces additive isolate)
    resolveVisibility(nodeId: string): boolean;
    toggleHidden(nodeId: string): void;
    solo(nodeId: string): void;
    unsolo(): void;
}
```

**DesignScene** — new data model for the Design tab (separate from `BotScene`):

```typescript
class DesignScene {
    tree: SceneTree;
    meshes: Map<string, THREE.Mesh>;  // meshId → Three.js mesh
    // Per-mesh state
    setMeshVisible(meshId: string, visible: boolean): void;
    getMeshVisible(meshId: string): boolean;
}
```

`BotScene` continues to exist unchanged for the Sim tab with body-level visibility.

### Component Tree UI

**Right-aligned ● / ○ toggles:**
- Filled circle (●) = visible, empty circle (○) = hidden
- Click to toggle, cascades to children
- Unhiding parent restores children to previous state

**Right-click Solo:**
- Right-click a node → hide everything except this subtree
- Right-click again or Escape → unsolo

**Design layers as children:**
Components with design layers show them as collapsible children, hidden by default:

```
STS3215 @ left_wheel                          ●
  ├── Bracket                                 ○  ← hidden, lazy-loaded
  ├── Coupler                                 ○
  ├── Clearance Envelope                      ○
  ├── Horn                                    ●
  └── 6× M3 SHC                              ●
```

### Component Browser Consolidation

The `?component=X` route currently loads `component-browser.ts` (~1400 lines). The Design Viewer replaces it.

**Features preserved in Design tab's component mode:**
- Layer toggles (body, bracket, coupler, clearance, insertion channel, fasteners, horn)
- Material rendering (multi-material support from workstream 1)
- Section cutter
- Camera focus/orbit

**Deferred to future work:**
- SVG overlay annotations (mounting points, wire ports as labeled circles)
- Measurement tool
- Technical drawing links

### Sim Tab

The existing MuJoCo viewer (`bot-viewer.ts`) becomes the Sim tab. It keeps Joint mode, IK mode, Assembly mode. Body-level visibility is fine here.

When Sim tab is active, the MuJoCo animation loop runs. When Design tab is active, it pauses.

### Tab Switching

Single `Viewport3D`. Switching tabs:
1. Pause/resume MuJoCo animation loop
2. Swap scene content (Design scene ↔ MuJoCo scene)
3. Camera position preserved (shared camera object)
4. Component tree state (expanded/collapsed) preserved
5. Visibility state is per-tab (independent)

## Files

| File | Change |
|------|--------|
| `viewer/design-viewer.ts` | **New.** Three.js scene loader, mesh positioning, per-mesh visibility |
| `viewer/design-scene.ts` | **New.** `DesignScene` data model, `SceneTree`, `SceneNode`, `NodeKind` |
| `viewer/build-scene-tree.ts` | **New.** Build SceneTree from manifest, including design layer nodes |
| `viewer/viewer.ts` | Tab routing: Design (default) + Sim. Shared Viewport3D. |
| `viewer/component-tree.ts` | Right-aligned ● / ○ toggles, right-click Solo, design layer children |
| `viewer/bot-viewer.ts` | Becomes Sim tab — wrap in tab lifecycle (pause/resume) |
| `botcad/emit/viewer.py` | Add pos/quat to bodies, comp parts, horns, fasteners. Design layer metadata. |
| `mindsim/server.py` | Add `/api/bots/{bot}/viewer_manifest` endpoint. Add bracket/coupler/clearance mesh handlers. |

## Dependencies

- Materials system (merged)
- Wire/fastener viz (integrates after — independent)

## Non-Goals

- Physics simulation in Design tab
- Joint manipulation in Design tab
- Full cable routing visualization
- Assembly sequence animation
- SVG annotation overlays (deferred from component browser)
