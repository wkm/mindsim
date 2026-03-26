# Design Viewer — Three.js Bot Viewer with Per-Mesh Visibility

**Date:** 2026-03-26
**Status:** Draft
**Replaces:** Component tree workstream (exp/260324-component-tree)

## Problem

The bot viewer uses MuJoCo WASM for rendering, which limits visibility to per-body (can't hide individual components/geoms). The component browser shows individual components with design layers but can't show the assembled bot. These are two halves of the same need.

## Design

### Two Viewer Tabs

| Tab | Renderer | Purpose |
|-----|----------|---------|
| **Design** (default) | Three.js, no MuJoCo | Component inspection, per-mesh visibility, materials, design layers, wiring, fasteners |
| **Sim** | MuJoCo WASM | Joint control, IK, physics, controller testing |

Both tabs share: camera controls, section cutter, component tree panel.

The Design tab replaces both the current bot viewer's Explore mode AND the standalone component browser. When `?bot=X`, it shows the full bot. When `?component=X`, it shows a single component with its design layers (effectively a "mini bot").

### Design Viewer Architecture

The Design Viewer loads meshes directly from the API — no MuJoCo involved.

**Loading flow:**
1. Fetch `viewer_manifest.json` from `/api/bots/{bot}/viewer_manifest`
2. Build `SceneTree` from manifest (reuse existing `buildSceneTree`)
3. For each body in the manifest:
   - Fetch body mesh STL from `/api/bots/{bot}/meshes/{body_name}.stl`
   - For each mounted component, fetch component mesh(es):
     - Single-material: `/api/bots/{bot}/meshes/{comp_id}.stl`
     - Multi-material: `/api/bots/{bot}/meshes/{comp_id}__{material}.stl` for each material
   - For each servo joint, fetch: servo mesh, horn mesh
   - Position each mesh using `pos` and `quat` from the manifest part entry
4. For design layers (hidden by default, loaded on demand):
   - Bracket: `/api/bots/{bot}/meshes/bracket_{joint}.stl`
   - Coupler: `/api/bots/{bot}/meshes/coupler_{joint}.stl`
   - Clearance: loaded when user toggles visibility
   - Fasteners: `/api/bots/{bot}/meshes/hardware_{spec}.stl`

**Key difference from MuJoCo viewer:** Each component/layer is its own Three.js Mesh with its own visibility flag. No body-level grouping constraint.

### Per-Mesh Visibility via SceneTree

Each `SceneNode` maps to one or more Three.js meshes (not MuJoCo bodies). The `bodyNames` field on SceneNode is replaced with `meshIds: string[]` — direct references to Three.js mesh objects.

**Visibility toggle (Fusion 360 style):**
- Right-aligned filled circle (●) = visible, empty circle (○) = hidden
- Click circle to toggle
- Cascades to children: hiding a parent hides all descendants
- Unhiding a parent restores children to their previous state (independently hidden children stay hidden)

**Solo (right-click):**
- Right-click a node → hide everything except this node's subtree
- Right-click again (or press Escape) → un-solo, restore previous state

**Design layers as tree children:**
Components that have design layers (servos with brackets/couplers/clearance) show them as collapsible children:

```
STS3215 @ left_wheel (Component)     ●
  ├── Bracket (Layer)                ●
  ├── Coupler (Layer)                ●
  ├── Clearance Envelope (Layer)     ○  ← hidden by default
  ├── Horn (SubPart)                 ●
  └── 6× M3 SHC (Fasteners)         ●
```

Design layers are hidden by default and loaded on-demand when the user toggles them visible (lazy STL fetch).

### Mesh Positioning

The manifest provides `pos` (position) and `quat` (quaternion) for each part entry. The Design Viewer uses these directly to place Three.js meshes — no MuJoCo transform pipeline.

For bodies: position at origin (body meshes are in body-local frame). Apply body's kinematic transform from the manifest's solved joint positions.

For components on bodies: position using the part's `pos` field (body-local offset).

For multi-material sub-meshes: same position as parent component (mount rotation already baked into STL).

### Sim Tab

The existing MuJoCo viewer becomes the Sim tab. It keeps:
- Joint mode (slider control)
- IK mode
- Assembly mode
- Physics simulation

Body-level visibility is fine in Sim (you're thinking in physics terms). The component tree in Sim mode falls back to the existing body-level visibility behavior.

### Tab Switching

Both tabs share the same component tree panel. When switching tabs:
- Camera position/orientation is preserved
- Component tree state (expanded/collapsed) is preserved
- Visibility state is per-tab (Design and Sim have independent visibility)

### Shared Infrastructure (reused from existing code)

- `SceneTree`, `NodeKind`, `SceneNode` — from component tree branch
- `buildSceneTree` — manifest → tree builder
- `ComponentTree` — DOM rendering with right-aligned indicators
- `FocusController` — camera focus on node
- `SectionCutter` — cross-section view
- Orbit controls, lighting, ground plane

## Mesh Loading Strategy

All meshes loaded on-demand from `/api/bots/{bot}/meshes/` endpoints. No static files.

**Eager load (on viewer init):**
- Body meshes (structural shells)
- Component meshes (servos, Pi, camera, etc.)
- Horn meshes

**Lazy load (when user toggles visible):**
- Design layers (bracket, coupler, clearance envelope, insertion channels)
- Fastener meshes
- Wire stub meshes

**Caching:** Browser HTTP cache handles repeat loads. Server caches generated solids in memory.

## Files Changed

| File | Change |
|------|--------|
| `viewer/design-viewer.ts` | **New.** Three.js scene loader, mesh positioning, per-mesh visibility |
| `viewer/viewer.ts` | Route to Design tab (default) or Sim tab |
| `viewer/bot-scene.ts` | SceneNode.meshIds replaces bodyNames for Design mode |
| `viewer/component-tree.ts` | Right-aligned ● / ○ toggles, right-click Solo, design layer children |
| `viewer/build-scene-tree.ts` | Add design layer nodes as children of components |
| `viewer/scene-sync.ts` | Design mode: per-mesh sync from SceneTree |
| `viewer/bot-viewer.ts` | Becomes Sim tab — minimal changes, just tab wrapper |
| `botcad/emit/viewer.py` | Ensure manifest has all positioning data (pos, quat per part) |

## Dependencies

- Materials system (merged — provides multi-material manifest + STL serving)
- Wire/fastener viz (can integrate after — independent workstream)

## Non-Goals

- Physics simulation in Design tab
- Joint manipulation in Design tab (that's Sim)
- Full cable routing visualization (future)
- Assembly sequence animation (future)
