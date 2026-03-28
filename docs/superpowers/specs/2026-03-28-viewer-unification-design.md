# Viewer Unification

**Date:** 2026-03-28
**Status:** Draft
**Branch:** `exp/260328-component-viewer-tree` (builds on manifest+tree work)

## Problem

The bot viewer (`design-viewer.ts`) and component viewer (`component-browser.ts`)
implement the same pipeline independently: manifest → SceneTree → DesignScene →
mesh loading → ComponentTree → click-to-select. This means:

- Duplicated mesh loading, raycasting, visibility sync code
- Features added to one viewer (section plane, measure) don't exist in the other
- Inconsistent UX between the two viewing experiences
- A component is conceptually a "simple bot" but is treated as a different thing

## Solution

Extract a shared `ManifestViewer` that owns the complete viewing experience.
Both the bot page and component page become thin callers that fetch a manifest,
create a ManifestViewer, and provide context-specific callbacks.

## Design

### 1. Component hierarchy

```
viewer.ts (URL routing, layout)
    │
    ├── ?bot=X ──────────┐
    └── ?component=X ────┤
                         ▼
                  ManifestViewer
                  (manifest + viewport + tree + scene)
                  (section, measure, presets, click-select)
                  (onNodeSelected callback → context panel)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Bot Context           Component Context
        (sim tab, joints)     (specs, markers, steps)
              │                     │
              └──────────┬──────────┘
                         ▼
                    Viewport3D
                    (Three.js engine)
```

### 2. ManifestViewer

New module: `viewer/manifest-viewer.ts`

```typescript
interface ManifestViewerOptions {
  container: HTMLElement;       // Canvas container
  treePanelEl: HTMLElement;     // Tree panel DOM element
  sidePanelEl: HTMLElement;     // Side panel DOM element
  manifest: ViewerManifest;
  resolveStlUrl: (mesh: string, context: StlUrlContext) => string;
  onNodeSelected?: (nodeId: string | null, manifest: ViewerManifest) => void;
}

interface StlUrlContext {
  kind: 'body' | 'mount' | 'mount-material' | 'part';
  name: string;         // body name, mount label, or part id
  componentName: string; // the manifest's bot_name
}

interface ManifestViewerContext {
  scene: DesignScene;
  tree: ComponentTree;
  viewport: Viewport3D;
  syncVisibility(): void;
  dispose(): void;       // Removes all event listeners, clears meshes
}

function initManifestViewer(options: ManifestViewerOptions): Promise<ManifestViewerContext>
```

The function:
1. Creates `Viewport3D` (orthographic by default, grid enabled).
   Viewport3D already has a perspective/orthographic toggle button — no new
   code needed for camera switching.
2. Calls `buildSceneTree(manifest)` → creates `DesignScene`
3. Loads all meshes: bodies, mounts (multi-material), parts (fasteners, wires)
   - STL URLs resolved via `resolveStlUrl` callback
   - Design layer mounts lazy-loaded on first toggle
4. Builds `ComponentTree` in `treePanelEl` with all callbacks wired
5. Sets up shared tools: section plane (S-key), measure (M-key)
6. Sets up click-to-select raycasting → calls `onNodeSelected`
7. Sets default visibility (fasteners, wires, design layers hidden)
8. Derives section cap colors from manifest (body/mount/material colors)
   and registers with Viewport3D
9. Frames camera on visible geometry

`dispose()` must clean up:
- Click-to-select pointer event listeners
- Section plane / measure keyboard listeners
- ComponentTree (calls `.dispose()`)
- All mesh groups (geometry + material disposal via `viewport.clearGroup`)

### 3. STL URL resolution

The bot viewer and component viewer construct STL URLs differently:
- Bot: `/api/bots/{bot}/stl/{body}` for bodies, various paths for mounts/parts
- Component: `/api/components/{name}/stl/{mesh}` for component-local meshes,
  `/api/components/{syntheticName}/stl/body` for shared parts (_screw_*, etc.)

ManifestViewer takes a `resolveStlUrl` callback. Each context provides its own:

**Bot context:**
```typescript
(mesh, ctx) => `/api/bots/${botName}/stl/${mesh}`
```

**Component context:**
```typescript
(mesh, ctx) => {
  // Shared synthetic parts: _screw_*, _connector_*, _wire_stub
  if (mesh.startsWith('_')) return `/api/components/${mesh}/stl/body`;
  // Multi-material mesh files (mat__*) and design layer meshes
  return `/api/components/${compName}/stl/${mesh}`;
}
```

The `context` parameter provides `kind` (body/mount/mount-material/part) and
`name` for cases where callers need to vary URL logic by what's being loaded.

### 4. Shared tools (move from component-browser into ManifestViewer)

**Section plane:** S-key toggle, axis buttons (X/Y/Z), position slider, flip.
Currently only in component-browser.ts. Moves into ManifestViewer. Uses
Viewport3D's existing section cap infrastructure.

**Measure tool:** M-key toggle. Currently only in component-browser.ts.
Moves into ManifestViewer. Uses Viewport3D's existing MeasureTool.

**View presets:** 1-7 keys already handled by Viewport3D (`_initKeys`).
Both viewers already delegate to Viewport3D for this. ManifestViewer
does NOT add its own preset handling — it uses Viewport3D's built-in
key mappings. Note: Viewport3D's key order (1=Iso, 2=Front, 3=Top,
4=Right, 5=Back, 6=Bottom, 7=Left) differs slightly from the old
component-browser order. We adopt Viewport3D's order as canonical.

**Camera toggle:** Already exists in Viewport3D (`_switchCamera` with
ortho/perspective buttons in the overlay). Not new — ManifestViewer
just ensures the default is orthographic.

**Click-to-select:** Already in both viewers (duplicated). Consolidates
into ManifestViewer with `onNodeSelected` callback.

**Section cap colors:** ManifestViewer derives cap colors directly from
the manifest. Each mesh group already has an associated color (body color
from `ManifestBody.color`, mount color from `ManifestMount.color`,
material color from the materials dict). Section caps for a group match
that group's color — no callback needed, the manifest is the source of
truth.

### 5. Context panels (what changes per viewer mode)

ManifestViewer fires `onNodeSelected(nodeId, manifest)` when a node is
clicked (in 3D or in tree). The caller decides what to show in the side panel.

**Component context** (`?component=X`):
- On load: side panel shows component specs (dimensions, mass, servo data,
  mounting points, wire ports) — same as today
- 3D markers for mounting points and wire ports — same as today
- Steps mode: a separate per-geometry-body tool, NOT part of
  ManifestViewer. It's a ShapeScript debugger that swaps out the
  manifest view for a step-by-step CAD construction visualization
  (step slider, tool toggle, per-step STLs). It uses the same
  Viewport3D instance for rendering but is otherwise independent.
  The component context wrapper manages the toggle between manifest
  view and steps view.

**Bot context** (`?bot=X`):
- On component node selected: side panel shows that component's specs
  (fetched from `/api/components` catalog by component name from manifest)
- On body node selected: side panel shows body info (mass, dimensions)
- On joint node selected: side panel shows joint info (range, servo)
- 3D markers appear for the selected component's mounting points
- Steps mode available per-component (future)

### 6. What gets removed

**`component-browser.ts`** shrinks to ~150-200 lines:
- Fetch component catalog + manifest
- Call `initManifestViewer` with component-specific `resolveStlUrl`
- Provide `onNodeSelected` handler that populates specs panel + markers
- Steps mode (ShapeScript debugger) — stays here as a separate tool
  that shares the Viewport3D but swaps out the manifest view. The
  component context manages the toggle between the two views.

**From `design-viewer.ts`:** most of the function body moves into
`manifest-viewer.ts`. What remains:
- Bot-specific manifest fetching (`/api/bots/{bot}/viewer_manifest`)
- Bot-specific `resolveStlUrl` and `sectionCapColorFn`
- Bot-specific `onNodeSelected` handler
- Sim tab integration (MuJoCo viewer coordination)

**From `viewer.ts`:** The `?bot` and `?component` code paths converge.
Both show tree panel (left) + side panel (right) + canvas (center).
The `#component-browser` wrapper div is no longer needed — both paths
use the same `#tree-panel` + `#side-panel` + `#canvas-container` layout.
The mode tabs (Design/Sim) only appear for bots.

**Quick switcher** (`Cmd+K`): already global in viewer.ts, not owned
by component-browser.ts. Stays as-is.

### 7. Layout changes

Both bot and component views use the same panel layout:
- Tree panel (left, 280px) — always visible, shows ComponentTree
- Side panel (right, 320px) — contextual content based on selection
- Canvas (center) — ManifestViewer's Viewport3D

The component viewer currently hides the tree panel and puts the
ComponentTree inside the side panel. After unification, it uses the
same left-tree / right-side layout as the bot viewer. This requires
`viewer.ts` to stop treating the component path as a separate layout —
the `#component-browser` wrapper div goes away. Both `?bot` and
`?component` paths show/hide the same `#tree-panel`, `#side-panel`,
and `#canvas-container` elements.

### 8. What stays unchanged

- `ComponentTree` — no changes needed
- `DesignScene`, `SceneTree`, `buildSceneTree` — no changes
- `Viewport3D` — default camera changes to orthographic; existing
  perspective toggle, view presets, measure tool, section cap infra
  all stay as-is
- `manifest-types.ts` — no changes
- Server endpoints — no changes
- Component manifest shape — no changes

### 9. Behaviors to preserve

These component-browser features must survive unification:

- **Steps mode** (ShapeScript debugger): a separate per-geometry-body
  tool with step slider, tool toggle, per-step STLs. NOT part of
  ManifestViewer — it shares the Viewport3D but is toggled independently
  by the component context wrapper.
- **3D markers**: mounting point cylinders and wire port spheres with
  hover-highlight from the side panel. Stays in component context.
- **Axis gizmo**: SVG orientation indicator (top-right). Already in
  Viewport3D's orientation cube — verify it works for both modes.
- **Arrow key navigation**: component-browser has arrow-key orbit/pan.
  Viewport3D's OrbitControls already supports mouse orbit/pan. Arrow
  key nav is a component-browser addition — evaluate whether to move
  it into Viewport3D (available everywhere) or drop it.

## Testing

- Verify bot viewer works: `?bot=wheeler_arm` loads, tree works, click-select works
- Verify component viewer works: `?component=STS3215` loads, specs panel, markers, section
- Verify shared tools: section plane, measure, view presets work in both modes
- Verify camera: orthographic default, perspective toggle works in both modes
- Verify context panel: selecting a component node in bot view shows specs
- Verify lazy-load: design layers load on first toggle in both modes
- Verify steps mode: full in-viewport ShapeScript debugger works for components
- Verify section caps: correct colors in both modes
- Verify dispose: switching between bot and component doesn't leak listeners
- Verify quick switcher: Cmd+K navigates between components
