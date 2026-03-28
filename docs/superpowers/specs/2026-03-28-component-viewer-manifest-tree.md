# Component Viewer: Manifest + Tree Unification

**Date:** 2026-03-28
**Status:** Draft
**Branch:** `exp/260327-fix-component-geometry` (builds on prior component geometry work)

## Problem

The component viewer (`/viewer/?component=X`) has its own ad-hoc layer system
(`LAYER_META`, `_loadLayer`, checkbox toggles) that duplicates functionality already
in the bot viewer's `ComponentTree`. This means:

- Two code paths for visibility toggles, solo mode, mesh loading
- The component viewer lacks search, category filters, and the tree hierarchy
- Adding new layer types (e.g., reference images) requires changes in both viewers

## Solution

Replace the component viewer's layer system with `ComponentTree`, driven by a
server-generated mini-manifest in the same `ViewerManifest` shape used by the
bot viewer.

## Design

### 1. Server: `/api/components/{name}/manifest`

Returns a `ViewerManifest`-compatible JSON for a single component. Field names
match `manifest-types.ts` exactly:

```json
{
  "bot_name": "BEC5V",
  "bodies": [
    {
      "name": "BEC5V",
      "parent": null,
      "role": "structure",
      "mesh": "body",
      "pos": [0, 0, 0],
      "quat": [1, 0, 0, 0]
    }
  ],
  "mounts": [
    {
      "body": "BEC5V",
      "label": "BEC5V",
      "component": "BEC5V",
      "category": "component",
      "mesh": "body",
      "pos": [0, 0, 0],
      "quat": [1, 0, 0, 0],
      "meshes": [
        {"file": "mat__fr4_green", "material": "fr4_green"},
        {"file": "mat__ic_package", "material": "ic_package"},
        {"file": "mat__nickel", "material": "nickel"}
      ]
    }
  ],
  "parts": [
    {
      "id": "fastener_BEC5V_m1",
      "name": "M2 screw",
      "category": "fastener",
      "parent_body": "BEC5V",
      "mesh": "_screw_0.0020",
      "pos": [-0.004, -0.004, 0],
      "quat": [1, 0, 0, 0]
    }
  ],
  "materials": {
    "fr4_green": {"color": [0.2, 0.6, 0.2], "metallic": 0.0, "roughness": 0.85, "opacity": 1.0},
    "ic_package": {"color": [0.15, 0.15, 0.15], "metallic": 0.0, "roughness": 0.9, "opacity": 1.0},
    "nickel": {"color": [0.6, 0.6, 0.6], "metallic": 0.85, "roughness": 0.25, "opacity": 1.0}
  },
  "joints": [],
  "assemblies": []
}
```

Key points:
- `bot_name` set to component name (used by ComponentTree for assembly label)
- One body, one mount (self-referential), no joints/assemblies
- Multi-material meshes in `mounts[].meshes` (same shape as bot manifest)
- Fasteners and wires as `parts[]` with `id`, `name`, `category`, `parent_body`
- STL URLs derived from `mesh` via `/api/components/{name}/stl/{mesh}`
- Materials dict includes `opacity` field per `ManifestMaterial`

### 2. ComponentTree reuse

The component browser constructs a `ComponentTree` from the mini-manifest:

```typescript
const resp = await fetch(`/api/components/${name}/manifest`);
const manifest = await resp.json();
this.tree = new ComponentTree(container, manifest, onSelect, {
  onToggleNodeHidden: (id) => this.toggleMesh(id),
  onSolo: (id) => this.soloMesh(id),
  onUnsolo: () => this.unsoloAll(),
});
this.tree.build();
```

The tree renders:
- Body node (with multi-material sub-meshes if available)
- Fastener nodes (grouped by size, same as bot viewer)
- Wire/connector nodes
- (Future: reference image nodes)

**Default visibility:** Non-body nodes (fasteners, wires) start hidden. The
component browser fires hide callbacks after tree construction so the initial
view shows only the component body. Users toggle other parts on as needed.

**ShapeScript links:** ComponentTree generates ShapeScript debug URLs using
`bot_name`. For the component viewer these link to
`/api/components/{name}/shapescript/...` instead of bot-scoped URLs. The
component browser provides an `onShapeScript` callback that routes to the
component shapescript endpoint.

### 3. Mesh loading

Meshes load lazily on first visibility toggle:

- **Body/materials:** Fetched from `/api/components/{name}/stl/{mesh}`
- **Fasteners:** Fetched from screw STL URLs in `parts[]`
- **Wires/connectors:** Fetched from connector STL URLs in `parts[]`

The component browser manages Three.js groups keyed by node ID, same pattern
as `DesignViewer.sync()` in the bot viewer.

### 4. What gets removed

From `component-browser.ts`:
- `LAYER_META` constant and all layer metadata
- `_loadLayer()`, `_addSTLMesh()`, `_loadFasteners()`, `_loadWires()`
- Layer checkbox HTML builder in `_updateSidePanel()`
- `layerGroups` management (replaced by tree-driven mesh groups)

### 5. What stays unchanged

- Side panel specs section (dimensions, mass, servo/mounting/wire details)
- 3D markers for mounting points and wire ports (hover-highlight from side panel)
- Steps mode (ShapeScript debugger)
- Section plane and measurements
- Axis gizmo
- Quick Switcher navigation
- Existing STL/materials/fasteners/wires server endpoints (manifest references them)

**Section cap colors:** Currently derived from `LAYER_META`. After removal, cap
colors are derived from node metadata in the manifest (material color for body
meshes, category-based defaults for fasteners/wires). The component browser
provides a `setSectionCapColorFn` that maps node IDs to colors using manifest
data instead of `LAYER_META`.

### 6. Category filter chips

Left as-is in `ComponentTree`. For single-component views they're mostly
inert but harmless. May revisit later.

## Future: Reference images

With the tree in place, reference images become a natural extension:

- Component definition gains a `reference_images` field listing image files
  with axis and real-world dimensions
- Manifest includes reference image entries as additional tree nodes
- Component browser creates textured `PlaneGeometry` meshes aligned to the
  specified axis, scaled to real dimensions
- Each image is independently toggleable from the tree, starts hidden

This is out of scope for this spec — will be a follow-up.

## Testing

- Verify component viewer renders BEC5V, WaveshareSerialBus, RaspberryPiZero2W,
  STS3215 with correct tree structure
- Verify toggle/solo/unsolo works for all node types
- Verify multi-material rendering matches current behavior
- Verify fasteners and wires render at correct positions
- Verify 3D markers for mounting points and wire ports still work
- Verify section plane caps have correct colors
- Verify steps mode still works (orthogonal to tree)
- Verify quick switcher navigation between components still works
