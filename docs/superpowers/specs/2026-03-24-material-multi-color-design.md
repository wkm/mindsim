# Material System & Multi-Color Components

**Date:** 2026-03-24
**Status:** Approved
**Workstream:** 1 of 3 (independent, parallel)

## Problem

Each component has a single `Appearance` (one RGBA color). A Pi board is flat green, a camera is flat dark gray, a wheel is flat gray. Real components have multiple materials — PCB substrate, IC packages, metal connectors, rubber tires, polycarbonate lenses.

Separately, `botcad/materials.py` tracks physical properties (density, print process) and `botcad/component.py` has `Appearance` (color, metallic, roughness). These are two halves of the same concept.

## Design

### Unified Material

Merge physical and visual properties into a single `Material` dataclass:

```python
@dataclass(frozen=True)
class Material:
    name: str                    # "fr4_green", "abs_black", "rubber_dark"
    # Visual
    color: RGBA
    metallic: float = 0.0
    roughness: float = 0.7
    # Physical
    density: float | None = None
    process: str | None = None   # "fdm", "purchased", etc.
```

### Material Catalog

Real-world materials with correct visual and physical properties:

- `MAT_FR4_GREEN` — PCB substrate (green, matte)
- `MAT_IC_PACKAGE` — chip epoxy (black, matte)
- `MAT_NICKEL` — connector pins, fastener plating (silver, metallic)
- `MAT_ABS_DARK` — servo housing, camera body (dark gray, slight sheen)
- `MAT_RUBBER` — tire surface (dark, high roughness)
- `MAT_POLYCARBONATE_CLEAR` — camera lens cover (dark, low roughness)
- `MAT_PLA_LIGHT` — 3D printed structural parts (light gray, matte)
- `MAT_STEEL` — fastener bodies (gray, metallic)
- `MAT_ALUMINUM` — servo horns, brackets (light gray, metallic)

### Material-Tagged ShapeScript Ops

Solid-producing ShapeScript ops get an optional `material: Material | None` field. Component geometry functions assign materials to sub-solids at design time (local frame).

Example — Pi compute board:
- PCB slab → `MAT_FR4_GREEN`
- Chip blocks → `MAT_IC_PACKAGE`
- GPIO header → `MAT_NICKEL`
- USB/ethernet ports → `MAT_NICKEL`

Example — Pololu wheel:
- Hub → `MAT_ABS_DARK` (or gray plastic)
- Tire ring → `MAT_RUBBER`

### STL Export

When exporting component meshes, group faces by material. Emit one STL per material group:
- `pi_compute__fr4_green.stl`
- `pi_compute__ic_package.stl`
- `pi_compute__nickel.stl`

### Viewer Manifest

Component mesh entry becomes a list of material-tagged meshes:

```json
{
  "name": "pi_compute",
  "meshes": [
    {"file": "pi_compute__fr4_green.stl", "material": "fr4_green"},
    {"file": "pi_compute__ic_package.stl", "material": "ic_package"},
    {"file": "pi_compute__nickel.stl", "material": "nickel"}
  ]
}
```

Manifest also includes a `materials` dictionary mapping name → visual properties (color, metallic, roughness).

### Viewer Rendering

`bot-viewer.ts` loads multi-mesh components as a Three.js `Group`. Each sub-mesh gets a `MeshPhysicalMaterial` derived from the material entry (color, metalness, roughness).

### Migration

- Delete `Appearance` dataclass from `component.py`
- Replace `appearance` field on `Component` with `material: Material` (default/fallback material)
- Existing `colors.py` palette becomes seed data for material catalog colors
- `materials.py` physical-only materials merge into unified `Material`

## Files Changed

| File | Change |
|------|--------|
| `botcad/materials.py` | Unified `Material` dataclass + catalog |
| `botcad/component.py` | Remove `Appearance`, add `material` field |
| `botcad/colors.py` | Keep as color constants, referenced by materials |
| `botcad/shapescript/ops.py` | Add `material` field to solid ops |
| `botcad/components/*.py` | Tag sub-solids with materials |
| `botcad/emit/viewer.py` | Multi-mesh manifest entries |
| `botcad/emit/mujoco.py` | Derive rgba from material.color |
| `viewer/bot-viewer.ts` | Load multi-mesh components, material-aware rendering |

## Dependencies

None — fully independent workstream. Workstream 3 (fasteners) will reference materials from this catalog.
