# Material System & Multi-Color Components

**Date:** 2026-03-24
**Status:** Approved
**Workstream:** 1 of 3 (independent, parallel)

## Problem

Each component has a single `Appearance` (one RGBA color). A Pi board is flat green, a camera is flat dark gray, a wheel is flat gray. Real components have multiple materials — PCB substrate, IC packages, metal connectors, rubber tires, polycarbonate lenses.

Separately, `botcad/materials.py` tracks physical properties (density, print process) and `botcad/component.py` has `Appearance` (color, metallic, roughness). These are two halves of the same concept.

## Scope & Non-Goals

**In scope:** Multi-material rendering for purchased component meshes in the viewer (servos, Pi boards, cameras, wheels). Unifying the Appearance/Material split.

**Non-goals:** Body solids (printed structural shells) are NOT affected — they remain single-mesh for MuJoCo collision. Multi-material applies only to purchased component visualization in the viewer. MuJoCo sim continues to use a single color per geom.

## Design

### Unified Material

Merge physical and visual properties into a single `Material` dataclass:

```python
@dataclass(frozen=True)
class Material:
    name: str                         # "fr4_green", "abs_black", "rubber_dark"
    # Visual
    color: RGBA
    metallic: float = 0.0
    roughness: float = 0.7
    opacity: float = 1.0
    # Physical
    density: float | None = None
    process: PrintProcess | None = None  # FDM params; None for purchased materials
```

Note: `process` is `PrintProcess | None` (preserving the existing structured type), not a bare string. Purchased-component materials (e.g., `MAT_FR4_GREEN`, `MAT_RUBBER`) always have `process=None` — print process only applies to fabricated body materials.

### Material Catalog

Real-world materials with correct visual and physical properties:

- `MAT_FR4_GREEN` — PCB substrate (green, matte)
- `MAT_IC_PACKAGE` — chip epoxy (black, matte)
- `MAT_NICKEL` — connector pins, fastener plating (silver, metallic)
- `MAT_ABS_DARK` — servo housing, camera body (dark gray, slight sheen)
- `MAT_RUBBER` — tire surface (dark, high roughness)
- `MAT_POLYCARBONATE_CLEAR` — camera lens cover (dark, low roughness, opacity < 1.0)
- `MAT_PLA_LIGHT` — 3D printed structural parts (light gray, matte, process=FDM)
- `MAT_STEEL` — fastener bodies (gray, metallic)
- `MAT_ALUMINUM` — servo horns, brackets (light gray, metallic)

### Multi-Material via Separate ShapeScript Programs

Each component emitter produces **one ShapeScript program per material region**. Sub-solids of different materials are never boolean-fused together — this avoids the problem of tracking per-face material origin through OCCT boolean operations (which is fragile and not exposed by build123d).

Example — Pi compute board emitter produces 3 programs:
- Program 1 (MAT_FR4_GREEN): PCB slab
- Program 2 (MAT_IC_PACKAGE): chip blocks (unioned together, same material)
- Program 3 (MAT_NICKEL): GPIO header + USB ports (unioned, same material)

Each program executes independently against the OCCT backend and exports to its own STL.

### STL Export

One STL per material group per component:
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

**Migration:** During transition, the viewer accepts both `"mesh"` (legacy single-mesh, used by body solids) and `"meshes"` (multi-material, used by components). Body solids continue to use the single `"mesh"` field.

Manifest also includes a `materials` dictionary mapping name → visual properties (color, metallic, roughness, opacity).

### Viewer Rendering

`bot-viewer.ts` loads multi-mesh components as a Three.js `Group`. Each sub-mesh gets a `MeshPhysicalMaterial` derived from the material entry (color, metalness, roughness, opacity).

### MuJoCo Emission

MuJoCo continues to use a single RGBA per geom. The `default_material.color` is used for `<geom rgba="...">`. Multi-material is viewer-only.

### Migration

- Delete `Appearance` dataclass from `component.py`
- Replace `appearance` field on `Component` with `default_material: Material` (used by MuJoCo and as fallback)
- Multi-material breakdown is encoded in the component's ShapeScript emitter (multiple programs, each tagged with a material)
- Existing `colors.py` palette becomes seed data for material catalog colors
- `materials.py` physical-only materials merge into unified `Material`

## Files Changed

| File | Change |
|------|--------|
| `botcad/materials.py` | Unified `Material` dataclass + catalog |
| `botcad/component.py` | Remove `Appearance`, add `default_material` field |
| `botcad/colors.py` | Keep as color constants, referenced by materials |
| `botcad/shapescript/ops.py` | Add `material` field to solid ops |
| `botcad/shapescript/emit_body.py` | Component emitters produce per-material programs |
| `botcad/components/*.py` | Tag sub-solids with materials via separate programs |
| `botcad/emit/viewer.py` | Multi-mesh manifest entries + materials dictionary |
| `botcad/emit/mujoco.py` | Derive rgba from `default_material.color` |
| `viewer/bot-viewer.ts` | Load multi-mesh components, material-aware rendering |

## Dependencies

None — fully independent workstream. Workstream 3 (fasteners) will reference materials from this catalog.
