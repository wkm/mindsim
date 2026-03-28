# Dimension Types for Physical Quantities

**Date:** 2026-03-28
**Goal:** Replace bare `float` with `NewType` wrappers for all dimensional quantities across botcad, so units are explicit in the type system and caught by type checkers.

## Context

Every component spec (`Component`, `ServoSpec`, `CameraSpec`, `BearingSpec`, `Material`, `MountPoint`, `WirePort`) stores physical quantities as bare `float` with comments indicating units. There's nothing preventing `mass=0.003` (kg) from being confused with `mass=3.0` (grams). The DFM spec (2026-03-28) sketched `Meters` and `Radians` NewTypes; this spec completes the type system and rolls it out project-wide.

## Design

### New file: `botcad/units.py`

All dimension types, compound types, and factory functions in one place.

#### Scalar types

```python
from typing import NewType

Meters    = NewType("Meters", float)     # length / position
Kg        = NewType("Kg", float)         # mass
Degrees   = NewType("Degrees", float)    # angle (human-facing: FOV, range limits)
Radians   = NewType("Radians", float)    # angle (math/physics: joint range, speed)
Volts     = NewType("Volts", float)      # voltage
Amps      = NewType("Amps", float)       # current
NewtonM   = NewType("NewtonM", float)    # torque
RadPerSec = NewType("RadPerSec", float)  # angular velocity
Pascals   = NewType("Pascals", float)    # pressure / stress / modulus
KgPerM3   = NewType("KgPerM3", float)    # density
```

#### Compound types

```python
Position = tuple[Meters, Meters, Meters]  # spatial coordinates (points in space)
Size3D   = tuple[Meters, Meters, Meters]  # bounding box extents (width, height, depth)
```

Both are `tuple[Meters, Meters, Meters]` — they're type aliases, not distinct NewTypes. The separate names exist for documentation: `Position` is a point, `Size3D` is an extent. If we later want the type checker to distinguish them, we can promote to NewType.

`Vec3 = tuple[float, float, float]` stays in `component.py` for dimensionless directions (shaft axis, mount normal, etc.).

`Quat` stays as-is — quaternions are dimensionless.

#### Convenience factory

```python
def mm3(x: float, y: float, z: float) -> tuple[Meters, Meters, Meters]:
    """Three mm values → (Meters, Meters, Meters). Works for both Position and Size3D."""
    return (mm(x), mm(y), mm(z))
```

#### Factory functions

Readable constructors for common datasheet units:

```python
def mm(val: float) -> Meters:
    """Millimeters → Meters."""
    return Meters(val / 1000.0)

def grams(val: float) -> Kg:
    """Grams → Kg."""
    return Kg(val / 1000.0)

def mpa(val: float) -> Pascals:
    """Megapascals → Pascals."""
    return Pascals(val * 1e6)

def gpa(val: float) -> Pascals:
    """Gigapascals → Pascals."""
    return Pascals(val * 1e9)

def deg_to_rad(val: float) -> Radians:
    """Degrees → Radians."""
    return Radians(val * math.pi / 180.0)
```

### Changes to `component.py`

#### `Pose`

| Field | Before | After |
|-------|--------|-------|
| `pos` | `Vec3` | `Position` |

#### `Component` base class

| Field | Before | After |
|-------|--------|-------|
| `dimensions` | `Vec3` | `Size3D` |
| `mass` | `float` | `Kg` |
| `voltage` | `float` | `Volts` |
| `typical_current` | `float` | `Amps` |

#### `MountPoint`

| Field | Before | After |
|-------|--------|-------|
| `pos` | `Vec3` | `Position` |
| `diameter` | `float` | `Meters` |

`axis` stays `Vec3` — it's a unit direction.

#### `MountingEar` factory

`pos` → `Position`, `hole_diameter` → `Meters`.

#### `WirePort`

| Field | Before | After |
|-------|--------|-------|
| `pos` | `Vec3` | `Position` |

#### `CameraSpec`

| Field | Before | After |
|-------|--------|-------|
| `fov_deg` | `float` | `fov: Degrees` (rename) |

`resolution` stays `tuple[int, int]` — pixels are dimensionless counts.

#### `ServoSpec`

| Field | Before | After |
|-------|--------|-------|
| `stall_torque` | `float` | `NewtonM` |
| `no_load_speed` | `float` | `RadPerSec` |
| `voltage` | `float` | `Volts` |
| `shaft_offset` | `Vec3` | `Position` |
| `range_rad` | `tuple[float, float]` | `tuple[Radians, Radians]` |
| `body_dimensions` | `Vec3` | `Size3D` |
| `shaft_boss_radius` | `float` | `Meters` |
| `shaft_boss_height` | `float` | `Meters` |
| `connector_pos` | `Vec3 \| None` | `Position \| None` |

`shaft_axis` stays `Vec3` — unit direction. `gear_ratio`, `continuous` stay untyped — dimensionless.

#### `BearingSpec`

| Field | Before | After |
|-------|--------|-------|
| `od` | `float` | `Meters` |
| `id` | `float` | `Meters` |
| `width` | `float` | `Meters` |

#### `BatterySpec`

Inherits `Component` changes. Own fields (`chemistry`, `cells_s`, `cells_p`) are dimensionless — no changes.

### Changes to `bracket.py`

#### `BracketSpec`

| Field | Before | After |
|-------|--------|-------|
| `wall` | `float` | `Meters` |
| `tolerance` | `float` | `Meters` |
| `shaft_clearance` | `float` | `Meters` |
| `cable_slot_width` | `float` | `Meters` |
| `cable_slot_height` | `float` | `Meters` |
| `coupler_thickness` | `float` | `Meters` |

#### `HornDiscParams`

| Field | Before | After |
|-------|--------|-------|
| `radius` | `float` | `Meters` |
| `thickness` | `float` | `Meters` |
| `center_z` | `float` | `Meters` |
| `center_xy` | `tuple[float, float]` | `tuple[Meters, Meters]` |

### Changes to `materials.py`

#### `PrintProcess`

| Field | Before | After |
|-------|--------|-------|
| `nozzle_width` | `float` | `Meters` |

`wall_layers` (count) and `infill` (ratio) stay `float`.

#### `Material`

| Field | Before | After |
|-------|--------|-------|
| `density` | `float \| None` | `KgPerM3 \| None` |
| `youngs_modulus` | `float` | `Pascals` |
| `yield_strength` | `float` | `Pascals` |

`poisson_ratio`, `metallic`, `roughness`, `opacity` stay `float` — dimensionless ratios.

### Changes to component factories

All factory functions (`OV5647()`, `PiCamera2()`, `STS3215()`, etc.) wrap their literals:

```python
# Before
CameraSpec(name="OV5647", dimensions=(0.025, 0.024, 0.009), mass=0.003, fov_deg=72.0, ...)

# After
CameraSpec(name="OV5647", dimensions=mm3(25, 24, 9), mass=grams(3), fov=Degrees(72.0), ...)
```

This makes datasheet values immediately readable — you see `mm3(25, 24, 9)` and know it came from a 25×24×9mm spec.

For computed values (e.g., `LiPo2S` battery where dimensions depend on capacity), wrap the result:

```python
length = 0.018 + capacity_mah * 0.000045  # linear model
BatterySpec(dimensions=(Meters(length), Meters(width), Meters(height)), mass=Kg(mass), ...)
```

### Changes to consumers

Callers that read these fields for math (ShapeScript emitters, MuJoCo export, physics calculations) continue to work unchanged at runtime — `NewType` erases to `float`. Type annotations on those functions should accept the typed values. No `.value` unwrapping needed.

Key consumer files that need annotation updates:
- `botcad/shapescript/emit_body.py` — reads component dimensions, positions
- `botcad/shapescript/emit_servo.py` — reads servo geometry fields
- `botcad/shapescript/emit_components.py` — reads component specs for mesh generation
- `botcad/shapescript/backend_occt.py` — receives Meters values from IR
- `botcad/bracket.py` — functions that accept dimensional parameters
- `botcad/mujoco_export.py` (or equivalent) — reads all specs for XML generation

This is a grep-and-annotate pass: find functions that receive values from typed fields and update their parameter annotations.

### CLAUDE.md update

Add to the Principles section:

> **Dimension types for physical quantities.** All physical quantities use `NewType` wrappers from `botcad/units.py` — never bare `float`. Use factory functions (`mm()`, `grams()`, `mpa()`) to convert from datasheet units. `Position` for spatial coordinates, `Size3D` for bounding box extents, `Vec3` for dimensionless directions. Dimensionless ratios and gains stay `float`. See `botcad/units.py` for the full list.

## Migration strategy

1. Create `botcad/units.py` with all types and factories
2. Update `component.py` and `materials.py` type annotations
3. Update all component factories to use typed constructors
4. Update consumers (emitters, exporters) to accept typed parameters
5. Update CLAUDE.md
6. Run `make validate` — all tests should pass since NewType is erased at runtime

The runtime behavior is identical before and after. The only breakage would be in type-checker runs where bare floats are passed to typed parameters — which is exactly what we want to catch.

## Non-goals

- **StringId typed references** — planned separately (sketched in DFM spec), not part of this change.
- **Runtime unit validation** — NewType is compile-time only. If we need runtime checks later, we can add them to factory functions.
- **Unit conversion arithmetic** (`Meters + Meters → Meters`) — Python's type system can't express this with NewType. Not worth the complexity.
