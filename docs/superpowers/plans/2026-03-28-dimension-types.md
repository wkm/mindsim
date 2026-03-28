# Dimension Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace bare `float` with `NewType` wrappers for all physical quantities across botcad, making units explicit in the type system.

**Architecture:** `botcad/units.py` defines all `NewType` scalars (`Meters`, `Kg`, `Degrees`, etc.), compound types (`Position`, `Size3D`), and factory functions (`mm()`, `grams()`, `mpa()`). Dataclass field annotations change; runtime behavior is identical since `NewType` erases to `float`.

**Tech Stack:** Python `typing.NewType`, frozen dataclasses, pyright/mypy.

**Spec:** `docs/superpowers/specs/2026-03-28-dimension-types-design.md`

---

### Task 1: Create `botcad/units.py` with types and factories

**Files:**
- Create: `botcad/units.py`
- Test: `tests/test_units.py`

- [ ] **Step 1: Write failing tests for unit types and factories**

```python
"""Tests for botcad.units — dimension NewTypes and factory functions."""

from __future__ import annotations

import math

import pytest

from botcad.units import (
    Amps,
    Degrees,
    Kg,
    KgPerM3,
    Meters,
    NewtonM,
    Pascals,
    Position,
    RadPerSec,
    Radians,
    Size3D,
    Volts,
    deg_to_rad,
    gpa,
    grams,
    mm,
    mm3,
    mpa,
)


def test_mm_converts_to_meters():
    assert mm(25) == pytest.approx(0.025)


def test_mm3_returns_tuple():
    result = mm3(25, 24, 9)
    assert result == pytest.approx((0.025, 0.024, 0.009))


def test_grams_converts_to_kg():
    assert grams(3) == pytest.approx(0.003)


def test_mpa_converts_to_pascals():
    assert mpa(40) == pytest.approx(40e6)


def test_gpa_converts_to_pascals():
    assert gpa(2.3) == pytest.approx(2.3e9)


def test_deg_to_rad_converts():
    assert deg_to_rad(180) == pytest.approx(math.pi)
    assert deg_to_rad(90) == pytest.approx(math.pi / 2)


def test_types_are_float_at_runtime():
    """NewType erases at runtime — values are plain float."""
    assert isinstance(Meters(1.0), float)
    assert isinstance(Kg(1.0), float)
    assert isinstance(Degrees(45.0), float)
    assert isinstance(Radians(1.0), float)
    assert isinstance(Volts(12.0), float)
    assert isinstance(Amps(0.5), float)
    assert isinstance(NewtonM(3.0), float)
    assert isinstance(RadPerSec(6.28), float)
    assert isinstance(Pascals(1e6), float)
    assert isinstance(KgPerM3(1200.0), float)


def test_arithmetic_works():
    """Typed values participate in normal float arithmetic."""
    a = Meters(0.025)
    b = Meters(0.010)
    assert a + b == pytest.approx(0.035)
    assert a * 2 == pytest.approx(0.050)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_units.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'botcad.units'`

- [ ] **Step 3: Implement `botcad/units.py`**

```python
"""Dimension types for physical quantities.

All physical quantities use NewType wrappers — never bare float.
Zero runtime cost; caught by type checkers (pyright/mypy).

Convention: values are always in SI base units.
Factory functions convert from common datasheet units.
"""

from __future__ import annotations

import math
from typing import NewType

# ── Scalar types ────────────────────────────────────────────────────────

Meters    = NewType("Meters", float)      # length
Kg        = NewType("Kg", float)          # mass
Degrees   = NewType("Degrees", float)     # angle (human-facing: FOV, display)
Radians   = NewType("Radians", float)     # angle (math/physics: joint range)
Volts     = NewType("Volts", float)       # voltage
Amps      = NewType("Amps", float)        # current
NewtonM   = NewType("NewtonM", float)     # torque
RadPerSec = NewType("RadPerSec", float)   # angular velocity
Pascals   = NewType("Pascals", float)     # pressure / stress / modulus
KgPerM3   = NewType("KgPerM3", float)     # density

# ── Compound types ──────────────────────────────────────────────────────

Position = tuple[Meters, Meters, Meters]  # spatial coordinates (point in space)
Size3D   = tuple[Meters, Meters, Meters]  # bounding box extents (w, h, d)

# ── Factory functions ───────────────────────────────────────────────────


def mm(val: float) -> Meters:
    """Millimeters → Meters."""
    return Meters(val / 1000.0)


def mm3(x: float, y: float, z: float) -> tuple[Meters, Meters, Meters]:
    """Three mm values → (Meters, Meters, Meters)."""
    return (mm(x), mm(y), mm(z))


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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_units.py -v`
Expected: All PASS

- [ ] **Step 5: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 6: Commit**

```
feat: add botcad/units.py — dimension NewTypes and factory functions
```

---

### Task 2: Update `component.py` type annotations

**Files:**
- Modify: `botcad/component.py`
- Modify: `tests/test_domain_model.py`

- [ ] **Step 1: Update imports and type annotations in `component.py`**

Add import:
```python
from botcad.units import Amps, Degrees, Kg, Meters, NewtonM, Position, RadPerSec, Radians, Size3D, Volts
```

Update fields:
- `Pose.pos: Vec3` → `Pose.pos: Position`
- `Component.dimensions: Vec3` → `Component.dimensions: Size3D`
- `Component.mass: float` → `Component.mass: Kg`
- `Component.voltage: float` → `Component.voltage: Volts`
- `Component.typical_current: float` → `Component.typical_current: Amps`
- `MountPoint.pos: Vec3` → `MountPoint.pos: Position`
- `MountPoint.diameter: float` → `MountPoint.diameter: Meters`
- `MountingEar` params: `pos: Vec3` → `Position`, `hole_diameter: float` → `Meters`
- `WirePort.pos: Vec3` → `WirePort.pos: Position`
- `CameraSpec.fov_deg: float` → `CameraSpec.fov: Degrees` (rename field)
- `BearingSpec.od/id/width: float` → `Meters`
- `ServoSpec.stall_torque: float` → `NewtonM`
- `ServoSpec.no_load_speed: float` → `RadPerSec`
- `ServoSpec.voltage: float` → `Volts`
- `ServoSpec.shaft_offset: Vec3` → `Position`
- `ServoSpec.range_rad: tuple[float, float]` → `tuple[Radians, Radians]`
- `ServoSpec.body_dimensions: Vec3` → `Size3D`
- `ServoSpec.shaft_boss_radius/shaft_boss_height: float` → `Meters`
- `ServoSpec.connector_pos: Vec3 | None` → `Position | None`

Update default values to use typed constructors:
- `Kg(0.0)`, `Volts(0.0)`, `Amps(0.0)`, `Meters(0.0)`, etc.
- `Pose.pos` default: `(Meters(0.0), Meters(0.0), Meters(0.0))`
- `POSE_IDENTITY` updated accordingly
- `ServoSpec.range_rad` default: `(Radians(-3.14159), Radians(3.14159))`

- [ ] **Step 2: Update `tests/test_domain_model.py` to use typed constructors**

Import `mm`, `grams`, `Meters`, `Degrees`, `mm3` etc. Update all `MountPoint`, `MountingEar`, `Component`, `CameraSpec` constructor calls to use typed values:

```python
# Before
MountPoint("m1", pos=(0.0, 0.0, 0.0), diameter=0.003)
# After
MountPoint("m1", pos=(Meters(0.0), Meters(0.0), Meters(0.0)), diameter=mm(3))
```

Update `fov_deg=` → `fov=` for CameraSpec.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_domain_model.py tests/test_units.py -v`
Expected: All PASS

- [ ] **Step 4: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 5: Commit**

```
refactor: type component.py fields with dimension NewTypes
```

---

### Task 3: Update `materials.py` type annotations

**Files:**
- Modify: `botcad/materials.py`

- [ ] **Step 1: Update imports and type annotations**

Add import:
```python
from botcad.units import KgPerM3, Meters, Pascals, gpa, mpa
```

Update fields:
- `PrintProcess.nozzle_width: float` → `Meters`
- `Material.density: float | None` → `KgPerM3 | None`
- `Material.youngs_modulus: float` → `Pascals`
- `Material.yield_strength: float` → `Pascals`

Update all `Material(...)` and `PrintProcess(...)` constructors in the module-level catalog (the material constants like `MAT_PLA`, `MAT_ABS_DARK`, etc.) to wrap values:
- `density=1240.0` → `density=KgPerM3(1240.0)`
- `youngs_modulus=2.3e9` → `youngs_modulus=gpa(2.3)`
- `yield_strength=40e6` → `yield_strength=mpa(40)`
- `nozzle_width=0.0004` → `nozzle_width=mm(0.4)`

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/ -k "material" -v`
Expected: PASS

- [ ] **Step 3: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 4: Commit**

```
refactor: type materials.py fields with dimension NewTypes
```

---

### Task 4: Update `bracket.py` type annotations

**Files:**
- Modify: `botcad/bracket.py`

- [ ] **Step 1: Update `BracketSpec` and `HornDiscParams` fields**

Add import:
```python
from botcad.units import Meters, mm
```

Update fields:
- `BracketSpec`: `wall`, `tolerance`, `shaft_clearance`, `cable_slot_width`, `cable_slot_height`, `coupler_thickness` → `Meters`
- `HornDiscParams`: `radius`, `thickness`, `center_z` → `Meters`; `center_xy` → `tuple[Meters, Meters]`

Update default values: `wall=mm(3)`, `tolerance=mm(0.3)`, etc.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/ -k "bracket or shapescript" -v`
Expected: PASS

- [ ] **Step 3: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 4: Commit**

```
refactor: type bracket.py fields with dimension NewTypes
```

---

### Task 5: Update supporting dataclasses (`fasteners.py`, `connectors.py`, `geometry.py`, `clearance.py`, `routing.py`, `skeleton.py`)

**Files:**
- Modify: `botcad/fasteners.py`
- Modify: `botcad/connectors.py`
- Modify: `botcad/geometry.py`
- Modify: `botcad/clearance.py`
- Modify: `botcad/routing.py`
- Modify: `botcad/skeleton.py`

- [ ] **Step 1: Update `fasteners.py`**

`FastenerSpec` fields → `Meters`: `thread_diameter`, `thread_pitch`, `head_diameter`, `head_height`, `socket_size`, `clearance_hole`, `close_fit_hole`.

Update all `FastenerSpec(...)` constructors in the catalog dictionaries (`_SOCKET_HEAD_CAP_CATALOG`, etc.) to wrap with `mm()`.

- [ ] **Step 2: Update `connectors.py`**

`ConnectorSpec`: `body_dimensions: Vec3` → `Size3D`, `wire_exit_offset: Vec3` → `Position`, `cable_bend_radius: float` → `Meters`.

`wire_exit_direction` and `mating_direction` stay `Vec3` — they're unit directions.

- [ ] **Step 3: Update `geometry.py`**

`MountRotation.yaw: float` → `Degrees`. Update `MOUNT_NO_ROTATION` default.

- [ ] **Step 4: Update `clearance.py`**

`ClearanceResult`: `intersection_volume: float` stays `float` (m³ — no CubicMeters type needed for now). `distance: float` → `Meters`. `min_distance: float` → `Meters`.

- [ ] **Step 5: Update `routing.py`**

`WireSegment`: `start/end: Vec3` → `Position`. `slack: float` → `Meters`.

- [ ] **Step 6: Update `skeleton.py`**

**Name collision:** `skeleton.py` already defines `Position = Literal["center", "bottom", "top", ...]` for mount face positions. Rename it to `FacePosition` and update all references in `skeleton.py` (the `Mount.position` field, `Body.mount()` parameter, `_FACE_ROTATION_TABLE` keys, etc.). Then import `Position` from `botcad.units`.

`ClearanceConstraint.min_distance: float` → `Meters`.

`Body` class (not frozen — has `# plint: disable=frozen-dataclass`):
- `radius`, `width`, `height`, `length`, `outer_r`, `jaw_length`, `jaw_width`, `jaw_thickness`, `padding` → `Meters`
- `explicit_dimensions: Vec3 | None` → `Size3D | None`
- `solved_dimensions: Vec3 | None` → `Size3D | None`
- `solved_mass: float` → `Kg`
- `solved_com: Vec3` → `Position`

`Bot.joint()` method: `min_distance: float` parameter → `Meters`.

Also grep for any imports of `Position` from `skeleton` in other files and update them to `FacePosition`.

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 8: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 9: Commit**

```
refactor: type supporting dataclasses with dimension NewTypes

FastenerSpec, ConnectorSpec, MountRotation, ClearanceResult,
WireSegment, ClearanceConstraint, Body fields.
```

---

### Task 6: Update component factories

**Files:**
- Modify: `botcad/components/camera.py`
- Modify: `botcad/components/servo.py`
- Modify: `botcad/components/battery.py`
- Modify: `botcad/components/bearing.py`
- Modify: `botcad/components/compute.py`
- Modify: `botcad/components/bec.py`
- Modify: `botcad/components/controller.py`
- Modify: `botcad/components/wheel.py`

- [ ] **Step 1: Update `camera.py`**

Import `mm`, `mm3`, `grams`, `Meters`, `Degrees` from `botcad.units`.

`OV5647()` and `PiCamera2()`:
- `dimensions=(0.025, 0.024, 0.009)` → `dimensions=mm3(25, 24, 9)`
- `mass=0.003` → `mass=grams(3)`
- `fov_deg=72.0` → `fov=Degrees(72.0)`
- `WirePort.pos=(0.0, -0.012, 0.0)` → `pos=(Meters(0.0), mm(-12), Meters(0.0))`
- `MountPoint.pos` and `diameter` values wrapped with `mm()`

- [ ] **Step 2: Update `servo.py`**

Import dimension types. Update module-level constants:
- `_STS3215_DIMS = mm3(45.23, 24.73, 35.0)` etc.
- `_STS3215_MASS = grams(55)`
- `_STS3215_STALL_TORQUE = NewtonM(2.942)`
- `_STS3215_VOLTAGE = Volts(12.0)`
- All `MountPoint` and `MountingEar` positions/diameters wrapped

Same pattern for STS3250 and SCS0009 variants.

- [ ] **Step 3: Update `battery.py`**

`LiPo2S()`: computed dimensions wrap with `Meters(...)`, mass with `Kg(...)`.

- [ ] **Step 4: Update `bearing.py`, `compute.py`, `bec.py`, `controller.py`, `wheel.py`**

Same pattern: wrap dimensional literals with `mm()`, `grams()`, `Meters()`, etc.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 7: Commit**

```
refactor: wrap component factory literals with dimension types

All component factories now use mm(), grams(), Degrees(), etc.
making datasheet values immediately readable.
```

---

### Task 7: Handle `fov_deg` → `fov` rename in emitters and viewer

The `CameraSpec.fov_deg` → `CameraSpec.fov` rename breaks consumers that access the attribute or emit it as a JSON key.

**Files:**
- Modify: `botcad/emit/viewer.py`
- Modify: `botcad/emit/mujoco.py`
- Modify: `viewer/semantic-viz.ts`
- Modify: `viewer/explore-mode.ts`

- [ ] **Step 1: Update `botcad/emit/viewer.py`**

`comp.fov_deg` → `comp.fov`. Keep the JSON key as `"fov_deg"` for now to avoid a viewer-side rename cascade:
```python
specs["fov_deg"] = comp.fov
```

- [ ] **Step 2: Update `botcad/emit/mujoco.py`**

`cam.fov_deg` → `cam.fov`:
```python
fovy=f"{cam.fov:.1f}",
```

- [ ] **Step 3: Run Python tests**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```
refactor: update emitters for CameraSpec.fov rename
```

---

### Task 8: Update remaining tests

**Files:**
- Modify: `tests/test_domain_model.py` (if not fully done in Task 2)
- Modify: `tests/test_botcad_pipeline.py`
- Modify: `tests/test_shapescript_components.py`
- Modify: `tests/test_shapescript_roundtrip.py`
- Modify: any other test files that construct component specs directly

- [ ] **Step 1: Grep for bare-float component constructors in tests**

Run: `rg "dimensions=\(" tests/ --type py` and `rg "mass=0\." tests/ --type py` to find remaining unwrapped values.

- [ ] **Step 2: Update test files**

Wrap constructor arguments with typed factories. Update `fov_deg=` → `fov=` references.

- [ ] **Step 3: Run full validation**

Run: `make validate`
Expected: PASS

- [ ] **Step 4: Commit**

```
test: update tests to use dimension types
```

---

### Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add dimension types principle to CLAUDE.md**

Add to the Principles section after the frozen dataclasses bullet:

```markdown
- **Dimension types for physical quantities.** All physical quantities use `NewType` wrappers from `botcad/units.py` — never bare `float`. Use factory functions (`mm()`, `grams()`, `mpa()`) to convert from datasheet units. `Position` for spatial coordinates, `Size3D` for bounding box extents, `Vec3` for dimensionless directions. Dimensionless ratios and gains stay `float`. See `botcad/units.py` for the full list.
```

- [ ] **Step 2: Run `make validate`**

Run: `make validate`
Expected: PASS (final full validation)

- [ ] **Step 3: Commit**

```
docs: add dimension types convention to CLAUDE.md
```
