# Component Kind Registry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all `isinstance` dispatch on component types. Replace with `ComponentKind` enum on every component + a metadata registry that maps kind → capabilities. Also name common rotation vectors to kill magic number tuples.

**Architecture:** Every `Component` gets a `kind: ComponentKind` field. A `COMPONENT_REGISTRY: dict[ComponentKind, ComponentMeta]` maps kind → metadata (layers, script emitter, mount orientation, category). All consumers look up the registry instead of type-checking. Named rotation/direction constants replace magic tuples.

**Tech Stack:** Python dataclasses, StrEnum, existing ShapeScript emitters

---

## Overview

Four tasks:

1. **Named rotation/direction constants** — kill magic number tuples like `(-90.0, 0.0, 0.0)`
2. **ComponentKind enum + kind field on all components** — data model change
3. **ComponentMeta registry** — centralized metadata, script emitter dispatch
4. **Kill isinstance dispatch everywhere** — replace with kind checks and registry lookups

## File Map

| Task | Files | Change |
|------|-------|--------|
| 1 | `botcad/geometry.py` | Add named Euler/direction constants |
| 1 | `botcad/skeleton.py` | Use named constants in `_FACE_ROTATION` table |
| 1 | `botcad/emit/mujoco.py` | Use named constants in `_CAMERA_XYAXES` |
| 1 | `botcad/packing.py` | Use named constants in `_POSITION_AXES` |
| 2 | `botcad/component.py` | Add `ComponentKind` enum, `MountOrientation` enum, `kind` field on `Component` |
| 2 | `botcad/components/servo.py` | Set `kind=ComponentKind.SERVO` |
| 2 | `botcad/components/camera.py` | Set `kind=ComponentKind.CAMERA` |
| 2 | `botcad/components/battery.py` | Set `kind=ComponentKind.BATTERY` |
| 2 | `botcad/components/compute.py` | Set `kind=ComponentKind.COMPUTE` |
| 2 | `botcad/components/wheel.py` | Set `kind=ComponentKind.WHEEL` |
| 2 | `botcad/components/bearing.py` | Set `kind=ComponentKind.BEARING` |
| 3 | `botcad/component.py` | Add `ComponentMeta` dataclass and `COMPONENT_REGISTRY` |
| 3 | `botcad/emit/cad.py` | Rewrite `make_component_solid()` to use registry |
| 3 | `botcad/skeleton.py` | Rewrite `_compute_shapescript_bbox()` to use registry |
| 4 | `botcad/skeleton.py` | Replace `isinstance` and `face_outward` with kind/registry |
| 4 | `botcad/emit/viewer.py` | Replace `isinstance` cascade in `_component_specs()` |
| 4 | `botcad/emit/mujoco.py` | Replace `isinstance` in camera emit |
| 4 | `botcad/emit/cad.py` | Replace remaining `isinstance` |
| 4 | `botcad/routing.py` | Replace `isinstance` checks |
| 4 | `botcad/emit/readme.py` | Replace `isinstance` check |
| 4 | `main.py` | Replace `isinstance` cascades in server endpoints |
| 4 | `tests/` | Update tests |

---

### Task 1: Named Rotation and Direction Constants

**Files:**
- Modify: `botcad/geometry.py`
- Modify: `botcad/skeleton.py`
- Modify: `botcad/emit/mujoco.py`
- Modify: `botcad/packing.py`
- Test: `tests/test_domain_model.py`

**Context:** Tuples like `(-90.0, 0.0, 0.0)` and `(0.0, 1.0, 0.0)` appear throughout the codebase with comments explaining what they mean. Name them once.

- [ ] **Step 1: Add named constants to geometry.py**

```python
# Named Euler rotations (degrees) — used by face rotation, camera orientation
EULER_RX_NEG90: tuple[float, float, float] = (-90.0, 0.0, 0.0)
EULER_RX_POS90: tuple[float, float, float] = (90.0, 0.0, 0.0)
EULER_RX_180: tuple[float, float, float] = (180.0, 0.0, 0.0)
EULER_RY_NEG90: tuple[float, float, float] = (0.0, -90.0, 0.0)
EULER_RY_POS90: tuple[float, float, float] = (0.0, 90.0, 0.0)
EULER_IDENTITY: tuple[float, float, float] = (0.0, 0.0, 0.0)

# Named direction vectors — mount face normals
DIR_POS_X: tuple[float, float, float] = (1.0, 0.0, 0.0)
DIR_NEG_X: tuple[float, float, float] = (-1.0, 0.0, 0.0)
DIR_POS_Y: tuple[float, float, float] = (0.0, 1.0, 0.0)
DIR_NEG_Y: tuple[float, float, float] = (0.0, -1.0, 0.0)
DIR_POS_Z: tuple[float, float, float] = (0.0, 0.0, 1.0)
DIR_NEG_Z: tuple[float, float, float] = (0.0, 0.0, -1.0)
```

- [ ] **Step 2: Update _FACE_ROTATION in skeleton.py to use named constants**

```python
from botcad.geometry import (
    EULER_RX_NEG90, EULER_RX_POS90, EULER_RX_180,
    EULER_RY_NEG90, EULER_RY_POS90, EULER_IDENTITY,
)

_FACE_ROTATION = {
    "front": (EULER_RX_NEG90, (0, 2, 1), lambda x, y, z: (x, z, -y)),
    "back":  (EULER_RX_POS90, (0, 2, 1), lambda x, y, z: (x, -z, y)),
    ...
}
_NO_EULER = EULER_IDENTITY
```

- [ ] **Step 3: Update _POSITION_AXES in packing.py to use named constants**

```python
from botcad.geometry import DIR_POS_X, DIR_NEG_X, DIR_POS_Y, DIR_NEG_Y, DIR_POS_Z, DIR_NEG_Z

_POSITION_AXES = {
    "top": DIR_POS_Z,
    "bottom": DIR_NEG_Z,
    "front": DIR_POS_Y,
    ...
}
```

- [ ] **Step 4: Update _CAMERA_XYAXES keys to reference position constants (comments only — values are strings for MuJoCo XML)**

- [ ] **Step 5: Run tests**

Run: `make lint && uv run pytest tests/test_domain_model.py -v`

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor: name rotation/direction constants, replace magic tuples"
```

---

### Task 2: ComponentKind Enum + kind Field

**Files:**
- Modify: `botcad/component.py`
- Modify: `botcad/components/servo.py`
- Modify: `botcad/components/camera.py`
- Modify: `botcad/components/battery.py`
- Modify: `botcad/components/compute.py`
- Modify: `botcad/components/wheel.py`
- Modify: `botcad/components/bearing.py`
- Modify: `botcad/components/test_fastener.py`
- Test: `tests/test_domain_model.py`

**Context:** Add `ComponentKind` and `MountOrientation` enums to component.py. Add `kind` field to `Component` base class (default `GENERIC`). Each subclass/factory sets the appropriate kind.

- [ ] **Step 1: Write failing tests**

```python
def test_ov5647_has_camera_kind():
    from botcad.components.camera import OV5647
    from botcad.component import ComponentKind
    cam = OV5647()
    assert cam.kind == ComponentKind.CAMERA

def test_sts3215_has_servo_kind():
    from botcad.components import STS3215
    from botcad.component import ComponentKind
    servo = STS3215()
    assert servo.kind == ComponentKind.SERVO

def test_lipo2s_has_battery_kind():
    from botcad.components import LiPo2S
    from botcad.component import ComponentKind
    bat = LiPo2S(1000)
    assert bat.kind == ComponentKind.BATTERY

def test_generic_component_has_generic_kind():
    from botcad.component import Component, ComponentKind
    comp = Component("test", dimensions=(0.01, 0.01, 0.01), mass=0.001)
    assert comp.kind == ComponentKind.GENERIC
```

- [ ] **Step 2: Add enums to component.py**

```python
class ComponentKind(StrEnum):
    SERVO = "servo"
    CAMERA = "camera"
    BATTERY = "battery"
    COMPUTE = "compute"
    WHEEL = "wheel"
    BEARING = "bearing"
    GENERIC = "generic"

class MountOrientation(StrEnum):
    FLAT = "flat"            # lies flat on mounting surface
    FACE_NORMAL = "face_normal"  # functional axis aligns with mount face normal
```

- [ ] **Step 3: Add kind field to Component**

```python
@dataclass(frozen=True)
class Component:
    name: str
    dimensions: Vec3
    mass: float
    kind: ComponentKind = ComponentKind.GENERIC
    ...
```

- [ ] **Step 4: Set kind in each subclass/factory**

- `CameraSpec`: `kind: ComponentKind = ComponentKind.CAMERA`
- `BatterySpec`: `kind: ComponentKind = ComponentKind.BATTERY`
- `ServoSpec`: `kind: ComponentKind = ComponentKind.SERVO`
- `BearingSpec`: `kind: ComponentKind = ComponentKind.BEARING`
- Wheel factories (`PololuWheel90mm`): pass `kind=ComponentKind.WHEEL`
- Compute factories (`RaspberryPiZero2W`): pass `kind=ComponentKind.COMPUTE`
- `TestFastenerPrism`: pass `kind=ComponentKind.GENERIC`

- [ ] **Step 5: Delete Component.is_wheel property**

Replace with `kind == ComponentKind.WHEEL` at call sites (or keep as a convenience, but prefer the enum).

- [ ] **Step 6: Run tests**

- [ ] **Step 7: Commit**

```bash
git commit -m "feat: add ComponentKind enum + kind field on all components"
```

---

### Task 3: ComponentMeta Registry

**Files:**
- Modify: `botcad/component.py`
- Modify: `botcad/emit/cad.py`
- Modify: `botcad/skeleton.py`
- Test: `tests/test_domain_model.py`

**Context:** Create a `ComponentMeta` dataclass and a `COMPONENT_REGISTRY` dict that maps `ComponentKind` → metadata. This centralizes script emitter dispatch, layer definitions, mount orientation, and BOM category.

- [ ] **Step 1: Define ComponentMeta**

```python
@dataclass(frozen=True)
class ComponentMeta:
    kind: ComponentKind
    category: str                          # BOM category: "electronics", "actuator", "structure"
    layers: tuple[str, ...]                # viewer STL layers
    mount_orientation: MountOrientation
    script_emitter: Callable | None = None # ShapeScript emitter, set after import
```

- [ ] **Step 2: Build COMPONENT_REGISTRY**

```python
# Populated at module level or via a build function to avoid circular imports
# (script emitters live in botcad.shapescript which imports botcad.component)
def _build_registry() -> dict[ComponentKind, ComponentMeta]:
    from botcad.shapescript.emit_components import (
        battery_script, bearing_script, camera_script,
        compute_script, wheel_component_script,
    )
    from botcad.shapescript.emit_servo import servo_script
    return {
        ComponentKind.SERVO: ComponentMeta(
            kind=ComponentKind.SERVO,
            category="actuator",
            layers=("servo", "bracket", "cradle", "coupler",
                    "bracket_envelope", "cradle_envelope", "horn", "fasteners"),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=servo_script,
        ),
        ComponentKind.CAMERA: ComponentMeta(
            kind=ComponentKind.CAMERA,
            category="electronics",
            layers=("body", "fasteners"),
            mount_orientation=MountOrientation.FACE_NORMAL,
            script_emitter=camera_script,
        ),
        ComponentKind.BATTERY: ComponentMeta(
            kind=ComponentKind.BATTERY,
            category="electronics",
            layers=("body", "fasteners"),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=battery_script,
        ),
        ComponentKind.COMPUTE: ComponentMeta(
            kind=ComponentKind.COMPUTE,
            category="electronics",
            layers=("body", "fasteners"),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=compute_script,
        ),
        ComponentKind.WHEEL: ComponentMeta(
            kind=ComponentKind.WHEEL,
            category="structure",
            layers=("body",),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=wheel_component_script,
        ),
        ComponentKind.BEARING: ComponentMeta(
            kind=ComponentKind.BEARING,
            category="structure",
            layers=("body",),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=bearing_script,
        ),
        ComponentKind.GENERIC: ComponentMeta(
            kind=ComponentKind.GENERIC,
            category="component",
            layers=("body",),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=None,
        ),
    }
```

Note: build lazily to avoid circular imports (shapescript modules import component.py).

- [ ] **Step 3: Rewrite make_component_solid() to use registry**

```python
def make_component_solid(component: Component):
    meta = get_component_meta(component.kind)
    if meta.script_emitter is None:
        raise ValueError(f"No script emitter for {component.kind}")
    prog = meta.script_emitter(component)
    return _exec_ir(prog)
```

- [ ] **Step 4: Rewrite _compute_shapescript_bbox() to use registry**

Replace the `isinstance` cascade with `meta.script_emitter(comp)` lookup.

- [ ] **Step 5: Write tests**

```python
def test_registry_has_all_kinds():
    from botcad.component import ComponentKind, get_component_meta
    for kind in ComponentKind:
        meta = get_component_meta(kind)
        assert meta.kind == kind

def test_camera_meta_has_face_normal_orientation():
    from botcad.component import ComponentKind, MountOrientation, get_component_meta
    meta = get_component_meta(ComponentKind.CAMERA)
    assert meta.mount_orientation == MountOrientation.FACE_NORMAL
```

- [ ] **Step 6: Run tests**

- [ ] **Step 7: Commit**

```bash
git commit -m "feat: ComponentMeta registry — centralized script emitters, layers, mount orientation"
```

---

### Task 4: Kill isinstance Dispatch Everywhere

**Files:**
- Modify: `botcad/skeleton.py`
- Modify: `botcad/emit/viewer.py`
- Modify: `botcad/emit/mujoco.py`
- Modify: `botcad/emit/cad.py`
- Modify: `botcad/routing.py`
- Modify: `botcad/emit/readme.py`
- Modify: `main.py`
- Test: various test files

**Context:** With `ComponentKind` on every component and the registry available, replace every `isinstance(comp, CameraSpec)` with `comp.kind == ComponentKind.CAMERA` and every dispatch cascade with a registry lookup. The goal: **zero `isinstance` checks on component subclasses in non-test code.**

- [ ] **Step 1: Replace skeleton.py isinstance checks**

- `Mount.face_outward`: `return get_component_meta(self.component.kind).mount_orientation == MountOrientation.FACE_NORMAL`
- `Body.is_wheel_body` assignment: `any(m.component.kind == ComponentKind.WHEEL for m in body.mounts)`
- `_compute_shapescript_bbox`: already done in Task 3

- [ ] **Step 2: Replace emit/viewer.py isinstance cascade**

`_component_specs()` currently does:
```python
if isinstance(comp, CameraSpec): specs["component_type"] = "camera"
elif isinstance(comp, BatterySpec): ...
```
Replace with:
```python
specs["component_type"] = comp.kind.value
# Add kind-specific fields from the component itself
if comp.kind == ComponentKind.CAMERA:
    specs["fov_deg"] = comp.fov_deg
    ...
```

- [ ] **Step 3: Replace emit/mujoco.py isinstance checks**

`_emit_camera()` uses `isinstance(mount.component, CameraSpec)` → `mount.component.kind == ComponentKind.CAMERA`

- [ ] **Step 4: Replace emit/cad.py isinstance checks**

Any remaining `isinstance` in cad.py → kind checks.

- [ ] **Step 5: Replace routing.py isinstance checks**

`isinstance(mount.component, CameraSpec)` → kind check
`isinstance(mount.component, BatterySpec)` → kind check

- [ ] **Step 6: Replace emit/readme.py isinstance checks**

- [ ] **Step 7: Replace main.py isinstance cascades**

`_component_layers()`: use `get_component_meta(comp.kind).layers`
`_component_to_json()`: use `comp.kind.value` instead of isinstance dispatch
`_generate_solid()`: use registry script_emitter

- [ ] **Step 8: Verify zero isinstance on component subclasses in non-test code**

```bash
grep -rn "isinstance.*CameraSpec\|isinstance.*BatterySpec\|isinstance.*BearingSpec\|isinstance.*ServoSpec" botcad/ main.py mindsim/ --include="*.py" | grep -v test
```
Expected: zero results.

- [ ] **Step 9: Remove now-unnecessary subclass imports from consumer modules**

Many files import `CameraSpec`, `BatterySpec` etc. just for isinstance checks. Remove those imports.

- [ ] **Step 10: Run full test suite**

Run: `make lint && uv run pytest tests/ -v --tb=short`

- [ ] **Step 11: Commit**

```bash
git commit -m "refactor: kill all isinstance dispatch — use ComponentKind enum + registry lookups"
```

---

## Dependency Graph

```
Task 1 (named constants)
  └── Task 2 (ComponentKind enum) — independent, but logically follows
        └── Task 3 (registry) — needs kind field
              └── Task 4 (kill isinstance) — needs registry
```

Tasks 1 and 2 are independent and could run in parallel.

## Testing Strategy

- **After Task 2:** All component factories produce components with correct `kind`
- **After Task 3:** Registry has entries for all kinds, `make_component_solid()` works via registry
- **After Task 4:** Zero `isinstance` on component subclasses in production code; full test suite passes

## Risk Mitigation

- **Circular imports:** The registry needs to import shapescript emitters which import component.py. Use lazy initialization (`_build_registry()` called on first access via `get_component_meta()`).
- **Frozen dataclass fields:** Adding `kind` to `Component` (frozen=True) requires all factories to pass it. If any factory misses it, the default `GENERIC` is safe.
- **Servo layers:** Servo `ComponentMeta.layers` includes horn/bracket/coupler, but whether horn is available depends on `horn_disc_params()`. The registry provides the superset; callers filter as needed.
