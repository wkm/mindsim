# Phase A1: Material & Appearance Data Types

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded PLA density and ad-hoc color computation with `Material` and `Appearance` data types on Body and Component.

**Architecture:** New `botcad/materials.py` for Material/PrintProcess. Appearance added to `botcad/component.py`. Component.color migrated to Component.appearance. Body gains material and appearance fields. Mass computation reads body.material; color emitters read body.appearance.

**Tech Stack:** Python dataclasses, frozen types, botcad

**Spec:** `docs/superpowers/specs/2026-03-22-data-oriented-refactor-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `botcad/materials.py` | Create | Material, PrintProcess, standard instances (PLA, TPU, ALUMINUM) |
| `botcad/component.py` | Modify | Add Appearance dataclass; migrate Component.color -> Component.appearance |
| `botcad/skeleton.py` | Modify | Add material and appearance fields to Body |
| `botcad/components/servo.py` | Modify | color= -> appearance= |
| `botcad/components/battery.py` | Modify | color= -> appearance= |
| `botcad/components/wheel.py` | Modify | color= -> appearance= |
| `botcad/components/bearing.py` | Modify | color= -> appearance= |
| `botcad/components/camera.py` | Modify | color= -> appearance= |
| `botcad/components/compute.py` | Modify | color= -> appearance= |
| `botcad/components/controller.py` | Modify | color= -> appearance= |
| `botcad/packing.py` | Modify | Read body.material.density instead of hardcoded 1200.0 |
| `botcad/emit/cad.py` | Modify | Read body.material for mass; read body.appearance for color |
| `botcad/emit/mujoco.py` | Modify | Read body.appearance for color |
| `tests/test_materials.py` | Create | Material/Appearance unit tests |

---

### Task 1: Create Material and PrintProcess data types

**Files:**
- Create: `botcad/materials.py`
- Create: `tests/test_materials.py`

- [ ] **Step 1: Write tests for Material and PrintProcess**

```python
# tests/test_materials.py
from __future__ import annotations

from botcad.materials import ALUMINUM, PLA, TPU, Material, PrintProcess


def test_pla_density():
    assert PLA.density == 1200.0


def test_pla_has_print_process():
    assert PLA.process is not None
    assert PLA.process.wall_layers == 2
    assert PLA.process.nozzle_width == 0.0004
    assert PLA.process.infill == 0.20


def test_tpu_lower_infill():
    assert TPU.process is not None
    assert TPU.process.infill == 0.15


def test_aluminum_no_print_process():
    assert ALUMINUM.process is None
    assert ALUMINUM.density == 2700.0


def test_material_is_frozen():
    import dataclasses
    assert dataclasses.fields(Material)
    try:
        PLA.density = 999  # type: ignore
        assert False, "Should be frozen"
    except dataclasses.FrozenInstanceError:
        pass


def test_print_process_wall_thickness():
    """Wall thickness = wall_layers * nozzle_width."""
    p = PLA.process
    assert p is not None
    assert p.wall_layers * p.nozzle_width == pytest.approx(0.0008)
```

Add `import pytest` at the top.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_materials.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'botcad.materials'`

- [ ] **Step 3: Implement materials.py**

```python
# botcad/materials.py
"""Material and print process definitions.

Data module — declares physical properties that affect mass computation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrintProcess:
    """FDM print parameters that determine effective mass."""

    wall_layers: int = 2
    nozzle_width: float = 0.0004  # 0.4mm
    infill: float = 0.20


@dataclass(frozen=True)
class Material:
    """Physical material with density and optional print process."""

    name: str
    density: float  # kg/m^3
    process: PrintProcess | None = None


# Standard instances
PLA = Material("PLA", 1200.0, PrintProcess())
TPU = Material("TPU", 1120.0, PrintProcess(infill=0.15))
ALUMINUM = Material("aluminum", 2700.0, None)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_materials.py -v`
Expected: All PASS

- [ ] **Step 5: Lint**

Run: `make lint`

- [ ] **Step 6: Commit**

```bash
git add botcad/materials.py tests/test_materials.py
git commit -m "feat: add Material and PrintProcess data types in botcad/materials.py"
```

---

### Task 2: Add Appearance dataclass to component.py

**Files:**
- Modify: `botcad/component.py:68-81`
- Modify: `tests/test_materials.py` (add Appearance tests)

- [ ] **Step 1: Write tests for Appearance**

Append to `tests/test_materials.py`:

```python
from botcad.component import Appearance


def test_appearance_defaults():
    a = Appearance(color=(1.0, 0.0, 0.0, 1.0))
    assert a.metallic == 0.0
    assert a.roughness == 0.7
    assert a.opacity == 1.0


def test_appearance_is_frozen():
    import dataclasses
    a = Appearance(color=(1.0, 0.0, 0.0, 1.0))
    try:
        a.color = (0.0, 0.0, 0.0, 1.0)  # type: ignore
        assert False, "Should be frozen"
    except dataclasses.FrozenInstanceError:
        pass


def test_appearance_with_metallic():
    a = Appearance(color=(0.8, 0.8, 0.8, 1.0), metallic=1.0, roughness=0.3)
    assert a.metallic == 1.0
    assert a.roughness == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_materials.py::test_appearance_defaults -v`
Expected: FAIL with `ImportError: cannot import name 'Appearance'`

- [ ] **Step 3: Add Appearance to component.py**

Add after the `RGBA` type alias (around line 13):

```python
@dataclass(frozen=True)
class Appearance:
    """Visual properties for rendering. Emitters read this, never compute colors."""

    color: RGBA
    metallic: float = 0.0    # 0=plastic, 1=metal
    roughness: float = 0.7   # surface roughness
    opacity: float = 1.0     # transparency
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_materials.py -v`
Expected: All PASS

- [ ] **Step 5: Lint**

Run: `make lint`

- [ ] **Step 6: Commit**

```bash
git add botcad/component.py tests/test_materials.py
git commit -m "feat: add Appearance dataclass to component.py"
```

---

### Task 3: Migrate Component.color to Component.appearance

**Files:**
- Modify: `botcad/component.py:68-81` — replace `color: RGBA` with `appearance: Appearance`
- Modify: all files in `botcad/components/` — update factory calls
- Modify: all files that read `.color` — update to `.appearance.color`

- [ ] **Step 1: Add backward-compatible `.color` property to Component**

In `botcad/component.py`, change the Component class:

```python
@dataclass(frozen=True)
class Component:
    name: str
    dimensions: Vec3
    mass: float
    mounting_points: tuple[MountPoint, ...] = ()
    wire_ports: tuple[WirePort, ...] = ()
    appearance: Appearance = Appearance(color=(0.541, 0.608, 0.659, 1.0))

    @property
    def color(self) -> RGBA:
        """Backward-compatible color accessor."""
        return self.appearance.color
```

Remove the old `color: RGBA` field. The `@property` ensures all existing `.color` reads still work without changes.

- [ ] **Step 2: Update all component factories**

Each factory changes `color=X.rgba` to `appearance=Appearance(color=X.rgba)`:

| File | Line | Before | After |
|------|------|--------|-------|
| `botcad/components/servo.py` | 138 | `color=COLOR_STRUCTURE_DARK.rgba` | `appearance=Appearance(color=COLOR_STRUCTURE_DARK.rgba)` |
| `botcad/components/servo.py` | 265 | `color=COLOR_STRUCTURE_DARK.rgba` | `appearance=Appearance(color=COLOR_STRUCTURE_DARK.rgba)` |
| `botcad/components/servo.py` | 413 | `color=COLOR_STRUCTURE_DARK.rgba` | `appearance=Appearance(color=COLOR_STRUCTURE_DARK.rgba)` |
| `botcad/components/battery.py` | 43 | `color=COLOR_POWER_BATTERY.rgba` | `appearance=Appearance(color=COLOR_POWER_BATTERY.rgba)` |
| `botcad/components/wheel.py` | 95 | `color=COLOR_STRUCTURE_RUBBER.rgba` | `appearance=Appearance(color=COLOR_STRUCTURE_RUBBER.rgba)` |
| `botcad/components/bearing.py` | 21 | `color=COLOR_METAL_STEEL.rgba` | `appearance=Appearance(color=COLOR_METAL_STEEL.rgba, metallic=1.0, roughness=0.3)` |
| `botcad/components/camera.py` | 63 | `color=COLOR_ELECTRONICS_DARK.rgba` | `appearance=Appearance(color=COLOR_ELECTRONICS_DARK.rgba)` |
| `botcad/components/camera.py` | 174 | `color=COLOR_ELECTRONICS_DARK.rgba` | `appearance=Appearance(color=COLOR_ELECTRONICS_DARK.rgba)` |
| `botcad/components/compute.py` | 58 | `color=COLOR_ELECTRONICS_PCB.rgba` | `appearance=Appearance(color=COLOR_ELECTRONICS_PCB.rgba)` |
| `botcad/components/controller.py` | 36 | `color=COLOR_ELECTRONICS_CONTROLLER.rgba` | `appearance=Appearance(color=COLOR_ELECTRONICS_CONTROLLER.rgba)` |

Each file also needs `from botcad.component import Appearance` added to imports.

- [ ] **Step 3: Run full test suite**

Run: `make validate`
Expected: All tests pass — the `.color` property ensures backward compatibility.

- [ ] **Step 4: Commit**

```bash
git add botcad/component.py botcad/components/
git commit -m "refactor: migrate Component.color to Component.appearance with backward-compatible property"
```

---

### Task 4: Add material and appearance fields to Body

**Files:**
- Modify: `botcad/skeleton.py:260-330` — add fields to Body

- [ ] **Step 1: Add fields to Body class**

After the existing `_component` field (line 326), add:

```python
    material: Material | None = None
    appearance: Appearance | None = None
```

Add imports at top of skeleton.py:

```python
from botcad.materials import Material
from botcad.component import Appearance
```

- [ ] **Step 2: Set default material=PLA for fabricated bodies**

In `Bot._collect_tree()` or wherever fabricated bodies are constructed, set `material=PLA` as the default. Check how bodies are created — they're defined in `bots/*/design.py`. Since Body is mutable, the simplest approach: set `material = PLA` as the field default for all bodies, since most are 3D-printed.

```python
    material: Material = field(default_factory=lambda: PLA)
    appearance: Appearance | None = None
```

Import `PLA` from `botcad.materials`.

- [ ] **Step 3: Assign appearance during solve()**

In `Bot._compute_world_transforms()` or `Bot._create_purchased_bodies()`, assign appearance:

For fabricated bodies (in `_compute_world_transforms` or a new loop after it):
```python
from botcad.colors import COLOR_STRUCTURE_BODY, COLOR_STRUCTURE_DARK, COLOR_STRUCTURE_RUBBER
from botcad.colors import BP_GRAY5

for body in self.all_bodies:
    if body.appearance is not None:
        continue  # already set (purchased bodies)
    if body.shape is BodyShape.CYLINDER and body.radius and body.radius > 0.03:
        body.appearance = Appearance(color=COLOR_STRUCTURE_DARK.rgba)
    elif body.shape is BodyShape.TUBE:
        body.appearance = Appearance(color=BP_GRAY5.rgba if hasattr(BP_GRAY5, 'rgba') else (*BP_GRAY5, 1.0))
    elif body.shape is BodyShape.JAW:
        body.appearance = Appearance(color=COLOR_STRUCTURE_BODY.rgba)
    else:
        body.appearance = Appearance(color=COLOR_STRUCTURE_BODY.rgba)
```

For purchased bodies (in `_create_purchased_bodies`):
```python
body.appearance = comp.appearance  # inherit from Component
```

- [ ] **Step 4: Run tests**

Run: `make validate`

- [ ] **Step 5: Commit**

```bash
git add botcad/skeleton.py
git commit -m "feat: add material and appearance fields to Body, assign during solve()"
```

---

### Task 5: Wire material into mass computation

**Files:**
- Modify: `botcad/packing.py:218-225` — read body.material.density
- Modify: `botcad/emit/cad.py:46,130-137` — read body.material

- [ ] **Step 1: Update packing.py to use body.material**

Replace lines 222-223:

```python
# Before:
wall_thickness = 0.001
density = 1200.0

# After:
wall_thickness = 0.001
density = body.material.density if body.material else 1200.0
```

- [ ] **Step 2: Update cad.py to use body.material**

Delete the `_PLA_DENSITY = 1200.0` constant (line 46).

In `_update_mass_from_solid()` (lines 130-137), replace hardcoded values:

```python
# Before:
line_width = 0.0004
n_walls = 2
infill_fraction = 0.20
wall_thickness = n_walls * line_width
...
struct_mass = (wall_volume + infill_volume) * _PLA_DENSITY

# After:
mat = body.material
if mat and mat.process:
    p = mat.process
    wall_thickness = p.wall_layers * p.nozzle_width
    infill_fraction = p.infill
else:
    wall_thickness = 0.0008
    infill_fraction = 0.20
density = mat.density if mat else 1200.0
...
struct_mass = (wall_volume + infill_volume) * density
```

- [ ] **Step 3: Run tests**

Run: `make validate`
Expected: All pass — same values, just read from data instead of hardcoded.

- [ ] **Step 4: Commit**

```bash
git add botcad/packing.py botcad/emit/cad.py
git commit -m "refactor: mass computation reads body.material instead of hardcoded PLA density"
```

---

### Task 6: Wire appearance into color emitters

**Files:**
- Modify: `botcad/emit/cad.py:779-787` — read body.appearance
- Modify: `botcad/emit/mujoco.py:797-813` — read body.appearance
- Modify: `botcad/emit/mujoco.py` — component color reads

- [ ] **Step 1: Update _body_color_rgb in mujoco.py**

Replace the shape-based logic with appearance lookup:

```python
def _body_color_rgb(body: Body) -> tuple[float, float, float]:
    """Read body color from appearance. Fallback to shape-based default."""
    if body.appearance is not None:
        r, g, b, _a = body.appearance.color
        return (r, g, b)
    # Fallback for bodies without appearance (shouldn't happen after solve)
    return COLOR_STRUCTURE_BODY.rgb
```

- [ ] **Step 2: Update cad.py delegation**

`_body_color_rgb` in cad.py already delegates to mujoco.py — no change needed, but verify it still works.

- [ ] **Step 3: Run tests**

Run: `make validate`

- [ ] **Step 4: Visual verification**

Run `make web`, check that bot colors in the viewer match before/after.

- [ ] **Step 5: Commit**

```bash
git add botcad/emit/cad.py botcad/emit/mujoco.py
git commit -m "refactor: color emitters read body.appearance instead of computing from shape"
```

---

### Task 7: Delete _compute_world_positions() from cad.py

**Files:**
- Modify: `botcad/emit/cad.py:723-755` — delete function
- Modify: callers of `_compute_world_positions()` — use `body.world_pos`

- [ ] **Step 1: Find all callers**

Search for `_compute_world_positions` in cad.py. Update each call site to read `body.world_pos` directly from the skeleton.

- [ ] **Step 2: Delete the function**

Remove `_compute_world_positions()` (lines 723-755).

- [ ] **Step 3: Run tests**

Run: `make validate`

- [ ] **Step 4: Commit**

```bash
git add botcad/emit/cad.py
git commit -m "refactor: delete _compute_world_positions(), read body.world_pos from skeleton"
```

---

### Task 8: Final validation and bot regeneration

- [ ] **Step 1: Run full validation**

Run: `make validate`

- [ ] **Step 2: Regenerate all bot meshes**

```bash
uv run mjpython main.py regen --all
```

- [ ] **Step 3: Visual check**

Run `make web`, verify so101_arm and wheeler_base look correct.

- [ ] **Step 4: Commit regenerated meshes**

```bash
git add bots/
git commit -m "chore: regenerate bot meshes after Material/Appearance refactor"
```

---

## Dependency Graph

```
Task 1 (Material types) ──→ Task 4 (Body fields) ──→ Task 5 (mass wiring)
Task 2 (Appearance type) ─┤                          Task 6 (color wiring)
Task 3 (Component migration) ─────────────────────→ Task 7 (delete world_pos dup)
                                                      ↓
                                                   Task 8 (validate + regen)
```

Tasks 1-3 are the foundation. Tasks 5-7 are independent of each other (can parallelize). Task 8 gates on all.
