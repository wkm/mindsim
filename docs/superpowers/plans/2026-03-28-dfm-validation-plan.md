# DFM Validation & Assembly Sequence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DFM validation system with assembly sequence model, interactive viewer, and sim validation for wheeler_base — printable by 2026-04-05.

**Architecture:** Foundational type cleanups (StringId, dimension types) first, then assembly sequence model, DFM check framework with registry pattern, async API, and viewer DFM mode. Sim validation runs as an independent parallel workstream.

**Tech Stack:** Python (botcad, FastAPI), TypeScript (Three.js viewer), build123d/OCCT geometry, MuJoCo simulation

**Spec:** `docs/superpowers/specs/2026-03-28-dfm-validation-design.md`

**Worktree:** `/Users/wkm/Code/mindsim-dfm-validation` (branch `exp/260328-dfm-validation`)

---

## Phase 0a: StringId Pattern

Introduce `BodyId` and `JointId` as frozen typed wrappers. Migrate all bare-string body/joint name usage across the codebase.

**Note:** Phase 0a modifies `skeleton.py` and bot design files. Phase 0b also modifies `skeleton.py` and bot design files. **Run 0a before 0b** to avoid merge conflicts. Phase 0c is independent of both.

### Task 0a-1: Create `botcad/ids.py` with BodyId and JointId

**Files:**
- Create: `botcad/ids.py`
- Test: `tests/test_ids.py`

- [ ] **Step 1: Write tests for BodyId and JointId**

```python
# tests/test_ids.py
from botcad.ids import BodyId, JointId

def test_body_id_frozen():
    bid = BodyId("base")
    assert bid.name == "base"
    # frozen — assignment raises
    import pytest
    with pytest.raises(AttributeError):
        bid.name = "other"

def test_body_id_hashable():
    a = BodyId("base")
    b = BodyId("base")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1

def test_body_id_not_equal_to_joint_id():
    bid = BodyId("base")
    jid = JointId("base")
    assert bid != jid

def test_body_id_str():
    bid = BodyId("base")
    assert str(bid) == "base"

def test_joint_id_frozen_and_hashable():
    a = JointId("left_wheel")
    b = JointId("left_wheel")
    assert a == b
    assert hash(a) == hash(b)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_ids.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement BodyId and JointId**

```python
# botcad/ids.py
"""Typed identifiers for skeleton entities.

The StringId pattern: frozen, hashable wrappers that prevent
accidental use of one entity type where another is expected.
A BodyId cannot be compared to a JointId.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BodyId:
    """Unique identifier for a Body in the skeleton."""
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class JointId:
    """Unique identifier for a Joint in the skeleton."""
    name: str

    def __str__(self) -> str:
        return self.name
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_ids.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add botcad/ids.py tests/test_ids.py
git commit -m "feat: introduce BodyId and JointId — the StringId pattern"
```

### Task 0a-2: Migrate `skeleton.py` core data classes

**Files:**
- Modify: `botcad/skeleton.py` (Body.name → BodyId, Joint.name → JointId, parent_body_name → BodyId | None)
- Modify: `botcad/ids.py` (add re-exports if needed)

This is the core change. All downstream modules reference these fields.

- [ ] **Step 1: Update Body, Joint, ClearanceConstraint in skeleton.py**

Key changes:
- `Body.name: str` → `Body.name: BodyId`
- `Body.parent_body_name: str | None` → `Body.parent_body_name: BodyId | None`
- `Joint.name: str` → `Joint.name: JointId`
- `ClearanceConstraint.body_a: str` → `ClearanceConstraint.body_a: BodyId`
- `ClearanceConstraint.body_b: str` → `ClearanceConstraint.body_b: BodyId`
- Update all internal string comparisons to use `.name` attribute or direct Id comparison
- Update `_build_implicit_tree()` and other skeleton methods that do name lookups

- [ ] **Step 2: Fix all bot design files that construct Body/Joint**

Update `bots/wheeler_base/design.py`, `bots/wheeler_arm/design.py`, `bots/so101_arm/design.py` and any other bot definitions to wrap name strings in `BodyId()`/`JointId()`.

- [ ] **Step 3: Run lint and tests**

Run: `make lint && uv run pytest tests/ -x -v`
Fix any type errors or test failures from the migration.

- [ ] **Step 4: Commit**

```bash
git commit -am "refactor: migrate skeleton.py Body/Joint names to BodyId/JointId"
```

### Task 0a-3: Migrate emit modules

**Files:**
- Modify: `botcad/emit/cad.py` — dict keys `str` → `BodyId`, name lookups
- Modify: `botcad/emit/mujoco.py` — body/joint name references in XML generation
- Modify: `botcad/emit/viewer.py` — manifest body name fields, `startswith()` patterns
- Modify: `botcad/emit/renders.py` — joint name lookups
- Modify: `botcad/emit/assembly_renders.py` — name-based entity tracking
- Modify: `botcad/emit/bom.py` — body/joint name keys
- Modify: `botcad/emit/readme.py` — name string interpolation

- [ ] **Step 1: Migrate `emit/cad.py`**

This file has the most dict[str, ...] usage. Change:
- `CadModel.body_solids: dict[str, ...]` → `dict[BodyId, ...]`
- `CadModel.parent_joint_map: dict[str, Joint]` → `dict[BodyId, Joint]`
- `CadModel.body_wire_segments: dict[str, ...]` → `dict[BodyId, ...]`
- `CadModel.multi_material_solids: dict[str, ...]` → `dict[BodyId, ...]`
- All local dicts similarly
- String prefix checks like `body.name.startswith("horn_")` → `body.name.name.startswith("horn_")` (or add a helper)

- [ ] **Step 2: Run tests after cad.py changes**

Run: `uv run pytest tests/ -x -v`

- [ ] **Step 3: Migrate `emit/mujoco.py`**

XML generation uses body/joint names for mesh names, actuator names, sensor names. These need `str(body.name)` or `body.name.name` when interpolating into XML strings, but `BodyId`/`JointId` in data structures.

- [ ] **Step 4: Run tests after mujoco.py changes**

Run: `uv run pytest tests/ -x -v`

- [ ] **Step 5: Migrate `emit/viewer.py`**

Manifest generation uses name parsing (`body.name[len("servo_"):]`). These string operations work on the `.name` attribute of the Id.

- [ ] **Step 6: Migrate remaining emit modules**

`emit/renders.py`, `emit/assembly_renders.py`, `emit/bom.py`, `emit/readme.py` — each is smaller.

- [ ] **Step 7: Run full test suite**

Run: `make lint && make validate`

- [ ] **Step 8: Commit**

```bash
git commit -am "refactor: migrate all emit modules to BodyId/JointId"
```

### Task 0a-4: Migrate routing, clearance, FEA modules

**Files:**
- Modify: `botcad/routing.py` — `WireSegment.body_name`, `WireSegment.joint_name`
- Modify: `botcad/clearance.py` — `ClearanceResult.body_a/body_b`
- Modify: `botcad/fea/joint_analytical.py` — `StressResult.joint_name/body_name`

- [ ] **Step 1: Migrate routing.py**

- `WireSegment.body_name: str` → `WireSegment.body_name: BodyId`
- `WireSegment.joint_name: str | None` → `WireSegment.joint_name: JointId | None`
- Update `_find_body()` and other lookup functions

- [ ] **Step 2: Migrate clearance.py and fea/joint_analytical.py**

- `ClearanceResult.body_a/body_b: str` → `BodyId`
- `StressResult.joint_name: str` → `JointId`
- `StressResult.body_name: str` → `BodyId`

- [ ] **Step 3: Run full validation**

Run: `make lint && make validate`

- [ ] **Step 4: Commit**

```bash
git commit -am "refactor: migrate routing, clearance, FEA to BodyId/JointId"
```

---

## Phase 0b: Dimension Types

Introduce `Meters` and `Radians` NewType wrappers. Migrate dimension fields in all data model classes.

### Task 0b-1: Create `botcad/units.py`

**Files:**
- Create: `botcad/units.py`
- Test: `tests/test_units.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_units.py
from botcad.units import Meters, Radians

def test_meters_runtime_transparent():
    """NewType is zero-cost at runtime — Meters(...) returns a plain float."""
    m = Meters(0.003)
    assert isinstance(m, float)
    assert m == 0.003

def test_radians_runtime_transparent():
    r = Radians(1.5708)
    assert isinstance(r, float)

def test_meters_arithmetic():
    a = Meters(0.003)
    b = Meters(0.002)
    assert a + b == 0.005
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_units.py -v`

- [ ] **Step 3: Implement**

```python
# botcad/units.py
"""Dimension types to prevent unit confusion.

Zero-cost at runtime (NewType), caught by type checkers.
All values are SI: meters for lengths, radians for angles.
"""
from typing import NewType

Meters = NewType("Meters", float)
Radians = NewType("Radians", float)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_units.py -v`

- [ ] **Step 5: Commit**

```bash
git add botcad/units.py tests/test_units.py
git commit -m "feat: introduce Meters and Radians dimension types"
```

### Task 0b-2: Migrate component data classes

**Files:**
- Modify: `botcad/component.py` — MountPoint.diameter, Component.dimensions, BearingSpec, ServoSpec fields
- Modify: `botcad/connectors.py` — ConnectorSpec dimension fields
- Modify: `botcad/fasteners.py` — FastenerSpec dimension fields
- Modify: `botcad/bracket.py` — BracketSpec, HornDiscParams dimension fields

- [ ] **Step 1: Migrate component.py**

Key changes (~15 fields):
- `MountPoint.diameter: float` → `Meters`
- `Component.dimensions: Vec3` stays Vec3 (implicitly meters)
- `BearingSpec.od/id/width: float` → `Meters`
- `ServoSpec.range_rad: tuple[float, float]` → `tuple[Radians, Radians]`
- `ServoSpec.no_load_speed: float` → stays float (rad/s is a derived unit)
- `ServoSpec.shaft_boss_radius/height: float` → `Meters`

- [ ] **Step 2: Migrate connectors.py**

- `ConnectorSpec.cable_bend_radius: float` → `Meters`
- Update all connector instances (MOLEX_5264_3PIN, etc.) to wrap values

- [ ] **Step 3: Migrate fasteners.py**

- All dimension fields (`thread_diameter`, `head_diameter`, `head_height`, `socket_size`, `clearance_hole`, `close_fit_hole`, `thread_pitch`) → `Meters`
- Update all fastener instances (M2, M2_5, M3 variants)

- [ ] **Step 4: Migrate bracket.py**

- `BracketSpec.wall/tolerance/shaft_clearance/cable_slot_width/cable_slot_height/coupler_thickness` → `Meters`
- `HornDiscParams.radius/thickness/center_z` → `Meters`

- [ ] **Step 5: Run full validation**

Run: `make lint && make validate`

- [ ] **Step 6: Commit**

```bash
git commit -am "refactor: migrate component/connector/fastener/bracket specs to Meters/Radians"
```

### Task 0b-3: Migrate skeleton and geometry data classes

**Files:**
- Modify: `botcad/skeleton.py` — Body dimension fields, Joint.range_rad, ClearanceConstraint.min_distance
- Modify: `botcad/geometry.py` — MountRotation.yaw, Euler constants

- [ ] **Step 1: Migrate skeleton.py**

Body dimension fields (~12 fields):
- `Body.radius/width/height/length/outer_r/jaw_length/jaw_width/jaw_thickness/padding` → `Meters`
- `Joint.range_rad: tuple[float, float]` → `tuple[Radians, Radians]`
- `ClearanceConstraint.min_distance: float` → `Meters`

- [ ] **Step 2: Migrate geometry.py**

- `MountRotation.yaw: float` → assess if this should be `Radians` (currently in degrees — note the inconsistency, flag for discussion)
- Euler constants — these are in degrees, used with build123d which expects degrees. Keep as-is but add comment.

- [ ] **Step 3: Update bot design files**

All `bots/*/design.py` files that set dimension values — wrap in `Meters()`.

- [ ] **Step 4: Run full validation**

Run: `make lint && make validate`

- [ ] **Step 5: Commit**

```bash
git commit -am "refactor: migrate skeleton/geometry dimension fields to Meters/Radians"
```

---

## Phase 0c: Remove Assembly Mode

### Task 0c-1: Remove assembly mode from viewer

**Files:**
- Delete: `viewer/assembly-mode.ts`
- Delete: `viewer/tests/assembly-mode.spec.mjs`
- Modify: `viewer/bot-viewer.ts` (lines 12, 653, 675, 680)
- Modify: `viewer/bot-scene.ts` (line 23 — remove 'assembly' from ViewerMode union)
- Modify: `viewer/index.html` (remove assembly button, step-info CSS)
- Modify: `viewer/manifest-types.ts` (remove ManifestAssembly interface, assemblies field)
- Modify: `viewer/test_viewer.mjs` (remove assembly test block, update modeNames)

- [ ] **Step 1: Delete assembly-mode.ts and its test**

- [ ] **Step 2: Remove imports and references in bot-viewer.ts**

Remove import, mode instantiation (`modes.assembly`), and entry from `simModeNames`/`simModeLabels`.

- [ ] **Step 3: Remove 'assembly' from ViewerMode type in bot-scene.ts**

- [ ] **Step 4: Remove assembly button and step-info CSS from index.html**

- [ ] **Step 5: Remove ManifestAssembly from manifest-types.ts**

- [ ] **Step 6: Update viewer tests**

Remove assembly test block from `test_viewer.mjs`, update `modeNames` array.

- [ ] **Step 7: Run viewer lint and tests**

Run: `pnpm exec biome check --write viewer/ && pnpm exec tsc --noEmit`

- [ ] **Step 8: Commit**

```bash
git commit -am "cleanup: remove assembly mode from viewer — superseded by DFM mode"
```

### Task 0c-2: Remove assembly step generation from Python

**Files:**
- Modify: `botcad/emit/viewer.py` (remove `_build_assembly_tree()`, assembly_steps generation)
- Modify: `tests/test_viewer_api.py` (remove `test_assembly_steps()`)

- [ ] **Step 1: Remove `_build_assembly_tree()` and assembly_steps from manifest**

In `viewer.py`:
- Delete `_build_assembly_tree()` function (lines ~67-82)
- Remove `"assemblies"` key from manifest dict
- Remove `"assembly_steps"` initialization and population in `_walk_body()`

- [ ] **Step 2: Remove test_assembly_steps from test_viewer_api.py**

- [ ] **Step 3: Run full validation**

Run: `make lint && make validate`

- [ ] **Step 4: Commit**

```bash
git commit -am "cleanup: remove assembly step generation from Python manifest"
```

---

## Phase 1: Assembly Sequence Model

**Note:** Create `botcad/assembly/__init__.py` (empty) before any tasks in this phase.

### Task 1-1: Create assembly ref types

**Files:**
- Create: `botcad/assembly/__init__.py` (empty)
- Create: `botcad/assembly/refs.py`
- Test: `tests/test_assembly_refs.py`

- [ ] **Step 1: Write tests for ComponentRef, FastenerRef, WireRef**

```python
# tests/test_assembly_refs.py
from botcad.ids import BodyId
from botcad.assembly.refs import ComponentRef, FastenerRef, WireRef

def test_component_ref_frozen():
    ref = ComponentRef(body=BodyId("base"), mount_label="battery")
    assert ref.body == BodyId("base")
    assert ref.mount_label == "battery"

def test_component_ref_hashable():
    a = ComponentRef(body=BodyId("base"), mount_label="battery")
    b = ComponentRef(body=BodyId("base"), mount_label="battery")
    assert a == b
    assert len({a, b}) == 1

def test_fastener_ref():
    ref = FastenerRef(body=BodyId("base"), index=0)
    assert ref.body.name == "base"

def test_wire_ref():
    ref = WireRef(label="left_wheel_uart")
    assert ref.label == "left_wheel_uart"
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement refs.py**
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -am "feat: assembly ref types — ComponentRef, FastenerRef, WireRef"
```

### Task 1-2: Create tool library

**Files:**
- Create: `botcad/assembly/tools.py`
- Test: `tests/test_assembly_tools.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_assembly_tools.py
from botcad.assembly.tools import ToolKind, TOOL_LIBRARY

def test_tool_library_complete():
    """Every ToolKind has a corresponding ToolSpec."""
    for kind in ToolKind:
        assert kind in TOOL_LIBRARY

def test_hex_key_2_5_dimensions():
    spec = TOOL_LIBRARY[ToolKind.HEX_KEY_2_5]
    assert spec.shaft_diameter == 0.0025  # 2.5mm in meters
    assert spec.shaft_length == 0.050     # 50mm

def test_tool_spec_frozen():
    spec = TOOL_LIBRARY[ToolKind.FINGERS]
    import pytest
    with pytest.raises(AttributeError):
        spec.shaft_diameter = 0.1
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement tool library with all ToolKind entries and ToolSpec**

Include `ToolSpec.solid` callables that return simple build123d geometry (cylinder for hex keys, box for fingers/tweezers). These are used for clearance ray-casting and viewer visualization.

- [ ] **Step 4: Write test for tool solid generation**

```python
def test_tool_solid_generates_shape():
    spec = TOOL_LIBRARY[ToolKind.HEX_KEY_2_5]
    shape = spec.solid()
    assert shape is not None
    # Volume should be reasonable for a 2.5mm hex key
    assert shape.volume > 0
```

- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Commit**

```bash
git commit -am "feat: tool library — ToolKind enum, ToolSpec, clearance dimensions, solid geometry"
```

### Task 1-3: Create AssemblyOp and AssemblySequence

**Files:**
- Create: `botcad/assembly/sequence.py`
- Test: `tests/test_assembly_sequence.py`

- [ ] **Step 1: Write tests for AssemblyOp, AssemblyState, AssemblySequence**

```python
# tests/test_assembly_sequence.py
from botcad.ids import BodyId, JointId
from botcad.units import Radians
from botcad.assembly.refs import ComponentRef, FastenerRef
from botcad.assembly.sequence import (
    AssemblyAction, AssemblyOp, AssemblyState, AssemblySequence,
)
from botcad.assembly.tools import ToolKind

def test_assembly_op_frozen():
    op = AssemblyOp(
        step=0,
        action=AssemblyAction.INSERT,
        target=ComponentRef(body=BodyId("base"), mount_label="battery"),
        body=BodyId("base"),
        tool=ToolKind.FINGERS,
        approach_axis=(0, 0, -1),
        angle=None,
        prerequisites=(),
        description="Insert battery into base pocket",
    )
    assert op.action == AssemblyAction.INSERT

def test_state_at_accumulates():
    ops = [
        AssemblyOp(step=0, action=AssemblyAction.INSERT,
                   target=ComponentRef(BodyId("base"), "battery"),
                   body=BodyId("base"), tool=ToolKind.FINGERS,
                   approach_axis=(0,0,-1), angle=None,
                   prerequisites=(), description="Insert battery"),
        AssemblyOp(step=1, action=AssemblyAction.FASTEN,
                   target=FastenerRef(BodyId("base"), 0),
                   body=BodyId("base"), tool=ToolKind.HEX_KEY_2_5,
                   approach_axis=(0,0,1), angle=None,
                   prerequisites=(0,), description="Fasten bracket"),
    ]
    seq = AssemblySequence(ops=ops)
    state = seq.state_at(0)
    assert ComponentRef(BodyId("base"), "battery") in state.installed_components
    assert len(state.installed_fasteners) == 0

    state = seq.state_at(1)
    assert FastenerRef(BodyId("base"), 0) in state.installed_fasteners

def test_articulate_sets_joint_angle():
    ops = [
        AssemblyOp(step=0, action=AssemblyAction.ARTICULATE,
                   target=JointId("shoulder_yaw"),
                   body=BodyId("turntable"), tool=None,
                   approach_axis=None, angle=Radians(1.5708),
                   prerequisites=(), description="Rotate shoulder"),
    ]
    seq = AssemblySequence(ops=ops)
    state = seq.state_at(0)
    assert state.joint_angles[JointId("shoulder_yaw")] == Radians(1.5708)
```

- [ ] **Step 2: Write test for geometry_at()**

```python
def test_geometry_at_includes_body_shells():
    """Body shells are always present from step 0 — they're what you print."""
    from unittest.mock import MagicMock
    body_solids = {BodyId("base"): MagicMock()}
    ops = [
        AssemblyOp(step=0, action=AssemblyAction.INSERT,
                   target=ComponentRef(BodyId("base"), "battery"),
                   body=BodyId("base"), tool=ToolKind.FINGERS,
                   approach_axis=(0,0,-1), angle=None,
                   prerequisites=(), description="Insert battery"),
    ]
    seq = AssemblySequence(ops=ops)
    # Even at step -1 (before any ops), body shells should be present
    state = seq.state_at(-1)
    assert len(state.installed_components) == 0
```

- [ ] **Step 3: Run tests, verify fail**
- [ ] **Step 4: Implement sequence.py** — include `state_at()` and `geometry_at()`. `geometry_at()` takes `dict[BodyId, Shape]` of printed body shells (always present) and adds component/fastener solids incrementally based on `state_at(step)`.
- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Commit**

```bash
git commit -am "feat: AssemblyOp, AssemblyState, AssemblySequence with state_at() and geometry_at()"
```

### Task 1-4: Build assembly sequence generator

**Files:**
- Create: `botcad/assembly/build.py`
- Test: `tests/test_assembly_build.py`

- [ ] **Step 1: Write integration test using wheeler_base**

```python
# tests/test_assembly_build.py
from botcad.assembly.build import build_assembly_sequence
from bots.wheeler_base.design import wheeler_base

def test_wheeler_base_sequence_structure():
    bot = wheeler_base()
    seq = build_assembly_sequence(bot)
    # Every op has a valid action
    for op in seq.ops:
        assert op.action is not None
    # Steps are sequential
    assert [op.step for op in seq.ops] == list(range(len(seq.ops)))
    # Prerequisites reference valid earlier steps
    for op in seq.ops:
        for prereq in op.prerequisites:
            assert prereq < op.step

def test_wheeler_base_has_fasten_ops():
    bot = wheeler_base()
    seq = build_assembly_sequence(bot)
    fasten_ops = [op for op in seq.ops if op.action == AssemblyAction.FASTEN]
    assert len(fasten_ops) > 0
    # Every fasten op has a tool
    for op in fasten_ops:
        assert op.tool is not None

def test_wheeler_base_has_insert_ops():
    bot = wheeler_base()
    seq = build_assembly_sequence(bot)
    insert_ops = [op for op in seq.ops if op.action == AssemblyAction.INSERT]
    # At least battery, Pi, camera, waveshare, BEC
    assert len(insert_ops) >= 5

def test_wheeler_base_has_route_and_connect_ops():
    """Wire checks depend on ROUTE_WIRE and CONNECT ops being present."""
    bot = wheeler_base()
    seq = build_assembly_sequence(bot)
    route_ops = [op for op in seq.ops if op.action == AssemblyAction.ROUTE_WIRE]
    connect_ops = [op for op in seq.ops if op.action == AssemblyAction.CONNECT]
    assert len(route_ops) > 0, "Expected ROUTE_WIRE ops for servo wiring"
    assert len(connect_ops) > 0, "Expected CONNECT ops for servo connectors"
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement `build_assembly_sequence()`**

Walk the bot skeleton tree. For each body:
1. Emit INSERT ops for each mounted component (approach_axis from mount insertion_axis)
2. Emit FASTEN ops for each fastener (tool from fastener spec, approach_axis from MountPoint.axis negated)
3. Emit ROUTE_WIRE ops for wire segments in this body
4. Emit CONNECT ops for connector ports

Order: structural bodies first, then joints/children. Within a body: servos → brackets → components → fasteners → wires.

Derive tool selection from fastener spec: M2/M2.5 socket head → HEX_KEY_2/HEX_KEY_2_5, M3 → HEX_KEY_3, pan head phillips → PHILLIPS_0/PHILLIPS_1.

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Run full validation**

Run: `make lint && make validate`

- [ ] **Step 6: Commit**

```bash
git commit -am "feat: build_assembly_sequence() — walks skeleton to emit typed ops"
```

---

## Phase 2: DFM Check Framework + Tier 1

**Note:** Create `botcad/dfm/__init__.py` and `botcad/dfm/checks/__init__.py` (both empty) before any tasks in this phase.

### Task 2-1: Create DFM check base class and finding model

**Files:**
- Create: `botcad/dfm/__init__.py` (empty)
- Create: `botcad/dfm/checks/__init__.py` (empty)
- Create: `botcad/dfm/check.py`
- Test: `tests/test_dfm_check.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_dfm_check.py
from botcad.dfm.check import DFMFinding, DFMSeverity, DFMCheck

def test_finding_id_deterministic():
    from botcad.ids import BodyId
    from botcad.assembly.refs import FastenerRef
    f = DFMFinding(
        check_name="fastener_tool_clearance",
        severity=DFMSeverity.ERROR,
        body=BodyId("base"),
        target=FastenerRef(BodyId("base"), 0),
        assembly_step=3,
        title="M3 screw inaccessible",
        description="...",
        pos=(0.065, 0.0, -0.017),
        direction=(0, 0, 1),
        measured=Meters(0.0),
        threshold=Meters(0.025),
        has_overlay=True,
    )
    # ID format: "{check_name}:{body}:{target_type}:{target_key}"
    # FastenerRef serializes as "fastener:base:0"
    assert f.id == "fastener_tool_clearance:base:fastener:base:0"

def test_finding_frozen():
    # ... same as above, verify frozen
    import pytest
    with pytest.raises(AttributeError):
        f.severity = DFMSeverity.WARNING
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement check.py with DFMCheck ABC, DFMSeverity, DFMFinding**
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -am "feat: DFM check framework — DFMCheck ABC, DFMFinding, DFMSeverity"
```

### Task 2-2: Implement fastener tool clearance check

**Files:**
- Create: `botcad/dfm/checks/fastener_clearance.py`
- Test: `tests/test_dfm_fastener_clearance.py`

- [ ] **Step 1: Write test using wheeler_base (known failure)**

```python
# tests/test_dfm_fastener_clearance.py
from botcad.emit.cad import build_cad  # existing CAD pipeline

def test_wheeler_base_has_fastener_clearance_errors():
    """Wheeler_base servo ears extend beyond body walls — known blocker."""
    bot = wheeler_base()
    cad_model = build_cad(bot)  # returns CadModel with .body_solids dict
    seq = build_assembly_sequence(bot)
    check = FastenerToolClearance()
    findings = check.run(bot, seq, cad_model.body_solids)
    errors = [f for f in findings if f.severity == DFMSeverity.ERROR]
    assert len(errors) > 0  # we know this is broken
```

- [ ] **Step 2: Run tests, verify fail (check not implemented)**
- [ ] **Step 3: Implement FastenerToolClearance check**

For each FASTEN op in the assembly sequence:
- Get fastener position and approach_axis
- Get required tool from TOOL_LIBRARY
- Build tool envelope (cylinder: head_diameter, shaft_length)
- Sample points on envelope, ray-cast against `geometry_at(step)`
- Report findings with clearance measurements

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -am "feat: fastener tool clearance DFM check"
```

### Task 2-3: Implement wire channel sizing check

**Files:**
- Create: `botcad/dfm/checks/wire_channel_sizing.py`
- Test: `tests/test_dfm_wire_channel.py`

- [ ] **Step 1: Write test**
- [ ] **Step 2: Implement check** — compare channel cross-section to ConnectorSpec.body_dimensions at entry/exit. Lookup chain: WireRoute → WirePort → connector_type → ConnectorSpec.
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -am "feat: wire channel sizing DFM check"
```

### Task 2-4: Implement wire bend radius check

**Files:**
- Create: `botcad/dfm/checks/wire_bend_radius.py`
- Test: `tests/test_dfm_wire_bend.py`

- [ ] **Step 1: Write test**
- [ ] **Step 2: Implement check** — at each WireSegment junction, compute angle between consecutive segment directions. Effective bend radius = segment_length / (2 * sin(angle/2)). Compare to 5x cable OD (static) or 10x (joint-crossing).
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -am "feat: wire bend radius DFM check"
```

### Task 2-5: Implement component retention check

**Files:**
- Create: `botcad/dfm/checks/component_retention.py`
- Test: `tests/test_dfm_retention.py`

- [ ] **Step 1: Write test (wheeler_base battery has no retention)**
- [ ] **Step 2: Implement check** — for each mounted component, check if any FASTEN ops target it or if a retention feature exists.
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -am "feat: component retention DFM check"
```

### Task 2-6: Implement connector mating access check

**Files:**
- Create: `botcad/dfm/checks/connector_access.py`
- Test: `tests/test_dfm_connector_access.py`

- [ ] **Step 1: Write test**
- [ ] **Step 2: Implement check** — for each connector port, ray-cast along mating axis, check 15mm clear + 10mm lateral.
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -am "feat: connector mating access DFM check"
```

### Task 2-7: DFM runner with check discovery

**Files:**
- Create: `botcad/dfm/runner.py`
- Test: `tests/test_dfm_runner.py`

- [ ] **Step 1: Write test**

```python
def test_runner_discovers_all_checks():
    from botcad.dfm.runner import discover_checks
    checks = discover_checks()
    names = {c.name for c in checks}
    expected = {"fastener_tool_clearance", "wire_channel_sizing", "wire_bend_radius",
                "component_retention", "connector_mating_access"}
    assert names.issuperset(expected), f"Missing checks: {expected - names}"

def test_runner_produces_findings():
    bot = wheeler_base()
    findings = run_dfm(bot)
    assert len(findings) > 0
```

- [ ] **Step 2: Implement runner** — discover DFMCheck subclasses, run each, collect findings.
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -am "feat: DFM runner with subclass discovery"
```

### Task 2-8: Wheeler_base DFM regression test

**Files:**
- Create: `tests/test_dfm_wheeler_base.py`

- [ ] **Step 1: Write regression test**

```python
# tests/test_dfm_wheeler_base.py
"""Regression test: run all DFM checks against wheeler_base, assert known findings."""
from botcad.dfm.runner import run_dfm
from bots.wheeler_base.design import wheeler_base

def test_wheeler_base_has_known_errors():
    bot = wheeler_base()
    findings = run_dfm(bot)
    errors = [f for f in findings if f.severity == DFMSeverity.ERROR]
    # Known blockers: fastener access, wire channel sizing
    assert len(errors) > 0
    check_names = {f.check_name for f in errors}
    assert "fastener_tool_clearance" in check_names

def test_wheeler_base_battery_no_retention():
    bot = wheeler_base()
    findings = run_dfm(bot)
    retention = [f for f in findings if f.check_name == "component_retention"]
    assert len(retention) > 0
```

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -am "test: wheeler_base DFM regression — known findings baseline"
```

### Task 2-9: Deprecate prose assembly guide

**Files:**
- Modify: `botcad/emit/readme.py` — add deprecation notice at top

- [ ] **Step 1: Add deprecation comment to readme.py**

Add at the top of the file:
```python
# DEPRECATED: This prose assembly guide is superseded by the structured
# AssemblySequence in botcad/assembly/. Will be removed in a future cleanup.
# See docs/superpowers/specs/2026-03-28-dfm-validation-design.md
```

- [ ] **Step 2: Commit**

```bash
git commit -am "chore: deprecate prose assembly guide (readme.py) — replaced by AssemblySequence"
```

---

## Phase 3: API Surface

### Task 3-1: Assembly sequence endpoint

**Files:**
- Modify: `mindsim/server.py`
- Test: `tests/test_dfm_api.py`

- [ ] **Step 1: Write test**

```python
def test_assembly_sequence_endpoint(client):
    resp = client.get("/api/bots/wheeler_base/assembly-sequence")
    assert resp.status_code == 200
    data = resp.json()
    assert "ops" in data
    assert "tool_library" in data
    assert len(data["ops"]) > 0
```

- [ ] **Step 2: Implement endpoint**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -am "feat: GET /api/bots/{bot}/assembly-sequence endpoint"
```

### Task 3-2: Async DFM run/status/findings endpoints

**Files:**
- Modify: `mindsim/server.py`
- Test: `tests/test_dfm_api.py`

- [ ] **Step 1: Write tests for POST run, GET status, GET findings**

```python
def test_dfm_run_returns_run_id(client):
    resp = client.post("/api/bots/wheeler_base/dfm/run")
    assert resp.status_code == 200
    assert "run_id" in resp.json()

def test_dfm_status_tracks_progress(client):
    run_id = client.post("/api/bots/wheeler_base/dfm/run").json()["run_id"]
    # poll until complete (with timeout)
    ...
    resp = client.get(f"/api/bots/wheeler_base/dfm/{run_id}/status")
    assert resp.json()["state"] in ("running", "complete")

def test_dfm_findings_returns_results(client):
    run_id = client.post("/api/bots/wheeler_base/dfm/run").json()["run_id"]
    # wait for completion...
    resp = client.get(f"/api/bots/wheeler_base/dfm/{run_id}/findings")
    assert len(resp.json()["findings"]) > 0
```

- [ ] **Step 2: Implement async run management** — background thread per run, stores state in dict keyed by run_id. Each check runs independently, reports findings incrementally.
- [ ] **Step 3: Implement overlay mesh endpoint** — `GET /api/bots/{bot}/dfm/{run_id}/overlays/{finding_id}.stl`
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git commit -am "feat: async DFM run/status/findings/overlay API endpoints"
```

---

## Phase 4: Viewer DFM Mode

### Task 4-1: DFM panel (findings table)

**Files:**
- Create: `viewer/dfm-panel.ts`

- [ ] **Step 1: Implement DFM panel** — sortable table of findings. Columns: severity icon, check name, body, title, measured vs threshold. Click row fires event with finding data.
- [ ] **Step 2: Run viewer lint**

Run: `pnpm exec biome check --write viewer/ && pnpm exec tsc --noEmit`

- [ ] **Step 3: Commit**

```bash
git commit -am "feat: DFM findings panel — sortable, filterable table"
```

### Task 4-2: Assembly step scrubber

**Files:**
- Create: `viewer/assembly-scrubber.ts`

- [ ] **Step 1: Implement scrubber** — slider control that emits step-change events. Shows current step description, total step count.
- [ ] **Step 2: Run viewer lint**
- [ ] **Step 3: Commit**

```bash
git commit -am "feat: assembly step scrubber — slider for incremental build-up"
```

### Task 4-3: DFM mode (orchestration)

**Files:**
- Create: `viewer/dfm-mode.ts`
- Modify: `viewer/bot-viewer.ts` (register DFM mode)
- Modify: `viewer/bot-scene.ts` (add 'dfm' to ViewerMode)
- Modify: `viewer/index.html` (add DFM button)

- [ ] **Step 1: Implement DFM mode**

On activate:
- Fetch assembly sequence from API
- POST dfm/run, poll status, stream findings to panel
- Wire panel click → camera fly-to + body isolation + overlay load
- Wire scrubber step-change → update body visibility per `state_at(step)` + filter findings

On deactivate:
- Remove overlays, restore visibility

- [ ] **Step 2: Register mode in bot-viewer.ts, add to ViewerMode type, add button**
- [ ] **Step 3: Run viewer lint and type check**

Run: `pnpm exec biome check --write viewer/ && pnpm exec tsc --noEmit`

- [ ] **Step 4: Manual test** — load wheeler_base in viewer, activate DFM mode, verify findings appear, step scrubber works, click finding flies camera to location.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: DFM viewer mode — findings, step scrubber, overlays, tool viz"
```

---

## Phase 5: Sim Validation (Parallel Workstream)

Independent of Phases 0-4. Can start immediately.

### Task 5-1: Wheeler_base sim smoke test

**Files:**
- Create: `tests/test_sim_validation.py`

- [ ] **Step 1: Write test — loads without error**

```python
# tests/test_sim_validation.py
import mujoco
from pathlib import Path

_ROOT = Path(__file__).parent.parent
SCENE_XML = _ROOT / "bots/wheeler_base/scene.xml"

def test_loads_without_error():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)  # one step without crash

def test_mass_matches_bom():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    total_mass = sum(model.body_mass)
    assert abs(total_mass - 0.468) / 0.468 < 0.02  # within 2%
```

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -am "test: wheeler_base sim loads and mass matches BOM"
```

### Task 5-2: Driving behavior tests

**Files:**
- Modify: `tests/test_sim_validation.py`

- [ ] **Step 1: Write test — drives straight**

```python
def test_drives_straight():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    # Apply equal velocity to both wheels
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_motor")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_motor")
    for _ in range(1000):  # ~1 second
        data.ctrl[left_id] = 1.0
        data.ctrl[right_id] = 1.0
        mujoco.mj_step(model, data)
    # Bot should have moved forward (positive Y or X depending on orientation)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    pos = data.xpos[base_id]
    # Should have moved significantly, and not veered far off-axis
    forward_dist = abs(pos[0]) + abs(pos[1])  # adjust axis based on bot orientation
    assert forward_dist > 0.05  # moved at least 5cm
```

- [ ] **Step 2: Write test — turns, doesn't tip, ground contact stable**
- [ ] **Step 3: Run tests, debug any physics issues**
- [ ] **Step 4: Generate filmstrip visualization for visual confirmation**
- [ ] **Step 5: Commit**

```bash
git commit -am "test: wheeler_base driving behavior — straight, turn, stability"
```

---

## Dependency Graph

```
Phase 0a (StringId) ──→ Phase 0b (Dimensions) ──┐
                                                 ├──→ Phase 1 (Assembly Seq) ──→ Phase 2 (DFM Checks) ──→ Phase 3 (API) ──→ Phase 4 (Viewer)
Phase 0c (Remove Asm) ──────────────────────────┘

Phase 5 (Sim Validation) ──→ (independent, start anytime)
```

Phase 0a must complete before 0b (both modify `skeleton.py` and bot design files). Phase 0c is independent of both. Phase 5 is fully independent.

## Commit Strategy

- One commit per task step (as marked)
- Merge `exp/260328-dfm-validation` → `master` when Phase 4 complete
- Phase 5 can merge independently
