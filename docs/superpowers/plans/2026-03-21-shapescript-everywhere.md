# ShapeScript Everywhere Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port all geometry factories to emit ShapeScript programs so every piece of geometry in the pipeline is inspectable, cacheable, and debuggable in the web viewer.

**Architecture:** Component factories (bracket_solid, battery_solid, servo_solid, etc.) gain a `*_script()` companion that returns a `ShapeScript` program. The body emitter uses `CallOp` to invoke sub-programs instead of `PrebuiltOp`. The web viewer shows expandable sub-programs. Shape references enable define-once-use-many semantics.

**Tech Stack:** Python 3.11+, build123d (OCCT), ShapeScript (`botcad/shapescript/`), Three.js web viewer.

**Review fixes applied:** (1) Extend `Align3` to per-axis alignment before porting components — camera needs MAX_Y, battery needs MIN_X. (2) `to_json()` and `content_hash()` both updated for sub-programs. (3) `cad_steps.py` guards `op.ref` access with `hasattr`. (4) `bracket_envelope_script` uses PrebuiltOp wrapper initially (don't re-derive math). (5) `CallOp` backend execution is intentionally uncached — correctness over performance for now.

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `botcad/shapescript/emit_components.py` | ShapeScript emitters for all component factories (battery, camera, bearing, horn, wire) |
| `botcad/shapescript/emit_bracket.py` | ShapeScript emitters for bracket_solid, bracket_envelope, cradle, coupler |
| `botcad/shapescript/emit_servo.py` | ShapeScript emitters for servo_solid (_sts_series, _scs0009) |
| `tests/test_shapescript_components.py` | Roundtrip tests: ShapeScript vs direct build123d for every component |
| `tests/test_shapescript_bracket.py` | Roundtrip tests for bracket/coupler/cradle |

### Modified files

| File | Change |
|------|--------|
| `botcad/shapescript/ops.py` | Add `CallOp` for sub-program invocation |
| `botcad/shapescript/program.py` | Add `call()` builder method, `sub_programs` dict |
| `botcad/shapescript/backend_occt.py` | Handle `CallOp` execution |
| `botcad/shapescript/emit_body.py` | Replace `PrebuiltOp` with `CallOp` for brackets/components |
| `botcad/shapescript/cad_steps.py` | Expand `CallOp` into nested step groups |
| `main.py` | Update cad-steps API to include sub-program metadata |
| `viewer/cad-steps-mode.js` | Expandable sub-program groups in step list |

---

## Phase 1: Language Extension — Align3 + CallOp + Sub-Programs

### Task 1: Extend Align3 + Add CallOp to ShapeScript

**Files:**
- Modify: `botcad/shapescript/ops.py`
- Modify: `botcad/shapescript/program.py`
- Modify: `botcad/shapescript/backend_occt.py`
- Test: `tests/test_shapescript_ops.py`
- Test: `tests/test_shapescript_backend.py`

Two language extensions:

**Align3 expansion:** Currently only `CENTER` and `MIN_Z`. Component factories need per-axis alignment (camera connector uses `Align.MAX` on Y, battery exit uses `Align.MIN` on X). Replace the enum with a tuple approach:

```python
# In ops.py, replace Align3 enum with:
@dataclass(frozen=True)
class Align3:
    """Per-axis alignment for primitives. Maps to build123d Align enum."""
    x: str = "center"  # "center" | "min" | "max"
    y: str = "center"
    z: str = "center"

# Convenience constants:
ALIGN_CENTER = Align3()  # centered on all axes
ALIGN_MIN_Z = Align3(z="min")  # centered XY, min Z
ALIGN_MIN_X = Align3(x="min")  # min X, centered YZ
ALIGN_MAX_Y = Align3(y="max")  # centered XZ, max Y
```

Update `backend_occt.py`'s `_align()` function to map per-axis strings to `build123d.Align` values.

**CallOp:** Invokes a sub-program (another `ShapeScript`) and produces a shape. This is how a body program references bracket geometry — the bracket's construction is a sub-program that can be inspected independently.

- [ ] **Step 1: Write failing tests for Align3 expansion and CallOp**

Add to `tests/test_shapescript_ops.py`:

```python
from botcad.shapescript.ops import CallOp


class TestCallOp:
    def test_call_fields(self):
        op = CallOp(
            ref=ShapeRef("call_0"),
            sub_program_key="bracket_sts3215",
            tag="bracket",
        )
        assert op.sub_program_key == "bracket_sts3215"
        assert op.tag == "bracket"

    def test_call_is_frozen(self):
        op = CallOp(ref=ShapeRef("c0"), sub_program_key="test")
        with pytest.raises(AttributeError):
            op.sub_program_key = "other"
```

Add to `TestCadProgram` class:

```python
    def test_call_sub_program(self):
        # Create a sub-program (a simple box)
        sub = ShapeScript()
        b = sub.box(1, 1, 1)
        sub.output_ref = b

        # Create main program that calls the sub-program
        main = ShapeScript()
        main.sub_programs["my_box"] = sub
        ref = main.call("my_box", tag="the_box")
        assert isinstance(ref, ShapeRef)
        assert len(main.ops) == 1
        assert isinstance(main.ops[0], CallOp)
        assert main.ops[0].sub_program_key == "my_box"

    def test_content_hash_includes_sub_programs(self):
        p1 = ShapeScript()
        sub1 = ShapeScript()
        sub1.box(1, 1, 1)
        sub1.output_ref = sub1.ops[0].ref
        p1.sub_programs["a"] = sub1
        p1.call("a")

        p2 = ShapeScript()
        sub2 = ShapeScript()
        sub2.box(2, 2, 2)  # different dims
        sub2.output_ref = sub2.ops[0].ref
        p2.sub_programs["a"] = sub2
        p2.call("a")

        assert p1.content_hash() != p2.content_hash()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_shapescript_ops.py::TestCallOp tests/test_shapescript_ops.py::TestCadProgram::test_call_sub_program -v`
Expected: ImportError — `CallOp` does not exist

- [ ] **Step 3: Implement CallOp**

In `botcad/shapescript/ops.py`, add after `PrebuiltOp`:

```python
@dataclass(frozen=True)
class CallOp:
    """Invoke a sub-program and produce its output shape.

    The sub-program is stored in ShapeScript.sub_programs[sub_program_key].
    The backend executes it and stores the result at ref.
    """
    ref: ShapeRef
    sub_program_key: str
    tag: str | None = None
```

Update `PrimitiveOp` and `ShapeOp` union types to include `CallOp`.

In `botcad/shapescript/program.py`, add:
- `sub_programs: dict[str, ShapeScript] = field(default_factory=dict)` field
- `call(key, tag=None) -> ShapeRef` builder method
- Update `content_hash()` to include sub-program hashes:
  ```python
  def content_hash(self) -> str:
      parts = [self.to_json()]
      for key in sorted(self.sub_programs):
          parts.append(f"{key}:{self.sub_programs[key].content_hash()}")
      return hashlib.sha256("|".join(parts).encode()).hexdigest()
  ```

In `botcad/shapescript/backend_occt.py`, add `CallOp` case:

```python
case CallOp(ref=ref, sub_program_key=key, tag=tag):
    sub_prog = program.sub_programs.get(key)
    if sub_prog is None:
        raise ValueError(f"CallOp: sub-program '{key}' not found")
    sub_result = self.execute(sub_prog)
    if sub_prog.output_ref is None:
        raise ValueError(f"CallOp: sub-program '{key}' has no output_ref")
    shapes[ref.id] = sub_result.shapes[sub_prog.output_ref.id]
    if tag:
        tags.declare(tag, ref)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_shapescript_ops.py -v`
Expected: All pass

- [ ] **Step 5: Write backend integration test for CallOp**

Add to `tests/test_shapescript_backend.py`:

```python
class TestCallOp:
    def test_call_produces_solid(self):
        """CallOp should execute sub-program and produce its output."""
        sub = ShapeScript()
        b = sub.box(1, 1, 1)
        sub.output_ref = b

        main = ShapeScript()
        main.sub_programs["box_maker"] = sub
        ref = main.call("box_maker")
        main.query_volume(ref)

        r = _exec(main)
        assert r.queries[0] == pytest.approx(1.0, rel=1e-6)

    def test_call_then_cut(self):
        """CallOp result should be usable in subsequent ops."""
        sub = ShapeScript()
        b = sub.box(1, 1, 1)
        sub.output_ref = b

        main = ShapeScript()
        main.sub_programs["box_maker"] = sub
        box = main.call("box_maker")
        hole = main.cylinder(0.1, 2)
        result = main.cut(box, hole)
        main.query_volume(result)

        r = _exec(main)
        expected = 1.0 - math.pi * 0.1**2 * 1.0
        assert r.queries[0] == pytest.approx(expected, rel=0.01)

    def test_call_with_locate(self):
        """CallOp result can be moved."""
        sub = ShapeScript()
        b = sub.box(1, 1, 1)
        sub.output_ref = b

        main = ShapeScript()
        main.sub_programs["box_maker"] = sub
        ref = main.call("box_maker")
        moved = main.locate(ref, pos=(10, 0, 0))
        main.query_centroid(moved)

        r = _exec(main)
        assert r.queries[0][0] == pytest.approx(10.0, abs=0.01)

    def test_same_sub_program_called_twice(self):
        """Two calls to same sub-program produce independent shapes."""
        sub = ShapeScript()
        b = sub.box(1, 1, 1)
        sub.output_ref = b

        main = ShapeScript()
        main.sub_programs["box_maker"] = sub
        a = main.call("box_maker")
        a = main.locate(a, pos=(5, 0, 0))
        b = main.call("box_maker")
        b = main.locate(b, pos=(-5, 0, 0))
        c = main.fuse(a, b)
        main.query_volume(c)

        r = _exec(main)
        assert r.queries[0] == pytest.approx(2.0, rel=0.001)
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/test_shapescript_ops.py tests/test_shapescript_backend.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add botcad/shapescript/ops.py botcad/shapescript/program.py botcad/shapescript/backend_occt.py tests/test_shapescript_ops.py tests/test_shapescript_backend.py
git commit -m "feat(shapescript): add CallOp for sub-program invocation

Sub-programs are stored in ShapeScript.sub_programs dict, invoked via
call() builder. The backend recursively executes sub-programs. Same
sub-program can be called multiple times (define once, use many)."
```

---

## Phase 2: Port Simple Component Factories

### Task 2: Port battery_solid, camera_solid, bearing_solid, horn_solid

**Files:**
- Create: `botcad/shapescript/emit_components.py`
- Create: `tests/test_shapescript_components.py`

These are the simplest factories — Box/Cylinder primitives, fuse, cut, locate. No fillets (except battery which uses fillet — we'll use PrebuiltOp as a fillet wrapper for now).

- [ ] **Step 1: Write roundtrip tests**

```python
# tests/test_shapescript_components.py
"""Roundtrip tests: ShapeScript component emitters must match direct build123d."""
from __future__ import annotations

import math

import pytest

b3d = pytest.importorskip("build123d")

from botcad.shapescript.backend_occt import OcctBackend


def _exec(prog):
    return OcctBackend().execute(prog)


def _total_volume(solid):
    solids = solid.solids() if hasattr(solid, "solids") else [solid]
    return sum(abs(s.volume) for s in solids)


class TestCameraSolidScript:
    def test_volume_matches_direct(self):
        from botcad.components.camera import OV5647, camera_solid
        from botcad.shapescript.emit_components import camera_script

        spec = OV5647()
        direct_solid = camera_solid(spec)
        direct_vol = _total_volume(direct_solid)

        prog = camera_script(spec)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        ir_vol = _total_volume(ir_solid)

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"camera: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )


class TestBatterySolidScript:
    def test_volume_matches_direct(self):
        from botcad.components.battery import LiPo2S, battery_solid
        from botcad.shapescript.emit_components import battery_script

        spec = LiPo2S(1000)
        direct_solid = battery_solid(spec)
        direct_vol = _total_volume(direct_solid)

        prog = battery_script(spec)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        ir_vol = _total_volume(ir_solid)

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"battery: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )


class TestBearingSolidScript:
    def test_volume_matches_direct(self):
        from botcad.shapescript.emit_components import bearing_script

        # Simple bearing: outer cylinder - inner cylinder
        from botcad.component import BearingSpec
        spec = BearingSpec(
            name="test_bearing",
            dimensions=(0.01, 0.01, 0.004),
            mass=0.002,
            od=0.01, id=0.004, width=0.004,
        )
        from botcad.emit.cad import _make_bearing_solid
        direct_solid = _make_bearing_solid(spec)
        direct_vol = _total_volume(direct_solid)

        prog = bearing_script(spec)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        ir_vol = _total_volume(ir_solid)

        assert ir_vol == pytest.approx(direct_vol, rel=0.001)


class TestHornSolidScript:
    def test_volume_matches_direct(self):
        from botcad.components.servo import STS3215
        from botcad.emit.cad import _horn_solid
        from botcad.shapescript.emit_components import horn_script

        servo = STS3215()
        direct_solid = _horn_solid(servo)
        direct_vol = _total_volume(direct_solid)

        prog = horn_script(servo)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        ir_vol = _total_volume(ir_solid)

        assert ir_vol == pytest.approx(direct_vol, rel=0.001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_shapescript_components.py -v`
Expected: ImportError

- [ ] **Step 3: Implement component emitters**

```python
# botcad/shapescript/emit_components.py
"""ShapeScript emitters for component geometry.

Each function returns a ShapeScript program that produces the same
solid as the corresponding build123d factory. Roundtrip tests verify
volume equivalence.
"""
from __future__ import annotations

from botcad.shapescript.ops import Align3
from botcad.shapescript.program import ShapeScript


def camera_script(spec) -> ShapeScript:
    """Emit ShapeScript for a camera module solid.

    Mirrors botcad/components/camera.py:camera_solid().
    """
    prog = ShapeScript()

    pcb_w, pcb_h, _pcb_t = spec.dimensions
    pcb_thick = 0.0016

    # PCB
    pcb = prog.box(pcb_w, pcb_h, pcb_thick, tag="pcb")

    # Mounting holes
    for mp in spec.mounting_points:
        hole = prog.cylinder(mp.diameter / 2, pcb_thick + 0.001)
        hole = prog.locate(hole, pos=mp.pos)
        pcb = prog.cut(pcb, hole)

    # Lens base
    base_size = 0.0085
    base_height = 0.0050
    lens_y_offset = 0.0025
    lens_base = prog.box(base_size, base_size, base_height, align=Align3.MIN_Z)
    lens_base = prog.locate(lens_base, pos=(0, lens_y_offset, pcb_thick / 2))

    # Lens barrel
    barrel_r = 0.00325
    barrel_h = 0.0025
    barrel = prog.cylinder(barrel_r, barrel_h, align=Align3.MIN_Z)
    barrel = prog.locate(barrel, pos=(0, lens_y_offset, pcb_thick / 2 + base_height))

    # CSI connector — Align.MAX on Y means connector's max-Y edge sits at the position
    conn_w = 0.016
    conn_t = 0.002
    conn_h = 0.005
    connector = prog.box(conn_w, conn_t, conn_h, align=Align3(y="max", z="min"))
    connector = prog.locate(connector, pos=(0, -pcb_h / 2, pcb_thick / 2))

    # Fuse all
    result = prog.fuse(pcb, lens_base)
    result = prog.fuse(result, barrel)
    result = prog.fuse(result, connector)

    prog.output_ref = result
    return prog


def battery_script(spec) -> ShapeScript:
    """Emit ShapeScript for a battery pack solid.

    Mirrors botcad/components/battery.py:battery_solid().
    Battery uses fillet on cell edges — since fillet requires edge
    selection, we use PrebuiltOp for the filleted cell and compose
    the rest in ShapeScript.
    """
    from build123d import Align, Box, Location

    from botcad.cad_utils import as_solid as _as_solid

    prog = ShapeScript()
    w, length, h = spec.dimensions

    # Filleted cells are PrebuiltOp (fillet needs edge selection)
    C = (Align.CENTER, Align.CENTER, Align.CENTER)
    if spec.cells_s == 2:
        cell_w = length / 2 - 0.001
        cell = Box(w, cell_w, h, align=C)
        cell = _as_solid(cell).fillet(0.003, cell.edges())
        body = cell.moved(Location((0, -length / 4, 0))).fuse(
            cell.moved(Location((0, length / 4, 0)))
        )
    else:
        body = Box(w, length, h, align=C)
        body = _as_solid(body).fillet(0.003, body.edges())
    body_ref = prog.prebuilt(body, tag="battery_body")

    # Label (pure ShapeScript)
    label_w = w * 0.7
    label_l = length * 0.6
    label_t = 0.0005
    label = prog.box(label_w, label_l, label_t, align=Align3.MIN_Z, tag="label")
    label = prog.locate(label, pos=(0, 0, h / 2 - 0.0001))

    # Cable exit
    exit_w = 0.012
    exit_l = 0.006
    exit_h = h * 0.8
    # Direct path uses align=(Align.MIN, CENTER, CENTER) — MIN on X axis
    exit_block = prog.box(exit_l, exit_w, exit_h, align=Align3(x="min"), tag="cable_exit")
    exit_block = prog.locate(exit_block, pos=(w / 2 - 0.001, 0, 0))

    result = prog.fuse(body_ref, label)
    result = prog.fuse(result, exit_block)

    prog.output_ref = result
    return prog


def bearing_script(spec) -> ShapeScript:
    """Emit ShapeScript for a bearing solid (outer ring - inner bore)."""
    prog = ShapeScript()
    outer = prog.cylinder(spec.od / 2, spec.width, tag="outer_ring")
    inner = prog.cylinder(spec.id / 2, spec.width + 0.001, tag="inner_bore")
    result = prog.cut(outer, inner)
    prog.output_ref = result
    return prog


def horn_script(servo) -> ShapeScript:
    """Emit ShapeScript for a horn disc solid."""
    from botcad.bracket import horn_disc_params

    prog = ShapeScript()
    params = horn_disc_params(servo)
    if params is None:
        # Fallback: small disc
        disc = prog.cylinder(0.01, 0.002, tag="horn_disc")
    else:
        disc = prog.cylinder(params.radius, params.thickness, tag="horn_disc")
    prog.output_ref = disc
    return prog
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_shapescript_components.py -v`
Expected: All pass. If volume mismatches exist, debug by comparing op-by-op.

Note: `camera_script` may need alignment tweaks for the CSI connector (Align.MAX vs MIN). Adjust until volume matches within 0.1%.

- [ ] **Step 5: Commit**

```bash
git add botcad/shapescript/emit_components.py tests/test_shapescript_components.py
git commit -m "feat(shapescript): port camera, battery, bearing, horn to ShapeScript

Roundtrip tests verify volume equivalence with direct build123d.
Battery uses PrebuiltOp for filleted cells (fillet needs edge selection)."
```

---

### Task 3: Port bracket_envelope, cradle_envelope, wire_channel

**Files:**
- Create: `botcad/shapescript/emit_bracket.py`
- Create: `tests/test_shapescript_bracket.py`

These are simple box/cylinder + locate — no booleans needed.

- [ ] **Step 1: Write roundtrip tests**

```python
# tests/test_shapescript_bracket.py
"""Roundtrip tests for bracket/cradle ShapeScript emitters."""
from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")

from botcad.shapescript.backend_occt import OcctBackend

def _exec(prog):
    return OcctBackend().execute(prog)

def _total_volume(solid):
    solids = solid.solids() if hasattr(solid, "solids") else [solid]
    return sum(abs(s.volume) for s in solids)


class TestBracketEnvelopeScript:
    def test_volume_matches_direct(self):
        from botcad.bracket import BracketSpec, bracket_envelope
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_bracket import bracket_envelope_script

        servo = STS3215()
        spec = BracketSpec()
        direct = bracket_envelope(servo, spec)
        direct_vol = _total_volume(direct)

        prog = bracket_envelope_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001)


class TestCradleEnvelopeScript:
    def test_volume_matches_direct(self):
        from botcad.bracket import BracketSpec, cradle_envelope
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_bracket import cradle_envelope_script

        servo = STS3215()
        spec = BracketSpec()
        direct = cradle_envelope(servo, spec)
        direct_vol = _total_volume(direct)

        prog = cradle_envelope_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001)
```

- [ ] **Step 2: Implement bracket envelope emitters**

```python
# botcad/shapescript/emit_bracket.py
"""ShapeScript emitters for bracket, cradle, and coupler geometry."""
from __future__ import annotations

from botcad.shapescript.ops import Align3
from botcad.shapescript.program import ShapeScript


def bracket_envelope_script(servo, spec) -> ShapeScript:
    """Emit ShapeScript for bracket envelope (insertion channel).

    Wraps botcad/bracket.py:bracket_envelope() as PrebuiltOp for now.
    Don't re-derive the math — use the existing factory and wrap it.
    Will be ported to native ShapeScript ops when bracket_solid_script
    is validated.
    """
    from botcad.bracket import bracket_envelope
    prog = ShapeScript()
    solid = bracket_envelope(servo, spec)
    ref = prog.prebuilt(solid, tag="bracket_envelope")
    prog.output_ref = ref
    return prog


def cradle_envelope_script(servo, spec) -> ShapeScript:
    """Emit ShapeScript for cradle envelope.

    Mirrors botcad/bracket.py:cradle_envelope().
    """
    # Cradle is more complex — use PrebuiltOp for now
    from botcad.bracket import cradle_envelope
    prog = ShapeScript()
    solid = cradle_envelope(servo, spec)
    ref = prog.prebuilt(solid, tag="cradle_envelope")
    prog.output_ref = ref
    return prog
```

- [ ] **Step 3: Run tests, iterate until passing**

Run: `uv run pytest tests/test_shapescript_bracket.py -v`

- [ ] **Step 4: Commit**

```bash
git add botcad/shapescript/emit_bracket.py tests/test_shapescript_bracket.py
git commit -m "feat(shapescript): port bracket_envelope and cradle_envelope"
```

---

### Task 4: Port bracket_solid (the proof-of-concept)

**Files:**
- Modify: `botcad/shapescript/emit_bracket.py`
- Modify: `tests/test_shapescript_bracket.py`

This is the complex one — 20+ build123d ops including pocket cut, horn clearance hole, shaft boss, fastener holes, cable slot, and connector port. It proves that complex bracket geometry can be fully expressed in ShapeScript.

- [ ] **Step 1: Write roundtrip test**

Add to `tests/test_shapescript_bracket.py`:

```python
class TestBracketSolidScript:
    def test_sts3215_volume_matches(self):
        from botcad.bracket import BracketSpec, bracket_solid
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_bracket import bracket_solid_script

        servo = STS3215()
        spec = BracketSpec()
        direct = bracket_solid(servo, spec)
        direct_vol = _total_volume(direct)

        prog = bracket_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"bracket: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )

    def test_scs0009_volume_matches(self):
        from botcad.bracket import BracketSpec, bracket_solid
        from botcad.components.servo import SCS0009
        from botcad.shapescript.emit_bracket import bracket_solid_script

        servo = SCS0009()
        spec = BracketSpec()
        direct = bracket_solid(servo, spec)
        direct_vol = _total_volume(direct)

        prog = bracket_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001)
```

- [ ] **Step 2: Implement bracket_solid_script**

This is the most complex emitter. Follow `bracket.py:bracket_solid()` line by line:

1. Call `_bracket_outer()` → translate to box + locate ops (or PrebuiltOp if SCS0009 dispatch)
2. Servo body pocket → box + locate + cut
3. Horn clearance hole → cylinder + locate + cut
4. Shaft boss clearance → cylinder + locate + cut (conditional)
5. Fastener holes → loop of cylinder + locate + cut (via `_cut_fastener_hole`)
6. Connector port → PrebuiltOp for `_connector_port()` result (complex shaped passage)
7. Fallback cable slot → box + locate + cut

For parts that use edge selection or complex OCCT-specific operations (like the connector port's shaped cutout), use `PrebuiltOp` to wrap the direct build123d result. This is honest — the viewer shows "PrebuiltOp: connector_port" for the parts we haven't ported yet, and native ShapeScript ops for everything else.

```python
def bracket_solid_script(servo, spec=None) -> ShapeScript:
    """Emit ShapeScript for a complete bracket solid."""
    from botcad.bracket import BracketSpec
    if spec is None:
        spec = BracketSpec()

    # For SCS0009, use PrebuiltOp (different geometry entirely)
    if servo.name == "SCS0009":
        from botcad.bracket import bracket_solid as _direct
        prog = ShapeScript()
        prog.output_ref = prog.prebuilt(_direct(servo, spec), tag="scs0009_bracket")
        return prog

    prog = ShapeScript()
    # ... translate bracket_solid line by line to ShapeScript ops ...
    # See botcad/bracket.py:388-500 for the reference implementation
    prog.output_ref = shell
    return prog
```

The full implementation follows the ~100-line function in `bracket.py:388-500`. Each build123d operation maps to a ShapeScript op:
- `Box(w, l, h, align=C)` → `prog.box(w, l, h)`
- `solid.locate(Location(pos))` → `prog.locate(ref, pos=pos)`
- `shell - pocket` → `prog.cut(shell, pocket)`
- `Cylinder(r, h, align=...)` → `prog.cylinder(r, h, align=...)`

For `_cut_fastener_hole()` and `_connector_port()`, use PrebuiltOp wrappers calling the existing functions.

- [ ] **Step 3: Run tests, iterate until both servos pass**

Run: `uv run pytest tests/test_shapescript_bracket.py -v`

- [ ] **Step 4: Run existing roundtrip tests to verify no regression**

Run: `uv run pytest tests/test_shapescript_roundtrip.py -v`

- [ ] **Step 5: Commit**

```bash
git add botcad/shapescript/emit_bracket.py tests/test_shapescript_bracket.py
git commit -m "feat(shapescript): port bracket_solid to ShapeScript

STS3215 bracket fully expressed as ShapeScript ops. SCS0009 and
connector ports remain as PrebuiltOp (different geometry / OCCT-specific).
Roundtrip tests verify volume equivalence."
```

---

### Task 5: Wire CallOp into emit_body_ir

**Files:**
- Modify: `botcad/shapescript/emit_body.py`
- Modify: `tests/test_shapescript_roundtrip.py`

Replace PrebuiltOp usage in emit_body_ir with CallOp where ShapeScript emitters exist. Bracket is the first target — instead of:
```python
brk_solid = bracket_solid(servo, spec).moved(Location(center, euler))
brk_ref = prog.prebuilt(brk_solid, tag="bracket_left_wheel")
```

Emit:
```python
# Register bracket sub-program once (keyed by servo name)
key = f"bracket_{servo.name}"
if key not in prog.sub_programs:
    prog.sub_programs[key] = bracket_solid_script(servo, spec)
brk_ref = prog.call(key, tag=f"bracket_{joint.name}")
brk_ref = prog.locate(brk_ref, pos=center, euler_deg=euler)
```

This achieves define-once-use-many: same bracket ShapeScript is defined once, called at each joint, located independently.

- [ ] **Step 1: Update emit_body_ir to use CallOp for brackets**

Replace the bracket loop in section 4 of `emit_body.py`.

- [ ] **Step 2: Run roundtrip tests**

Run: `uv run pytest tests/test_shapescript_roundtrip.py -v`
Expected: All pass — volumes must match within 0.1%

- [ ] **Step 3: Commit**

```bash
git add botcad/shapescript/emit_body.py
git commit -m "feat(shapescript): use CallOp for brackets in emit_body_ir

Brackets are defined once as sub-programs, called at each joint.
Same bracket spec → same sub-program → define once, use many."
```

---

## Phase 3: Web Viewer Integration

### Task 6: Update cad-steps API for ShapeScript sub-programs

**Files:**
- Modify: `botcad/shapescript/cad_steps.py`
- Modify: `main.py` (API endpoints)
- Modify: `viewer/cad-steps-mode.js`

The cad-steps API currently returns flat step lists. Add sub-program support:
- Each step can have a `sub_steps` array (for CallOp expansions)
- The viewer renders sub-steps as collapsible groups

- [ ] **Step 1: Update cad_steps.py to expand CallOps**

Modify `shapescript_to_cad_steps()` to produce nested structures for CallOps:

```python
{
    "index": 5,
    "label": "Call: bracket_STS3215",
    "op": "call",
    "has_tool": false,
    "sub_steps": [
        {"index": 0, "label": "Outer box", "op": "create", ...},
        {"index": 1, "label": "Cut pocket", "op": "cut", ...},
        ...
    ]
}
```

- [ ] **Step 2: Update cad-steps-mode.js to render sub-program groups**

Add collapsible group rendering in the step list:
- CallOp steps show as expandable headers with a disclosure triangle
- Click to expand shows the sub-program's steps indented
- Clicking a sub-step loads that step's STL

- [ ] **Step 3: Test in browser**

Run: `python main.py web --bot wheeler_arm`
Navigate to `?cadsteps=wheeler_arm:base`
Verify: bracket steps appear as expandable groups with sub-steps inside.

- [ ] **Step 4: Commit**

```bash
git add botcad/shapescript/cad_steps.py main.py viewer/cad-steps-mode.js
git commit -m "feat(viewer): expandable sub-program groups in CAD steps

CallOp steps render as collapsible groups. Sub-program steps visible
when expanded. Click to load intermediate geometry at any level."
```

---

### Task 7: Add ShapeScript view to component browser

**Files:**
- Modify: `viewer/component-browser.js`
- Modify: `main.py` (add `/api/components/{name}/shapescript` endpoint)

When viewing a component, show a "ShapeScript" tab/panel that displays the program ops.

- [ ] **Step 1: Add API endpoint**

In `main.py`, add `/api/components/{name}/shapescript` that:
1. Imports the component's `*_script()` emitter
2. Executes it to get the ShapeScript program
3. Returns JSON with the op list + step metadata (same format as cad-steps)

- [ ] **Step 2: Add ShapeScript panel to component browser**

In `component-browser.js`, add a "Steps" mode alongside existing view modes. When active, it fetches the shapescript metadata and renders the step list with slider (reusing CadStepsViewer logic).

- [ ] **Step 3: Test in browser**

Run: `python main.py web`
Navigate to component browser → select OV5647 camera → click "Steps"
Verify: ShapeScript ops displayed, step slider works.

- [ ] **Step 4: Commit**

```bash
git add viewer/component-browser.js main.py
git commit -m "feat(viewer): ShapeScript step viewer in component browser

Components show their construction steps via ShapeScript.
Step slider scrubs through intermediate geometry."
```

---

## Phase 4: Port Remaining Factories (Parallel Work)

Tasks 8-11 can be done in parallel by separate agents since each factory is independent.

### Task 8: Port servo_solid (_sts_series_solid, _scs0009_solid)

**Files:**
- Create: `botcad/shapescript/emit_servo.py`
- Modify: `tests/test_shapescript_components.py`

Complex geometry with fillets. Use PrebuiltOp for filleted sub-assemblies, native ops for the composition. Test both STS3215 and SCS0009 variants.

### Task 9: Port coupler_solid, cradle_solid

**Files:**
- Modify: `botcad/shapescript/emit_bracket.py`
- Modify: `tests/test_shapescript_bracket.py`

Coupler is the most complex single factory (25+ ops, C-shape with D-clip plates). Port what can be expressed as ShapeScript, PrebuiltOp for clip geometry.

### Task 10: Port _make_wheel_solid

**Files:**
- Modify: `botcad/shapescript/emit_components.py`
- Modify: `tests/test_shapescript_components.py`

30+ ops (tire, treads, rim, hub, spokes, bore). Heavy use of rotation loops for treads and spokes. Port the structural composition, PrebuiltOp for individual rotated pieces if needed.

### Task 11: Port _child_clearance_volume, _wire_segment_solid, connector_solid, fastener_solid

**Files:**
- Modify: `botcad/shapescript/emit_components.py`
- Modify: `tests/test_shapescript_components.py`

Mixed complexity. Clearance volume involves rotation sweep. Fastener uses chamfer + hex socket extrude. Connector has 5 type variants. Port what's clean, PrebuiltOp the rest.

---

## Testing Strategy

Every task has roundtrip tests that compare ShapeScript output volumes against direct build123d within 0.1%. The body-level roundtrip tests (`test_shapescript_roundtrip.py`) catch regressions from wiring changes. Component-level tests catch individual factory regressions.

**Run the full suite after each task:**
```bash
uv run pytest tests/test_shapescript_*.py -v
```

**Run the full pipeline validation periodically:**
```bash
uv run pytest tests/ -v -k "not smoketest and not tui"
```
