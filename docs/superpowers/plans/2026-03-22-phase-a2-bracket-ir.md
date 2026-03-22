# Phase A2: Bracket IR Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify bracket geometry on ShapeScript IR — bracket.py emits ShapeScript instead of calling build123d, eliminating ~970 lines of duplication (emit_bracket.py + _build_body_solid).

**Architecture:** bracket.py's geometry functions (`bracket_solid`, `cradle_solid`, `coupler_solid`, `servo_solid`, envelopes) switch from returning `build123d.Solid` to returning `ShapeScript`. `emit_bracket.py` merges into bracket.py. `cad.py`'s `_build_body_solid()` is replaced by executing ShapeScript IR through backend_occt. A volume equivalence test validates the migration before deleting the old path.

**Tech Stack:** ShapeScript IR, OCCT backend, build123d (validation only)

**Spec:** `docs/superpowers/specs/2026-03-22-data-oriented-refactor-design.md`

**Prerequisite:** Phase A1 (Material/Appearance) should be complete, but is not strictly blocking — the bracket IR migration is structurally independent.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `botcad/bracket.py` | Modify | Geometry functions return ShapeScript; absorb emit_bracket.py logic |
| `botcad/shapescript/emit_bracket.py` | Delete | Merged into bracket.py |
| `botcad/emit/cad.py` | Modify | Replace `_build_body_solid()` with IR execution; delete dual code path |
| `botcad/shapescript/emit_body.py` | Modify | Import bracket IR from bracket.py instead of emit_bracket.py |
| `botcad/shapescript/emit_components.py` | Modify | Update imports if needed |
| `mindsim/server.py` | Modify | Update imports from emit_bracket → bracket |
| `tests/test_shapescript_bracket.py` | Modify | Update imports; add volume equivalence validation |
| `tests/test_shapescript_snapshots.py` | Modify | Update imports |
| `tests/shapescript_baselines/*.shapescript` | Regenerate | New baselines from bracket.py |

---

### Task 1: Volume equivalence validation test

Before changing anything, write a test that asserts the current ShapeScript emission produces volumes matching the direct build123d path. This test validates the migration — if it passes before and after, the migration is correct.

**Files:**
- Create: `tests/test_bracket_equivalence.py`

- [ ] **Step 1: Write equivalence test**

```python
"""Validate that ShapeScript bracket emission matches direct build123d.

This is a migration validation test. It asserts that
backend_occt.execute(bracket_script) produces volumes within 0.1%
of the direct build123d bracket_solid() for all bracket types.

Delete this test after the migration is complete and bracket.py
no longer has a build123d code path.
"""
from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")

from botcad.bracket import (
    BracketSpec,
    bracket_envelope,
    bracket_solid,
    coupler_solid,
    cradle_envelope,
    cradle_solid,
    servo_solid,
)
from botcad.shapescript.backend_occt import OcctBackend
from botcad.shapescript.emit_bracket import (
    bracket_envelope_script,
    bracket_solid_script,
    coupler_solid_script,
    cradle_envelope_script,
    cradle_solid_script,
)
from botcad.shapescript.emit_servo import servo_script


def _servo():
    from botcad.components.servo import STS3215
    return STS3215()


def _exec(prog):
    """Execute a ShapeScript and return the output solid.

    OcctBackend().execute() returns an ExecutionResult with a .shapes dict
    mapping ref IDs to build123d Solids. We extract the output solid.
    """
    result = OcctBackend().execute(prog)
    return result.shapes[prog.output_ref.id]


def _vol(solid):
    return abs(solid.volume)


def _bbox(solid):
    """Return bounding box as ((xmin,ymin,zmin), (xmax,ymax,zmax))."""
    bb = solid.bounding_box()
    return (bb.min, bb.max)


def _com(solid):
    """Return center of mass as (x, y, z)."""
    return solid.center()


class TestBracketEquivalence:
    """ShapeScript emission must match direct build123d within 0.1%.

    Volume alone won't catch positional bugs (e.g. a bracket translated
    to the wrong location). We also compare bounding box extents and
    center-of-mass to catch spatial regressions.
    """

    TOLERANCE = 0.001  # 0.1%

    def _assert_equiv(self, direct_solid, ir_solid, label: str):
        """Assert volume, bounding box, and center-of-mass equivalence."""
        dv, iv = _vol(direct_solid), _vol(ir_solid)
        assert abs(dv - iv) / dv < self.TOLERANCE, f"{label} volume mismatch"

        dbb, ibb = _bbox(direct_solid), _bbox(ir_solid)
        for axis in range(3):
            assert abs(dbb[0][axis] - ibb[0][axis]) < 1e-4, f"{label} bbox min[{axis}]"
            assert abs(dbb[1][axis] - ibb[1][axis]) < 1e-4, f"{label} bbox max[{axis}]"

        dc, ic = _com(direct_solid), _com(ir_solid)
        for axis in range(3):
            assert abs(dc[axis] - ic[axis]) < 1e-4, f"{label} COM[{axis}]"

    def test_bracket_envelope(self):
        servo, spec = _servo(), BracketSpec()
        direct = bracket_envelope(servo, spec)
        ir = _exec(bracket_envelope_script(servo, spec))
        self._assert_equiv(direct, ir, "bracket_envelope")

    def test_bracket_solid(self):
        servo, spec = _servo(), BracketSpec()
        direct = bracket_solid(servo, spec)
        ir = _exec(bracket_solid_script(servo, spec))
        self._assert_equiv(direct, ir, "bracket_solid")

    def test_cradle_envelope(self):
        servo, spec = _servo(), BracketSpec()
        direct = cradle_envelope(servo, spec)
        ir = _exec(cradle_envelope_script(servo, spec))
        self._assert_equiv(direct, ir, "cradle_envelope")

    def test_cradle_solid(self):
        servo, spec = _servo(), BracketSpec()
        direct = cradle_solid(servo, spec)
        ir = _exec(cradle_solid_script(servo, spec))
        self._assert_equiv(direct, ir, "cradle_solid")

    def test_coupler_solid(self):
        servo, spec = _servo(), BracketSpec()
        direct = coupler_solid(servo, spec)
        ir = _exec(coupler_solid_script(servo, spec))
        self._assert_equiv(direct, ir, "coupler_solid")

    def test_servo_solid(self):
        servo = _servo()
        direct = servo_solid(servo)
        ir = _exec(servo_script(servo))
        self._assert_equiv(direct, ir, "servo_solid")
```

- [ ] **Step 2: Run equivalence tests**

Run: `uv run pytest tests/test_bracket_equivalence.py -v`
Expected: All PASS (both paths exist and should agree)

- [ ] **Step 3: Commit**

```bash
git add tests/test_bracket_equivalence.py
git commit -m "test: bracket volume equivalence — validates ShapeScript matches build123d"
```

---

### Task 2: Migrate bracket.py envelope functions to ShapeScript

Start with the simplest functions: `bracket_envelope` and `cradle_envelope`. These are single-box geometries.

**Files:**
- Modify: `botcad/bracket.py:365-384,1028-1071`

- [ ] **Step 1: Add ShapeScript imports to bracket.py**

At the top of bracket.py, add:

```python
from botcad.shapescript.ops import ShapeRef
from botcad.shapescript.program import ShapeScript
```

- [ ] **Step 2: Create `bracket_envelope_ir()` alongside existing `bracket_envelope()`**

Add a new function that returns ShapeScript, keeping the old one temporarily:

```python
def bracket_envelope_ir(servo: ServoSpec, spec: BracketSpec | None = None) -> ShapeScript:
    """Bracket envelope as ShapeScript IR."""
    # Copy logic from emit_bracket.py:bracket_envelope_script()
    ...
```

Copy the implementation from `botcad/shapescript/emit_bracket.py:bracket_envelope_script()` into this function.

- [ ] **Step 3: Create `cradle_envelope_ir()` alongside existing `cradle_envelope()`**

Same pattern — copy from `emit_bracket.py:cradle_envelope_script()`.

- [ ] **Step 4: Update emit_bracket.py to delegate to bracket.py**

In `emit_bracket.py`, replace `bracket_envelope_script` and `cradle_envelope_script` with thin wrappers:

```python
def bracket_envelope_script(servo, spec=None):
    from botcad.bracket import bracket_envelope_ir
    return bracket_envelope_ir(servo, spec)

def cradle_envelope_script(servo, spec=None):
    from botcad.bracket import cradle_envelope_ir
    return cradle_envelope_ir(servo, spec)
```

- [ ] **Step 5: Run equivalence tests**

Run: `uv run pytest tests/test_bracket_equivalence.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `make validate`

- [ ] **Step 7: Commit**

```bash
git add botcad/bracket.py botcad/shapescript/emit_bracket.py
git commit -m "refactor: bracket/cradle envelope IR emitted from bracket.py"
```

---

### Task 3: Migrate bracket_solid and cradle_solid to ShapeScript

These are more complex — they have pocket cuts, fastener holes, connector passages.

**Files:**
- Modify: `botcad/bracket.py`

- [ ] **Step 1: Create `bracket_solid_ir()` in bracket.py**

Copy `bracket_solid_script()` from `emit_bracket.py:178-351` into bracket.py as `bracket_solid_ir()`.

- [ ] **Step 2: Create `cradle_solid_ir()` in bracket.py**

Copy `cradle_solid_script()` from `emit_bracket.py:398-496` into bracket.py as `cradle_solid_ir()`.

- [ ] **Step 3: Update emit_bracket.py to delegate**

```python
def bracket_solid_script(servo, spec=None):
    from botcad.bracket import bracket_solid_ir
    return bracket_solid_ir(servo, spec)

def cradle_solid_script(servo, spec=None):
    from botcad.bracket import cradle_solid_ir
    return cradle_solid_ir(servo, spec)
```

- [ ] **Step 4: Run equivalence tests**

Run: `uv run pytest tests/test_bracket_equivalence.py -v`

- [ ] **Step 5: Run full test suite**

Run: `make validate`

- [ ] **Step 6: Commit**

```bash
git add botcad/bracket.py botcad/shapescript/emit_bracket.py
git commit -m "refactor: bracket/cradle solid IR emitted from bracket.py"
```

---

### Task 4: Migrate coupler_solid and servo_solid to ShapeScript

**Files:**
- Modify: `botcad/bracket.py`
- Modify: `botcad/shapescript/emit_bracket.py`
- Modify: `botcad/shapescript/emit_servo.py`

- [ ] **Step 1: Create `coupler_solid_ir()` in bracket.py**

Copy `coupler_solid_script()` from `emit_bracket.py:499-720` into bracket.py.

- [ ] **Step 2: Note on emit_servo.py**

`emit_servo.py` contains `servo_script()`, `sts_series_script()`, `scs0009_script()`, and helpers (`_group_ears_by_y_side`, `_emit_servo_connector`). Multiple callers import the sub-functions directly (e.g. `tests/test_shapescript_components.py` imports `sts_series_script` and `scs0009_script`; `botcad/skeleton.py` imports `servo_script`).

**Do NOT move `servo_script` to bracket.py.** `emit_servo.py` is a self-contained ShapeScript emitter with its own helpers and is already in the right place. Leave it as-is for this migration. A future cleanup could consolidate all component emitters, but that is out of scope here. The equivalence test still validates `servo_solid` against `servo_script` — that is sufficient.

- [ ] **Step 3: Update emit_bracket.py to delegate**

- [ ] **Step 4: Run equivalence tests**

Run: `uv run pytest tests/test_bracket_equivalence.py -v`

- [ ] **Step 5: Run full test suite**

Run: `make validate`

- [ ] **Step 6: Commit**

```bash
git add botcad/bracket.py botcad/shapescript/emit_bracket.py botcad/shapescript/emit_servo.py
git commit -m "refactor: coupler/servo solid IR emitted from bracket.py"
```

---

### Task 5: Make bracket.py functions return ShapeScript as primary interface

Now that bracket.py has `*_ir()` functions, rename them to replace the old build123d functions. The old functions become convenience wrappers that execute the IR.

**Files:**
- Modify: `botcad/bracket.py`

- [ ] **Step 1: Rename functions**

For each function pair:
- `bracket_envelope()` (old, returns Solid) -> `_bracket_envelope_solid()` (private, legacy)
- `bracket_envelope_ir()` -> `bracket_envelope()` (public, returns ShapeScript)

Add a convenience helper:

```python
def bracket_envelope_solid(servo, spec=None):
    """Execute bracket_envelope IR and return a Solid. Convenience wrapper."""
    from botcad.shapescript.backend_occt import OcctBackend
    prog = bracket_envelope(servo, spec)
    result = OcctBackend().execute(prog)
    return result.shapes[prog.output_ref.id]
```

Repeat for: `bracket_solid`, `cradle_solid`, `cradle_envelope`, `coupler_solid`, `servo_solid`.

- [ ] **Step 2: Update all callers**

Search for every call to `bracket_solid()`, `cradle_solid()`, etc. that expects a Solid. Update them to call the `*_solid()` convenience wrapper or execute the IR themselves.

Key callers (exhaustive — search for `from botcad.bracket import` and `from botcad.shapescript.emit_bracket import`):
- `botcad/emit/cad.py` — calls `bracket_solid()`, `cradle_solid()`, `coupler_solid()`, `bracket_envelope()`, `cradle_envelope()`, `servo_solid()`
- `botcad/emit/drawings.py` — imports `bracket_solid`, `servo_solid`
- `botcad/emit/component_renders.py` — imports `bracket_envelope`, `bracket_solid`, `cradle_envelope`, `cradle_solid`, `coupler_solid`, `servo_solid`
- `botcad/clearance.py` — may call bracket functions
- `mindsim/server.py` — `_generate_solid()` calls bracket functions
- `tests/test_shapescript_bracket.py` — imports from emit_bracket
- `tests/test_shapescript_snapshots.py` — imports from emit_bracket
- `tests/test_render_svg.py` — imports `servo_solid`, `bracket_solid`
- `tests/test_bracket_orientations.py` — imports `bracket_envelope`, `bracket_solid`, `servo_solid`
- `tests/test_shapescript_roundtrip.py` — may import bracket functions
- `tests/test_shapescript_components.py` — imports `servo_solid`
- `scripts/compare_cad.py` — imports `servo_solid`
- `scripts/regen_test_renders.py` — imports `coupler_solid`, `cradle_solid`, `servo_solid`

- [ ] **Step 3: Run equivalence tests**

Run: `uv run pytest tests/test_bracket_equivalence.py -v`

- [ ] **Step 4: Run full test suite**

Run: `make validate`

- [ ] **Step 5: Commit**

```bash
git add botcad/bracket.py botcad/emit/cad.py botcad/clearance.py mindsim/server.py tests/
git commit -m "refactor: bracket.py primary interface returns ShapeScript; Solid via convenience wrappers"
```

---

### Task 6: Delete emit_bracket.py and emit_servo.py delegation

Now that bracket.py is the single source, delete the delegation wrappers.

**Files:**
- Delete: `botcad/shapescript/emit_bracket.py`
- Modify: `botcad/shapescript/emit_body.py` — update imports
- Modify: `tests/test_shapescript_bracket.py` — update imports
- Modify: `tests/test_shapescript_snapshots.py` — update imports
- Modify: any other files importing from emit_bracket

- [ ] **Step 1: Find all imports of emit_bracket**

```bash
uv run ruff check --select F401 .  # unused imports will surface
```

Also search: `from botcad.shapescript.emit_bracket import`

- [ ] **Step 2: Update all imports to point to bracket.py**

Every `from botcad.shapescript.emit_bracket import X_script` becomes `from botcad.bracket import X`.

Key files:
- `botcad/shapescript/emit_body.py` — imports bracket/cradle/coupler scripts
- `tests/test_shapescript_bracket.py` — imports bracket/cradle/coupler scripts
- `tests/test_shapescript_snapshots.py` — if it imports emit_bracket

- [ ] **Step 3: Delete emit_bracket.py**

```bash
git rm botcad/shapescript/emit_bracket.py
```

- [ ] **Step 4: Run full test suite**

Run: `make validate`

- [ ] **Step 5: Commit**

```bash
git rm botcad/shapescript/emit_bracket.py
git add botcad/bracket.py botcad/shapescript/emit_body.py botcad/emit/cad.py \
    botcad/emit/drawings.py botcad/emit/component_renders.py \
    tests/test_shapescript_bracket.py tests/test_shapescript_snapshots.py \
    tests/test_render_svg.py tests/test_bracket_orientations.py \
    tests/test_shapescript_components.py scripts/compare_cad.py scripts/regen_test_renders.py
git commit -m "refactor: delete emit_bracket.py — bracket.py is the single source of bracket IR"
```

---

### Task 7: Replace _build_body_solid() with IR execution

This is the biggest win — replacing ~250 lines of direct build123d with ShapeScript IR execution.

**Files:**
- Modify: `botcad/emit/cad.py:1028-1278`

- [ ] **Step 1: Understand the current flow**

Read `_build_body_solid()` in cad.py. It:
1. Gets body dimensions
2. Creates the outer shell (box/cylinder/tube/sphere)
3. Loops through joints, cutting bracket envelopes and fusing brackets
4. Loops through mounts, cutting component envelopes
5. Returns the final solid

`emit_body_ir()` in emit_body.py does the same thing but produces ShapeScript.

- [ ] **Step 2: Replace `_make_body_solid` with IR execution**

**Important:** `_build_body_solid()` returns `list[CadStep]` (debug step history), not a Solid. `_make_body_solid()` wraps it and extracts `steps[-1].solid`. There are two options:

- **Option A (simpler):** Replace `_make_body_solid()` only. This preserves `_build_body_solid()` for debug viz (CadStep generation) while the production path goes through IR.
- **Option B (full):** Replace both, but then debug step visualization (`make_body_solid_with_steps`, `?cadsteps=` viewer) needs a ShapeScript-based replacement.

**Recommended: Option A.** Replace `_make_body_solid`:

```python
def _make_body_solid(body, parent_joint=None, wire_segments=None):
    """Build a body solid by executing its ShapeScript IR."""
    from botcad.shapescript.backend_occt import OcctBackend
    from botcad.shapescript.emit_body import emit_body_ir

    prog = emit_body_ir(body, parent_joint, wire_segments)
    result = OcctBackend().execute(prog)
    return result.shapes[prog.output_ref.id]
```

Note the correct `emit_body_ir` signature: `emit_body_ir(body, parent_joint, wire_segments)` — NOT `(body, bot, bracket_spec)`. The function takes a single Body, its parent Joint (or None for root), and optional wire segment tuples. BracketSpec is created internally by `emit_body_ir`.

This replaces `_make_body_solid`'s delegation to `_build_body_solid` with direct IR execution. `_build_body_solid` and `make_body_solid_with_steps` remain for debug CadStep generation until a future task replaces them.

- [ ] **Step 3: Run equivalence test**

Before this change, add a test that compares old `_build_body_solid` output vs IR execution for each body of each bot. After confirming equivalence, make the switch.

- [ ] **Step 4: Delete the old build123d body-building code**

Remove all the direct build123d calls that were in `_build_body_solid`.

- [ ] **Step 5: Run full test suite**

Run: `make validate`

- [ ] **Step 6: Commit**

```bash
git add botcad/emit/cad.py
git commit -m "refactor: _make_body_solid() executes ShapeScript IR — no more direct build123d"
```

---

### Task 8: Delete old build123d functions from bracket.py

Now that everything goes through IR, delete the old build123d implementations that are no longer called.

**Files:**
- Modify: `botcad/bracket.py` — remove build123d code paths

- [ ] **Step 1: Identify dead code**

The old build123d implementations (`_bracket_envelope_solid` private wrappers, old `@lru_cache` functions) should now be unused. Search for callers.

- [ ] **Step 2: Remove dead code**

Delete functions that are no longer called. Remove `from build123d import ...` if no longer needed. Remove `@lru_cache` decorators (ShapeScript's content-hash caching replaces them).

- [ ] **Step 3: Run full test suite**

Run: `make validate`

- [ ] **Step 4: Commit**

```bash
git add botcad/bracket.py
git commit -m "refactor: remove old build123d bracket code — ShapeScript IR is the only path"
```

---

### Task 9: Delete equivalence test and regenerate

The equivalence test was a migration tool — the old code path no longer exists, so delete it.

**Files:**
- Delete: `tests/test_bracket_equivalence.py`
- Regenerate: test baselines, bot meshes

- [ ] **Step 1: Delete equivalence test**

```bash
git rm tests/test_bracket_equivalence.py
```

- [ ] **Step 2: Regenerate shapescript baselines**

```bash
uv run pytest tests/test_shapescript_snapshots.py --snapshot-update -v
```

- [ ] **Step 3: Regenerate bot meshes**

```bash
uv run mjpython main.py regen --all
```

- [ ] **Step 4: Run full validation**

Run: `make validate`

- [ ] **Step 5: Visual check**

Run `make web`, verify so101_arm and wheeler_base look correct.

- [ ] **Step 6: Commit**

```bash
git rm tests/test_bracket_equivalence.py
git add tests/shapescript_baselines/ botcad/bracket.py
git commit -m "chore: delete migration test, regenerate baselines and bot meshes after bracket IR migration"
```

---

## Dependency Graph

```
Task 1 (equivalence test) ──→ Task 2 (envelopes) ──→ Task 3 (solids) ──→ Task 4 (coupler/servo)
                                                                              ↓
                                                                         Task 5 (rename to primary)
                                                                              ↓
                                                                         Task 6 (delete emit_bracket)
                                                                              ↓
                                                                         Task 7 (replace _build_body_solid)
                                                                              ↓
                                                                         Task 8 (delete old build123d)
                                                                              ↓
                                                                         Task 9 (cleanup + regen)
```

Tasks are strictly sequential — each builds on the previous, and the equivalence test validates correctness throughout.
