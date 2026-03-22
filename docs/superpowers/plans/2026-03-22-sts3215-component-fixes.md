# STS3215 Component Geometry Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four geometry/viewer issues with the STS3215 servo component: horn Z-positioning, coupler Z-positioning, bracket envelope sizing, and cradle envelope sizing + naming.

**Architecture:** All four issues are in `botcad/bracket.py` (geometry) and `main.py` (viewer STL generation). The viewer layer system (`viewer/component-browser.js`) needs a new "coupled assembly" composite view. Changes propagate to all bots using STS3215 (notably so101_arm with 4 COUPLER joints), so test baselines and bot meshes must be regenerated.

**Tech Stack:** build123d (CAD), ShapeScript IR, Three.js viewer

---

## Issue Summary

1. **Horn intersects servo in viewer** — `_generate_solid()` in `main.py:808-822` positions horn at `center_z` (shaft face + thickness/2), but this is the horn's *center* in the servo's local frame. The horn disc sits *on top of* the shaft face, so its bottom should be at `shaft_offset[2]` (the +Z face), not overlapping the servo body. Additionally, this code uses `.locate()` on a potentially cached solid — violates the `.moved()` rule.
2. **Coupler intersects servo in viewer** — The coupler solid is generated in shaft-centered frame but displayed without offset. In the layer viewer, it needs to be positioned so it clears both the servo body and the top horn disc.
3. **Bracket envelope too large** — Currently 5x bracket height insertion clearance. User wants ~3x.
4. **Cradle envelope too small** — Envelope is sized to the cradle footprint, but should be sized to the servo cross-section (the thing being inserted). Rename to `cradle_servo_insertion_envelope` or similar for clarity.

---

### Task 1: Fix horn Z-positioning in component viewer

The horn disc should sit above the servo shaft face, not intersect it. The issue is in `_generate_solid()` in `main.py`.

**Files:**
- Modify: `main.py:808-822` — fix horn positioning
- Test: manual visual check in component viewer + existing `tests/test_shapescript_bracket.py`

- [ ] **Step 1: Write a test for horn positioning**

Add to `tests/test_shapescript_bracket.py`:
```python
def test_horn_disc_does_not_intersect_servo():
    """Horn disc bottom face must be at or above servo shaft face Z."""
    from botcad.components.servo import sts3215
    from botcad.bracket import horn_disc_params

    servo = sts3215()
    params = horn_disc_params(servo)
    assert params is not None

    # Horn bottom face = center_z - thickness/2
    horn_bottom_z = params.center_z - params.thickness / 2
    shaft_face_z = servo.shaft_offset[2]

    # Horn must sit on or above the shaft face, not inside the servo
    assert horn_bottom_z >= shaft_face_z - 1e-9, (
        f"Horn bottom {horn_bottom_z:.4f} is below shaft face {shaft_face_z:.4f}"
    )
```

- [ ] **Step 2: Run test to verify it passes (this tests the bracket.py geometry, which is already correct)**

Run: `uv run pytest tests/test_shapescript_bracket.py::test_horn_disc_does_not_intersect_servo -v`

The `horn_disc_params()` geometry is actually correct — `center_z = sz + thickness/2` places the horn above the shaft face. The bug is in `_generate_solid()` in `main.py` which uses `.locate()` (mutates cached solid) and may double-offset.

- [ ] **Step 3: Fix `_generate_solid` horn case in `main.py`**

Replace lines 808-822 with:
```python
elif part == "horn":
    from botcad.bracket import horn_disc_params
    from botcad.emit.cad import _horn_solid
    from build123d import Location

    params = horn_disc_params(comp)
    if params is not None:
        horn = _horn_solid(comp)
        if horn is not None:
            # .moved() not .locate() — horn_solid may be cached
            solid = horn.moved(
                Location(
                    (params.center_xy[0], params.center_xy[1], params.center_z)
                )
            )
```

Key change: `.locate()` → `.moved()` to avoid corrupting cached solid.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_shapescript_bracket.py -v`

- [ ] **Step 5: Visually verify in component viewer**

Run `make web`, navigate to STS3215, toggle horn layer — horn should sit on top of servo, not intersect it.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_shapescript_bracket.py
git commit -m "fix: horn uses .moved() not .locate(), preventing cache corruption in viewer"
```

---

### Task 2: Fix coupler Z-positioning in component viewer

The coupler is built in shaft-centered frame. In the viewer, it shows at the origin, intersecting the servo. It needs to be positioned so the front plate clears the top horn and the rear plate clears the bottom horn.

**Files:**
- Modify: `main.py:825-826` — position coupler correctly relative to servo body
- Test: visual check in viewer

- [ ] **Step 1: Understand coupler frame**

Read `botcad/bracket.py` `coupler_solid()` to confirm the coupler's coordinate frame. The coupler front plate Z should be at the front horn face, rear plate at the rear horn face. The coupler is already in shaft-centered frame — it just needs the shaft offset applied when displaying in the viewer.

- [ ] **Step 2: Fix `_generate_solid` coupler case**

The coupler solid is built in shaft-centered frame. To display it correctly relative to the servo body (which is in servo local frame), apply the shaft offset:

```python
elif part == "coupler":
    from build123d import Location

    raw = coupler_solid(comp, spec)
    if raw is not None:
        # Coupler is in shaft-centered frame; shift to servo local frame
        sx, sy, sz = comp.shaft_offset
        solid = raw.moved(Location((sx, sy, 0)))
```

Note: Z offset may not be needed since the coupler plates are already positioned at the horn Z coordinates relative to shaft center. Only X/Y offset needed.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_shapescript_bracket.py -v`

- [ ] **Step 4: Visual verification**

In component viewer, toggle coupler + servo + horn layers. Coupler should bridge between front and rear horns without intersecting the servo body.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "fix: position coupler in servo local frame in component viewer"
```

---

### Task 3: Add "coupled assembly" composite view to component viewer

The user wants to see the full coupled assembly: servo + top horn + bottom horn + coupler + fasteners. Currently the viewer only shows individual layers.

**Files:**
- Modify: `viewer/component-browser.js` — add composite layer or multi-layer toggle
- Modify: `main.py` — potentially add a rear horn STL endpoint

- [ ] **Step 1: Add `coupled_assembly` to LAYER_META**

```javascript
const LAYER_META = {
  // ... existing layers ...
  coupled_assembly: { label: 'Coupled Assembly', colorHex: null, opts: {},
                      composite: ['servo', 'horn', 'coupler', 'cradle', 'fasteners'] },
};
```

- [ ] **Step 2: Handle composite layers in `_loadLayer`**

When a layer has a `composite` property, load all sub-layers:

```javascript
async _loadLayer(layerId) {
    const meta = LAYER_META[layerId] || { colorHex: null, opts: {} };
    if (meta.composite) {
        for (const subId of meta.composite) {
            await this._loadLayer(subId);
            this.layerGroups[subId].visible = true;
        }
        return;
    }
    // ... existing code
}
```

- [ ] **Step 3: Ensure layer checkboxes handle composite toggling**

When the coupled_assembly checkbox is toggled, it should show/hide all sub-layers as a group. Update the checkbox handler to iterate `meta.composite` if present.

- [ ] **Step 4: Visual verification**

Toggle "Coupled Assembly" — should show servo body, both horns, coupler bridge, cradle tray, and all fasteners as a complete rotational joint.

- [ ] **Step 5: Commit**

```bash
git add viewer/component-browser.js
git commit -m "feat: add coupled assembly composite view to component browser"
```

---

### Task 4: Reduce bracket envelope insertion clearance to 3x

**Files:**
- Modify: `botcad/bracket.py:379-383` — change multiplier from 5 to 3
- Modify: `botcad/shapescript/emit_bracket.py:124-175` — update matching ShapeScript emission
- Update: `tests/shapescript_baselines/bracket_envelope_sts3215.shapescript` — regenerate baseline
- Test: `tests/test_shapescript_bracket.py`

- [ ] **Step 1: Update test expectations**

The bracket envelope volume will shrink. Update any volume-checking tests in `tests/test_shapescript_bracket.py::TestBracketEnvelopeRoundtrip` to expect the smaller volume.

- [ ] **Step 2: Change multiplier in `bracket.py`**

In `bracket_envelope()` (line 383):
```python
# Before:
outer, _, _ = _bracket_outer(servo, spec, insertion_clearance=bracket_height * 5)
# After:
outer, _, _ = _bracket_outer(servo, spec, insertion_clearance=bracket_height * 3)
```

Update docstring (lines 371-372) from "5x" to "3x".

- [ ] **Step 3: Change multiplier in `emit_bracket.py`**

Find the matching `5` multiplier in `bracket_envelope_script()` and change to `3`.

- [ ] **Step 4: Regenerate shapescript baselines**

Run: `uv run pytest tests/test_shapescript_snapshots.py --snapshot-update -v`

- [ ] **Step 5: Run full test suite**

Run: `make validate`

- [ ] **Step 6: Commit**

```bash
git add botcad/bracket.py botcad/shapescript/emit_bracket.py tests/
git commit -m "fix: reduce bracket envelope insertion clearance from 5x to 3x"
```

---

### Task 5: Resize and rename cradle envelope

The cradle envelope should be based on the **servo cross-section** (what's being inserted), not the cradle footprint. Rename for clarity.

**Files:**
- Modify: `botcad/bracket.py:1028-1071` — resize envelope based on servo dims, rename function
- Modify: `botcad/shapescript/emit_bracket.py:354-395` — update ShapeScript emission
- Modify: `botcad/shapescript/emit_body.py` — update references to renamed function
- Modify: `botcad/emit/cad.py` — update import
- Modify: `main.py` — update import and usage
- Modify: `viewer/component-browser.js` — update layer ID
- Update: test baselines and test files
- Update: all other files importing `cradle_envelope`

- [ ] **Step 1: Search for all references to `cradle_envelope`**

Find every import and usage across the codebase. Plan the rename.

- [ ] **Step 2: Rename function to `cradle_insertion_envelope`**

In `botcad/bracket.py`, rename `cradle_envelope` → `cradle_insertion_envelope`. Update docstring to explain this is the volume needed to insert the servo into the cradle, sized by servo cross-section + insertion axis clearance.

- [ ] **Step 3: Resize the envelope**

The envelope should be based on **servo body cross-section** perpendicular to the insertion axis (+X), not cradle footprint:

```python
def cradle_insertion_envelope(servo: ServoSpec, spec: BracketSpec | None = None):
    """Envelope for inserting servo into cradle.

    Sized by the servo cross-section (Y×Z body dims + tolerance) extended
    along the insertion axis (+X). The servo slides in from the +X side,
    so the envelope must clear the full servo body width.
    """
    from build123d import Align, Box, Location

    if spec is None:
        spec = BracketSpec()

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall

    ear_bottom_z = _ear_bottom_z(servo, wall)

    # Y extent: full servo width + tolerance + wall (same as cradle)
    outer_ly = body_y + 2 * (tol + wall)

    # Z extent: from ear bottom to grip margin (same as cradle solid)
    grip_margin = 0.004
    outer_top_z = -body_z / 2 + grip_margin
    outer_bottom_z = ear_bottom_z
    outer_lz = outer_top_z - outer_bottom_z
    outer_cz = (outer_top_z + outer_bottom_z) / 2

    # X extent: full servo body width as insertion path, extended 3x
    # The servo inserts from +X, so envelope covers from cradle min_x
    # out past the body for insertion clearance
    sx = servo.shaft_offset[0]
    cradle_min_x = -body_x / 2 - tol - wall
    cradle_nominal_max_x = sx - 0.002
    insertion_length = body_x  # servo must pass through fully
    cradle_max_x = cradle_nominal_max_x + insertion_length * 3
    cradle_lx = cradle_max_x - cradle_min_x
    cradle_cx = (cradle_min_x + cradle_max_x) / 2

    envelope = Box(
        cradle_lx, outer_ly, outer_lz,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    return envelope.moved(Location((cradle_cx, 0, outer_cz)))
```

- [ ] **Step 4: Update ShapeScript emission**

In `emit_bracket.py`, rename `cradle_envelope_script` → `cradle_insertion_envelope_script` and match the new sizing logic.

- [ ] **Step 5: Update all references**

Update imports/references in:
- `botcad/shapescript/emit_body.py`
- `botcad/emit/cad.py`
- `main.py` (both import and `_generate_solid` usage)
- `viewer/component-browser.js` (layer ID: `cradle_envelope` → `cradle_insertion_envelope`)

- [ ] **Step 6: Update tests and baselines**

- Update `tests/test_shapescript_bracket.py` test class names and references
- Regenerate shapescript baselines: `uv run pytest tests/test_shapescript_snapshots.py --snapshot-update -v`

- [ ] **Step 7: Run full validation**

Run: `make validate`

- [ ] **Step 8: Commit**

```bash
git add botcad/ main.py viewer/ tests/
git commit -m "fix: rename cradle_envelope → cradle_insertion_envelope, size by servo cross-section"
```

---

### Task 6: Regenerate bot meshes and validate downstream

Changes to bracket/cradle envelopes affect all bots. Regenerate and verify.

**Files:**
- Regenerate: `bots/so101_arm/` meshes and manifests
- Regenerate: `bots/wheeler_base/` and `bots/wheeler_arm/` meshes
- Test: clearance validation via `make validate`

- [ ] **Step 1: Regenerate all bot meshes**

```bash
uv run mjpython main.py regen --all
```

- [ ] **Step 2: Run clearance validation**

```bash
uv run mjpython main.py validate-clearances
```

Check that no new clearance violations appeared from the envelope changes.

- [ ] **Step 3: Run full test suite**

```bash
make validate
```

- [ ] **Step 4: Visually inspect so101_arm in viewer**

Run `make web`, navigate to so101_arm. Check each body — cradle cuts should be clean, no intersections visible between adjacent bodies.

- [ ] **Step 5: Commit regenerated meshes**

```bash
git add bots/
git commit -m "chore: regenerate bot meshes after envelope fixes"
```

---

## Dependency Graph

```
Task 1 (horn Z-fix) ─────────┐
Task 2 (coupler Z-fix) ──────┤
Task 3 (coupled assembly) ───┤──→ Task 6 (regenerate + validate)
Task 4 (bracket envelope 3x) ┤
Task 5 (cradle rename+resize) ┘
```

Tasks 1-5 are independent of each other and can be parallelized. Task 6 must run after all others.
