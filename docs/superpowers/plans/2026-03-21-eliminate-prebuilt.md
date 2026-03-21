# Eliminate PrebuiltOp Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all PrebuiltOp usage from production ShapeScript emitters so every piece of geometry is fully expressed in native ShapeScript ops — inspectable, cacheable, and debuggable.

**Architecture:** Each remaining PrebuiltOp exists because ShapeScript lacks a specific operation. Extending ShapeScript with these operations eliminates PrebuiltOp at the source.

**Tech Stack:** Python 3.11+, ShapeScript (`botcad/shapescript/`), build123d (OCCT).

---

## Remaining PrebuiltOp Usage — Categorized by Root Cause

### Category 1: Fillet (edge selection needed) — 5 usages

ShapeScript has `FilletOp` with tag-based edge selection, but the current fillet resolution heuristic (bounding-box proximity) doesn't work reliably for the filleted sub-assemblies. These factories build the shape in build123d, fillet it with explicit edge selection, then inject the result.

| File | Usage | What's filleted |
|------|-------|----------------|
| `emit_servo.py:55` | `prog.prebuilt(middle_solid)` | STS servo middle section — Z-edge fillets (4mm radii) |
| `emit_servo.py:72` | `prog.prebuilt(top_solid)` | STS servo top cap — edge fillets |
| `emit_servo.py:88` | `prog.prebuilt(bot_solid)` | STS servo bottom cap — edge fillets |
| `emit_servo.py:153` | `prog.prebuilt(body_solid)` | SCS0009 body — 1mm edge fillets |
| `emit_components.py:106` | `prog.prebuilt(body_solid)` | Battery cells — 3mm edge fillets |

**Fix:** Add a `FilletAllEdgesOp` or `FilletByAxisOp` that fillets all edges (or edges aligned with a given axis) on a shape. This covers the common pattern: "fillet all Z-aligned edges" or "fillet all edges with radius R". No tag-based edge selection needed.

### Category 2: SCS0009 dispatch (whole shape wrapped) — 2 usages

The SCS0009 servo has different geometry from STS-series. Rather than porting it, the entire shape is wrapped as PrebuiltOp.

| File | Usage | What's wrapped |
|------|-------|---------------|
| `emit_bracket.py:47` | `prog.prebuilt(bracket_envelope(...))` | SCS0009 bracket envelope |
| `emit_bracket.py:102` | `prog.prebuilt(bracket_solid(...))` | SCS0009 bracket solid |

**Fix:** Port `_scs0009_bracket_solid()` and `_scs0009_bracket_envelope()` to ShapeScript. They use the same primitives (Box, Cylinder, cut) as the STS bracket — just different dimensions and layout. This is straightforward porting work, not a language gap.

### Category 3: Connector port (complex shaped passage) — 2 usages

The connector port cut through the bracket wall is a shaped passage with type-specific features (polarization key, etc.). The `_connector_port()` function builds it with multiple boxes and cylinders.

| File | Usage | What's wrapped |
|------|-------|---------------|
| `emit_bracket.py:189` | `prog.prebuilt(cut_solid)` | Connector port cut in bracket |
| `emit_bracket.py:337` | `prog.prebuilt(cut_solid)` | Connector port cut in cradle |

**Fix:** Port `_connector_port()` to ShapeScript. It's Box + Cylinder + fuse/cut — no special operations needed. The emitter just needs to translate the function line by line.

### Category 4: Coupler D-clip geometry — 1 usage

The coupler's D-clip plates (semicircle + rectangle clipped together) can't be expressed with Box/Cylinder primitives alone. Needs arc/semicircle or a clipping operation.

| File | Usage | What's wrapped |
|------|-------|---------------|
| `emit_bracket.py:435` | `prog.prebuilt(direct_coupler)` | Entire coupler structural shape |

**Fix:** Two options:
1. Add `SemicircleOp` or `ArcExtrudeOp` to ShapeScript — express the D-clip as a native op
2. Approximate the semicircle as a cylinder cut in half (cylinder + box cut) — expressible with existing ops

Option 2 is simpler and accurate enough for FDM printing. A cylinder cut in half by a box IS a semicircle.

### Category 5: Servo connector receptacle — 1 usage

The receptacle solid (placed on the servo body) is built by `receptacle_solid()` from connectors.py, then located with `.moved()`.

| File | Usage | What's wrapped |
|------|-------|---------------|
| `emit_servo.py:234` | `prog.prebuilt(rcpt_placed)` | Servo connector receptacle |

**Fix:** Use `CallOp` to invoke `receptacle_script()` from emit_components.py (already ported). Then locate it. This is just a wiring change — the ShapeScript emitter exists.

### Category 6: Custom solid (body.custom_solid) — 1 usage

When a bot design provides a custom build123d solid directly, it gets wrapped as PrebuiltOp.

| File | Usage | What's wrapped |
|------|-------|---------------|
| `emit_body.py:57` | `prog.prebuilt(body.custom_solid)` | User-provided custom solid |

**Fix:** This is the escape hatch for user-defined geometry not expressible in ShapeScript. Keep PrebuiltOp for this case — it's the correct abstraction. Long-term, custom solids should be expressed as ShapeScript too, but that requires the user to write ShapeScript instead of build123d.

### Category 7: Coupler in body emitter — 1 usage

The coupler solid for coupler-style joints is built and located before injection.

| File | Usage | What's wrapped |
|------|-------|---------------|
| `emit_body.py:152` | `prog.prebuilt(coupler)` | Coupler solid at joint |

**Fix:** Use `CallOp` to invoke `coupler_solid_script()` from emit_bracket.py (already exists). Then locate it. Wiring change only.

---

## Task Breakdown

### Task 1: Add FilletAllEdgesOp to ShapeScript

**Files:**
- Modify: `botcad/shapescript/ops.py` — add `FilletAllEdgesOp(ref, target, radius)` and `FilletByAxisOp(ref, target, axis, radius)`
- Modify: `botcad/shapescript/program.py` — add `fillet_all(target, radius)` and `fillet_by_axis(target, axis, radius)` builders
- Modify: `botcad/shapescript/backend_occt.py` — handle new ops: `solid.fillet(radius, solid.edges())` and `solid.fillet(radius, [e for e in solid.edges() if aligned_with(e, axis)])`
- Test: `tests/test_shapescript_backend.py` — volume tests for fillet ops
- Then update `emit_servo.py` and `emit_components.py` to use native fillet ops instead of PrebuiltOp

### Task 2: Port SCS0009 bracket to ShapeScript

**Files:**
- Modify: `botcad/shapescript/emit_bracket.py` — replace PrebuiltOp wrappers with native ops for `_scs0009_bracket_solid()` and `_scs0009_bracket_envelope()`
- Test: `tests/test_shapescript_bracket.py` — roundtrip volume tests

### Task 3: Port connector port to ShapeScript

**Files:**
- Modify: `botcad/shapescript/emit_bracket.py` — translate `_connector_port()` to ShapeScript ops
- Test: `tests/test_shapescript_bracket.py` — roundtrip volume tests

### Task 4: Fix coupler D-clip with cylinder+box approximation

**Files:**
- Modify: `botcad/shapescript/emit_bracket.py` — replace PrebuiltOp with cylinder-cut-in-half approach
- Test: `tests/test_shapescript_bracket.py` — roundtrip volume tests (may need relaxed tolerance for the semicircle approximation)

### Task 5: Wire CallOp for servo connector and body coupler

**Files:**
- Modify: `botcad/shapescript/emit_servo.py` — use CallOp for receptacle_script()
- Modify: `botcad/shapescript/emit_body.py` — use CallOp for coupler_solid_script()
- Test: `tests/test_shapescript_roundtrip.py` — existing roundtrip tests must pass

### Task 6: Update snapshot baselines + verify

**Files:**
- Run: `uv run pytest tests/test_shapescript_snapshots.py --update-shapescript-baselines`
- Verify no PrebuiltOp in production emitters (except custom_solid escape hatch)

---

## Success Criteria

After all tasks, `grep -r "\.prebuilt\(" botcad/shapescript/emit_*.py` should return only:
- `emit_body.py` — `body.custom_solid` (the intentional escape hatch)

Every other piece of geometry is native ShapeScript ops — fully inspectable in the viewer.

---

## What We're NOT Doing

- **Removing PrebuiltOp from the language** — it remains as the escape hatch for user-defined custom solids
- **Porting debug_drawing.py / render_svg.py** — these read geometry, they don't create it
- **Porting _occt_cut / _subprocess_bool_cut** — these are backend implementation details

## Ordering

Tasks 1-4 are independent (can run in parallel). Task 5 depends on Tasks 3-4 (coupler needs to be ported first). Task 6 runs last.
