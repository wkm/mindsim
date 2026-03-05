# Plan: Servo Solids in STEP + Higher-Fidelity Brackets

## Goal
Make the generated STEP assembly show seated servos and produce bracket geometry that's dimensionally correct for fabrication — "ugly but correct" is fine.

## Current State
- `emit_cad.py` generates per-body shells (hollow boxes/tubes) + bracket geometry (pocket, shaft hole, screw holes, cable slot) unioned into parent body
- Servo bodies are **absent** from the STEP — only the negative-space pocket exists
- Body shells use generic dimensions from the skeleton DSL, not shaped around the bracket+servo envelope

## Changes

### 1. Add `servo_solid()` to `botcad/bracket.py`
Generate a build123d Solid representing the physical servo body:
- Main rectangular block from `body_dimensions`
- Mounting ear flanges extending below (from `mounting_ears` positions)
- Output shaft cylinder stub on +Z face (from `shaft_offset`)
- Colored dark gray (matches real servo)

This is a simple box + ear tabs + cylinder. No need for filleted edges or cosmetic detail.

### 2. Include servo solids in STEP assembly (`botcad/emit/cad.py`)
In `emit_cad()`, for each joint:
- Build the servo solid via `servo_solid()`
- Position it using `servo_placement()` (same math already used for brackets)
- Add it to the assembly as a separate dark-gray part (not unioned into the bracket — servos are purchased, not printed)

This means the STEP will have: printed brackets (per rigid group) + servo bodies (separate parts, colored differently).

### 3. Size body shells to the bracket envelope
Currently body shells are sized by `packing.py` from component dimensions. The bracket outer dimensions should inform body sizing too — where a bracket protrudes beyond the body shell, the shell looks wrong.

Approach: In `_make_body_solid()`, after computing the base shell, check if any bracket extends beyond the shell bounds. If so, add material (fillet/extend the shell) to fully enclose the bracket. Simpler version: just make the body shell at least as large as the bracket outer envelope in each axis.

**Actually, this may be over-engineering for now.** The bracket already unions into the body solid via boolean `+`. The visual result in the STEP is that brackets stick out of thin body shells, which looks odd but is geometrically correct — the bracket IS the structural part, the thin shell just connects brackets. This is fine for "ugly but correct". Skip this for now.

### 4. Regenerate bot outputs
- Run `design.py` for so101_arm and wheeler_arm
- Verify STEP shows servo bodies seated in brackets
- Verify STLs unchanged (servo solids are assembly-only, not part of per-body meshes)

### 5. Commit and push

## Files Modified
- `botcad/bracket.py` — add `servo_solid()` function
- `botcad/emit/cad.py` — include servo solids in STEP assembly as separate colored parts
- `bots/so101_arm/assembly.step` — regenerated
- `bots/wheeler_arm/assembly.step` — regenerated

## What We're NOT Doing (yet)
- Contoured bracket surfaces, fillets, chamfers
- Mounting features (screw bosses, alignment pins)
- Body shells shaped to bracket envelope
- Horn/spline geometry on servo shaft
