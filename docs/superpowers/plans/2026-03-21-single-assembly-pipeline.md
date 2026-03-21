# Single Assembly Pipeline — Body as Source of Truth

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every physical object (fabricated and purchased) is a `Body` in the Assembly tree with a ShapeScript, world position, and world orientation. All exporters (STL, STEP, MuJoCo, viewer manifest) read from this single list. No exporter computes transforms independently.

**Architecture:** During `bot.solve()`, ALL bodies are placed in world frame — structural bodies, servos, horns, fasteners, mounted components. Each body gets a ShapeScript during `build_cad()`. Exporters iterate `bot.all_bodies` and use `body.world_pos`, `body.world_quat`, `body.shapescript` — no independent placement logic.

**Tech Stack:** Python, ShapeScript, existing emitters.

---

## The Problem

Currently, three separate code paths compute positions/rotations for the same objects:
- `emit_cad()` builds `AssemblyPart(solid=positioned_solid)` for STEP export
- `emit_cad()` separately exports STLs (sometimes forgetting rotation)
- `emit_mujoco()` independently computes servo/horn/fastener positions from servo specs
- `emit_viewer_manifest()` independently computes part positions from servo specs

This leads to bugs where one exporter forgets a transform that another has.

## The Fix

### 1. Expand `bot.all_bodies` to include ALL physical objects

Currently `all_bodies` only has structural bodies (base, wheels, arm segments). Purchased parts (servos, horns, fasteners, components) aren't bodies — they're computed on the fly by each exporter.

After the fix, `all_bodies` includes:
```
bot.all_bodies = [
    Body("base", kind=FABRICATED, shapescript=base_script, world_pos=..., world_quat=...),
    Body("STS3215_left_wheel", kind=PURCHASED, shapescript=servo_script, world_pos=..., world_quat=...),
    Body("horn_left_wheel", kind=PURCHASED, shapescript=horn_script, world_pos=..., world_quat=...),
    Body("M3_screw_left_wheel_1", kind=PURCHASED, shapescript=fastener_script, world_pos=..., world_quat=...),
    ...
]
```

### 2. Add world transform fields to Body

```python
@dataclass
class Body:
    # ... existing fields ...
    world_pos: Vec3 = (0.0, 0.0, 0.0)      # set by solve()
    world_quat: Quat4 = (1.0, 0.0, 0.0, 0.0)  # set by solve()
    shapescript: ShapeScript | None = None   # set by build_cad()
```

### 3. Populate during solve() + build_cad()

`bot.solve()` already computes world positions for structural bodies. Extend it to also place:
- Servo bodies at `joint.solved_servo_center` with `joint.solved_servo_quat`
- Horn bodies at joint position + axis offset
- Mounted components at `mount.resolved_pos`
- Fasteners at mounting ear positions

`bot.build_cad()` assigns ShapeScript to each body:
- Structural bodies: `emit_body_ir(body, ...)`
- Servos: `servo_script(servo_spec)`
- Horns: `horn_script(servo_spec)`
- Fasteners: `fastener_script(fastener_spec, length)`
- Components: `camera_script(spec)`, `battery_script(spec)`, etc.

### 4. Simplify ALL exporters to iterate all_bodies

Each exporter becomes:
```python
def emit_X(bot):
    for body in bot.all_bodies:
        solid = execute_cached(body.shapescript)  # cache by content hash
        positioned = solid.moved(Location(body.world_pos, quat_to_euler(body.world_quat)))
        # export positioned solid in X format
```

No exporter has its own servo/horn/fastener placement logic.

---

## Task Breakdown

### Task 1: Add world transform + shapescript fields to Body

**Files:** `botcad/skeleton.py`

Add to Body dataclass:
```python
world_pos: Vec3 = (0.0, 0.0, 0.0)
world_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
shapescript: object | None = None  # ShapeScript program
```

### Task 2: Create purchased Body instances during solve()

**Files:** `botcad/skeleton.py` (in `_collect_tree` or `solve`)

After the kinematic tree walk, create Body instances for all purchased parts:
- For each joint: create servo body, horn body, fastener bodies
- For each mount: create component body
- Compute world_pos and world_quat for each from the existing servo_placement / mount position data

These go into `bot.all_bodies` alongside structural bodies. Add a field to distinguish: `body.parent_body` pointing to the structural body they're associated with.

### Task 3: Assign ShapeScript during build_cad()

**Files:** `botcad/emit/cad.py`

In `build_cad()`, after executing each body's ShapeScript, store it:
```python
body.shapescript = prog
```

For purchased parts, assign the appropriate script:
```python
servo_body.shapescript = servo_script(servo_spec)
horn_body.shapescript = horn_script(servo_spec)
```

### Task 4: Refactor emit_cad() to use body.world_pos/quat

**Files:** `botcad/emit/cad.py`

Replace the independent STEP assembly positioning and STL export with a single loop over `bot.all_bodies`. Each body's shapescript is executed (cached), the solid is positioned using `body.world_pos`/`body.world_quat`, and exported.

### Task 5: Refactor emit_mujoco() to use body.world_pos/quat

**Files:** `botcad/emit/mujoco.py`

Replace the independent servo/horn/fastener positioning with reads from the body's world transform. The MuJoCo body tree structure stays (it mirrors the kinematic tree), but positions come from `body.world_pos` instead of being recomputed.

### Task 6: Refactor emit_viewer_manifest() to use body data

**Files:** `botcad/emit/viewer.py`

The `parts` list is already generated by iterating joints/mounts. Replace with a simple loop over purchased bodies in `bot.all_bodies`, reading pos/quat from the body.

### Task 7: Remove AssemblyPart

**Files:** `botcad/emit/cad.py`

`AssemblyPart` becomes unnecessary — it was a positioned solid with metadata. Now `Body` carries all of that. Delete the class.

---

## Success Criteria

After all tasks:
1. `bot.all_bodies` contains every physical object
2. Every body has `world_pos`, `world_quat`, `shapescript`
3. No exporter computes positions independently
4. The horn rotation bug is impossible (position/orientation is computed once in solve, used everywhere)
5. All existing tests pass
6. `make validate` passes

---

## Ordering

Task 1 → Task 2 → Task 3 → Tasks 4-6 (parallel) → Task 7

Tasks 4-6 are independent refactors of different emitters.
