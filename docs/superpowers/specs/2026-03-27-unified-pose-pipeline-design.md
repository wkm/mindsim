# Unified Pose Pipeline — Single Orientation System for All Placed Components

**Date:** 2026-03-27
**Status:** Draft

## Problem

Component placement uses two incompatible orientation systems:

1. **Quaternion-based** (servos): `joint.solved_servo_quat` computed in packing, composed via `quat_multiply` and `rotate_vec`. Works correctly.
2. **Lambda-based** (mounted components): coordinate-swap functions in `_FACE_ROTATION` table, applied point-by-point via `mount.rotate_point()`. No stored quaternion. Breaks repeatedly.

This causes recurring bugs:
- Mount fasteners were invisible in the Design Viewer (node ID mismatch)
- Pi fasteners faced wrong direction (shank up instead of down)
- Camera fasteners pointed forward instead of backward
- Wheel fasteners pointed down instead of sideways (viewer ignores `body.frame_quat`)

The lambda system is fragile because:
- It doesn't compose (can't multiply two lambdas like quaternions)
- It's not inspectable (can't print/debug a lambda like a quaternion)
- It treats positions and directions identically
- Every new consumer re-derives orientation from scratch
- There's no single source of truth for "what orientation is this mount in?"

Additionally, `MountPoint.axis` has inconsistent conventions: MountingEar values implicitly mean "head direction" while component MountPoints mean "insertion direction." The MuJoCo emitter compensates by negating one but not the other.

Finally, the data model mixes design-time inputs and solver outputs on the same mutable objects (`mount.resolved_pos`, `joint.solved_servo_quat`), violating the project's declarative data principle.

## Design

### New Types

All new types are frozen dataclasses in `botcad/geometry.py`:

```python
@dataclass(frozen=True)
class Pose:
    """Position + orientation in a single frame. The atomic unit of placement."""
    pos: Vec3
    quat: Quat  # (w, x, y, z), identity = (1, 0, 0, 0)

POSE_IDENTITY = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
```

Free functions operating on `Pose`:

- `pose_transform_point(pose, local_point) -> Vec3` — rotate then translate
- `pose_transform_dir(pose, local_dir) -> Vec3` — rotate only (for axes/normals)
- `pose_compose(parent, child) -> Pose` — compose two poses

```python
@dataclass(frozen=True)
class MountRotation:
    """Design-time rotation of a component on its mounting surface."""
    yaw: float = 0.0  # degrees around component local Z

MOUNT_NO_ROTATION = MountRotation()
MOUNT_YAW_90 = MountRotation(yaw=90.0)
```

### Solver Output Types

```python
@dataclass(frozen=True)
class Placement:
    """Solver output for any placed component — servo or mount."""
    pose: Pose       # component origin + orientation in world frame
    bbox: Vec3       # axis-aligned bounding box in world frame

@dataclass(frozen=True)
class PackingResult:
    """Complete solver output. Returned by pack(), consumed by all emitters."""
    placements: dict[Mount | Joint, Placement]
```

`pack(bot)` returns a `PackingResult` instead of mutating skeleton objects. Emitters receive `(bot, packing_result)`.

### Data Model Changes

**`Mount`** becomes frozen. Design-time input only:

```python
@dataclass(frozen=True)
class Mount:
    component: Component
    label: str
    position: Position | Vec3
    rotation: MountRotation = MOUNT_NO_ROTATION
    insertion_axis: Vec3 | None = None
```

Removed fields: `resolved_pos`, `resolved_insertion_axis`, `solved_bbox`, `rotate_z`.
Removed methods/properties: `rotate_point()`, `face_euler_deg`, `placed_dimensions`, `_face_rotation_entry`, `face_outward`.

The `face_outward` property (which checked `MountOrientation.FACE_NORMAL`) moves into the packing solver — it's a solver concern, not a data property.

**`Joint`** drops solver state: `solved_servo_center`, `solved_servo_quat` removed (lives in `PackingResult`).

**`Body`** gains `world_pose: Pose` replacing separate `world_pos: Vec3` + `world_quat: Quat`. `frame_quat` stays — MuJoCo needs body-local frame orientation separately.

Note: `Body` remains mutable (not frozen) because tree building mutates many fields incrementally. Freezing Body is a separate follow-up.

### Axis Convention

**One convention everywhere:** `MountPoint.axis` = insertion direction (where the shank goes into material).

`MountingEar` factory default changes from `(0,0,-1)` to `(0,0,1)` to match this convention. All existing MountingEar and MountPoint axis values in servo definitions are audited and flipped as needed.

One function computes fastener orientation for both joints and mounts:

```python
def fastener_pose(parent: Pose, mp: MountPoint) -> Pose:
    """Compute world-frame fastener pose from parent placement and mount point.

    MountPoint.axis = insertion direction (where shank goes).
    Screw STL head faces +Z, so align +Z with head direction = -axis.
    """
    pos = pose_transform_point(parent, mp.pos)
    head_dir = (-mp.axis[0], -mp.axis[1], -mp.axis[2])
    axis_align = rotation_between((0.0, 0.0, 1.0), head_dir)
    quat = quat_multiply(parent.quat, axis_align)
    return Pose(pos, quat)
```

### Removed Code

- `_FACE_ROTATION` lambda table and `_FaceRotEntry` type (skeleton.py)
- `Mount.rotate_point()` method
- `Mount.placed_dimensions` property
- `Mount.face_euler_deg` property
- `Mount._face_rotation_entry` property
- `Mount.face_outward` property (logic moves to packing solver)
- `Mount.resolved_pos`, `Mount.resolved_insertion_axis`, `Mount.solved_bbox` fields
- `Joint.solved_servo_center`, `Joint.solved_servo_quat` fields
- `Body.world_pos`, `Body.world_quat` (replaced by `Body.world_pose`)
- `_build_joint_fastener_entry()` in viewer.py
- `_build_mount_fastener_entry()` in viewer.py
- `_transform_fastener_pos()` in viewer.py
- `_z_to_axis_quat()` in mujoco.py

### Packing Solver Changes

`pack(bot) -> PackingResult`:

For each joint:
- `servo_placement()` already returns `(center, quat)` — wrap in `Pose`
- Compute bbox from servo dimensions rotated by pose quat
- Store as `placements[joint] = Placement(pose, bbox)`

For each mount:
- Compute body-frame dimensions first (needed for position resolution):
  rotate component dimensions by yaw + face quat, take axis-aligned extents.
  This replaces `placed_dimensions` and happens as an intermediate step
  *before* the final `Placement` is produced.
- Resolve position via existing heuristic using the body-frame dimensions
- Compute mount quat by composing:
  1. `euler_to_quat((0, 0, mount.rotation.yaw))` — yaw rotation
  2. Face rotation quat from position string (new `_FACE_QUAT` dict)
  3. `body.frame_quat` — handles rims/cylinders
- Compute world-frame bbox via `rotate_vec(full_quat, dim_vec)` then take axis-aligned extents
- Store as `placements[mount] = Placement(Pose(pos, quat), bbox)`

Note: `Placement.bbox` is in **world frame** (axis-aligned after full rotation).

**`_FACE_QUAT`** lives in `botcad/packing.py` (replacing `_FACE_ROTATION` from skeleton.py). It is a `dict[str, Quat]` mapping position strings to quaternions. Explicit values derived from the Euler angles in the current table:

| Position | Euler (current) | Quaternion |
|----------|----------------|------------|
| `"front"` | (-90, 0, 0) | (0.7071, -0.7071, 0, 0) |
| `"back"` | (90, 0, 0) | (0.7071, 0.7071, 0, 0) |
| `"left"` | (0, -90, 0) | (0.7071, 0, -0.7071, 0) |
| `"right"` | (0, 90, 0) | (0.7071, 0, 0.7071, 0) |
| `"bottom"` | (180, 0, 0) | (0, 1, 0, 0) |

`"top"` and `"center"` are identity — no entry needed.

Whether a mount needs face rotation is determined by the component's `MountOrientation` (checked via `get_component_meta`), same logic as the current `face_outward` property but moved into the solver.

New utility: `euler_to_quat()` added to `geometry.py` as the inverse of existing `quat_to_euler()`.

**Resolved insertion axis:** No longer a stored field. Consumers that need it (assembly renders, ShapeScript) derive it from the placement pose: `pose_transform_dir(placement.pose, (0, 0, 1))` gives the component's Z axis in world frame, which is the mount face normal.

### Emitter and Consumer Changes

All emitters and consumers that use placement data change signature to receive `(bot, packing: PackingResult)`.

**Complete list of affected modules:**

| Module | Old fields used | Migration |
|--------|----------------|-----------|
| `botcad/emit/viewer.py` | `solved_servo_quat`, `solved_servo_center`, `mount.rotate_point()` | `packing.placements[joint\|mount].pose` + `fastener_pose()` |
| `botcad/emit/mujoco.py` | `solved_servo_quat`, `solved_servo_center`, `mount.rotate_point()`, `body.to_body_frame()`, `_z_to_axis_quat()` | Same pattern. `_z_to_axis_quat` removed — `fastener_pose()` handles orientation. `body.to_body_frame()` no longer needed for mount points since `body.frame_quat` is composed into the mount's placement pose |
| `botcad/emit/cad.py` | `mount.face_euler_deg`, `mount.rotate_z` | `quat_to_euler(packing.placements[mount].pose.quat)` |
| `botcad/shapescript/emit_body.py` | `mount.resolved_pos`, `mount.placed_dimensions`, `mount.resolved_insertion_axis`, `joint.solved_servo_quat`, `joint.solved_servo_center` | Pose lookups from `PackingResult` |
| `botcad/routing.py` | `joint.solved_servo_center`, `joint.solved_servo_quat`, `mount.resolved_pos` | Pose lookups from `PackingResult` |
| `botcad/emit/assembly_renders.py` | `mount.resolved_insertion_axis` | `pose_transform_dir(placement.pose, (0,0,1))` |
| `botcad/clearance.py` | `body.world_pos`, `body.world_quat` | `body.world_pose.pos`, `body.world_pose.quat` |
| `botcad/skeleton.py` (`_solve_purchased_body_positions`) | `joint.solved_servo_center`, `joint.solved_servo_quat` | Reads from `PackingResult` to set `body.world_pose` |
| `mindsim/server.py` | `joint.solved_servo_center`, `joint.solved_servo_quat` | Pose lookups from `PackingResult` |

**Viewer (`viewer.py`):**

Both fastener builders collapse into one call to `fastener_pose()`:

```python
# Joint fasteners
servo_pose = packing.placements[joint].pose
for mp in joint.servo.mounting_ears:
    fp = fastener_pose(servo_pose, mp)
    manifest["parts"].append(_fastener_entry(..., fp))

# Mount fasteners — identical pattern
mount_pose = packing.placements[mount].pose
for mp in mount.component.mounting_points:
    fp = fastener_pose(mount_pose, mp)
    manifest["parts"].append(_fastener_entry(..., fp))
```

One `_fastener_entry()` function replaces `_build_joint_fastener_entry` and `_build_mount_fastener_entry`.

Mounted component body entries use `packing.placements[mount].pose` instead of hardcoded identity quat.

Wire stub placement uses `pose_transform_point(mount_pose, wp.pos)` and `pose_transform_dir(mount_pose, exit_dir)` instead of `mount.rotate_point()`.

**MuJoCo (`mujoco.py`):**

Same pattern — looks up poses from `PackingResult`, calls `fastener_pose()` for screw geoms. The `body.to_body_frame(mount.rotate_point(...))` chain goes away because `body.frame_quat` is already composed into the mount's placement pose. `_z_to_axis_quat()` is removed — `fastener_pose()` handles all orientation.

**CAD (`cad.py`):**

Where it uses `mount.face_euler_deg` and `mount.rotate_z` for build123d `Location()`, it uses `quat_to_euler(packing.placements[mount].pose.quat)` instead.

### Bot Design API

Bot definitions update from:

```python
base.mount(RaspberryPiZero2W(), position="top", label="pi", rotate_z=True)
```

To:

```python
base.mount(RaspberryPiZero2W(), position="top", label="pi", rotation=MOUNT_YAW_90)
```

### Testing

**Unit tests (`test_pose.py`):**
- `Pose` composition: `pose_compose(parent, child)` matches manual quat math
- `euler_to_quat` round-trips with `quat_to_euler` for all face rotation angles
- `MountRotation(yaw=90)` produces the same quaternion as the old `rotate_z` lambda
- `pose_transform_point` and `pose_transform_dir` correctness

**Fastener orientation tests (extend `test_fastener_orientation.py`):**
- `fastener_pose()` with insertion axis `(0,0,-1)` — head faces `+Z`, shank faces `-Z`
- `fastener_pose()` with rotated parent pose — head/shank rotate correctly
- MountingEar axis values under new convention produce same world-frame results as before

**Cross-emitter consistency test (`test_emitter_consistency.py` — new):**
- For each bot: generate viewer manifest AND MuJoCo XML
- For every fastener, extract world-frame position and orientation from both
- Assert they match within tolerance
- Key guardrail — if emitters ever diverge again, this catches it

**Mount quaternion regression (`test_mount_orientation.py` — new):**
- For each face position (front, back, left, right, bottom, top, center):
  - Compute quaternion via new `_face_quat()` lookup
  - Apply to test points via `rotate_vec(quat, point)`
  - Assert results match old lambda outputs
- Validates migration didn't change any orientations

**Packing completeness:**
- Assert every mount and joint in the bot has an entry in `PackingResult.placements`

## Follow-Up: Custom AST Linter

After this work lands, add a custom Python AST linter (`tools/lint.py`) to enforce conventions. Initial rules:
- Ban calls to removed methods/references (`rotate_point`, `_FACE_ROTATION`, old solved fields)
- Ban raw `rotation_between` for fastener orientation outside `geometry.py`
- Warn on `world_pos` / `world_quat` (should be `world_pose.pos` / `world_pose.quat`)

See memory: `project_custom_linter.md`.

### Axis Convention Worked Example

To verify the MountingEar axis flip preserves behavior:

**Before (current):** MountingEar axis = `(0,0,-1)` (head direction convention)
```
axis_align = rotation_between((0,0,1), (0,0,-1))  →  180° flip
final_quat = quat_multiply(servo_quat, 180°_flip)
```

**After (new):** MountingEar axis = `(0,0,1)` (insertion direction convention)
```
head_dir = -(0,0,1) = (0,0,-1)
axis_align = rotation_between((0,0,1), (0,0,-1))  →  180° flip  ← same rotation
final_quat = quat_multiply(servo_quat, 180°_flip)               ← same result
```

The negation in `fastener_pose()` compensates for the flipped axis value. World-frame orientation is identical.

## Migration Strategy

Each step is independently testable and committable. The dual-write approach keeps the code working between steps.

1. Add `Pose`, `MountRotation`, `Placement`, `PackingResult`, `euler_to_quat`, `fastener_pose` to `geometry.py` with unit tests
2. Update `Mount`: add `rotation: MountRotation` field, keep old fields temporarily. Update bot definitions (`rotate_z=True` → `rotation=MOUNT_YAW_90`). Backward compat: `rotate_z` becomes a derived property reading from `rotation.yaw`.
3. Update `pack()` to return `PackingResult` AND still write old fields (`resolved_pos`, `solved_servo_quat`, etc.) so existing consumers keep working. Add mount quaternion computation. Update `tests/test_domain_model.py` for new fields.
4. Migrate `viewer.py` to use `PackingResult` and `fastener_pose()`
5. Migrate `mujoco.py` — replace `_z_to_axis_quat` and `body.to_body_frame(mount.rotate_point(...))` chains
6. Migrate `cad.py`, `shapescript/emit_body.py`, `routing.py`, `assembly_renders.py`, `server.py`
7. Update `Body` to use `world_pose: Pose`, migrate `clearance.py` and `skeleton.py:_solve_purchased_body_positions()`
8. Standardize MountingEar axis values to insertion convention
9. Remove dual-write from `pack()`. Remove dead code: `rotate_point()`, `_FACE_ROTATION`, old fields, old `_build_*_fastener_entry` functions, `_z_to_axis_quat`. Remove old tests from `test_domain_model.py`.
10. Add cross-emitter consistency tests
11. Verify all bots: `make validate`, compare manifests before/after
