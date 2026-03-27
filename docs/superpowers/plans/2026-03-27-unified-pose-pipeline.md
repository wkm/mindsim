# Unified Pose Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dual orientation system (quaternion for servos, coordinate-swap lambdas for mounts) with a single `Pose`-based pipeline that all emitters share.

**Architecture:** New `Pose(pos, quat)` type in `geometry.py`. Packing solver returns immutable `PackingResult` instead of mutating skeleton. One `fastener_pose()` function for all fastener orientation. Dual-write bridge keeps old fields alive until all consumers migrate.

**Tech Stack:** Python dataclasses, quaternion math (existing `geometry.py` utilities), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-27-unified-pose-pipeline-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `botcad/geometry.py` | Modify | Add `Pose`, `MountRotation`, `Placement`, `PackingResult`, `euler_to_quat`, `fastener_pose`, pose helpers |
| `botcad/skeleton.py` | Modify | Update `Mount` (freeze, add `rotation`), update `Body` (`world_pose`), remove lambda table |
| `botcad/packing.py` | Modify | Return `PackingResult`, compute mount quaternions, `_FACE_QUAT` dict |
| `botcad/component.py` | Modify | Flip `MountingEar` default axis |
| `botcad/components/servo.py` | Modify | Audit MountingEar/MountPoint axis values |
| `botcad/emit/viewer.py` | Modify | Use `PackingResult` + `fastener_pose()` |
| `botcad/emit/mujoco.py` | Modify | Use `PackingResult` + `fastener_pose()` |
| `botcad/emit/cad.py` | Modify | Use `PackingResult` for mount orientation |
| `botcad/shapescript/emit_body.py` | Modify | Use `PackingResult` for mount/servo poses |
| `botcad/routing.py` | Modify | Use `PackingResult` for servo/mount positions |
| `botcad/emit/assembly_renders.py` | Modify | Derive insertion axis from placement pose |
| `botcad/clearance.py` | Modify | Use `body.world_pose` |
| `mindsim/server.py` | Modify | Use `PackingResult` for servo poses |
| `bots/wheeler_base/design.py` | Modify | `rotate_z=True` to `rotation=MOUNT_YAW_90` |
| `tests/test_pose.py` | Create | Unit tests for Pose, euler_to_quat, fastener_pose |
| `tests/test_mount_orientation.py` | Create | Regression: new quaternions match old lambda outputs |
| `tests/test_emitter_consistency.py` | Create | Cross-emitter fastener position/orientation check |
| `tests/test_domain_model.py` | Modify | Update tests for new Mount API |
| `tools/lint_project.py` | Modify | Update suppressions as old code is removed |

---

## Task 1: Add Pose type and helpers to geometry.py

**Files:**
- Modify: `botcad/geometry.py`
- Create: `tests/test_pose.py`

- [ ] Write failing tests for `euler_to_quat` (round-trips with `quat_to_euler` for all face rotation angles, identity, yaw 90)
- [ ] Run test to verify it fails (ImportError)
- [ ] Implement `euler_to_quat` — inverse of existing `quat_to_euler`, intrinsic XYZ convention
- [ ] Run test to verify it passes
- [ ] Write failing tests for `Pose`, `pose_transform_point`, `pose_transform_dir`, `pose_compose` — identity, translation-only, rotation-only, composition, frozen
- [ ] Run test to verify it fails (ImportError)
- [ ] Implement `Pose` dataclass (frozen) and free functions
- [ ] Run test to verify it passes
- [ ] Write failing tests for `fastener_pose` — insertion axis (0,0,-1) head faces +Z, (0,0,1) head faces -Z, position from parent, rotated parent
- [ ] Run test to verify it fails (ImportError)
- [ ] Implement `fastener_pose(parent: Pose, mp) -> Pose` — negates axis, rotation_between, quat_multiply with parent
- [ ] Run all tests, run `make lint`
- [ ] Commit: `feat: add Pose type, euler_to_quat, fastener_pose, pose helpers`

---

## Task 2: Add MountRotation type and update Mount + bot definitions

**Files:**
- Modify: `botcad/geometry.py`
- Modify: `botcad/skeleton.py` (Mount dataclass, Body.mount method)
- Modify: `bots/wheeler_base/design.py`
- Modify: `tests/test_domain_model.py`

- [ ] Add `MountRotation` frozen dataclass to geometry.py with `yaw: float = 0.0`, plus `MOUNT_NO_ROTATION` and `MOUNT_YAW_90` constants
- [ ] Add `rotation: MountRotation` field to Mount, make `rotate_z` a derived `@property` (reads `rotation.yaw == 90.0`)
- [ ] Update `Body.mount()` to accept `rotation: MountRotation` param, support both `rotation` and deprecated `rotate_z` for bridge period
- [ ] Update bot definitions: `rotate_z=True` -> `rotation=MOUNT_YAW_90` in `bots/wheeler_base/design.py` and any others
- [ ] Update `tests/test_domain_model.py` for new API
- [ ] Run tests and lint
- [ ] Commit: `feat: add MountRotation type, update Mount and bot definitions`

---

## Task 3: Add PackingResult, update packing solver with dual-write

**Files:**
- Modify: `botcad/geometry.py` (add Placement, PackingResult)
- Modify: `botcad/packing.py`
- Modify: `botcad/skeleton.py` (store PackingResult on Bot)
- Create: `tests/test_mount_orientation.py`

- [ ] Add `Placement(pose, bbox)` and `PackingResult(placements)` frozen dataclasses to geometry.py
- [ ] Write regression tests (`test_mount_orientation.py`): for each face position (front/back/left/right/bottom), verify `euler_to_quat` + `rotate_vec` matches the old lambda output. Also test yaw=90 matches old rotate_z swap `(-y, x, z)`.
- [ ] Run regression tests to verify they pass (they test existing math, should pass)
- [ ] Add `_FACE_QUAT` dict to packing.py mapping position strings to quaternions. Add `_mount_quat(mount, body)` helper composing yaw + face rotation + body.frame_quat
- [ ] Update `solve_packing()` to return `PackingResult` AND still write old fields (dual-write). For joints: wrap (center, quat) in Pose. For mounts: compute quat via `_mount_quat`, compute bbox.
- [ ] Store `PackingResult` on `Bot` (add field, set in `_rebuild`)
- [ ] Write packing completeness test: every mount and joint in bot has a placement entry
- [ ] Run all tests, run `make lint`
- [ ] Commit: `feat: packing solver returns PackingResult with dual-write bridge`

---

## Task 4: Migrate viewer.py to PackingResult + fastener_pose()

**Files:**
- Modify: `botcad/emit/viewer.py`
- Modify: `tools/lint_project.py` (remove viewer suppressions)

- [ ] Update `build_viewer_manifest` signature to accept `packing: PackingResult`, update callers
- [ ] Replace `_build_joint_fastener_entry` — use `packing.placements[joint].pose` + `fastener_pose()` + new unified `_fastener_entry()` helper
- [ ] Replace `_build_mount_fastener_entry` — same pattern with `packing.placements[mount].pose`
- [ ] Fix mounted component body quat: replace hardcoded `[1.0, 0.0, 0.0, 0.0]` with `packing.placements[mount].pose.quat`
- [ ] Migrate wire stub placement: `pose_transform_point`/`pose_transform_dir` instead of `mount.rotate_point()`
- [ ] Delete old functions: `_build_joint_fastener_entry`, `_build_mount_fastener_entry`, `_transform_fastener_pos`
- [ ] Remove all `plint: disable` comments from viewer.py
- [ ] Run `make validate`
- [ ] Commit: `refactor: viewer.py uses PackingResult + fastener_pose()`

---

## Task 5: Migrate mujoco.py to PackingResult + fastener_pose()

**Files:**
- Modify: `botcad/emit/mujoco.py`

- [ ] Update emitter signature to accept `PackingResult`
- [ ] Replace joint fastener orientation: use `fastener_pose()` instead of `rotate_vec` + `_z_to_axis_quat`
- [ ] Replace mount fastener orientation: use `fastener_pose()` instead of `body.to_body_frame(mount.rotate_point(mp.axis))` + negate + `_z_to_axis_quat`
- [ ] Delete `_z_to_axis_quat()`
- [ ] Remove `plint: disable` comments from mujoco.py
- [ ] Run `make validate`
- [ ] Commit: `refactor: mujoco.py uses PackingResult + fastener_pose()`

---

## Task 6: Migrate remaining consumers

**Files:**
- Modify: `botcad/emit/cad.py`, `botcad/shapescript/emit_body.py`, `botcad/routing.py`, `botcad/emit/assembly_renders.py`, `mindsim/server.py`

- [ ] Migrate `cad.py`: replace `mount.face_euler_deg`/`mount.rotate_z` with `quat_to_euler(packing.placements[mount].pose.quat)`
- [ ] Migrate `shapescript/emit_body.py`: replace all `mount.resolved_pos`, `mount.placed_dimensions`, `mount.resolved_insertion_axis`, `joint.solved_servo_center`, `joint.solved_servo_quat` with PackingResult lookups
- [ ] Migrate `routing.py`: replace `joint.solved_servo_center/quat` and `mount.resolved_pos` with PackingResult lookups
- [ ] Migrate `assembly_renders.py`: replace `mount.resolved_insertion_axis` with `pose_transform_dir(placement.pose, (0,0,1))`
- [ ] Migrate `server.py`: replace `joint.solved_servo_center/quat` with PackingResult lookups
- [ ] Run `make validate`
- [ ] Commit: `refactor: migrate cad, shapescript, routing, renders, server to PackingResult`

---

## Task 7: Update Body to use world_pose

**Files:**
- Modify: `botcad/skeleton.py` (Body class, `_compute_world_transforms`, `_create_purchased_bodies`)
- Modify: `botcad/clearance.py`
- Modify: all files using `body.world_pos` or `body.world_quat` (viewer.py, mujoco.py, cad.py, etc.)

Depends on Task 3 (`packing_result` stored on Bot).

- [ ] Add `world_pose: Pose` to Body, keep `world_pos`/`world_quat` as `@property` getters from `world_pose`
- [ ] Update `_compute_world_transforms` and `_create_purchased_bodies` to set `body.world_pose`
- [ ] Migrate `clearance.py` to `body.world_pose.pos`/`.quat`
- [ ] Grep for all remaining `body.world_pos`/`body.world_quat` across codebase (including viewer.py, mujoco.py, cad.py) and migrate
- [ ] Run `make validate`
- [ ] Commit: `refactor: Body uses world_pose: Pose`

---

## Task 8: Standardize MountingEar axis convention

**Files:**
- Modify: `botcad/component.py`
- Modify: `tests/test_pose.py`, `tests/test_domain_model.py`

- [ ] Write test proving axis flip preserves orientation: old axis (0,0,-1) with no-negate == new axis (0,0,1) with fastener_pose negate
- [ ] Grep codebase to verify no consumer reads `MountingEar.axis` directly outside `fastener_pose()` — if any found, migrate first
- [ ] Flip `MountingEar` default axis from `(0,0,-1)` to `(0,0,1)`, update docstring
- [ ] Audit servo horn/rear mount points in `botcad/components/servo.py` (horn +Z insertion = correct, rear -Z insertion = correct)
- [ ] Update any affected tests
- [ ] Run `make validate`
- [ ] Commit: `refactor: standardize MountingEar axis to insertion direction convention`

---

## Task 9: Remove dual-write bridge and dead code

**Files:**
- Modify: `botcad/packing.py`, `botcad/skeleton.py`, `botcad/emit/viewer.py`, `botcad/emit/mujoco.py`, `tests/test_domain_model.py`, `tools/lint_project.py`

- [ ] Remove dual-write from packing (stop writing `mount.resolved_pos`, `joint.solved_servo_center`, etc.)
- [ ] Remove old fields from Mount: `resolved_pos`, `resolved_insertion_axis`, `solved_bbox`. Remove `rotate_z` compat property. Make Mount `@dataclass(frozen=True)`.
- [ ] Remove old methods from Mount: `rotate_point()`, `placed_dimensions`, `face_euler_deg`, `_face_rotation_entry`, `face_outward`
- [ ] Remove `_FACE_ROTATION` table, `_FaceRotEntry` type from skeleton.py
- [ ] Remove `solved_servo_center`/`solved_servo_quat` from Joint
- [ ] Remove `world_pos`/`world_quat` compat properties from Body
- [ ] Remove `rotate_z` parameter from `Body.mount()`
- [ ] Remove old tests from `test_domain_model.py`
- [ ] Verify no `plint: disable` comments remain: `grep -rn 'plint: disable' botcad/`
- [ ] Run `make validate`
- [ ] Commit: `cleanup: remove dual-write bridge, old fields, lambda table, freeze Mount`

---

## Task 10: Add cross-emitter consistency tests

**Files:**
- Create: `tests/test_emitter_consistency.py`

- [ ] Write test: for each bot, generate viewer manifest and MuJoCo XML, extract all fastener positions/orientations from both, assert they match within tolerance (1e-4). Key: both emitters now use `fastener_pose()` so they must agree.
- [ ] Run the test
- [ ] Commit: `tests: add cross-emitter consistency test for fastener positions`

---

## Task 11: Final validation

- [ ] Run `make validate` — all tests pass, all lint clean
- [ ] Run `make web`, visually inspect: Pi fasteners (head up), camera fasteners (head outward), wheel fasteners (sideways into hub), bracket fasteners (unchanged)
- [ ] Compare viewer manifests before/after for all bots — positions should match except wheel fasteners (were broken, now fixed)
- [ ] Commit any fixups: `chore: final validation for unified Pose pipeline`
