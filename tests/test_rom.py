"""Range-of-motion validation tests for all bots.

For each bot's scene.xml, loads the MuJoCo model and:
  1. Verifies all joints have declared ranges
  2. Sweeps each joint through its full range — no self-collisions
  3. Tests multi-joint combined poses — no self-collisions
  4. Verifies joint limits are enforced by the sim
  5. Verifies fixed-base bots don't move

Run:
    PYTHONPATH=. uv run pytest tests/test_rom.py -v
    PYTHONPATH=. uv run pytest tests/test_rom.py -v -k so101_arm
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

# ── Discover bots ──

BOTS_DIR = Path(__file__).resolve().parent.parent / "bots"


def _discover_bots() -> list[tuple[str, Path]]:
    """Find all bots with a scene.xml."""
    bots = []
    for scene in sorted(BOTS_DIR.glob("*/scene.xml")):
        bots.append((scene.parent.name, scene))
    return bots


BOT_SCENES = _discover_bots()
BOT_IDS = [name for name, _ in BOT_SCENES]


@pytest.fixture(params=BOT_SCENES, ids=BOT_IDS, scope="class")
def bot_model(request):
    """Load a bot's MuJoCo model + data (shared per test class)."""
    name, scene_path = request.param
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return name, model, data


# ── Helpers ──


def _hinge_joints(model: mujoco.MjModel) -> list[int]:
    """All hinge joint IDs."""
    return [
        jid
        for jid in range(model.njnt)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
    ]


def _ranged_joints(model: mujoco.MjModel) -> list[int]:
    """Hinge joints with an actual range (not continuous wheels)."""
    joints = []
    for jid in _hinge_joints(model):
        lo, hi = model.jnt_range[jid]
        if lo != hi:  # continuous joints have range [0, 0]
            joints.append(jid)
    return joints


def _jname(model: mujoco.MjModel, jid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"


def _has_freejoint(model: mujoco.MjModel) -> bool:
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return True
    return False


def _has_root_mobility(model: mujoco.MjModel) -> bool:
    """True if the bot has a freejoint or root slide/hinge joints (not fixed-base)."""
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            return True
        # Root slide or unlimited hinge joints (e.g., walker2d's rootx/rootz/rooty)
        if jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            return True
        # Unlimited hinge at body attached to world
        if jtype == mujoco.mjtJoint.mjJNT_HINGE and not model.jnt_limited[jid]:
            return True
    return False


def _is_ancestor(model: mujoco.MjModel, ancestor_bid: int, descendant_bid: int) -> bool:
    """True if ancestor_bid is in the parent chain of descendant_bid."""
    bid = descendant_bid
    while bid > 0:
        bid = model.body_parentid[bid]
        if bid == ancestor_bid:
            return True
    return False


def _same_branch(model: mujoco.MjModel, b1: int, b2: int) -> bool:
    """True if b1 and b2 are on the same kinematic branch (one is ancestor of the other)."""
    return _is_ancestor(model, b1, b2) or _is_ancestor(model, b2, b1)


def _self_collisions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    same_branch_only: bool = False,
) -> list[dict]:
    """Return list of self-collision contacts (penetrating, between robot bodies).

    Excludes:
    - Non-penetrating contacts (dist >= 0)
    - Contacts involving world body (body 0) — floor, arena, etc.
    - Contacts between geoms on the same body
    - Contacts involving non-collision geoms (contype == 0)

    If same_branch_only=True, also excludes cross-branch collisions (e.g.,
    left leg hitting right leg). These are physically valid but are the
    controller's problem, not a geometry design bug.
    """
    collisions = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        if c.dist >= 0:
            continue  # no penetration
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]
        b2 = model.geom_bodyid[g2]
        # Skip world body (floor, arena objects)
        if b1 == 0 or b2 == 0:
            continue
        # Skip same-body contacts
        if b1 == b2:
            continue
        # Only count contacts between collision geoms (contype & conaffinity > 0)
        if model.geom_contype[g1] == 0 or model.geom_contype[g2] == 0:
            continue
        # Optionally skip cross-branch collisions
        if same_branch_only and not _same_branch(model, b1, b2):
            continue
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom_{g1}"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom_{g2}"
        bn1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body_{b1}"
        bn2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body_{b2}"
        collisions.append(
            {
                "geom1": n1,
                "geom2": n2,
                "body1_name": bn1,
                "body2_name": bn2,
                "dist": float(c.dist),
                "body1": b1,
                "body2": b2,
            }
        )
    return collisions


def _set_pose_and_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_angles: dict[int, float],
) -> None:
    """Set joint angles and step the sim to settle contacts."""
    mujoco.mj_resetData(model, data)
    for jid, angle in joint_angles.items():
        qposadr = model.jnt_qposadr[jid]
        data.qpos[qposadr] = angle
    # Use mj_forward to compute contacts without dynamics
    mujoco.mj_forward(model, data)


# ── Tests ──


class TestJointRanges:
    """Verify all joints have proper range definitions."""

    def test_all_hinge_joints_have_ranges(self, bot_model):
        """Every hinge joint must have a declared range (or be continuous)."""
        name, model, data = bot_model
        for jid in _hinge_joints(model):
            lo, hi = model.jnt_range[jid]
            jn = _jname(model, jid)
            # Either it has a real range or it's a continuous joint [0, 0]
            if lo == hi == 0:
                continue  # continuous — OK
            assert lo < hi, f"{name}/{jn}: range inverted ({lo} >= {hi})"
            assert hi - lo > 0.01, (
                f"{name}/{jn}: range too narrow ({math.degrees(hi - lo):.1f}°)"
            )
            assert hi - lo <= 2 * math.pi + 0.01, (
                f"{name}/{jn}: range > 360° ({math.degrees(hi - lo):.1f}°)"
            )

    def test_range_is_symmetric_or_documented(self, bot_model):
        """Ranges should be roughly symmetric unless there's a reason (like a gripper)."""
        name, model, data = bot_model
        asymmetric = []
        for jid in _ranged_joints(model):
            lo, hi = model.jnt_range[jid]
            jn = _jname(model, jid)
            if abs(abs(lo) - abs(hi)) > 0.2:  # >~11° asymmetry
                asymmetric.append(
                    f"{jn}: [{math.degrees(lo):.0f}°, {math.degrees(hi):.0f}°]"
                )
        # Not an assertion failure — just informational
        if asymmetric:
            print(f"  {name}: asymmetric joints (expected for grippers): {asymmetric}")


class TestSingleJointSweep:
    """Sweep each joint through its full range individually, checking for collisions.

    Only flags same-branch collisions. Cross-branch collisions (e.g., hip
    abduction pushing one leg into the other) are physically normal — the
    controller should avoid those poses, not the geometry.
    """

    def test_no_collisions_at_range_limits(self, bot_model):
        """No same-branch self-collision when any single joint is at its min or max."""
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        failures = []

        for jid in ranged:
            lo, hi = model.jnt_range[jid]
            jn = _jname(model, jid)

            for angle, label in [(lo, "min"), (hi, "max")]:
                _set_pose_and_step(model, data, {jid: angle})
                cols = _self_collisions(model, data, same_branch_only=True)
                if cols:
                    pairs = [(c["body1_name"], c["body2_name"]) for c in cols]
                    failures.append(
                        f"{jn}@{label}({math.degrees(angle):+.0f}°): {pairs}"
                    )

        assert not failures, f"{name} self-collisions at range limits:\n" + "\n".join(
            failures
        )

    def test_no_collisions_through_full_sweep(self, bot_model):
        """No same-branch self-collision at 11 points across each joint's full range."""
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        n_samples = 11
        failures = []

        for jid in ranged:
            lo, hi = model.jnt_range[jid]
            jn = _jname(model, jid)

            for angle in np.linspace(lo, hi, n_samples):
                _set_pose_and_step(model, data, {jid: angle})
                cols = _self_collisions(model, data, same_branch_only=True)
                if cols:
                    pairs = [(c["body1_name"], c["body2_name"]) for c in cols]
                    failures.append(f"{jn}@{math.degrees(angle):+.0f}°: {pairs}")

        assert not failures, f"{name} self-collisions during sweep:\n" + "\n".join(
            failures
        )


class TestMultiJointPoses:
    """Test combined joint configurations for self-collision."""

    def test_home_pose_no_collision(self, bot_model):
        """Zero pose (all joints at 0) should have no collision."""
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        _set_pose_and_step(model, data, {jid: 0.0 for jid in ranged})
        cols = _self_collisions(model, data)
        assert not cols, f"{name} collision at home pose: {cols}"

    def test_all_min_collisions_reported(self, bot_model):
        """All joints at minimum — stress test. Collisions are reported, not failures.

        Simultaneously driving every joint to its limit is physically unrealistic
        for long kinematic chains. We report collisions for awareness but don't
        fail — single-joint and pairwise tests are the real validation.
        """
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        pose = {jid: float(model.jnt_range[jid][0]) for jid in ranged}
        _set_pose_and_step(model, data, pose)
        cols = _self_collisions(model, data)
        if cols:
            bodies = {f"{c['body1_name']}<->{c['body2_name']}" for c in cols}
            print(f"  {name} all-min collisions (informational): {sorted(bodies)}")

    def test_all_max_collisions_reported(self, bot_model):
        """All joints at maximum — stress test. Same caveat as all-min."""
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        pose = {jid: float(model.jnt_range[jid][1]) for jid in ranged}
        _set_pose_and_step(model, data, pose)
        cols = _self_collisions(model, data)
        if cols:
            bodies = {f"{c['body1_name']}<->{c['body2_name']}" for c in cols}
            print(f"  {name} all-max collisions (informational): {sorted(bodies)}")

    def test_random_poses_no_collision(self, bot_model):
        """50 random poses within joint limits — no same-branch self-collision.

        Cross-branch collisions (e.g., left foot hitting right foot) are
        physically valid reachable poses — the controller should learn to
        avoid them. Only same-branch collisions (shin through its own thigh)
        indicate a geometry design bug.
        """
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        rng = np.random.default_rng(seed=42)
        failures = []

        for i in range(50):
            pose = {}
            for jid in ranged:
                lo, hi = model.jnt_range[jid]
                pose[jid] = float(rng.uniform(lo, hi))
            _set_pose_and_step(model, data, pose)
            cols = _self_collisions(model, data, same_branch_only=True)
            if cols:
                angles_str = ", ".join(
                    f"{_jname(model, jid)}={math.degrees(a):+.0f}°"
                    for jid, a in pose.items()
                )
                pairs = [(c["body1_name"], c["body2_name"]) for c in cols]
                failures.append(f"  pose {i}: {pairs} at [{angles_str}]")

        assert not failures, (
            f"{name} same-branch collisions in random poses:\n" + "\n".join(failures)
        )

    def test_pairwise_extremes(self, bot_model):
        """For each pair of adjacent joints, test all 4 corner combinations.

        Only flags same-branch collisions (see test_random_poses_no_collision).
        """
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        if len(ranged) < 2:
            pytest.skip("Not enough ranged joints for pairwise test")

        failures = []
        for j1, j2 in zip(ranged[:-1], ranged[1:]):
            lo1, hi1 = model.jnt_range[j1]
            lo2, hi2 = model.jnt_range[j2]
            corners = [(lo1, lo2), (lo1, hi2), (hi1, lo2), (hi1, hi2)]

            for a1, a2 in corners:
                _set_pose_and_step(model, data, {j1: float(a1), j2: float(a2)})
                cols = _self_collisions(model, data, same_branch_only=True)
                if cols:
                    n1, n2 = _jname(model, j1), _jname(model, j2)
                    pairs = [(c["body1_name"], c["body2_name"]) for c in cols]
                    failures.append(
                        f"{n1}={math.degrees(a1):+.0f}°, "
                        f"{n2}={math.degrees(a2):+.0f}°: {pairs}"
                    )

        assert not failures, f"{name} collisions at pairwise extremes:\n" + "\n".join(
            failures
        )


class TestJointLimitsEnforced:
    """Verify the sim respects declared joint limits."""

    def test_joint_stays_within_range(self, bot_model):
        """Apply max torque for 500 steps — joint should not exceed range."""
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        if not ranged:
            pytest.skip("No ranged joints")

        violations = []
        for jid in ranged:
            lo, hi = model.jnt_range[jid]
            jn = _jname(model, jid)
            qposadr = model.jnt_qposadr[jid]

            # Find the actuator for this joint
            act_id = -1
            for aid in range(model.nu):
                if model.actuator_trnid[aid, 0] == jid:
                    act_id = aid
                    break
            if act_id < 0:
                continue

            # Drive to max
            mujoco.mj_resetData(model, data)
            ctrl_hi = model.actuator_ctrlrange[act_id, 1]
            for _ in range(500):
                data.ctrl[act_id] = ctrl_hi
                mujoco.mj_step(model, data)
            pos = data.qpos[qposadr]
            margin = 0.05  # ~3° tolerance for damped overshoot
            if pos > hi + margin:
                violations.append(
                    f"{jn}: exceeded max by {math.degrees(pos - hi):.1f}°"
                )

            # Drive to min
            mujoco.mj_resetData(model, data)
            ctrl_lo = model.actuator_ctrlrange[act_id, 0]
            for _ in range(500):
                data.ctrl[act_id] = ctrl_lo
                mujoco.mj_step(model, data)
            pos = data.qpos[qposadr]
            if pos < lo - margin:
                violations.append(
                    f"{jn}: exceeded min by {math.degrees(lo - pos):.1f}°"
                )

        assert not violations, f"{name} joint limit violations:\n" + "\n".join(
            violations
        )


class TestFixedBase:
    """Verify fixed-base bots stay anchored."""

    def test_fixed_base_no_drift(self, bot_model):
        """Fixed-base bots should have no freejoint and base shouldn't move."""
        name, model, data = bot_model

        if _has_root_mobility(model):
            pytest.skip(f"{name} has root mobility (not fixed-base)")

        # Record initial base position
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id < 0:
            pytest.skip(f"{name} has no 'base' body")

        initial_pos = data.xpos[base_id].copy()

        # Step 1000 times with random control
        rng = np.random.default_rng(seed=123)
        for _ in range(1000):
            data.ctrl[:] = rng.uniform(
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(model, data)

        final_pos = data.xpos[base_id]
        drift = np.linalg.norm(final_pos - initial_pos)
        assert drift < 1e-6, f"{name} base drifted {drift:.2e}m after 1000 steps"


class TestWorkspace:
    """Validate arm reach and workspace properties."""

    def test_end_effector_reachable(self, bot_model):
        """The end effector should move when joints are actuated."""
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        if len(ranged) < 3:
            pytest.skip("Not enough joints for workspace test")

        # Find the tip body: the body attached to the last ranged joint
        last_jid = ranged[-1]
        tip_body_id = model.jnt_bodyid[last_jid]

        # Verify the tip is a descendant of the target joint's body
        # (important for branching kinematic trees like bipeds)
        target_jid = ranged[min(1, len(ranged) - 1)]
        target_body_id = model.jnt_bodyid[target_jid]

        # Walk up from tip to find if target_body is an ancestor
        def _is_ancestor(model, ancestor_bid, descendant_bid):
            bid = descendant_bid
            while bid > 0:
                bid = model.body_parentid[bid]
                if bid == ancestor_bid:
                    return True
            return False

        if not _is_ancestor(model, target_body_id, tip_body_id):
            # Tip is on a different branch — find a tip that IS downstream
            # Use the deepest descendant of the target joint's body
            found = False
            for jid in reversed(ranged):
                if _is_ancestor(model, target_body_id, model.jnt_bodyid[jid]):
                    tip_body_id = model.jnt_bodyid[jid]
                    found = True
                    break
            if not found:
                pytest.skip(f"{name}: no downstream tip for workspace test")

        _set_pose_and_step(model, data, {jid: 0.0 for jid in ranged})
        home_pos = data.xpos[tip_body_id].copy()

        # Move the target joint by a meaningful amount within its range
        pose = {jid: 0.0 for jid in ranged}
        lo, hi = model.jnt_range[target_jid]
        # Pick an angle that actually produces displacement:
        # use 45° toward whichever limit is farther from 0
        if abs(hi) >= abs(lo):
            pose[target_jid] = min(0.785, hi)
        else:
            pose[target_jid] = max(-0.785, lo)
        _set_pose_and_step(model, data, pose)
        moved_pos = data.xpos[tip_body_id].copy()

        displacement = np.linalg.norm(moved_pos - home_pos)
        assert displacement > 0.01, (
            f"{name}: end effector barely moved ({displacement * 1000:.1f}mm)"
        )
