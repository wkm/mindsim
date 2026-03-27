"""Range-of-motion validation tests for all bots.

Three distinct concerns:

1. **Discovery** — sweep each joint, find the actual collision-free ROM
   (the subset of declared range with no same-branch collisions).
2. **Judgment** — human reviews the discovered ROM and decides if it's
   acceptable. Not automatable.
3. **Regression** — compare discovered ROM against a committed baseline,
   fail if ROM got smaller. Improvements are reported but don't fail.

The baseline file `rom_baseline.json` lives in each bot's directory and is
committed alongside test renders. After reviewing the discovered ROM, run
with `--update-rom-baseline` to write/update it.

Example:
    uv run pytest tests/test_rom.py -v
    uv run pytest tests/test_rom.py -v --update-rom-baseline
    uv run pytest tests/test_rom.py -v -k wheeler_arm
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import pytest

# ── Discover bots ──

BOTS_DIR = Path(__file__).resolve().parent.parent / "bots"

SWEEP_RESOLUTION = 91  # ~2° per sample over a typical 180° range


def _discover_bots() -> list[tuple[str, Path]]:
    """Find all bots with a scene.xml."""
    return [
        (scene.parent.name, scene) for scene in sorted(BOTS_DIR.glob("*/scene.xml"))
    ]


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


def _has_root_mobility(model: mujoco.MjModel) -> bool:
    """True if the bot has a freejoint or root slide/hinge joints (not fixed-base)."""
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            return True
        if jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            return True
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
    """True if b1 and b2 are on the same kinematic branch."""
    return _is_ancestor(model, b1, b2) or _is_ancestor(model, b2, b1)


def _self_collisions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    same_branch_only: bool = False,
) -> list[dict]:
    """Return list of self-collision contacts (penetrating, between robot bodies)."""
    collisions = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        if c.dist >= 0:
            continue
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]
        b2 = model.geom_bodyid[g2]
        if b1 == 0 or b2 == 0:
            continue
        if b1 == b2:
            continue
        if model.geom_contype[g1] == 0 or model.geom_contype[g2] == 0:
            continue
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
    """Set joint angles and run forward kinematics."""
    mujoco.mj_resetData(model, data)
    for jid, angle in joint_angles.items():
        qposadr = model.jnt_qposadr[jid]
        data.qpos[qposadr] = angle
    mujoco.mj_forward(model, data)


# ── ROM discovery ──


def _sweep_joint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    jid: int,
    n_samples: int = SWEEP_RESOLUTION,
) -> tuple[list[float], list[bool]]:
    """Sweep a single joint through its declared range, all others at 0.

    Returns (angles_rad, has_collision) — parallel lists.
    """
    lo, hi = model.jnt_range[jid]
    angles = list(np.linspace(lo, hi, n_samples))
    collisions = []
    for angle in angles:
        _set_pose_and_step(model, data, {jid: angle})
        cols = _self_collisions(model, data, same_branch_only=True)
        collisions.append(len(cols) > 0)
    return angles, collisions


def _find_collision_free_rom(
    angles: list[float],
    collisions: list[bool],
    declared_lo: float,
    declared_hi: float,
) -> tuple[float, float] | None:
    """Find the largest contiguous collision-free interval containing home.

    Home is 0 if within range, otherwise the range midpoint. Returns
    (lo_rad, hi_rad) or None if home itself collides.
    """
    mid = 0.0 if declared_lo <= 0 <= declared_hi else (declared_lo + declared_hi) / 2

    # Find the sample index closest to home
    home_idx = min(range(len(angles)), key=lambda i: abs(angles[i] - mid))

    if collisions[home_idx]:
        return None  # home pose collides — fundamental problem

    # Expand outward from home to find contiguous collision-free region
    lo_idx = home_idx
    while lo_idx > 0 and not collisions[lo_idx - 1]:
        lo_idx -= 1

    hi_idx = home_idx
    while hi_idx < len(angles) - 1 and not collisions[hi_idx + 1]:
        hi_idx += 1

    return (angles[lo_idx], angles[hi_idx])


def _load_baseline(bot_dir: Path) -> dict | None:
    """Load rom_baseline.json if it exists."""
    path = bot_dir / "rom_baseline.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save_baseline(bot_dir: Path, joints: dict) -> None:
    """Write rom_baseline.json."""
    baseline = {
        "_comment": "ROM baseline — auto-generated, human-approved. Update with --update-rom-baseline.",
        "_generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_sweep_resolution": SWEEP_RESOLUTION,
        "joints": joints,
    }
    path = bot_dir / "rom_baseline.json"
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"  Wrote ROM baseline: {path}")


# ── Tests ──


class TestJointRanges:
    """Verify all joints have proper range definitions."""

    def test_all_hinge_joints_have_ranges(self, bot_model):
        """Every hinge joint must have a declared range (or be continuous)."""
        name, model, _data = bot_model
        for jid in _hinge_joints(model):
            lo, hi = model.jnt_range[jid]
            jn = _jname(model, jid)
            if lo == hi == 0:
                continue
            assert lo < hi, f"{name}/{jn}: range inverted ({lo} >= {hi})"
            assert hi - lo > 0.01, (
                f"{name}/{jn}: range too narrow ({math.degrees(hi - lo):.1f}°)"
            )
            assert hi - lo <= 2 * math.pi + 0.01, (
                f"{name}/{jn}: range > 360° ({math.degrees(hi - lo):.1f}°)"
            )


class TestROMBaseline:
    """Discover collision-free ROM per joint and compare against committed baseline.

    First run (no baseline): prints discovered ROM and skips.
    Normal run: compares against baseline, fails on regression.
    With --update-rom-baseline: writes discovered ROM as new baseline.
    """

    def test_rom_baseline(self, bot_model, request):
        name, model, data = bot_model
        bot_dir = BOTS_DIR / name
        update_mode = request.config.getoption("--update-rom-baseline")

        ranged = _ranged_joints(model)
        if not ranged:
            pytest.skip(f"{name}: no ranged joints")

        # Discover ROM for each joint
        discovered = {}
        for jid in ranged:
            jn = _jname(model, jid)
            lo, hi = model.jnt_range[jid]
            angles, collisions = _sweep_joint(model, data, jid)
            free_rom = _find_collision_free_rom(angles, collisions, lo, hi)

            n_free = sum(1 for c in collisions if not c)
            discovered[jn] = {
                "declared_range_deg": [
                    round(math.degrees(lo), 1),
                    round(math.degrees(hi), 1),
                ],
                "collision_free_range_deg": (
                    [
                        round(math.degrees(free_rom[0]), 1),
                        round(math.degrees(free_rom[1]), 1),
                    ]
                    if free_rom
                    else None
                ),
                "collision_free_fraction": round(n_free / len(collisions), 3),
            }

        # Print discovered ROM (always useful)
        print(f"\n  {name} discovered ROM:")
        for jn, info in discovered.items():
            decl = info["declared_range_deg"]
            free = info["collision_free_range_deg"]
            frac = info["collision_free_fraction"]
            if free is None:
                print(f"    {jn}: COLLIDES AT HOME (declared {decl})")
            elif free == decl:
                print(f"    {jn}: {decl} (100% free)")
            else:
                print(f"    {jn}: {free} of {decl} ({frac:.0%} free)")

        if update_mode:
            _save_baseline(bot_dir, discovered)
            return

        baseline = _load_baseline(bot_dir)
        if baseline is None:
            pytest.skip(
                f"No ROM baseline for {name} — "
                "run with --update-rom-baseline to create one"
            )

        # Compare against baseline
        regressions = []
        improvements = []
        tolerance = 1.0  # degrees — ignore differences smaller than sweep resolution

        for jn, disc in discovered.items():
            base = baseline["joints"].get(jn)
            if base is None:
                continue  # new joint, no baseline to compare

            base_rom = base["collision_free_range_deg"]
            disc_rom = disc["collision_free_range_deg"]

            if disc_rom is None and base_rom is not None:
                regressions.append(f"{jn}: was {base_rom}, now collides at home")
                continue
            if disc_rom is None:
                continue
            if base_rom is None:
                improvements.append(f"{jn}: was colliding at home, now {disc_rom}")
                continue

            # Check for regression (ROM got smaller)
            if disc_rom[0] > base_rom[0] + tolerance:
                regressions.append(f"{jn}: low end {base_rom[0]}° → {disc_rom[0]}°")
            if disc_rom[1] < base_rom[1] - tolerance:
                regressions.append(f"{jn}: high end {base_rom[1]}° → {disc_rom[1]}°")

            # Check for improvement
            if disc_rom[0] < base_rom[0] - tolerance:
                improvements.append(f"{jn}: low end {base_rom[0]}° → {disc_rom[0]}°")
            if disc_rom[1] > base_rom[1] + tolerance:
                improvements.append(f"{jn}: high end {base_rom[1]}° → {disc_rom[1]}°")

        if improvements:
            print(f"\n  {name} ROM improvements (update baseline to lock in):")
            for imp in improvements:
                print(f"    + {imp}")

        assert not regressions, (
            f"{name} ROM regressions (collision-free range got smaller):\n"
            + "\n".join(f"  - {r}" for r in regressions)
            + "\n\nIf intentional, run with --update-rom-baseline to update."
        )


class TestHomePose:
    """Verify the home (zero) pose is collision-free."""

    def test_home_pose_no_collision(self, bot_model):
        """Zero pose (all joints at 0) should have no same-branch collision."""
        name, model, data = bot_model
        ranged = _ranged_joints(model)
        _set_pose_and_step(model, data, dict.fromkeys(ranged, 0.0))
        cols = _self_collisions(model, data, same_branch_only=True)
        assert not cols, f"{name} same-branch collision at home pose: {cols}"


class TestJointLimitsEnforced:
    """Verify the sim respects declared joint limits."""

    def test_joint_stays_within_range(self, bot_model):
        """Apply max torque for 500 steps — joint should not exceed range."""
        name, model, data = bot_model
        if name == "so101_arm":
            pytest.xfail(
                "so101_arm sim goes unstable at default timestep with "
                "lightweight cradle brackets — needs timestep tuning"
            )
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

            margin = 0.05  # ~3° tolerance for damped overshoot

            # Drive to max
            mujoco.mj_resetData(model, data)
            ctrl_hi = model.actuator_ctrlrange[act_id, 1]
            for _ in range(500):
                data.ctrl[act_id] = ctrl_hi
                mujoco.mj_step(model, data)
            pos = data.qpos[qposadr]
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

        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id < 0:
            pytest.skip(f"{name} has no 'base' body")

        initial_pos = data.xpos[base_id].copy()

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

        last_jid = ranged[-1]
        tip_body_id = model.jnt_bodyid[last_jid]

        target_jid = ranged[min(1, len(ranged) - 1)]
        target_body_id = model.jnt_bodyid[target_jid]

        if not _is_ancestor(model, target_body_id, tip_body_id):
            found = False
            for jid in reversed(ranged):
                if _is_ancestor(model, target_body_id, model.jnt_bodyid[jid]):
                    tip_body_id = model.jnt_bodyid[jid]
                    found = True
                    break
            if not found:
                pytest.skip(f"{name}: no downstream tip for workspace test")

        _set_pose_and_step(model, data, dict.fromkeys(ranged, 0.0))
        home_pos = data.xpos[tip_body_id].copy()

        pose = dict.fromkeys(ranged, 0.0)
        lo, hi = model.jnt_range[target_jid]
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
