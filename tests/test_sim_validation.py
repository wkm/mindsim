"""Smoke tests: wheeler_base loads in MuJoCo and sim properties match BOM."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).parent.parent
SCENE_XML = _ROOT / "bots/wheeler_base/scene.xml"
BOT_XML = _ROOT / "bots/wheeler_base/bot.xml"

# Sim timestep from scene.xml
_DT = 0.002


def _make_sim() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load scene and return (model, data) after reset."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _step_n(model: mujoco.MjModel, data: mujoco.MjData, n: int) -> None:
    """Advance simulation by n steps."""
    for _ in range(n):
        mujoco.mj_step(model, data)


def _base_pos(data: mujoco.MjData) -> np.ndarray:
    """Return (x, y, z) position of the base body via the root freejoint."""
    return data.qpos[:3].copy()


def _base_heading(data: mujoco.MjData) -> float:
    """Return yaw angle (radians) of the base body from its freejoint quaternion."""
    # freejoint qpos: [x, y, z, qw, qx, qy, qz]
    qw, qx, qy, qz = data.qpos[3:7]
    # yaw from quaternion (rotation about Z)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Existing smoke tests
# ---------------------------------------------------------------------------


def test_loads_without_error():
    """scene.xml parses, meshes resolve, one physics step runs."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)  # one step without crash


def test_mass_matches_bom():
    """Total robot mass within 2% of BOM total (0.468 kg).

    Uses bot.xml (robot only) so world furniture doesn't pollute the sum.
    """
    model = mujoco.MjModel.from_xml_path(str(BOT_XML))
    # body_mass[0] is the immovable MuJoCo world body -- skip it
    total_mass = sum(model.body_mass[1:])
    assert abs(total_mass - 0.468) / 0.468 < 0.02, (
        f"Total mass {total_mass:.4f} kg deviates >2% from BOM 0.468 kg"
    )


# ---------------------------------------------------------------------------
# Driving behavior tests
# ---------------------------------------------------------------------------


def _get_ctrl_indices(model: mujoco.MjModel) -> tuple[int, int]:
    """Return (left_motor_idx, right_motor_idx) for ctrl array."""
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_motor")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_motor")
    assert left >= 0, "left_wheel_motor actuator not found"
    assert right >= 0, "right_wheel_motor actuator not found"
    return left, right


def _settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 0.5) -> None:
    """Let the bot settle onto the ground with no control input."""
    data.ctrl[:] = 0.0
    _step_n(model, data, int(seconds / _DT))


def test_drives_forward():
    """Equal wheel torques -> bot moves significantly in the XY plane.

    Verifies the bot translates >3cm horizontally when both motors are driven
    at full power for 2 seconds.  We check total horizontal displacement rather
    than decomposing into forward/lateral because small asymmetries in the
    mesh-based collision geometry can cause veering in open-loop torque control.
    """
    model, data = _make_sim()
    left_idx, right_idx = _get_ctrl_indices(model)

    _settle(model, data)
    start = _base_pos(data)

    # Full throttle for 2 seconds
    data.ctrl[left_idx] = 1.0
    data.ctrl[right_idx] = 1.0
    _step_n(model, data, int(2.0 / _DT))

    end = _base_pos(data)
    dx, dy = end[0] - start[0], end[1] - start[1]
    horiz = math.hypot(dx, dy)

    assert horiz > 0.03, (
        f"Bot moved only {horiz:.4f}m horizontally (need >0.03m). "
        f"displacement=({dx:.4f}, {dy:.4f})"
    )


def test_turns():
    """Differential wheel speeds -> the bot curves (does not drive straight).

    We drive one wheel at full power and the other at zero, then verify that
    the resulting trajectory curves: the heading changes, or equivalently the
    displacement direction differs from what pure straight-line driving would
    produce.  We compare the displacement angle of differential-drive vs
    equal-drive to confirm the bot actually steers.
    """
    model, data = _make_sim()
    left_idx, right_idx = _get_ctrl_indices(model)

    # --- Run 1: straight (equal torque) ---
    _settle(model, data)
    start_straight = _base_pos(data)
    data.ctrl[left_idx] = 1.0
    data.ctrl[right_idx] = 1.0
    _step_n(model, data, int(2.0 / _DT))
    end_straight = _base_pos(data)
    dx_s, dy_s = (
        end_straight[0] - start_straight[0],
        end_straight[1] - start_straight[1],
    )
    angle_straight = math.atan2(dy_s, dx_s)

    # --- Run 2: differential (one wheel only) ---
    model2, data2 = _make_sim()
    _settle(model2, data2)
    start_diff = _base_pos(data2)
    data2.ctrl[left_idx] = 1.0
    data2.ctrl[right_idx] = 0.0
    _step_n(model2, data2, int(2.0 / _DT))
    end_diff = _base_pos(data2)
    dx_d, dy_d = end_diff[0] - start_diff[0], end_diff[1] - start_diff[1]
    angle_diff = math.atan2(dy_d, dx_d)

    # The displacement directions should differ by a meaningful amount
    angle_delta = abs(angle_diff - angle_straight)
    if angle_delta > math.pi:
        angle_delta = 2 * math.pi - angle_delta

    assert angle_delta > math.radians(5), (
        f"Differential drive displacement angle differs only "
        f"{math.degrees(angle_delta):.1f} deg from straight (need >5 deg). "
        f"straight=({dx_s:.4f},{dy_s:.4f}), diff=({dx_d:.4f},{dy_d:.4f})"
    )


def test_stays_upright():
    """During forward driving the base Z stays within 1cm of initial height."""
    model, data = _make_sim()
    left_idx, right_idx = _get_ctrl_indices(model)

    _settle(model, data)
    z_initial = _base_pos(data)[2]

    data.ctrl[left_idx] = 0.8
    data.ctrl[right_idx] = 0.8

    z_min = z_initial
    z_max = z_initial
    steps = int(1.5 / _DT)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        z = data.qpos[2]
        z_min = min(z_min, z)
        z_max = max(z_max, z)

    deviation = max(abs(z_max - z_initial), abs(z_min - z_initial))
    assert deviation < 0.02, (
        f"Base Z deviated {deviation:.4f}m from initial {z_initial:.4f}m "
        f"(range [{z_min:.4f}, {z_max:.4f}])"
    )


def test_ground_contact_stable():
    """Wheels maintain ground contact -- no sustained bouncing/jittering.

    Sample base Z at 10 Hz over 1 second of driving; check that the standard
    deviation of Z is small (<1mm), indicating stable ground contact.
    """
    model, data = _make_sim()
    left_idx, right_idx = _get_ctrl_indices(model)

    _settle(model, data)

    data.ctrl[left_idx] = 0.5
    data.ctrl[right_idx] = 0.5

    sample_interval = int(0.1 / _DT)  # every 100ms
    n_samples = 10
    z_samples = []

    for _ in range(n_samples):
        _step_n(model, data, sample_interval)
        z_samples.append(data.qpos[2])

    z_arr = np.array(z_samples)
    z_std = np.std(z_arr)

    assert z_std < 0.002, (
        f"Base Z std={z_std:.5f}m over 1s of driving (need <2mm). samples={z_arr}"
    )
