#!/usr/bin/env python3
"""Interactive balance demo for wheeler_arm.

Balance controller keeps the robot upright on its two wheels while you
control the arm joints via the MuJoCo viewer's Control panel sliders.

The controller uses three feedback loops:
  1. PID on pitch angle (inner loop — keeps the robot vertical)
  2. Feedforward from arm joint positions (proactively leans into CoM shifts)
  3. Wheel position feedback (outer loop — prevents translational drift)

Usage:
    uv run mjpython bots/wheeler_arm/demo.py

In the viewer:
    - Right panel → "Control" section has sliders for all actuators
    - Drag shoulder_yaw/pitch, elbow, wrist sliders to pose the arm
    - The balance controller compensates for the shifting center of mass
    - Ctrl+right-click on any body to apply external forces (perturbation)
    - Space to pause/unpause
"""

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

# Arm link geometry (meters) and masses (kg) for feedforward CoM estimation
_UPPER_ARM_LEN = 0.12
_FOREARM_LEN = 0.10
_UPPER_ARM_MASS = 0.069
_FOREARM_MASS = 0.067
_HAND_MASS = 0.013
_ARM_TOTAL_MASS = _UPPER_ARM_MASS + _FOREARM_MASS + _HAND_MASS


def quat_to_pitch_around_x(quat: np.ndarray) -> float:
    """Extract pitch angle (rotation around X axis) from a quaternion."""
    w, x, y, z = quat
    sin_r = 2.0 * (w * x + y * z)
    cos_r = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(sin_r, cos_r)


def estimate_arm_com_y(shoulder_pitch: float, elbow: float) -> float:
    """Estimate Y-offset of arm CoM from shoulder_pitch and elbow angles.

    Uses simplified 2-link forward kinematics in the pitch plane.
    Shoulder_pitch rotates the upper arm from vertical toward +Y.
    Elbow is relative to the upper arm.
    """
    ua_com_y = (_UPPER_ARM_LEN / 2) * math.sin(shoulder_pitch)
    fa_origin_y = _UPPER_ARM_LEN * math.sin(shoulder_pitch)
    fa_angle = shoulder_pitch + elbow
    fa_com_y = fa_origin_y + (_FOREARM_LEN / 2) * math.sin(fa_angle)
    hand_y = fa_origin_y + _FOREARM_LEN * math.sin(fa_angle)
    return (
        _UPPER_ARM_MASS * ua_com_y + _FOREARM_MASS * fa_com_y + _HAND_MASS * hand_y
    ) / _ARM_TOTAL_MASS


def main() -> None:
    model = mujoco.MjModel.from_xml_path("bots/wheeler_arm/scene.xml")
    data = mujoco.MjData(model)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    # Sensor addresses
    gyro_adr = model.sensor_adr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    ]
    sp_adr = model.sensor_adr[
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "shoulder_pitch_pos_sensor"
        )
    ]
    el_adr = model.sensor_adr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "elbow_pos_sensor")
    ]
    lw_pos_adr = model.sensor_adr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wheel_pos_sensor")
    ]
    rw_pos_adr = model.sensor_adr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wheel_pos_sensor")
    ]

    # Actuator indices
    act_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    left_wheel_idx = act_names.index("left_wheel_motor")
    right_wheel_idx = act_names.index("right_wheel_motor")

    # Balance controller gains — tuned for wheeler_arm with:
    #   wheel gear = 2.94 N-m (STS3215 stall torque)
    #   wheel damping = 0.01 (bearing friction only)
    #   arm forcerange = [-2.94, 2.94] (realistic servo torque limits)
    #   timestep = 0.002 (500 Hz control)
    #
    # Inner loop (pitch PID):
    kp = 10.0  # proportional: reacts to pitch error
    kd = 0.1  # derivative: damps oscillation via gyro rate
    ki = 2.0  # integral: corrects steady-state pitch offset
    # Feedforward (arm CoM compensation):
    k_ff = 10.0  # lean angle per meter of arm CoM offset
    # Outer loop (drift prevention):
    k_pos = 0.05  # lean per radian of average wheel rotation

    integral = 0.0
    integral_max = 1.0

    dt = model.opt.timestep

    print("Wheeler Arm — Interactive Balance Demo")
    print("=" * 42)
    print()
    print("The robot balances on two wheels while you control the arm.")
    print("Feedforward compensation estimates arm CoM shifts.")
    print()
    print("Controls (in MuJoCo viewer):")
    print("  Right panel → 'Control' → drag arm joint sliders")
    print("  Ctrl+Right-click a body to push it (perturbation)")
    print("  Space to pause/unpause")
    print("  Double-click a body to track it with the camera")
    print()
    print("Tip: move arm sliders gradually for best stability.")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        while viewer.is_running():
            # --- Read state ---
            pitch = quat_to_pitch_around_x(data.xquat[base_id])
            gyro_x = data.sensordata[gyro_adr]

            # Arm joint positions for feedforward
            sp_angle = data.sensordata[sp_adr]
            el_angle = data.sensordata[el_adr]

            # Average wheel position for drift compensation
            # Left wheel axis is -X, right is +X → negate left for consistent sign
            w_pos = 0.5 * (-data.sensordata[lw_pos_adr] + data.sensordata[rw_pos_adr])

            # --- Feedforward: estimate arm CoM shift → pitch setpoint ---
            com_y = estimate_arm_com_y(sp_angle, el_angle)
            pitch_target = -k_ff * com_y - k_pos * w_pos
            pitch_target = max(-0.3, min(0.3, pitch_target))

            # --- PID on pitch error ---
            pitch_error = pitch - pitch_target
            integral += pitch_error * dt
            integral = max(-integral_max, min(integral_max, integral))

            wheel_cmd = kp * pitch_error + kd * gyro_x + ki * integral
            wheel_cmd = max(-1.0, min(1.0, wheel_cmd))

            # Both wheels get same command for balance (not turning)
            data.ctrl[left_wheel_idx] = -wheel_cmd
            data.ctrl[right_wheel_idx] = wheel_cmd

            # Arm actuators are left to the viewer's Control panel.
            # Position actuators hold whatever angle the user sets.

            # --- Step simulation ---
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - start
            sim_time = data.time
            if sim_time > elapsed:
                time.sleep(sim_time - elapsed)


if __name__ == "__main__":
    main()
