#!/usr/bin/env python3
"""SO-101 Arm — 6-DOF manipulator arm with gripper and eye-in-hand camera.

Inspired by the SO-101 (SO-ARM100) open-source robot arm:
- 5 revolute joints + 1 gripper, all STS3215 servos
- Shoulder pan, shoulder lift, elbow, wrist flex, wrist roll, gripper
- PiCamera2 on wrist for eye-in-hand vision
- Raspberry Pi Zero 2W + LiPo battery in the base

Run this script to generate all outputs:
    python bots/so101_arm/design.py
"""

from pathlib import Path

from botcad.components import (
    STS3215,
    LiPo2S,
    PiCamera2,
    RaspberryPiZero2W,
)
from botcad.skeleton import Bot


def build() -> Bot:
    """Define the SO-101 arm robot."""
    bot = Bot("so101_arm")
    base_mod = bot.module("base")
    arm_mod = bot.module("arm")

    # Base body — houses Pi and battery
    base = base_mod.body("base", shape="box", padding=0.008)
    base.mount(RaspberryPiZero2W(), position="center", label="pi")
    base.mount(LiPo2S(1000), position="bottom", label="battery")

    # --- 6-DOF Arm ---

    # Shoulder pan: rotates the whole arm around Z (vertical)
    shoulder_pan = base.joint(
        "shoulder_pan",
        servo=STS3215(),
        axis="z",
        pos=(0.0, 0.0, 0.031),
        range=(-1.92, 1.92),
    )
    turntable = shoulder_pan.body(
        "turntable",
        shape="box",
        dimensions=(0.06, 0.04, 0.04),
        module=arm_mod,
    )

    # Shoulder lift: tilts the arm up/down, axis = X
    shoulder_lift = turntable.joint(
        "shoulder_lift",
        servo=STS3215(),
        axis="x",
        pos=(0.0, 0.0, 0.02),
        range=(-1.745, 1.745),
    )
    upper_arm = shoulder_lift.body(
        "upper_arm", shape="tube", length=0.116, outer_r=0.018
    )

    # Elbow: bends the forearm, axis = X
    elbow_flex = upper_arm.joint(
        "elbow_flex",
        servo=STS3215(),
        axis="x",
        pos=(0.0, 0.0, 0.116),
        range=(-1.69, 1.69),
    )
    forearm = elbow_flex.body("forearm", shape="tube", length=0.135, outer_r=0.016)

    # Wrist flex: bends the wrist, axis = X
    wrist_flex = forearm.joint(
        "wrist_flex",
        servo=STS3215(),
        axis="x",
        pos=(0.0, 0.0, 0.135),
        range=(-1.658, 1.658),
    )
    wrist = wrist_flex.body("wrist", shape="box", dimensions=(0.04, 0.035, 0.064))

    # Wrist roll: rotates the end effector, axis = Z
    wrist_roll = wrist.joint(
        "wrist_roll",
        servo=STS3215(),
        axis="z",
        pos=(0.0, 0.0, 0.032),
        range=(-2.744, 2.841),
    )
    wrist_roll_body = wrist_roll.body(
        "wrist_roll", shape="box", dimensions=(0.035, 0.035, 0.037)
    )

    # Eye-in-hand camera
    wrist_roll_body.mount(PiCamera2(), position="front", label="camera")

    # Gripper: force-limited hinge, axis = X
    gripper = wrist_roll_body.joint(
        "gripper",
        servo=STS3215(),
        axis="x",
        pos=(0.0, 0.0, 0.018),
        range=(-0.175, 1.745),
        grip=True,
    )
    gripper.body(
        "jaw",
        shape="jaw",
        jaw_length=0.04,
        jaw_width=0.03,
        jaw_thickness=0.005,
    )

    return bot


def main() -> None:
    bot = build()
    bot.solve()

    output_dir = Path(__file__).parent
    bot.emit(str(output_dir))

    print(f"\nGenerated so101_arm bot in {output_dir}")
    print("  bot.xml       — MuJoCo robot definition")
    print("  scene.xml     — Scene wrapper (bot + room)")
    print("  meshes/       — Per-body STL meshes")
    print("  bom.md        — Bill of materials")
    print("  assembly_guide.md — Assembly instructions")
    print("\nTo view in MuJoCo:")
    print("  uv run mjpython main.py view --bot so101_arm")


if __name__ == "__main__":
    main()
