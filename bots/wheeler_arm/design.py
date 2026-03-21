#!/usr/bin/env python3
"""Wheeler Arm — 2-wheel differential drive + 4-DOF manipulator arm.

A simple2wheeler rebuilt with real components:
- 2x STS3215 servos for wheels (continuous rotation)
- 4x STS3215 servos for arm (shoulder yaw, shoulder pitch, elbow, wrist)
- Raspberry Pi Zero 2W for compute
- OV5647 camera on the wrist (eye-in-hand)
- LiPo 2S 1000mAh battery

Run this script to generate all outputs:
    python bots/wheeler_arm/design.py
"""

from pathlib import Path

from botcad.components import (
    OV5647,
    STS3215,
    LiPo2S,
    PololuWheel90mm,
    RaspberryPiZero2W,
)
from botcad.skeleton import BodyShape, Bot, BracketStyle


def build() -> Bot:
    """Define the wheeler_arm robot."""
    bot = Bot("wheeler_arm")
    base_mod = bot.assembly("base")
    arm_mod = bot.assembly("arm")

    # Base body — houses electronics, battery on bottom, Pi in center
    base = base_mod.body("base", shape=BodyShape.BOX, padding=0.008)
    base.mount(RaspberryPiZero2W(), position="center", label="pi")
    base.mount(LiPo2S(1000), position="bottom", label="battery")

    # --- Wheels (STS3215 in continuous rotation mode) ---

    # Left wheel: servo at left side of base, axis = X (rolls forward on Y)
    left_joint = base.joint(
        "left_wheel",
        servo=STS3215(continuous=True),
        axis="-x",
        pos=(-0.06, 0.0, 0.0),
    )
    left_rim = left_joint.body(
        "left_rim", shape=BodyShape.CYLINDER, radius=0.045, width=0.010
    )
    left_rim.mount(PololuWheel90mm(), label="wheel")

    # Right wheel: servo at right side of base, axis = X (opposite direction)
    right_joint = base.joint(
        "right_wheel",
        servo=STS3215(continuous=True),
        axis="x",
        pos=(0.06, 0.0, 0.0),
    )
    right_rim = right_joint.body(
        "right_rim", shape=BodyShape.CYLINDER, radius=0.045, width=0.010
    )
    right_rim.mount(PololuWheel90mm(), label="wheel")

    # --- 4-DOF Arm ---

    # Shoulder yaw: rotates around Z (vertical), mounted on top of base
    shoulder_yaw = base.joint(
        "shoulder_yaw",
        servo=STS3215(),
        axis="z",
        pos=(0.0, 0.0, 0.04),
        range=(-1.5, 1.5),
        bracket_style=BracketStyle.COUPLER,
    )
    turntable = shoulder_yaw.body(
        "turntable",
        shape=BodyShape.CYLINDER,
        radius=0.03,
        height=0.02,
        assembly=arm_mod,
    )

    # Shoulder pitch: tilts the arm up/down, axis = X
    shoulder_pitch = turntable.joint(
        "shoulder_pitch",
        servo=STS3215(),
        axis="x",
        pos=(0.0, 0.0, 0.02),
        range=(-1.5, 1.5),
        bracket_style=BracketStyle.COUPLER,
    )
    upper_arm = shoulder_pitch.body(
        "upper_arm", shape=BodyShape.TUBE, length=0.12, outer_r=0.018
    )

    # Elbow: bends the forearm, axis = X
    elbow = upper_arm.joint(
        "elbow",
        servo=STS3215(),
        axis="x",
        pos=(0.0, 0.0, 0.12),
        range=(-1.92, 0.0),
        bracket_style=BracketStyle.COUPLER,
    )
    forearm = elbow.body("forearm", shape=BodyShape.TUBE, length=0.10, outer_r=0.016)

    # Wrist: rotates the hand/camera, axis = Z
    wrist = forearm.joint(
        "wrist",
        servo=STS3215(),
        axis="z",
        pos=(0.0, 0.0, 0.10),
        range=(-1.5, 1.5),
        bracket_style=BracketStyle.COUPLER,
    )
    hand = wrist.body("hand", shape=BodyShape.BOX, dimensions=(0.04, 0.04, 0.03))
    hand.mount(OV5647(), position="front", label="camera")

    return bot


def main() -> None:
    bot = build()
    bot.solve()

    output_dir = Path(__file__).parent
    bot.emit(str(output_dir))

    print(f"\nGenerated wheeler_arm bot in {output_dir}")
    print("  bot.xml       — MuJoCo robot definition")
    print("  scene.xml     — Scene wrapper (bot + room)")
    print("  meshes/       — Per-body STL meshes")
    print("  bom.md        — Bill of materials")
    print("  assembly_guide.md — Assembly instructions")
    print("\nTo view in MuJoCo:")
    print("  uv run mjpython main.py view --bot wheeler_arm")


if __name__ == "__main__":
    main()
