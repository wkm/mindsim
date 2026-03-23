#!/usr/bin/env python3
"""Wheeler Base — 2-wheel differential drive platform.

Drive-only variant of wheeler_arm for independent simulation and training:
- 2x STS3215 servos for wheels (continuous rotation)
- Raspberry Pi Zero 2W for compute
- LiPo 2S 1000mAh battery

Run this script to generate all outputs:
    python bots/wheeler_base/design.py
"""

from pathlib import Path

from botcad.components import (
    STS3215,
    LiPo2S,
    PololuWheel90mm,
    RaspberryPiZero2W,
)
from botcad.components.camera import OV5647
from botcad.skeleton import BodyShape, Bot


def build() -> Bot:
    """Define the wheeler_base robot."""
    bot = Bot("wheeler_base")

    # Base body — houses electronics + camera.
    # Pi and battery rotated 90° so their long axis runs front-to-back (Y),
    # keeping X extent narrow to clear wheel servos at ±65mm.
    # Camera faces forward (lens +Z rotated to +Y), making it 24mm tall;
    # extra padding prevents Z-overlap with the bottom battery.
    # No custom_solid — standard pipeline generates the body shell with
    # bracket pockets, component pockets, and wire channels.
    # Let the packing solver determine dimensions from actual component
    # geometry (derived from ShapeScript bounding boxes during solve()).
    base = bot.body(
        "base",
        shape=BodyShape.BOX,
        padding=0.014,
    )
    base.mount(LiPo2S(1000), position="bottom", label="battery", rotate_z=True)
    base.mount(OV5647(), position="front", label="camera")
    base.mount(RaspberryPiZero2W(), position="top", label="pi", rotate_z=True)

    # --- Wheels (STS3215 in continuous rotation mode) ---
    # Joints at ±65mm to clear rotated electronics with margin.

    # Left wheel: servo at left side of base, axis = -X (rolls forward on Y)
    left_joint = base.joint(
        "left_wheel",
        servo=STS3215(continuous=True),
        axis="-x",
        pos=(-0.065, 0.0, 0.0),
    )
    left_rim = left_joint.body(
        "left_rim", shape=BodyShape.CYLINDER, radius=0.045, width=0.010
    )
    left_rim.mount(PololuWheel90mm(), label="wheel")

    # Right wheel: servo at right side of base, axis = +X
    right_joint = base.joint(
        "right_wheel",
        servo=STS3215(continuous=True),
        axis="x",
        pos=(0.065, 0.0, 0.0),
    )
    right_rim = right_joint.body(
        "right_rim", shape=BodyShape.CYLINDER, radius=0.045, width=0.010
    )
    right_rim.mount(PololuWheel90mm(), label="wheel")

    # Clearance constraints — checked during build_cad()
    bot.clearance(
        "left_rim",
        "servo_left_wheel",
        min_distance=0.0005,
        label="left wheel-servo gap",
    )
    bot.clearance(
        "right_rim",
        "servo_right_wheel",
        min_distance=0.0005,
        label="right wheel-servo gap",
    )

    return bot


def main() -> None:
    bot = build()
    bot.solve()

    output_dir = Path(__file__).parent
    bot.emit(str(output_dir))

    print(f"\nGenerated wheeler_base bot in {output_dir}")
    print("  bot.xml       — MuJoCo robot definition")
    print("  scene.xml     — Scene wrapper (bot + room)")
    print("  meshes/       — Per-body STL meshes")
    print("  bom.md        — Bill of materials")
    print("  assembly_guide.md — Assembly instructions")
    print("\nTo view in MuJoCo:")
    print("  uv run mjpython main.py view --bot wheeler_base")


if __name__ == "__main__":
    main()
