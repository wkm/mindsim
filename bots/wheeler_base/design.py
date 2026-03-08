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
from botcad.skeleton import Bot


def build() -> Bot:
    """Define the wheeler_base robot."""
    bot = Bot("wheeler_base")

    # Base body — houses electronics + camera.
    # Pi and battery rotated 90° so their long axis runs front-to-back (Y),
    # keeping X extent narrow to clear wheel servos at ±65mm.
    # Minimum height 45mm to stack battery (bottom) + camera + Pi (top)
    # without Z-axis overlap.
    base = bot.body("base", shape="box", padding=0.008)
    base.height = 0.045
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
    left_rim = left_joint.body("left_rim", shape="cylinder", radius=0.045, width=0.010)
    left_rim.mount(PololuWheel90mm(), label="wheel")

    # Right wheel: servo at right side of base, axis = +X
    right_joint = base.joint(
        "right_wheel",
        servo=STS3215(continuous=True),
        axis="x",
        pos=(0.065, 0.0, 0.0),
    )
    right_rim = right_joint.body(
        "right_rim", shape="cylinder", radius=0.045, width=0.010
    )
    right_rim.mount(PololuWheel90mm(), label="wheel")

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
