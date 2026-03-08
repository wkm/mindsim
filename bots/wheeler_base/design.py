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
from botcad.skeleton import Bot


def build() -> Bot:
    """Define the wheeler_base robot."""
    bot = Bot("wheeler_base")

    # Base body — houses electronics, battery on bottom, Pi in center
    base = bot.body("base", shape="box", padding=0.008)
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
    left_rim = left_joint.body("left_rim", shape="cylinder", radius=0.045, width=0.010)
    left_rim.mount(PololuWheel90mm(), label="wheel")

    # Right wheel: servo at right side of base, axis = X (opposite direction)
    right_joint = base.joint(
        "right_wheel",
        servo=STS3215(continuous=True),
        axis="x",
        pos=(0.06, 0.0, 0.0),
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
