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


def wheeler_base_solid():
    from build123d import Align, Box, Cylinder, Location

    # Dimensional constants
    width = 0.130  # X: exactly spans to the servos at +/- 65mm
    length = 0.080  # Y: long enough for battery and Pi Zero (rotated to run along Y)
    height = 0.045  # Z

    C = (Align.CENTER, Align.CENTER, Align.CENTER)

    # 1. Base block with filleted corners in Z
    base = Box(width, length, height, align=C)

    # Fillet vertical edges
    vertical_edges = [
        e for e in base.edges() if e.geom_type == "LINE" and abs(e.direction.Z) > 0.99
    ]
    if vertical_edges:
        base = base.fillet(0.008, vertical_edges)

    # 2. Main cavity for battery + electronics + camera
    # We want approx 5mm thick walls, maybe thinner
    wall = 0.005
    cavity = Box(
        width - wall * 2,
        length - wall,
        height,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    # cut from top, leaving a floor at the bottom
    # We'll cut it such that it leaves exactly 'wall' (5mm) floor at the bottom
    # Actually wait, align=MAX puts top face at Z=0. We'll shift it up
    cavity = cavity.locate(Location((0, 0, height / 2)))
    base = base - cavity

    # 3. Pi Zero Mounting Standoffs
    # The Pi is mounted at the top (top means exactly flush or slightly recessed)
    # Pi Zero holes are at +/- 29mm X, +/- 11.5mm Y (rotated, so +/- 11.5mm X, +/- 29mm Y)
    hole_pitch_x = 0.0115
    hole_pitch_y = 0.029

    # The Pi is positioned at "top", which resolves to Z = height/2 - pi_thickness/2
    # pi dimensions are (0.065, 0.030, 0.005), rotated so (0.030, 0.065, 0.005)
    # so Z = 0.045 / 2 - 0.005 / 2 = 0.020
    # Standoff needs to reach Z = 0.020. Then the Pi sits on it.
    pi_z = 0.020

    standoffs = []
    # 4 standoffs
    for hx in [-hole_pitch_x, hole_pitch_x]:
        for hy in [-hole_pitch_y, hole_pitch_y]:
            # Cylinder from floor to pi_z
            # Floor is at -height/2 + wall = -0.0225 + 0.005 = -0.0175
            # So length = 0.020 - (-0.0175) = 0.0375
            # We'll just build a full cylinder and rely on the base floor anchoring it
            sh = pi_z - (-height / 2)  # from absolute bottom up to pi_z
            standoff = Cylinder(
                0.003, sh, align=(Align.CENTER, Align.CENTER, Align.MAX)
            )
            standoff = standoff.locate(Location((hx, hy, pi_z)))

            # Screw hole for M2.5 (radius 1.1mm) cut from top of standoff down
            hole = Cylinder(
                0.0011, 0.010, align=(Align.CENTER, Align.CENTER, Align.MAX)
            )
            hole = hole.locate(Location((hx, hy, pi_z)))

            standoff = standoff - hole
            standoffs.append(standoff)

    for s in standoffs:
        base = base + s

    # Remove the generic camera + battery cuts because botcad.cad already subtracts
    # their bounding boxes from the solid exactly where they are mounted!
    return base


def build() -> Bot:
    """Define the wheeler_base robot."""
    bot = Bot("wheeler_base")

    # Base body — houses electronics + camera.
    # Pi and battery rotated 90° so their long axis runs front-to-back (Y),
    # keeping X extent narrow to clear wheel servos at ±65mm.
    # Minimum height 45mm to stack battery (bottom) + camera + Pi (top)
    # without Z-axis overlap.
    # using a custom_solid for the filleted shell, precise cavity, and exact Pi standoffs
    base = bot.body(
        "base",
        shape=BodyShape.BOX,
        padding=0.008,
        dimensions=(0.130, 0.080, 0.045),
        custom_solid=wheeler_base_solid(),
    )
    # base.height is implicitly 0.045 through dimensions
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
