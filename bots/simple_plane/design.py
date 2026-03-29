#!/usr/bin/env python3
"""Simple Plane — fixed-wing drone with conventional control surfaces.

A simple fixed-wing RC-style plane with real components:
- 1x MT2213 brushless motor (nose-mounted, tractor configuration)
- 1x 9x4.5 propeller
- 1x 30A ESC
- 4x SCS0009 micro servos (ailerons, elevator, rudder)
- 1x Matek F405-Wing flight controller
- 1x LiPo 3S 1300mAh battery

Control surfaces:
- Left aileron (roll)
- Right aileron (roll, opposite deflection)
- Elevator (pitch)
- Rudder (yaw)

Approximate dimensions:
- Fuselage: ~120mm wide x 200mm long x 60mm tall
- Control surfaces hinged at fuselage edges
- All-up weight: ~550g

The fuselage body houses all electronics and servos. Control surfaces
are thin box bodies attached via hinge joints at the fuselage edges.
In simulation, aerodynamic forces would be applied via a plugin — the
geometry here represents the physical structure for CAD and assembly.

Run this script to generate all outputs:
    python bots/simple_plane/design.py
"""

from pathlib import Path

from botcad.components import (
    MT2213,
    SCS0009,
    LiPo3S,
    MatekF405Wing,
    Propeller9x45,
    SimonK30A,
)
from botcad.skeleton import BodyShape, Bot, BracketStyle


def build() -> Bot:
    """Define the simple_plane drone."""
    bot = Bot("simple_plane")

    # ── Fuselage ────────────────────────────────────────────────────
    # Central body housing electronics and servos.
    # Oriented: X = lateral, Y = forward (nose), Z = up.
    fuselage = bot.body(
        "fuselage",
        shape=BodyShape.BOX,
        dimensions=(0.12, 0.20, 0.06),  # 120mm wide, 200mm long, 60mm tall
        padding=0.005,
    )

    # Mount electronics
    fuselage.mount(MatekF405Wing(), position="top", label="fc")
    fuselage.mount(LiPo3S(1300), position="center", label="battery")
    fuselage.mount(SimonK30A(), position="bottom", label="esc")
    fuselage.mount(MT2213(), position="front", label="motor")
    fuselage.mount(Propeller9x45(), position="front", label="prop")

    # ── Left aileron ────────────────────────────────────────────────
    # Hinge at left edge of fuselage, trailing edge.
    # Coupler bracket avoids cutting pockets into the thin fuselage.
    left_aileron_joint = fuselage.joint(
        "left_aileron",
        servo=SCS0009(),
        axis="x",
        pos=(-0.05, -0.05, 0.0),
        range=(-0.35, 0.35),  # ~±20 degrees
        bracket_style=BracketStyle.COUPLER,
    )
    left_aileron_joint.body(
        "left_aileron",
        shape=BodyShape.BOX,
        dimensions=(0.18, 0.04, 0.008),
    )

    # ── Right aileron ───────────────────────────────────────────────
    right_aileron_joint = fuselage.joint(
        "right_aileron",
        servo=SCS0009(),
        axis="x",
        pos=(0.05, -0.05, 0.0),
        range=(-0.35, 0.35),
        bracket_style=BracketStyle.COUPLER,
    )
    right_aileron_joint.body(
        "right_aileron",
        shape=BodyShape.BOX,
        dimensions=(0.18, 0.04, 0.008),
    )

    # ── Elevator ────────────────────────────────────────────────────
    # Horizontal stabilizer — pitch control at rear.
    elevator_joint = fuselage.joint(
        "elevator",
        servo=SCS0009(),
        axis="x",
        pos=(0.0, -0.09, 0.0),
        range=(-0.52, 0.52),  # ~±30 degrees
        bracket_style=BracketStyle.COUPLER,
    )
    elevator_joint.body(
        "elevator",
        shape=BodyShape.BOX,
        dimensions=(0.20, 0.04, 0.008),
    )

    # ── Rudder ──────────────────────────────────────────────────────
    # Vertical stabilizer — yaw control at rear/top.
    rudder_joint = fuselage.joint(
        "rudder",
        servo=SCS0009(),
        axis="z",
        pos=(0.0, -0.09, 0.025),
        range=(-0.52, 0.52),
        bracket_style=BracketStyle.COUPLER,
    )
    rudder_joint.body(
        "rudder",
        shape=BodyShape.BOX,
        dimensions=(0.008, 0.04, 0.07),
    )

    return bot


def main() -> None:
    bot = build()
    bot.solve()

    output_dir = Path(__file__).parent
    bot.emit(str(output_dir))

    print(f"\nGenerated simple_plane bot in {output_dir}")
    print("  bot.xml       — MuJoCo robot definition")
    print("  scene.xml     — Scene wrapper (bot + room)")
    print("  meshes/       — Per-body STL meshes")
    print("  bom.md        — Bill of materials")
    print("  assembly_guide.md — Assembly instructions")
    print("\nTo view in MuJoCo:")
    print("  uv run mjpython main.py view --bot simple_plane")


if __name__ == "__main__":
    main()
