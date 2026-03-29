#!/usr/bin/env python3
"""Simple Plane — RC trainer-style fixed-wing aircraft (~820mm wingspan).

A 3D-printable fixed-wing RC plane with real components:
- 1x MT2213 935KV brushless motor (nose-mounted, tractor config)
- 1x 9x4.5 propeller
- 1x 30A ESC
- 4x SCS0009 micro servos (2 ailerons, 1 elevator, 1 rudder)
- 1x Matek F405-Wing flight controller
- 1x LiPo 3S 1300mAh battery

Structure (high-wing, conventional tail):
- Fuselage: 60mm W × 500mm L × 50mm H box
- Wings: 380mm span each (820mm total effective), 150mm chord, 15mm thick
- Tail boom: 250mm tube connecting fuselage to empennage
- Horizontal stabilizer: 250mm span × 80mm chord
- Vertical fin: 80mm chord × 100mm tall
- 4 hinged control surfaces: ailerons, elevator, rudder

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
    """Define the simple_plane aircraft."""
    bot = Bot("simple_plane")

    # ── Fuselage ────────────────────────────────────────────────────
    # Central body housing electronics.
    # Oriented: X = lateral, Y = forward (nose +Y), Z = up.
    fuselage = bot.body(
        "fuselage",
        shape=BodyShape.BOX,
        dimensions=(0.060, 0.500, 0.050),  # 60mm W × 500mm L × 50mm H
        padding=0.005,
    )

    # Electronics inside fuselage — CG near wing quarter-chord
    fuselage.mount(MatekF405Wing(), position="top", label="fc")
    fuselage.mount(LiPo3S(1300), position="center", label="battery")
    fuselage.mount(SimonK30A(), position="bottom", label="esc")
    fuselage.mount(MT2213(), position="front", label="motor")
    fuselage.mount(Propeller9x45(), position="front", label="prop")

    # ── Left wing (rigid attachment) ────────────────────────────────
    # High-wing config: attaches to top of fuselage, slightly ahead of CG.
    # Wing root is at fuselage edge (x=±30mm), extends outboard 380mm.
    left_wing_att = fuselage.attach(
        "left_wing_attach",
        pos=(-0.220, 0.050, 0.025),  # outboard left, slightly forward, top
    )
    left_wing = left_wing_att.body(
        "left_wing",
        shape=BodyShape.BOX,
        dimensions=(0.380, 0.150, 0.015),  # 380mm span × 150mm chord × 15mm
    )

    # Left aileron — hinged at trailing edge of left wing
    left_aileron_joint = left_wing.joint(
        "left_aileron",
        servo=SCS0009(),
        axis="x",
        pos=(0.0, -0.055, 0.0),  # trailing edge of wing
        range=(-0.35, 0.35),  # ~±20 degrees
        bracket_style=BracketStyle.COUPLER,
    )
    left_aileron_joint.body(
        "left_aileron_surface",
        shape=BodyShape.BOX,
        dimensions=(0.180, 0.040, 0.008),
    )

    # ── Right wing (rigid attachment) ───────────────────────────────
    right_wing_att = fuselage.attach(
        "right_wing_attach",
        pos=(0.220, 0.050, 0.025),  # outboard right, slightly forward, top
    )
    right_wing = right_wing_att.body(
        "right_wing",
        shape=BodyShape.BOX,
        dimensions=(0.380, 0.150, 0.015),
    )

    # Right aileron
    right_aileron_joint = right_wing.joint(
        "right_aileron",
        servo=SCS0009(),
        axis="x",
        pos=(0.0, -0.055, 0.0),
        range=(-0.35, 0.35),
        bracket_style=BracketStyle.COUPLER,
    )
    right_aileron_joint.body(
        "right_aileron_surface",
        shape=BodyShape.BOX,
        dimensions=(0.180, 0.040, 0.008),
    )

    # ── Tail boom (rigid attachment) ────────────────────────────────
    # Tube from rear of fuselage to tail section.
    tail_boom_att = fuselage.attach(
        "tail_boom_attach",
        pos=(0.0, -0.375, 0.0),  # rear of fuselage
    )
    tail_boom = tail_boom_att.body(
        "tail_boom",
        shape=BodyShape.TUBE,
        length=0.250,  # 250mm long
        outer_r=0.012,  # 12mm outer radius
    )

    # ── Horizontal stabilizer (rigid, at end of tail boom) ─────────
    # TUBE shape has Z as the length axis, so "end of tube" = +Z
    h_stab_att = tail_boom.attach(
        "h_stab_attach",
        pos=(0.0, 0.0, 0.250),  # end of tail boom (Z = tube length axis)
    )
    h_stab = h_stab_att.body(
        "horizontal_stab",
        shape=BodyShape.BOX,
        dimensions=(0.250, 0.080, 0.008),  # 250mm span × 80mm chord × 8mm
    )

    # Elevator — hinged at trailing edge of horizontal stabilizer
    elevator_joint = h_stab.joint(
        "elevator",
        servo=SCS0009(),
        axis="x",
        pos=(0.0, -0.025, 0.0),  # trailing edge
        range=(-0.52, 0.52),  # ~±30 degrees
        bracket_style=BracketStyle.COUPLER,
    )
    elevator_joint.body(
        "elevator_surface",
        shape=BodyShape.BOX,
        dimensions=(0.250, 0.030, 0.008),
    )

    # ── Vertical fin (rigid, at end of tail boom) ──────────────────
    v_fin_att = tail_boom.attach(
        "v_fin_attach",
        pos=(0.0, 0.0, 0.250),  # same location as h_stab
    )
    v_fin = v_fin_att.body(
        "vertical_fin",
        shape=BodyShape.BOX,
        dimensions=(0.008, 0.080, 0.100),  # 8mm × 80mm chord × 100mm tall
    )

    # Rudder — hinged at trailing edge of vertical fin
    rudder_joint = v_fin.joint(
        "rudder",
        servo=SCS0009(),
        axis="z",
        pos=(0.0, -0.025, 0.0),  # trailing edge
        range=(-0.52, 0.52),
        bracket_style=BracketStyle.COUPLER,
    )
    rudder_joint.body(
        "rudder_surface",
        shape=BodyShape.BOX,
        dimensions=(0.008, 0.030, 0.100),
    )

    return bot


def main() -> None:
    bot = build()
    bot.solve()

    output_dir = Path(__file__).parent
    bot.write_mujoco(str(output_dir))
    bot.write_step(str(output_dir))
    bot.write_docs(str(output_dir))
    bot.write_viewer_manifest(str(output_dir))

    print(f"\nGenerated simple_plane bot in {output_dir}")
    print("  bot.xml       — MuJoCo robot definition")
    print("  scene.xml     — Scene wrapper (bot + room)")
    print("  meshes/       — Per-body STL meshes")
    print("  assembly.step — Full CAD assembly")
    print("  bom.md        — Bill of materials")
    print("  assembly_guide.md — Assembly instructions")
    print("\nTo view in MuJoCo:")
    print("  uv run mjpython main.py view --bot simple_plane")


if __name__ == "__main__":
    main()
