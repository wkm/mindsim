#!/usr/bin/env python3
"""Simple Plane — RC trainer-style fixed-wing aircraft (~1000mm wingspan).

A 3D-printable fixed-wing RC plane with real components:
- 1x MT2213 935KV brushless motor (nose firewall-mounted, tractor config)
- 1x 9x4.5 propeller
- 1x 30A ESC
- 4x SCS0009 micro servos (2 ailerons, 1 elevator, 1 rudder)
- 1x Matek F405-Wing flight controller
- 1x LiPo 3S 1300mAh battery

Structure (high-wing, conventional tail):
- Fuselage: 50mm W × 400mm L × 45mm H box (PLA_LIGHT)
- Motor firewall: 40mm × 5mm × 40mm plate at nose (PLA for strength)
- Wings: 470mm span each (1000mm total effective), 140mm chord, 6mm thick
- Tail boom: 250mm tube connecting fuselage to empennage
- Horizontal stabilizer: 220mm span × 70mm chord
- Vertical fin: 70mm chord × 90mm tall
- 4 hinged control surfaces: ailerons, elevator, rudder
- Landing gear: main gear plate + nose skid

Target AUW: ~600-700g (components ~263g, structure ~340g max)

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
from botcad.materials import PLA, PLA_LIGHT
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
        dimensions=(0.050, 0.400, 0.045),  # 50mm W × 400mm L × 45mm H
        padding=0.005,
    )
    fuselage.set_material(PLA_LIGHT)

    # Electronics inside fuselage — CG near wing quarter-chord
    fuselage.mount(MatekF405Wing(), position="top", label="fc")
    fuselage.mount(LiPo3S(1300), position="center", label="battery")
    fuselage.mount(SimonK30A(), position="bottom", label="esc")

    # ── Motor mount (firewall) ──────────────────────────────────────
    # Thin firewall plate at nose of fuselage for motor + prop.
    firewall_att = fuselage.attach(
        "firewall_attach",
        pos=(0.0, 0.200, 0.0),  # front of fuselage (Y+)
    )
    firewall = firewall_att.body(
        "firewall",
        shape=BodyShape.BOX,
        dimensions=(0.040, 0.005, 0.040),  # 40mm W × 5mm L × 40mm H
    )
    firewall.set_material(PLA)  # standard PLA for strength at motor mount
    firewall.mount(MT2213(), position="front", label="motor")
    firewall.mount(Propeller9x45(), position="front", label="prop")

    # ── Left wing (rigid attachment) ────────────────────────────────
    # High-wing config: thin flat-plate wing, 470mm span × 140mm chord × 6mm
    left_wing_att = fuselage.attach(
        "left_wing_attach",
        pos=(-0.270, 0.050, 0.022),  # outboard left, slightly forward, top
    )
    left_wing = left_wing_att.body(
        "left_wing",
        shape=BodyShape.BOX,
        dimensions=(0.470, 0.140, 0.006),  # 470mm span × 140mm chord × 6mm
    )
    left_wing.set_material(PLA_LIGHT)

    # Left aileron — hinged at trailing edge of left wing
    left_aileron_joint = left_wing.joint(
        "left_aileron",
        servo=SCS0009(),
        axis="x",
        pos=(0.0, -0.055, 0.0),  # trailing edge of wing
        range=(-0.35, 0.35),  # ~±20 degrees
        bracket_style=BracketStyle.COUPLER,
    )
    left_aileron_body = left_aileron_joint.body(
        "left_aileron_surface",
        shape=BodyShape.BOX,
        dimensions=(0.220, 0.035, 0.005),
    )
    left_aileron_body.set_material(PLA_LIGHT)

    # ── Right wing (rigid attachment) ───────────────────────────────
    right_wing_att = fuselage.attach(
        "right_wing_attach",
        pos=(0.270, 0.050, 0.022),  # outboard right, slightly forward, top
    )
    right_wing = right_wing_att.body(
        "right_wing",
        shape=BodyShape.BOX,
        dimensions=(0.470, 0.140, 0.006),
    )
    right_wing.set_material(PLA_LIGHT)

    # Right aileron
    right_aileron_joint = right_wing.joint(
        "right_aileron",
        servo=SCS0009(),
        axis="x",
        pos=(0.0, -0.055, 0.0),
        range=(-0.35, 0.35),
        bracket_style=BracketStyle.COUPLER,
    )
    right_aileron_body = right_aileron_joint.body(
        "right_aileron_surface",
        shape=BodyShape.BOX,
        dimensions=(0.220, 0.035, 0.005),
    )
    right_aileron_body.set_material(PLA_LIGHT)

    # ── Tail boom (rigid attachment) ────────────────────────────────
    # Tube from rear of fuselage to tail section.
    tail_boom_att = fuselage.attach(
        "tail_boom_attach",
        pos=(0.0, -0.300, 0.0),  # rear of fuselage (Y-)
    )
    tail_boom = tail_boom_att.body(
        "tail_boom",
        shape=BodyShape.TUBE,
        length=0.250,  # 250mm long
        outer_r=0.010,  # 10mm outer radius
    )
    tail_boom.set_material(PLA_LIGHT)

    # ── Horizontal stabilizer (rigid, at end of tail boom) ─────────
    # TUBE shape has Z as the length axis, so "end of tube" = +Z
    h_stab_att = tail_boom.attach(
        "h_stab_attach",
        pos=(0.0, 0.0, 0.250),  # end of tail boom (Z = tube length axis)
    )
    h_stab = h_stab_att.body(
        "horizontal_stab",
        shape=BodyShape.BOX,
        dimensions=(0.220, 0.070, 0.006),  # 220mm span × 70mm chord × 6mm
    )
    h_stab.set_material(PLA_LIGHT)

    # Elevator — hinged at trailing edge of horizontal stabilizer
    elevator_joint = h_stab.joint(
        "elevator",
        servo=SCS0009(),
        axis="x",
        pos=(0.0, -0.025, 0.0),  # trailing edge
        range=(-0.52, 0.52),  # ~±30 degrees
        bracket_style=BracketStyle.COUPLER,
    )
    elevator_body = elevator_joint.body(
        "elevator_surface",
        shape=BodyShape.BOX,
        dimensions=(0.220, 0.025, 0.005),
    )
    elevator_body.set_material(PLA_LIGHT)

    # ── Vertical fin (rigid, at end of tail boom) ──────────────────
    v_fin_att = tail_boom.attach(
        "v_fin_attach",
        pos=(0.0, 0.0, 0.250),  # same location as h_stab
    )
    v_fin = v_fin_att.body(
        "vertical_fin",
        shape=BodyShape.BOX,
        dimensions=(0.006, 0.070, 0.090),  # 6mm × 70mm chord × 90mm tall
    )
    v_fin.set_material(PLA_LIGHT)

    # Rudder — hinged at trailing edge of vertical fin
    rudder_joint = v_fin.joint(
        "rudder",
        servo=SCS0009(),
        axis="z",
        pos=(0.0, -0.025, 0.0),  # trailing edge
        range=(-0.52, 0.52),
        bracket_style=BracketStyle.COUPLER,
    )
    rudder_body = rudder_joint.body(
        "rudder_surface",
        shape=BodyShape.BOX,
        dimensions=(0.006, 0.025, 0.090),
    )
    rudder_body.set_material(PLA_LIGHT)

    # ── Landing gear ───────────────────────────────────────────────
    # Main gear: simple plate under fuselage, just behind wing quarter-chord
    main_gear_att = fuselage.attach(
        "main_gear_attach",
        pos=(0.0, 0.030, -0.022),  # under fuselage, behind wing
    )
    main_gear = main_gear_att.body(
        "main_gear",
        shape=BodyShape.BOX,
        dimensions=(0.200, 0.005, 0.060),  # 200mm W × 5mm L × 60mm H
    )
    main_gear.set_material(PLA)  # standard PLA for strength

    # Nose skid: small block under nose
    nose_skid_att = fuselage.attach(
        "nose_skid_attach",
        pos=(0.0, 0.180, -0.022),  # under nose
    )
    nose_skid = nose_skid_att.body(
        "nose_skid",
        shape=BodyShape.BOX,
        dimensions=(0.015, 0.030, 0.015),  # 15mm W × 30mm L × 15mm H
    )
    nose_skid.set_material(PLA)  # standard PLA for strength

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
