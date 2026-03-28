"""Technical drawing emission — 2D section SVGs at every design scale.

Component level:  bracket-servo interface sections for each servo type
Bot level:        per-joint sections showing bracket, servo, and body clearances

All drawings are generated deterministically from the parametric model.
They serve as visual regression tests alongside the 3D renders — review
them in git diff to catch clearance and geometry regressions.
"""

from __future__ import annotations

from pathlib import Path

from botcad.debug_drawing import DebugDrawing

# ── Component-level drawings ──


def emit_component_drawings(servo_name: str, out_dir: Path) -> list[Path]:
    """Generate technical drawings for a servo's bracket variants.

    Produces one SVG per bracket type, each containing multiple section
    views side by side. Called alongside component PNG renders.
    """
    from build123d import Location, Plane

    from botcad.bracket import (
        BracketSpec,
        bracket_solid_solid,
        coupler_solid_solid,
        cradle_solid_solid,
        servo_solid,
    )
    from botcad.components import STS3215

    servo_map = {"STS3215": STS3215}
    if servo_name not in servo_map:
        return []

    servo = servo_map[servo_name]()
    spec = BracketSpec()
    sx, sy, sz = servo.shaft_offset

    servo_body = servo_solid(servo)  # body-centered frame
    pocket = bracket_solid_solid(servo, spec)  # body-centered frame
    cradle = cradle_solid_solid(servo, spec)  # body-centered frame
    coupler = coupler_solid_solid(servo, spec)  # shaft-centered frame

    # Shift body-centered solids into shaft-centered frame
    to_shaft = Location((-sx, -sy, -sz))
    servo_shaft = servo_body.moved(to_shaft)
    cradle_shaft = cradle.moved(to_shaft)

    # Z planes in servo body frame (pocket bracket)
    _body_x, _body_y, body_z = servo.effective_body_dims
    shaft_top_z = body_z / 2
    ear_z = servo.mounting_ears[0].pos[2] if servo.mounting_ears else -body_z / 2

    # Z planes in shaft-centered frame (coupler/cradle)
    front_z = servo.horn_mounting_points[0].pos[2] - sz
    rear_z = servo.rear_horn_mounting_points[0].pos[2] - sz

    safe = servo_name.lower()
    outputs: list[Path] = []

    # --- Pocket bracket + servo ---
    d = DebugDrawing("pocket_bracket")
    d.add_part("servo", servo_body, color=(60, 60, 60))
    d.add_part("bracket", pocket, color=(50, 120, 190))
    d.add_section("top_shaft", Plane.XY.offset(shaft_top_z))
    d.add_section("mid_body", Plane.XY.offset(0.0))
    d.add_section("ears", Plane.XY.offset(ear_z))
    d.add_projection("side_XZ", origin=(0, -10, 0))
    d.add_projection("side_YZ", origin=(10, 0, 0))
    outputs.append(d.save(out_dir / f"drawing_pocket_{safe}.svg"))

    # --- Coupler + servo (shaft-centered frame) ---
    d = DebugDrawing("coupler")
    d.add_part("servo", servo_shaft, color=(60, 60, 60))
    d.add_part("coupler", coupler, color=(220, 60, 40))
    d.add_section("front_horn", Plane.XY.offset(front_z))
    d.add_section("rear_horn", Plane.XY.offset(rear_z))
    d.add_section("mid_body", Plane.XY.offset((front_z + rear_z) / 2))
    d.add_projection("side_XZ", origin=(0, -10, 0))
    d.add_projection("side_YZ", origin=(10, 0, 0))
    outputs.append(d.save(out_dir / f"drawing_coupler_{safe}.svg"))

    # --- Cradle + servo (shaft-centered frame) ---
    d = DebugDrawing("cradle")
    d.add_part("servo", servo_shaft, color=(60, 60, 60))
    d.add_part("cradle", cradle_shaft, color=(50, 120, 190))
    d.add_section("front_horn", Plane.XY.offset(front_z))
    d.add_section("rear_horn", Plane.XY.offset(rear_z))
    d.add_section("mid_body", Plane.XY.offset((front_z + rear_z) / 2))
    d.add_projection("side_XZ", origin=(0, -10, 0))
    d.add_projection("side_YZ", origin=(10, 0, 0))
    outputs.append(d.save(out_dir / f"drawing_cradle_{safe}.svg"))

    # --- Full coupler assembly (shaft-centered frame) ---
    d = DebugDrawing("coupler_assembly")
    d.add_part("cradle", cradle_shaft, color=(50, 120, 190))
    d.add_part("servo", servo_shaft, color=(60, 60, 60))
    d.add_part("coupler", coupler, color=(220, 60, 40))
    d.add_section("front_horn", Plane.XY.offset(front_z))
    d.add_section("mid_body", Plane.XY.offset((front_z + rear_z) / 2))
    d.add_section("rear_horn", Plane.XY.offset(rear_z))
    d.add_projection("side_XZ", origin=(0, -10, 0))
    outputs.append(d.save(out_dir / f"drawing_coupler_assembly_{safe}.svg"))

    return outputs


# ── Bot-level drawings ──


def emit_drawings(bot, output_dir: Path) -> list[Path]:
    """Generate technical drawings for a bot's joints.

    One SVG per joint, containing shaft-face, mid-body, and side sections.
    """
    from build123d import Plane

    from botcad.bracket import BracketSpec, bracket_solid_solid, servo_solid

    drawings_dir = output_dir / "drawings"
    drawings_dir.mkdir(exist_ok=True)

    outputs: list[Path] = []

    for body in bot.all_bodies:
        if not body.joints:
            continue

        for joint in body.joints:
            servo = joint.servo
            spec = BracketSpec()

            servo_body = servo_solid(servo)
            bracket = bracket_solid_solid(servo, spec)

            _body_x, _body_y, body_z = servo.effective_body_dims
            shaft_top_z = body_z / 2

            safe_name = joint.name.replace(" ", "_").lower()

            d = DebugDrawing(f"joint_{joint.name}")
            d.add_part("servo", servo_body, color=(60, 60, 60))
            d.add_part("bracket", bracket, color=(50, 120, 190))
            d.add_section("shaft_face", Plane.XY.offset(shaft_top_z))
            d.add_section("mid_body", Plane.XY.offset(0.0))
            d.add_projection("side_XZ", origin=(0, -10, 0))
            outputs.append(d.save(drawings_dir / f"drawing_joint_{safe_name}.svg"))

    if outputs:
        print(f"    Drawings: {len(outputs)} SVGs in {drawings_dir}")

    return outputs
