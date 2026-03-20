"""ShapeScript emitters for bracket geometry.

bracket_envelope_script / cradle_envelope_script wrap pre-computed
envelope solids via PrebuiltOp.

bracket_solid_script translates bracket_solid() line-by-line into
ShapeScript ops: outer box, servo pocket, horn clearance, shaft boss,
fastener holes, and connector/cable passage. SCS0009 and connector
ports that are hard to express purely in ShapeScript use PrebuiltOp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.shapescript.program import ShapeScript

if TYPE_CHECKING:
    from botcad.bracket import BracketSpec
    from botcad.component import ServoSpec


def bracket_envelope_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Build a ShapeScript wrapping bracket_envelope() as a PrebuiltOp."""
    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import bracket_envelope

    if spec is None:
        spec = BS()
    prog = ShapeScript()
    ref = prog.prebuilt(bracket_envelope(servo, spec), tag="bracket_envelope")
    prog.output_ref = ref
    return prog


def bracket_solid_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Build a ShapeScript expressing the full bracket_solid geometry.

    STS3215 (and similar) brackets are fully expressed as ShapeScript ops:
    outer box, servo pocket cut, horn clearance hole, shaft boss clearance,
    fastener holes, and connector/cable passage. SCS0009 uses PrebuiltOp
    since its geometry differs substantially.
    """
    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import (
        _cable_slot_dims,
        _connector_port,
        _ear_bottom_z,
        bracket_solid,
        horn_disc_params,
    )
    from botcad.fasteners import HeadType, resolve_fastener
    from botcad.shapescript.ops import ALIGN_MIN_Z

    if spec is None:
        spec = BS()

    # SCS0009 has unique geometry — wrap as PrebuiltOp
    if servo.name == "SCS0009":
        prog = ShapeScript()
        ref = prog.prebuilt(bracket_solid(servo, spec), tag="bracket_solid")
        prog.output_ref = ref
        return prog

    prog = ShapeScript()

    # ── Outer box ──
    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall
    ear_bottom_z = _ear_bottom_z(servo, wall)

    outer_x = body_x + 2 * (tol + wall)
    outer_y = body_y + 2 * (tol + wall)
    outer_top_z = body_z / 2 + wall
    outer_z = outer_top_z - ear_bottom_z
    outer_center_z = (outer_top_z + ear_bottom_z) / 2

    outer = prog.box(outer_x, outer_y, outer_z, tag="bracket_outer")
    outer = prog.locate(outer, pos=(0, 0, outer_center_z))

    # ── Servo body pocket (open on +Z for insertion) ──
    pocket_x = body_x + 2 * tol
    pocket_y = body_y + 2 * tol
    pocket_z = body_z + tol + wall + 0.001
    pocket_center_z = -body_z / 2 + pocket_z / 2 - tol

    pocket = prog.box(pocket_x, pocket_y, pocket_z, tag="pocket")
    pocket = prog.locate(pocket, pos=(0, 0, pocket_center_z))
    shell = prog.cut(outer, pocket)

    # ── Horn clearance hole (+Z face) ──
    horn_radius = 0.011
    params = horn_disc_params(servo)
    if params is not None:
        horn_radius = params.radius + spec.shaft_clearance

    shaft_hole_h = wall + 0.002
    shaft_hole_z = body_z / 2 - 0.001
    shaft_hole = prog.cylinder(
        horn_radius, shaft_hole_h, align=ALIGN_MIN_Z, tag="horn_hole"
    )
    shaft_hole = prog.locate(
        shaft_hole,
        pos=(servo.shaft_offset[0], servo.shaft_offset[1], shaft_hole_z),
    )
    shell = prog.cut(shell, shaft_hole)

    # ── Shaft boss clearance (bearing housing above body top) ──
    if servo.shaft_boss_radius > 0 and servo.shaft_boss_height > 0:
        boss_r = servo.shaft_boss_radius + tol
        boss_h = servo.shaft_boss_height + tol + 0.001
        boss = prog.cylinder(boss_r, boss_h, align=ALIGN_MIN_Z, tag="shaft_boss")
        boss = prog.locate(
            boss,
            pos=(servo.shaft_offset[0], servo.shaft_offset[1], body_z / 2 - 0.001),
        )
        shell = prog.cut(shell, boss)

    # ── Fastener holes at each mounting ear ──
    for ear in servo.mounting_ears:
        fspec = resolve_fastener(ear)
        hole_r = fspec.clearance_hole / 2
        hole_depth = abs(ear.pos[2] - ear_bottom_z) + wall + 0.002
        hole = prog.cylinder(hole_r, hole_depth, align=ALIGN_MIN_Z, tag="fastener_hole")
        hole = prog.locate(hole, pos=(ear.pos[0], ear.pos[1], ear_bottom_z - 0.001))
        shell = prog.cut(shell, hole)

        # Counterbore for socket head cap screws
        if fspec.head_type == HeadType.SOCKET_HEAD_CAP:
            cb_r = fspec.head_diameter / 2 + 0.0002
            cb_depth = fspec.head_height + 0.0005
            cb = prog.cylinder(cb_r, cb_depth, align=ALIGN_MIN_Z, tag="counterbore")
            cb = prog.locate(cb, pos=(ear.pos[0], ear.pos[1], ear_bottom_z - 0.001))
            shell = prog.cut(shell, cb)

    # ── Connector port / cable slot ──
    if servo.connector_pos is not None:
        wall_x = -outer_x / 2
        cut_solid, _housing = _connector_port(
            servo,
            spec,
            wall_center=(wall_x, 0, 0),
            exit_axis=(-1.0, 0.0, 0.0),
        )
        if cut_solid is not None:
            # Complex shaped passage — use PrebuiltOp
            port = prog.prebuilt(cut_solid, tag="connector_port")
            shell = prog.cut(shell, port)
        else:
            # Fallback: rectangular cable slot
            cx, cy, cz = servo.connector_pos
            slot_w, slot_h = _cable_slot_dims(servo, spec)
            slot_depth = wall + tol + 0.002
            slot = prog.box(slot_depth, slot_w, slot_h, tag="cable_slot")
            slot_x = wall_x + slot_depth / 2 - 0.001
            slot = prog.locate(slot, pos=(slot_x, cy, cz))
            shell = prog.cut(shell, slot)

    prog.output_ref = shell
    return prog


def cradle_envelope_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Build a ShapeScript wrapping cradle_envelope() as a PrebuiltOp."""
    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import cradle_envelope

    if spec is None:
        spec = BS()
    prog = ShapeScript()
    ref = prog.prebuilt(cradle_envelope(servo, spec), tag="cradle_envelope")
    prog.output_ref = ref
    return prog
