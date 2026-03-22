"""ShapeScript emitters for bracket geometry.

bracket_envelope_script / cradle_envelope_script emit native Box + locate
ops (no PrebuiltOp).

bracket_solid_script translates bracket_solid() line-by-line into
ShapeScript ops: outer box, servo pocket, horn clearance, shaft boss,
fastener holes, and connector/cable passage. Both STS3215 and SCS0009
use native ops. Connector ports that are hard to express purely in
ShapeScript use native Box + locate ops.

cradle_solid_script translates cradle_solid() into native ShapeScript ops:
outer box, servo pocket cut, fastener holes, and connector/cable passage.

coupler_solid_script builds the C-shaped coupler: front plate, rear plate,
side wall, horn holes, boss clearance. D-clip plate geometry uses native
cylinder-cut-in-half ops (no PrebuiltOp).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.shapescript.ops import ShapeRef
from botcad.shapescript.program import ShapeScript

if TYPE_CHECKING:
    from botcad.bracket import BracketSpec
    from botcad.component import ServoSpec


def _emit_connector_port(
    prog: ShapeScript,
    servo: ServoSpec,
    spec: BracketSpec,
    wall_center: tuple[float, float, float],
    exit_axis: tuple[float, float, float],
) -> ShapeRef | None:
    """Emit native ShapeScript ops for a connector passage through a bracket wall.

    Computes a bounding box over all wire port connector envelopes + clearance,
    then emits a single Box + locate positioned at wall_center. Returns a ShapeRef
    for the cut solid, or None if no connectors are present.

    This is the ShapeScript equivalent of bracket._connector_port().
    """
    from botcad.connectors import connector_spec

    # Collect all wire ports with connectors
    ports = []
    for wp in servo.wire_ports:
        if wp.connector_type:
            try:
                cspec = connector_spec(wp.connector_type)
                ports.append((wp, cspec))
            except KeyError:
                pass

    if not ports:
        return None

    tol = spec.tolerance
    wall = spec.wall
    ax, ay, az = exit_axis

    clearance = tol + 0.001  # per side

    if abs(ax) > 0.5:
        # Passage through X wall — footprint in Y-Z plane
        y_coords: list[float] = []
        z_coords: list[float] = []
        for wp, cspec in ports:
            bx, by, bz = cspec.body_dimensions
            hw = max(bx, by) / 2 + clearance
            hh = max(min(bx, by), bz) / 2 + clearance
            y_coords.extend([wp.pos[1] - hw, wp.pos[1] + hw])
            z_coords.extend([wp.pos[2] - hh, wp.pos[2] + hh])
        cut_w = max(y_coords) - min(y_coords)
        cut_h = max(z_coords) - min(z_coords)
        center_y = (max(y_coords) + min(y_coords)) / 2
        center_z = (max(z_coords) + min(z_coords)) / 2
        passage_depth = wall + tol + 0.004
        cut = prog.box(passage_depth, cut_w, cut_h, tag="connector_port")
        cut_pos = (wall_center[0], center_y, center_z)
    elif abs(az) > 0.5:
        # Passage through Z wall — footprint in X-Y plane
        x_coords: list[float] = []
        y_coords2: list[float] = []
        for wp, cspec in ports:
            bx, by, bz = cspec.body_dimensions
            hw = max(bx, by) / 2 + clearance
            hh = max(min(bx, by), bz) / 2 + clearance
            x_coords.extend([wp.pos[0] - hw, wp.pos[0] + hw])
            y_coords2.extend([wp.pos[1] - hh, wp.pos[1] + hh])
        cut_w = max(x_coords) - min(x_coords)
        cut_h = max(y_coords2) - min(y_coords2)
        center_x = (max(x_coords) + min(x_coords)) / 2
        center_y = (max(y_coords2) + min(y_coords2)) / 2
        passage_depth = wall + tol + 0.004
        cut = prog.box(cut_w, cut_h, passage_depth, tag="connector_port")
        cut_pos = (center_x, center_y, wall_center[2])
    else:
        # Passage through Y wall — footprint in X-Z plane
        x_coords2: list[float] = []
        z_coords2: list[float] = []
        for wp, cspec in ports:
            bx, by, bz = cspec.body_dimensions
            hw = max(bx, by) / 2 + clearance
            hh = max(min(bx, by), bz) / 2 + clearance
            x_coords2.extend([wp.pos[0] - hw, wp.pos[0] + hw])
            z_coords2.extend([wp.pos[2] - hh, wp.pos[2] + hh])
        cut_w = max(x_coords2) - min(x_coords2)
        cut_h = max(z_coords2) - min(z_coords2)
        center_x = (max(x_coords2) + min(x_coords2)) / 2
        center_z = (max(z_coords2) + min(z_coords2)) / 2
        passage_depth = wall + tol + 0.004
        cut = prog.box(cut_w, passage_depth, cut_h, tag="connector_port")
        cut_pos = (center_x, wall_center[1], center_z)

    cut = prog.locate(cut, pos=cut_pos)
    return cut


def bracket_envelope_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Delegate to bracket.py's bracket_envelope_ir()."""
    from botcad.bracket import bracket_envelope_ir

    return bracket_envelope_ir(servo, spec)


def bracket_solid_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Delegate to bracket.py's bracket_solid_ir()."""
    from botcad.bracket import bracket_solid_ir

    return bracket_solid_ir(servo, spec)


def cradle_envelope_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Delegate to bracket.py's cradle_envelope_ir()."""
    from botcad.bracket import cradle_envelope_ir

    return cradle_envelope_ir(servo, spec)


def cradle_solid_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Delegate to bracket.py's cradle_solid_ir()."""
    from botcad.bracket import cradle_solid_ir

    return cradle_solid_ir(servo, spec)


def coupler_solid_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Build a ShapeScript for coupler_solid().

    All geometry uses native ShapeScript ops — no PrebuiltOps.
    Front plate + rear plate + side wall + horn holes + boss clearance,
    with D-clip plate shaping via cylinder-cut-in-half (semicircle + rect).
    """
    import math

    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import _body_collision_half_y, _horn_clip_radius
    from botcad.fasteners import resolve_fastener
    from botcad.shapescript.ops import ALIGN_MAX_Z, ALIGN_MIN_Z, Align3

    if spec is None:
        spec = BS()

    prog = ShapeScript()

    tol = spec.tolerance
    sx, sy, sz = servo.shaft_offset
    body_x, body_y, body_z = servo.effective_body_dims

    # ── Collect horn hole positions in shaft-centered frame ──
    front_holes = []
    if servo.horn_mounting_points:
        for mp in servo.horn_mounting_points:
            front_holes.append((mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz))

    rear_holes = []
    if servo.rear_horn_mounting_points:
        for mp in servo.rear_horn_mounting_points:
            rear_holes.append((mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz))

    if not front_holes or not rear_holes:
        # Degenerate case — tiny placeholder box
        tiny = prog.box(0.001, 0.001, 0.001, tag="coupler_placeholder")
        prog.output_ref = tiny
        return prog

    all_holes = front_holes + rear_holes

    # ── Key Z coordinates (in shaft frame) ──
    front_z = front_holes[0][2]
    rear_z = rear_holes[0][2]

    # ── Plate extent in XY ──
    body_plus_x = body_x / 2 - sx
    body_half_y = _body_collision_half_y(servo)
    body_diag = math.sqrt(body_plus_x**2 + body_half_y**2)
    wall_clear_x = body_diag + 0.001

    plate_t = spec.coupler_thickness
    hole_margin = 0.002
    plate_min_x = min(h[0] for h in all_holes) - hole_margin - plate_t
    plate_max_x = max(
        max(h[0] for h in all_holes) + hole_margin + plate_t,
        wall_clear_x + plate_t,
    )
    plate_min_y = min(h[1] for h in all_holes) - hole_margin - plate_t
    plate_max_y = max(h[1] for h in all_holes) + hole_margin + plate_t
    plate_lx = plate_max_x - plate_min_x
    plate_ly = plate_max_y - plate_min_y
    plate_cx = (plate_min_x + plate_max_x) / 2
    plate_cy = (plate_min_y + plate_max_y) / 2

    # ── Shaft boss clearance ──
    boss_clear_r = (
        servo.shaft_boss_radius + tol + 0.001 if servo.shaft_boss_radius > 0 else 0
    )

    horn_clip_r = _horn_clip_radius(servo, spec)

    def _clip_plate_to_d_shape(plate_ref, clip_r, z_base, z_align_max):
        """Clip plate to D-shape: semicircle (-X) + rectangle (+X).

        Mirrors bracket.py _clip_plate_to_horn_circle using native ops:
        cylinder cut in half → semicircle keep zone, fused with rect keep zone.
        """
        clip_h = plate_t + 0.002
        z_str = "max" if z_align_max else "min"
        align_z = ALIGN_MAX_Z if z_align_max else ALIGN_MIN_Z
        # Align3 with x="min" and matching z alignment (must match cylinders)
        align_xmin_z = Align3(x="min", z=z_str)
        clip_z = z_base + 0.001 if z_align_max else z_base - 0.001

        # Big cylinder covering entire plate (what we subtract from)
        clip_outer = prog.cylinder(plate_lx, clip_h, align=align_z)
        clip_outer = prog.locate(clip_outer, pos=(0, 0, clip_z))

        # Keep circle = cylinder of clip_r
        keep_circle = prog.cylinder(clip_r, clip_h + 0.002, align=align_z)
        keep_circle = prog.locate(keep_circle, pos=(0, 0, clip_z))

        # Cut away +X half to get -X semicircle
        cut_pos_x = prog.box(
            clip_r + 0.001,
            clip_r * 2 + 0.004,
            clip_h + 0.004,
            align=align_xmin_z,
        )
        cut_pos_x = prog.locate(cut_pos_x, pos=(0, plate_cy, clip_z))
        keep_semi = prog.cut(keep_circle, cut_pos_x)

        # Rectangle from X=0 to plate_max_x, width = 2*clip_r
        keep_rect = prog.box(
            plate_max_x + 0.001,
            clip_r * 2 + 0.002,
            clip_h + 0.002,
            align=align_xmin_z,
        )
        keep_rect = prog.locate(keep_rect, pos=(0, plate_cy, clip_z))

        # D-shape keep zone = semicircle + rectangle
        keep = prog.fuse(keep_semi, keep_rect)
        # Clip = everything outside D-shape
        clip_cut = prog.cut(clip_outer, keep)
        return prog.cut(plate_ref, clip_cut)

    # ── Front plate ──
    front_plate = prog.box(
        plate_lx, plate_ly, plate_t, align=ALIGN_MIN_Z, tag="front_plate"
    )
    front_plate = prog.locate(front_plate, pos=(plate_cx, plate_cy, front_z))

    # Boss clearance on front plate
    if boss_clear_r > 0:
        boss_hole = prog.cylinder(
            boss_clear_r,
            plate_t + 0.002,
            align=ALIGN_MIN_Z,
            tag="front_boss_hole",
        )
        boss_hole = prog.locate(boss_hole, pos=(0, 0, front_z - 0.001))
        front_plate = prog.cut(front_plate, boss_hole)

    front_plate = _clip_plate_to_d_shape(
        front_plate,
        horn_clip_r,
        front_z,
        z_align_max=False,
    )

    # ── Rear plate ──
    rear_plate = prog.box(
        plate_lx, plate_ly, plate_t, align=ALIGN_MAX_Z, tag="rear_plate"
    )
    rear_plate = prog.locate(rear_plate, pos=(plate_cx, plate_cy, rear_z))

    # Boss clearance on rear plate
    if boss_clear_r > 0:
        boss_hole = prog.cylinder(
            boss_clear_r,
            plate_t + 0.002,
            align=ALIGN_MAX_Z,
            tag="rear_boss_hole",
        )
        boss_hole = prog.locate(boss_hole, pos=(0, 0, rear_z + 0.001))
        rear_plate = prog.cut(rear_plate, boss_hole)

    rear_plate = _clip_plate_to_d_shape(
        rear_plate,
        horn_clip_r,
        rear_z,
        z_align_max=True,
    )

    # ── Side wall ──
    wall_z_lo = rear_z - plate_t
    wall_z_hi = front_z + plate_t
    wall_lz = wall_z_hi - wall_z_lo
    wall_cz = (wall_z_lo + wall_z_hi) / 2
    wall_cx = plate_max_x - plate_t / 2
    wall_width = horn_clip_r * 2

    side_wall = prog.box(plate_t, wall_width, wall_lz, tag="side_wall")
    side_wall = prog.locate(side_wall, pos=(wall_cx, plate_cy, wall_cz))

    # ── Fuse all three pieces ──
    shell = prog.fuse(front_plate, side_wall)
    shell = prog.fuse(shell, rear_plate)

    # ── Front horn holes ──
    if servo.horn_mounting_points:
        fspec = resolve_fastener(servo.horn_mounting_points[0])
        horn_hole_r = fspec.clearance_hole / 2
    else:
        horn_hole_r = 0.00125

    front_hole_proto = prog.cylinder(
        horn_hole_r,
        plate_t + 0.002,
        align=ALIGN_MIN_Z,
        tag="front_horn_hole",
    )
    for i, (hx, hy, _hz) in enumerate(front_holes):
        hole = front_hole_proto if i == 0 else prog.copy(front_hole_proto)
        hole = prog.locate(hole, pos=(hx, hy, front_z - 0.001))
        shell = prog.cut(shell, hole)

    # ── Rear horn holes ──
    if servo.rear_horn_mounting_points:
        rear_fspec = resolve_fastener(servo.rear_horn_mounting_points[0])
        rear_hole_r = rear_fspec.clearance_hole / 2
    else:
        rear_hole_r = horn_hole_r

    rear_hole_proto = prog.cylinder(
        rear_hole_r,
        plate_t + 0.002,
        align=ALIGN_MAX_Z,
        tag="rear_horn_hole",
    )
    for i, (hx, hy, _hz) in enumerate(rear_holes):
        hole = rear_hole_proto if i == 0 else prog.copy(rear_hole_proto)
        hole = prog.locate(hole, pos=(hx, hy, rear_z + 0.001))
        shell = prog.cut(shell, hole)

    prog.output_ref = shell
    return prog
