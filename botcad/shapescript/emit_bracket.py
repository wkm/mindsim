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
    """Build a ShapeScript for bracket_envelope() using native Box + locate.

    SCS0009 uses ear-tab geometry (U-shaped tray). STS3215 and similar
    use _bracket_outer() logic: Box at (0, 0, center_z) with insertion
    clearance extending 5x bracket height above.
    """
    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import _ear_bottom_z

    if spec is None:
        spec = BS()

    prog = ShapeScript()

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall

    if servo.name == "SCS0009":
        # SCS0009: U-shaped tray — envelope uses ear tab geometry
        ear_ext = 0.00465
        ear_thick = 0.0025
        ear_top_z = body_z / 2 - 0.00775
        ear_bot_z = ear_top_z - ear_thick

        outer_x = body_x + 2 * ear_ext + 2 * wall
        outer_y = body_y + 2 * (tol + wall)
        bracket_height = ear_bot_z - (-body_z / 2 - wall)
        outer_top_z = ear_bot_z + bracket_height * 5
        outer_bot_z = -body_z / 2 - wall
        outer_z = outer_top_z - outer_bot_z
        outer_center_z = (outer_top_z + outer_bot_z) / 2
    else:
        ear_bottom_z = _ear_bottom_z(servo, wall)

        # Matches _bracket_outer() with insertion_clearance = bracket_height * 5
        outer_x = body_x + 2 * (tol + wall)
        outer_y = body_y + 2 * (tol + wall)
        bracket_height = body_z / 2 + wall - ear_bottom_z
        insertion_clearance = bracket_height * 5
        outer_top_z = body_z / 2 + wall + insertion_clearance
        outer_z = outer_top_z - ear_bottom_z
        outer_center_z = (outer_top_z + ear_bottom_z) / 2

    outer = prog.box(outer_x, outer_y, outer_z, tag="bracket_envelope")
    outer = prog.locate(outer, pos=(0, 0, outer_center_z))

    prog.output_ref = outer
    return prog


def bracket_solid_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Build a ShapeScript expressing the full bracket_solid geometry.

    Both STS3215 and SCS0009 are expressed as native ShapeScript ops.
    STS3215: outer box, servo pocket, horn clearance, shaft boss, fastener holes,
    connector passage. SCS0009: U-shaped tray, body pocket, fastener holes,
    connector passage on -Z face.
    """
    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import (
        _cable_slot_dims,
        _ear_bottom_z,
        horn_disc_params,
    )
    from botcad.fasteners import HeadType, resolve_fastener
    from botcad.shapescript.ops import ALIGN_MIN_Z

    if spec is None:
        spec = BS()

    prog = ShapeScript()

    # ── Outer box ──
    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall

    if servo.name == "SCS0009":
        # SCS0009: U-shaped tray from bottom up to ear tabs
        ear_ext = 0.00465
        ear_thick = 0.0025
        ear_top_z = body_z / 2 - 0.00775
        ear_bot_z = ear_top_z - ear_thick

        outer_x = body_x + 2 * ear_ext + 2 * wall
        outer_y = body_y + 2 * (tol + wall)
        outer_top_z = ear_bot_z
        outer_bot_z = -body_z / 2 - wall
        outer_z = outer_top_z - outer_bot_z
        outer_center_z = (outer_top_z + outer_bot_z) / 2

        outer = prog.box(outer_x, outer_y, outer_z, tag="bracket_outer")
        outer = prog.locate(outer, pos=(0, 0, outer_center_z))

        # ── Body pocket (open on +Z for servo insertion) ──
        pocket_x = body_x + 2 * tol
        pocket_y = body_y + 2 * tol
        pocket_z = outer_z + 0.002  # through full height
        pocket = prog.box(pocket_x, pocket_y, pocket_z, tag="pocket")
        pocket = prog.locate(pocket, pos=(0, 0, outer_center_z))
        shell = prog.cut(outer, pocket)

        # ── Fastener holes at each ear position ──
        for ear in servo.mounting_ears:
            fspec = resolve_fastener(ear)
            hole_r = fspec.clearance_hole / 2
            hole = prog.cylinder(hole_r, outer_z + 0.002, tag="fastener_hole")
            hole = prog.locate(hole, pos=(ear.pos[0], ear.pos[1], outer_center_z))
            shell = prog.cut(shell, hole)

        # ── Connector passage (-Z face) ──
        if servo.connector_pos is not None:
            port = _emit_connector_port(
                prog,
                servo,
                spec,
                wall_center=(0, 0, outer_bot_z),
                exit_axis=(0.0, 0.0, -1.0),
            )
            if port is not None:
                shell = prog.cut(shell, port)
            else:
                _cx, cy, _cz = servo.connector_pos
                slot_w, slot_h = _cable_slot_dims(servo, spec)
                slot_depth = wall + tol + 0.002
                slot = prog.box(slot_w, slot_h, slot_depth, tag="cable_slot")
                slot_z = outer_bot_z + slot_depth / 2 - 0.001
                slot = prog.locate(slot, pos=(0, cy, slot_z))
                shell = prog.cut(shell, slot)
    else:
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
            hole = prog.cylinder(
                hole_r, hole_depth, align=ALIGN_MIN_Z, tag="fastener_hole"
            )
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
            port = _emit_connector_port(
                prog,
                servo,
                spec,
                wall_center=(wall_x, 0, 0),
                exit_axis=(-1.0, 0.0, 0.0),
            )
            if port is not None:
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
    """Build a ShapeScript for cradle_envelope() using native Box + locate.

    Covers the cradle footprint plus a 5x insertion channel in +X
    so the servo can slide in from the shaft side.
    """
    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import _ear_bottom_z

    if spec is None:
        spec = BS()

    prog = ShapeScript()

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall
    sx = servo.shaft_offset[0]

    ear_bottom_z = _ear_bottom_z(servo, wall)

    cradle_min_x = -body_x / 2 - tol - wall
    cradle_nominal_max_x = sx - 0.002
    cradle_lx_nominal = cradle_nominal_max_x - cradle_min_x
    cradle_max_x = cradle_nominal_max_x + cradle_lx_nominal * 5
    cradle_lx = cradle_max_x - cradle_min_x
    cradle_cx = (cradle_min_x + cradle_max_x) / 2

    outer_ly = body_y + 2 * (tol + wall)
    grip_margin = 0.004
    outer_top_z = -body_z / 2 + grip_margin
    outer_bottom_z = ear_bottom_z
    outer_lz = outer_top_z - outer_bottom_z
    outer_cz = (outer_top_z + outer_bottom_z) / 2

    envelope = prog.box(cradle_lx, outer_ly, outer_lz, tag="cradle_envelope")
    envelope = prog.locate(envelope, pos=(cradle_cx, 0, outer_cz))

    prog.output_ref = envelope
    return prog


def cradle_solid_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Build a ShapeScript for cradle_solid() using native ops.

    Outer box - pocket - fastener holes - connector passage.
    """
    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import (
        _cable_slot_dims,
        _ear_bottom_z,
        coupler_sweep_radius,
    )
    from botcad.fasteners import HeadType, resolve_fastener
    from botcad.shapescript.ops import ALIGN_MIN_Z

    if spec is None:
        spec = BS()

    prog = ShapeScript()

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall
    sx = servo.shaft_offset[0]

    ear_bottom_z = _ear_bottom_z(servo, wall)

    # ── Cradle extent in X ──
    cradle_min_x = -body_x / 2 - tol - wall
    sweep_r = coupler_sweep_radius(servo, spec)
    if sweep_r > 0:
        cradle_max_x = sx - sweep_r - 0.001
    else:
        cradle_max_x = sx - 0.002
    cradle_lx = cradle_max_x - cradle_min_x
    cradle_cx = (cradle_min_x + cradle_max_x) / 2

    # ── Cradle extent in Y ──
    outer_ly = body_y + 2 * (tol + wall)

    # ── Cradle extent in Z (shallow tray) ──
    grip_margin = 0.004
    outer_top_z = -body_z / 2 + grip_margin
    outer_bottom_z = ear_bottom_z
    outer_lz = outer_top_z - outer_bottom_z
    outer_cz = (outer_top_z + outer_bottom_z) / 2

    outer = prog.box(cradle_lx, outer_ly, outer_lz, tag="cradle_outer")
    outer = prog.locate(outer, pos=(cradle_cx, 0, outer_cz))

    # ── Pocket (servo body cavity) ──
    pocket_x = body_x + 2 * tol
    pocket_y = body_y + 2 * tol
    pocket_z = outer_lz + 0.002  # through full height
    pocket = prog.box(pocket_x, pocket_y, pocket_z, tag="cradle_pocket")
    pocket = prog.locate(pocket, pos=(0, 0, outer_cz))
    shell = prog.cut(outer, pocket)

    # ── Through-holes at each ear position ──
    for ear in servo.mounting_ears:
        if ear.pos[0] > cradle_max_x + 0.001:
            continue
        fspec = resolve_fastener(ear)
        hole_r = fspec.clearance_hole / 2
        hole_depth = abs(ear.pos[2] - ear_bottom_z) + wall + 0.002
        hole = prog.cylinder(hole_r, hole_depth, align=ALIGN_MIN_Z, tag="fastener_hole")
        hole = prog.locate(hole, pos=(ear.pos[0], ear.pos[1], ear_bottom_z - 0.001))
        shell = prog.cut(shell, hole)

        if fspec.head_type == HeadType.SOCKET_HEAD_CAP:
            cb_r = fspec.head_diameter / 2 + 0.0002
            cb_depth = fspec.head_height + 0.0005
            cb = prog.cylinder(cb_r, cb_depth, align=ALIGN_MIN_Z, tag="counterbore")
            cb = prog.locate(cb, pos=(ear.pos[0], ear.pos[1], ear_bottom_z - 0.001))
            shell = prog.cut(shell, cb)

    # ── Connector passage ──
    if servo.connector_pos is not None:
        port = _emit_connector_port(
            prog,
            servo,
            spec,
            wall_center=(cradle_min_x, 0, 0),
            exit_axis=(-1.0, 0.0, 0.0),
        )
        if port is not None:
            shell = prog.cut(shell, port)
        else:
            _cx, cy, cz = servo.connector_pos
            slot_w, slot_h = _cable_slot_dims(servo, spec)
            slot_depth = wall + tol + 0.002
            slot = prog.box(slot_depth, slot_w, slot_h, tag="cable_slot")
            slot_x = cradle_min_x + slot_depth / 2 - 0.001
            slot = prog.locate(slot, pos=(slot_x, cy, cz))
            shell = prog.cut(shell, slot)

    prog.output_ref = shell
    return prog


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


def _coupler_solid_script_native(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Native ShapeScript for coupler structural parts (without D-clip).

    Builds front plate + rear plate + side wall + horn holes + boss clearance
    using only native ops. The D-clip shaping is left to PrebuiltOp in
    coupler_solid_script() since it requires complex boolean clipping.

    This is kept as a reference for when the ShapeScript IR gains arc/fillet
    primitives that can express the D-clip natively.
    """
    import math

    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import _body_collision_half_y, _horn_clip_radius
    from botcad.fasteners import resolve_fastener
    from botcad.shapescript.ops import ALIGN_MAX_Z, ALIGN_MIN_Z

    if spec is None:
        spec = BS()

    prog = ShapeScript()

    tol = spec.tolerance
    sx, sy, sz = servo.shaft_offset
    body_x, body_y, body_z = servo.effective_body_dims

    # ── Collect horn holes in shaft-centered frame ──
    front_holes = []
    if servo.horn_mounting_points:
        for mp in servo.horn_mounting_points:
            front_holes.append((mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz))

    rear_holes = []
    if servo.rear_horn_mounting_points:
        for mp in servo.rear_horn_mounting_points:
            rear_holes.append((mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz))

    if not front_holes or not rear_holes:
        tiny = prog.box(0.001, 0.001, 0.001, tag="coupler_placeholder")
        prog.output_ref = tiny
        return prog

    all_holes = front_holes + rear_holes

    front_z = front_holes[0][2]
    rear_z = rear_holes[0][2]

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

    boss_clear_r = (
        servo.shaft_boss_radius + tol + 0.001 if servo.shaft_boss_radius > 0 else 0
    )

    horn_clip_r = _horn_clip_radius(servo, spec)

    # ── Front plate ──
    front_plate = prog.box(
        plate_lx, plate_ly, plate_t, align=ALIGN_MIN_Z, tag="front_plate"
    )
    front_plate = prog.locate(front_plate, pos=(plate_cx, plate_cy, front_z))

    # Boss clearance on front plate
    if boss_clear_r > 0:
        boss_hole = prog.cylinder(
            boss_clear_r, plate_t + 0.002, align=ALIGN_MIN_Z, tag="front_boss_hole"
        )
        boss_hole = prog.locate(boss_hole, pos=(0, 0, front_z - 0.001))
        front_plate = prog.cut(front_plate, boss_hole)

    # ── Rear plate ──
    rear_plate = prog.box(
        plate_lx, plate_ly, plate_t, align=ALIGN_MAX_Z, tag="rear_plate"
    )
    rear_plate = prog.locate(rear_plate, pos=(plate_cx, plate_cy, rear_z))

    # Boss clearance on rear plate
    if boss_clear_r > 0:
        boss_hole = prog.cylinder(
            boss_clear_r, plate_t + 0.002, align=ALIGN_MAX_Z, tag="rear_boss_hole"
        )
        boss_hole = prog.locate(boss_hole, pos=(0, 0, rear_z + 0.001))
        rear_plate = prog.cut(rear_plate, boss_hole)

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
        horn_hole_r, plate_t + 0.002, align=ALIGN_MIN_Z, tag="front_horn_hole"
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
        rear_hole_r, plate_t + 0.002, align=ALIGN_MAX_Z, tag="rear_horn_hole"
    )
    for i, (hx, hy, _hz) in enumerate(rear_holes):
        hole = rear_hole_proto if i == 0 else prog.copy(rear_hole_proto)
        hole = prog.locate(hole, pos=(hx, hy, rear_z + 0.001))
        shell = prog.cut(shell, hole)

    prog.output_ref = shell
    return prog
