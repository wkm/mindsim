"""Parametric servo bracket geometry.

Two bracket styles for STS-series servos:

**Bracket** (POCKET style) — for wheel/linear applications.
Wraps ±X, ±Y sides and -Z face. Powered horn (+Z) exposed through a
clearance hole. Fasteners through mounting ear holes from -Z side.
Wire cutout on the -Z face.

**Cradle** (COUPLER style) — for rotational joint applications.
Shallow tray cupping the servo from below: ±Y side walls + bottom wall
below the mounting ears. Both +Z and -Z horn faces fully exposed for
coupler attachment. Much smaller than the bracket — roughly half the
body height.

All dimensions are in meters (SI), matching the rest of botcad.

Servo local frame convention (same as servo.py):
    X = long axis, Y = width, Z = shaft axis
    Origin = servo body center
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botcad.component import ServoSpec
    from botcad.shapescript.program import ShapeScript


def _cable_slot_dims(servo: ServoSpec, spec: BracketSpec) -> tuple[float, float]:
    """Derive cable slot width and height from the servo's connector spec.

    Falls back to BracketSpec defaults if no connector info is available.
    """
    # Find the UART wire port with a connector_type
    for wp in servo.wire_ports:
        if wp.connector_type:
            try:
                from botcad.connectors import connector_spec

                cspec = connector_spec(wp.connector_type)
                bx, by, bz = cspec.body_dimensions
                # Slot should fit connector + 2mm tolerance per side
                tol = 0.002
                return max(bx, by) + tol, max(min(bx, by), bz) + tol
            except KeyError:
                break
    return spec.cable_slot_width, spec.cable_slot_height


def _ear_bottom_z(servo: ServoSpec, wall: float) -> float:
    """Compute the lowest Z extent of the bracket due to mounting ears."""
    body_z = servo.effective_body_dims[2]
    bottom = -body_z / 2
    if servo.mounting_ears:
        min_ear_z = min(ear.pos[2] for ear in servo.mounting_ears)
        max_hole_d = max(ear.diameter for ear in servo.mounting_ears)
        bottom = min_ear_z - max_hole_d / 2 - wall
    return bottom


def _all_horn_points(servo: ServoSpec) -> list:
    """Collect all horn mounting points (front + rear)."""
    return list(servo.horn_mounting_points or []) + list(
        servo.rear_horn_mounting_points or []
    )


def _body_collision_half_y(servo: ServoSpec) -> float:
    """Half-width of the servo collision envelope including ear flanges."""
    body_half_y = servo.effective_body_dims[1] / 2
    if servo.mounting_ears:
        max_ear_y = max(
            abs(ear.pos[1]) + ear.diameter / 2 + 0.001 for ear in servo.mounting_ears
        )
        body_half_y = max(body_half_y, max_ear_y)
    return body_half_y


def _horn_clip_radius(servo: ServoSpec, spec: BracketSpec) -> float:
    """Compute the horn disc clip radius, constrained by ear clearance.

    Used by coupler_solid, coupler_sweep_radius, and coupler_max_rom_rad
    to ensure consistent geometry.
    """
    import math

    sx, sy = servo.shaft_offset[0], servo.shaft_offset[1]
    hole_margin = 0.002
    all_horn = _all_horn_points(servo)
    if not all_horn:
        return 0.0

    horn_r = max(
        math.sqrt((mp.pos[0] - sx) ** 2 + (mp.pos[1] - sy) ** 2) for mp in all_horn
    )
    clip_r = horn_r + hole_margin + spec.wall
    if servo.mounting_ears:
        min_ear_r = min(
            math.sqrt((ear.pos[0] - sx) ** 2 + (ear.pos[1] - sy) ** 2)
            - ear.diameter / 2
            - spec.tolerance
            - 0.001
            for ear in servo.mounting_ears
        )
        clip_r = min(clip_r, min_ear_r)
    return clip_r


@dataclass(frozen=True)
class BracketSpec:
    """Parameters controlling bracket geometry around a servo."""

    wall: float = 0.003  # 3mm wall thickness
    tolerance: float = 0.0003  # 0.3mm clearance per side for FDM
    shaft_clearance: float = 0.001  # 1mm extra radius around horn
    cable_slot_width: float = 0.010  # 10mm wide cable exit
    cable_slot_height: float = 0.006  # 6mm tall cable exit
    coupler_thickness: float = 0.010  # 10mm plate thickness for PLA rigidity


def bracket_insertion_channel(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Return the bracket insertion channel as ShapeScript IR.

    The insertion channel is the volume subtracted from the parent body
    shell to create a path for servo insertion during assembly. It is NOT
    an enclosure around the servo or bracket. Its cross-section perpendicular
    to the insertion axis must clear the servo's cross-section on those axes.

    Used by the CAD emitter to cut the bracket footprint from the parent
    body shell. The insertion channel extends 5x the bracket height above.
    """
    from botcad.shapescript.program import ShapeScript

    if spec is None:
        spec = BracketSpec()

    prog = ShapeScript()

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall

    if servo.name == "SCS0009":
        # SCS0009: U-shaped tray — insertion channel uses ear tab geometry
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

    outer = prog.box(outer_x, outer_y, outer_z, tag="bracket_insertion_channel")
    outer = prog.locate(outer, pos=(0, 0, outer_center_z))

    prog.output_ref = outer
    return prog


def _emit_connector_port_ir(
    prog,
    servo: ServoSpec,
    spec: BracketSpec,
    wall_center: tuple[float, float, float],
    exit_axis: tuple[float, float, float],
):
    """Emit native ShapeScript ops for a connector passage through a bracket wall.

    Computes a bounding box over all wire port connector envelopes + clearance,
    then emits a single Box + locate positioned at wall_center. Returns a ShapeRef
    for the cut solid, or None if no connectors are present.
    """
    from botcad.connectors import connector_spec

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


def bracket_solid(servo: ServoSpec, spec: BracketSpec | None = None) -> ShapeScript:
    """Build a bracket solid for wheel/linear applications as ShapeScript IR.

    Wraps ±X, ±Y, and -Z (unpowered horn side). The +Z face is open
    with a clearance hole for the powered horn disc + shaft boss.
    """
    from botcad.shapescript.ops import ALIGN_MIN_Z
    from botcad.shapescript.program import ShapeScript

    if spec is None:
        spec = BracketSpec()

    prog = ShapeScript()

    # -- Outer box --
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

        # -- Body pocket (open on +Z for servo insertion) --
        pocket_x = body_x + 2 * tol
        pocket_y = body_y + 2 * tol
        pocket_z = outer_z + 0.002  # through full height
        pocket = prog.box(pocket_x, pocket_y, pocket_z, tag="pocket")
        pocket = prog.locate(pocket, pos=(0, 0, outer_center_z))
        shell = prog.cut(outer, pocket)

        # -- Fastener holes at each ear position --
        from botcad.fasteners import HeadType, resolve_fastener

        for ear in servo.mounting_ears:
            fspec = resolve_fastener(ear)
            hole_r = fspec.clearance_hole / 2
            hole = prog.cylinder(hole_r, outer_z + 0.002, tag="fastener_hole")
            hole = prog.locate(hole, pos=(ear.pos[0], ear.pos[1], outer_center_z))
            shell = prog.cut(shell, hole)

        # -- Connector passage (-Z face) --
        if servo.connector_pos is not None:
            port = _emit_connector_port_ir(
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

        # -- Servo body pocket (open on +Z for insertion) --
        pocket_x = body_x + 2 * tol
        pocket_y = body_y + 2 * tol
        pocket_z = body_z + tol + wall + 0.001
        pocket_center_z = -body_z / 2 + pocket_z / 2 - tol

        pocket = prog.box(pocket_x, pocket_y, pocket_z, tag="pocket")
        pocket = prog.locate(pocket, pos=(0, 0, pocket_center_z))
        shell = prog.cut(outer, pocket)

        # -- Horn clearance hole (+Z face) --
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

        # -- Shaft boss clearance (bearing housing above body top) --
        if servo.shaft_boss_radius > 0 and servo.shaft_boss_height > 0:
            boss_r = servo.shaft_boss_radius + tol
            boss_h = servo.shaft_boss_height + tol + 0.001
            boss = prog.cylinder(boss_r, boss_h, align=ALIGN_MIN_Z, tag="shaft_boss")
            boss = prog.locate(
                boss,
                pos=(servo.shaft_offset[0], servo.shaft_offset[1], body_z / 2 - 0.001),
            )
            shell = prog.cut(shell, boss)

        # -- Fastener holes at each mounting ear --
        from botcad.fasteners import HeadType, resolve_fastener

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

        # -- Connector port / cable slot --
        if servo.connector_pos is not None:
            wall_x = -outer_x / 2
            port = _emit_connector_port_ir(
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


@dataclass(frozen=True)
class HornDiscParams:
    """Dimensions for the metal horn disc (purchased part)."""

    radius: float  # outer radius of the disc
    thickness: float  # disc thickness along shaft axis
    center_z: float  # disc center Z in servo local frame
    center_xy: tuple[float, float]  # shaft XY in servo frame


def horn_disc_params(
    servo: ServoSpec, material_margin: float = 0.001
) -> HornDiscParams | None:
    """Compute horn disc dimensions from servo mounting point data.

    Returns None if the servo has no horn_mounting_points.
    """
    if not servo.horn_mounting_points:
        return None

    sx, sy, sz = servo.shaft_offset

    # Radius: max distance from shaft center to screw hole edge + margin
    max_r = 0.0
    for mp in servo.horn_mounting_points:
        dx = mp.pos[0] - sx
        dy = mp.pos[1] - sy
        r = (dx * dx + dy * dy) ** 0.5 + mp.diameter / 2
        if r > max_r:
            max_r = r
    radius = max_r + material_margin

    # Thickness: Z span from shaft face to top of horn screws.
    # Minimum 2mm — when all mounting points are at the same Z (on the horn
    # face), the span is 0 but the physical disc is ~2mm thick.
    max_z = max(mp.pos[2] for mp in servo.horn_mounting_points)
    thickness = max(max_z - sz, 0.002)

    # Horn sits on top of the shaft boss, not the body face.
    horn_base_z = sz + servo.shaft_boss_height
    center_z = horn_base_z + thickness / 2

    return HornDiscParams(
        radius=radius,
        thickness=thickness,
        center_z=center_z,
        center_xy=(sx, sy),
    )


# ── Coupler-style bracket: cradle (static) + coupler (moving) ──────────
#
# Cross-section looking from +X, Y horizontal, Z vertical:
#
#         (front horn +Z — exposed)
#
#     W   SSSSSSSSSS   W    <- side walls grip ±Y
#     W   SSSSSSSSSS   W    <- body mid-section
#     W===SSSSSSSSSS===W    <- ear shelf ledges
#     WWWWWWWWWWWWWWWWWWW    <- bottom wall (below ears)
#
#         (rear horn -Z — exposed)
#
# Cradle (W) is a shallow tray cupping the lower body from ±Y and below.
# Coupler (C) bridges both horn faces, rotating around the shaft.


def cradle_solid(servo: ServoSpec, spec: BracketSpec | None = None) -> ShapeScript:
    """Build a shallow-tray cradle as ShapeScript IR.

    Outer box - pocket - fastener holes - connector passage.
    """
    from botcad.shapescript.ops import ALIGN_MIN_Z
    from botcad.shapescript.program import ShapeScript

    if spec is None:
        spec = BracketSpec()

    prog = ShapeScript()

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall
    sx = servo.shaft_offset[0]

    ear_bottom_z = _ear_bottom_z(servo, wall)

    # -- Cradle extent in X --
    cradle_min_x = -body_x / 2 - tol - wall
    sweep_r = coupler_sweep_radius(servo, spec)
    if sweep_r > 0:
        cradle_max_x = sx - sweep_r - 0.001
    else:
        cradle_max_x = sx - 0.002
    cradle_lx = cradle_max_x - cradle_min_x
    cradle_cx = (cradle_min_x + cradle_max_x) / 2

    # -- Cradle extent in Y --
    outer_ly = body_y + 2 * (tol + wall)

    # -- Cradle extent in Z (shallow tray) --
    grip_margin = 0.004
    outer_top_z = -body_z / 2 + grip_margin
    outer_bottom_z = ear_bottom_z
    outer_lz = outer_top_z - outer_bottom_z
    outer_cz = (outer_top_z + outer_bottom_z) / 2

    outer = prog.box(cradle_lx, outer_ly, outer_lz, tag="cradle_outer")
    outer = prog.locate(outer, pos=(cradle_cx, 0, outer_cz))

    # -- Pocket (servo body cavity) --
    pocket_x = body_x + 2 * tol
    pocket_y = body_y + 2 * tol
    pocket_z = outer_lz + 0.002  # through full height
    pocket = prog.box(pocket_x, pocket_y, pocket_z, tag="cradle_pocket")
    pocket = prog.locate(pocket, pos=(0, 0, outer_cz))
    shell = prog.cut(outer, pocket)

    # -- Through-holes at each ear position --
    from botcad.fasteners import HeadType, resolve_fastener

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

    # -- Connector passage --
    if servo.connector_pos is not None:
        port = _emit_connector_port_ir(
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


def cradle_insertion_channel(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Cradle insertion channel as ShapeScript IR.

    The insertion channel is the volume subtracted from the parent body
    shell to create a path for servo insertion during assembly. It is NOT
    an enclosure around the servo or bracket. Its cross-section perpendicular
    to the insertion axis must clear the servo's cross-section on those axes.

    Covers the cradle footprint plus a 5x insertion channel in +X.
    """
    from botcad.shapescript.program import ShapeScript

    if spec is None:
        spec = BracketSpec()

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

    channel = prog.box(cradle_lx, outer_ly, outer_lz, tag="cradle_insertion_channel")
    channel = prog.locate(channel, pos=(cradle_cx, 0, outer_cz))

    prog.output_ref = channel
    return prog


def coupler_sweep_radius(servo: ServoSpec, spec: BracketSpec | None = None) -> float:
    """Compute the coupler's maximum sweep radius from shaft center.

    Used by the cradle to know how far it must retract. Returns the
    distance from shaft center to the farthest point of the coupler.

    With rounded plates, the sweep is the max of:
    - The horn circle clip radius (rounded plate boundary)
    - The wall strip corners (plate_max_x, ±plate_t/2)
    """
    import math

    if spec is None:
        spec = BracketSpec()
    plate_t = spec.coupler_thickness
    hole_margin = 0.002
    sx = servo.shaft_offset[0]
    body_x = servo.effective_body_dims[0]

    all_horn = _all_horn_points(servo)
    if not all_horn:
        return 0.0

    horn_clip_r = _horn_clip_radius(servo, spec)

    # Wall position
    holes_x = [mp.pos[0] - sx for mp in all_horn]
    body_plus_x = body_x / 2 - sx
    body_half_y = _body_collision_half_y(servo)
    body_diag = math.sqrt(body_plus_x**2 + body_half_y**2)
    wall_clear_x = body_diag + 0.001
    plate_max_x = max(
        max(holes_x) + hole_margin + plate_t,
        wall_clear_x + plate_t,
    )

    # Wall strip corners: (plate_max_x, ±horn_clip_r) — strip matches disc width
    wall_corner_r = math.sqrt(plate_max_x**2 + horn_clip_r**2)

    return max(horn_clip_r, wall_corner_r)


def coupler_max_rom_rad(servo: ServoSpec, spec: BracketSpec | None = None) -> float:
    """Compute the maximum safe ROM (half-range) for a coupler bracket.

    Returns the largest |angle| in radians where the coupler's side wall
    rib stays outside the servo body (including ear flanges) at all
    rotation angles within [-result, +result].

    The constraint: as the coupler rotates, its side wall rib (on the +X
    edge) sweeps through positions that may enter the servo body. The body
    is asymmetric (shaft offset from center), so the rib enters the body
    at large angles where sin(θ) isn't large enough to clear the body width.
    """
    import math

    if spec is None:
        spec = BracketSpec()

    sx = servo.shaft_offset[0]
    body_x, body_y, _body_z = servo.effective_body_dims

    # Body envelope from shaft center (including ear flanges)
    body_plus_x = body_x / 2 - sx
    body_minus_x = body_x / 2 + sx
    body_half_y = _body_collision_half_y(servo)

    # Wall geometry
    plate_t = spec.coupler_thickness
    body_diag = math.sqrt(body_plus_x**2 + body_half_y**2)
    wall_clear_x = body_diag + 0.001
    hole_margin = 0.002
    all_horn = _all_horn_points(servo)
    if not all_horn:
        return 0.0
    holes_x = [mp.pos[0] - sx for mp in all_horn]
    plate_max_x = max(
        max(holes_x) + hole_margin + plate_t,
        wall_clear_x + plate_t,
    )
    wall_inner = plate_max_x - plate_t

    horn_clip_r = _horn_clip_radius(servo, spec)
    wall_half_y = horn_clip_r

    # Wall corners in coupler local frame (XY only)
    rib_corners = [
        (wall_inner, -wall_half_y),
        (wall_inner, wall_half_y),
        (plate_max_x, -wall_half_y),
        (plate_max_x, wall_half_y),
    ]

    # Binary search for max safe angle
    lo, hi = 0.0, math.pi
    for _ in range(50):  # converge to ~1e-15 rad precision
        mid = (lo + hi) / 2
        safe = True
        for theta in [mid, -mid]:
            ct, st = math.cos(theta), math.sin(theta)
            for cx, cy in rib_corners:
                rx = cx * ct - cy * st
                ry = cx * st + cy * ct
                # Inside body if within all bounds
                if -body_minus_x < rx < body_plus_x and -body_half_y < ry < body_half_y:
                    safe = False
                    break
            if not safe:
                break
        if safe:
            lo = mid
        else:
            hi = mid

    return lo


def coupler_solid(servo: ServoSpec, spec: BracketSpec | None = None) -> ShapeScript:
    """Build a C-shaped coupler as ShapeScript IR.

    Front plate + rear plate + side wall + horn holes + boss clearance,
    with D-clip plate shaping.
    """
    import math

    from botcad.fasteners import resolve_fastener
    from botcad.shapescript.ops import ALIGN_MAX_Z, ALIGN_MIN_Z, Align3
    from botcad.shapescript.program import ShapeScript

    if spec is None:
        spec = BracketSpec()

    prog = ShapeScript()

    tol = spec.tolerance
    sx, sy, sz = servo.shaft_offset
    body_x, body_y, body_z = servo.effective_body_dims

    # -- Collect horn hole positions in shaft-centered frame --
    front_holes = []
    if servo.horn_mounting_points:
        for mp in servo.horn_mounting_points:
            front_holes.append((mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz))

    rear_holes = []
    if servo.rear_horn_mounting_points:
        for mp in servo.rear_horn_mounting_points:
            rear_holes.append((mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz))

    if not front_holes or not rear_holes:
        # Degenerate case - tiny placeholder box
        tiny = prog.box(0.001, 0.001, 0.001, tag="coupler_placeholder")
        prog.output_ref = tiny
        return prog

    all_holes = front_holes + rear_holes

    # -- Key Z coordinates (in shaft frame) --
    front_z = front_holes[0][2]
    rear_z = rear_holes[0][2]

    # -- Plate extent in XY --
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

    # -- Shaft boss clearance --
    boss_clear_r = (
        servo.shaft_boss_radius + tol + 0.001 if servo.shaft_boss_radius > 0 else 0
    )

    horn_clip_r = _horn_clip_radius(servo, spec)

    def _clip_plate_to_d_shape(plate_ref, clip_r, z_base, z_align_max):
        """Clip plate to D-shape: semicircle (-X) + rectangle (+X)."""
        clip_h = plate_t + 0.002
        z_str = "max" if z_align_max else "min"
        align_z = ALIGN_MAX_Z if z_align_max else ALIGN_MIN_Z
        align_xmin_z = Align3(x="min", z=z_str)
        clip_z = z_base + 0.001 if z_align_max else z_base - 0.001

        # Big cylinder covering entire plate
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
        clip_cut = prog.cut(clip_outer, keep)
        return prog.cut(plate_ref, clip_cut)

    # -- Front plate --
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

    # -- Rear plate --
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

    # -- Side wall --
    wall_z_lo = rear_z - plate_t
    wall_z_hi = front_z + plate_t
    wall_lz = wall_z_hi - wall_z_lo
    wall_cz = (wall_z_lo + wall_z_hi) / 2
    wall_cx = plate_max_x - plate_t / 2
    wall_width = horn_clip_r * 2

    side_wall = prog.box(plate_t, wall_width, wall_lz, tag="side_wall")
    side_wall = prog.locate(side_wall, pos=(wall_cx, plate_cy, wall_cz))

    # -- Fuse all three pieces --
    shell = prog.fuse(front_plate, side_wall)
    shell = prog.fuse(shell, rear_plate)

    # -- Front horn holes --
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

    # -- Rear horn holes --
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


# ── Convenience wrappers: execute ShapeScript IR and return a Solid ────────
#
# These exist so callers that need a build123d Solid can get one without
# knowing about ShapeScript. They import OcctBackend lazily to avoid
# circular imports.


def _exec_ir(prog: ShapeScript):
    """Execute ShapeScript IR and return the output Solid."""
    from botcad.shapescript.backend_occt import OcctBackend

    result = OcctBackend().execute(prog)
    return result.shapes[prog.output_ref.id]


def bracket_insertion_channel_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Execute bracket_insertion_channel IR and return a Solid."""
    return _exec_ir(bracket_insertion_channel(servo, spec))


def bracket_solid_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Execute bracket_solid IR and return a Solid."""
    return _exec_ir(bracket_solid(servo, spec))


def cradle_insertion_channel_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Execute cradle_insertion_channel IR and return a Solid."""
    return _exec_ir(cradle_insertion_channel(servo, spec))


def cradle_solid_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Execute cradle_solid IR and return a Solid."""
    return _exec_ir(cradle_solid(servo, spec))


def coupler_solid_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Execute coupler_solid IR and return a Solid."""
    return _exec_ir(coupler_solid(servo, spec))


def servo_solid(servo: ServoSpec):
    """Execute servo_script IR and return a Solid."""
    from botcad.shapescript.emit_servo import servo_script

    return _exec_ir(servo_script(servo))
