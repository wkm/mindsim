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
from functools import lru_cache
from typing import TYPE_CHECKING

from botcad.cad_utils import as_solid as _as_solid

if TYPE_CHECKING:
    from botcad.component import ServoSpec


def _cut_fastener_hole(shell, ear, ear_bottom_z: float, wall: float):
    """Cut a clearance hole + optional counterbore at a mounting ear position.

    Uses the fastener catalog for clearance diameter and head dimensions.
    Socket head cap screws get a counterbore so the head sits flush.
    """
    from build123d import Align, Cylinder, Location

    from botcad.fasteners import HeadType, resolve_fastener

    fspec = resolve_fastener(ear)
    hole_r = fspec.clearance_hole / 2
    hole_depth = abs(ear.pos[2] - ear_bottom_z) + wall + 0.002
    hole = Cylinder(hole_r, hole_depth, align=(Align.CENTER, Align.CENTER, Align.MIN))
    hole = hole.locate(Location((ear.pos[0], ear.pos[1], ear_bottom_z - 0.001)))
    shell = shell - hole

    if fspec.head_type == HeadType.SOCKET_HEAD_CAP:
        cb_r = fspec.head_diameter / 2 + 0.0002  # 0.2mm clearance
        cb_depth = fspec.head_height + 0.0005  # 0.5mm extra
        cb = Cylinder(cb_r, cb_depth, align=(Align.CENTER, Align.CENTER, Align.MIN))
        cb = cb.locate(Location((ear.pos[0], ear.pos[1], ear_bottom_z - 0.001)))
        shell = shell - cb

    return shell


def _fuse_servo_connector(body, servo: ServoSpec):
    """Fuse connector receptacles onto the servo body at each wire port.

    Each non-permanent wire port with a connector_type gets a receptacle
    fused at its position. The receptacle is rotated so pins run along X
    (servo long axis) and the mating cavity faces outward from the body.
    Permanent wires (soldered/molded) are skipped.
    """
    from build123d import Location

    from botcad.connectors import connector_spec, receptacle_solid

    for wp in servo.wire_ports:
        if not wp.connector_type or wp.permanent:
            continue
        try:
            cspec = connector_spec(wp.connector_type)
        except KeyError:
            continue

        rcpt = receptacle_solid(cspec)
        cx, cy, cz = wp.pos

        # Rotate receptacle: mating cavity faces outward from servo body,
        # and pins run along X (servo long axis). The 5264 connector's
        # default mating direction is +Z; we flip 180° for bottom-face
        # connectors and add 90° around Z so pins align with servo X.
        mx, my, mz = cspec.mating_direction
        if abs(mz) > 0.5:
            flip = 180 if cz < 0 else 0
            euler = (flip, 0, 90)  # 90° around Z so pins run along X
        elif abs(mx) > 0.5:
            euler = (0, -90 if cx < 0 else 90, 0)
        else:
            euler = (90 if cy < 0 else -90, 0, 0)

        rcpt_placed = rcpt.moved(Location((cx, cy, cz), euler))
        body = _as_solid(body.fuse(rcpt_placed))

    return body


def _connector_port(
    servo: ServoSpec,
    spec: BracketSpec,
    wall_center: tuple[float, float, float],
    exit_axis: tuple[float, float, float],
):
    """Build a shaped passage through a bracket wall for all connector plugs.

    Computes a single cut that covers all wire port positions with their
    connector envelopes + clearance. For servos with two side-by-side
    connectors (e.g. STS3215 daisy-chain), the passage spans both.

    Returns (cut_solid, None) positioned in servo-local frame,
    or (None, None) if no connector info is available.
    """
    from build123d import Align, Box, Location

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
        return None, None

    tol = spec.tolerance
    wall = spec.wall
    ax, ay, az = exit_axis

    # Compute bounding box of all connector envelopes in the passage plane
    # Each connector is at wp.pos with dimensions from cspec.body_dimensions
    # The connectors are rotated 90° (pins along X), so effective footprint
    # in the wall plane swaps dimensions accordingly.
    clearance = tol + 0.001  # per side

    # Accumulate min/max across all ports in the two axes perpendicular to exit
    if abs(ax) > 0.5:
        # Passage through X wall — footprint in Y-Z plane
        y_coords = []
        z_coords = []
        for wp, cspec in ports:
            bx, by, bz = cspec.body_dimensions
            # Connector rotated 90°: long dim (bx) runs along X, by along Y
            hw = max(bx, by) / 2 + clearance
            hh = max(min(bx, by), bz) / 2 + clearance
            y_coords.extend([wp.pos[1] - hw, wp.pos[1] + hw])
            z_coords.extend([wp.pos[2] - hh, wp.pos[2] + hh])
        cut_w = max(y_coords) - min(y_coords)
        cut_h = max(z_coords) - min(z_coords)
        center_y = (max(y_coords) + min(y_coords)) / 2
        center_z = (max(z_coords) + min(z_coords)) / 2
        passage_depth = wall + tol + 0.004
        cut = Box(
            passage_depth,
            cut_w,
            cut_h,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        cut_pos = (wall_center[0], center_y, center_z)
    elif abs(az) > 0.5:
        # Passage through Z wall — footprint in X-Y plane
        x_coords = []
        y_coords = []
        for wp, cspec in ports:
            bx, by, bz = cspec.body_dimensions
            hw = max(bx, by) / 2 + clearance
            hh = max(min(bx, by), bz) / 2 + clearance
            x_coords.extend([wp.pos[0] - hw, wp.pos[0] + hw])
            y_coords.extend([wp.pos[1] - hh, wp.pos[1] + hh])
        cut_w = max(x_coords) - min(x_coords)
        cut_h = max(y_coords) - min(y_coords)
        center_x = (max(x_coords) + min(x_coords)) / 2
        center_y = (max(y_coords) + min(y_coords)) / 2
        passage_depth = wall + tol + 0.004
        cut = Box(
            cut_w,
            cut_h,
            passage_depth,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        cut_pos = (center_x, center_y, wall_center[2])
    else:
        # Passage through Y wall — footprint in X-Z plane
        x_coords = []
        z_coords = []
        for wp, cspec in ports:
            bx, by, bz = cspec.body_dimensions
            hw = max(bx, by) / 2 + clearance
            hh = max(min(bx, by), bz) / 2 + clearance
            x_coords.extend([wp.pos[0] - hw, wp.pos[0] + hw])
            z_coords.extend([wp.pos[2] - hh, wp.pos[2] + hh])
        cut_w = max(x_coords) - min(x_coords)
        cut_h = max(z_coords) - min(z_coords)
        center_x = (max(x_coords) + min(x_coords)) / 2
        center_z = (max(z_coords) + min(z_coords)) / 2
        passage_depth = wall + tol + 0.004
        cut = Box(
            cut_w,
            passage_depth,
            cut_h,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        cut_pos = (center_x, wall_center[1], center_z)

    cut = cut.locate(Location(cut_pos))

    return cut, None


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


def _group_ears_by_y_side(ears) -> dict[str, list]:
    """Group mounting ears into +Y and -Y sides."""
    sides: dict[str, list] = {}
    for ear in ears:
        side = "pos" if ear.pos[1] > 0 else "neg"
        sides.setdefault(side, []).append(ear)
    return sides


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


def _bracket_outer(
    servo: ServoSpec, spec: BracketSpec, insertion_clearance: float = 0.0
):
    """Compute the bracket's outer box solid and ear_bottom_z.

    Shared between bracket_solid() and bracket_envelope().
    insertion_clearance extends the box upward (+Z, horn/shaft side) to cut
    a clear servo insertion path through the parent body shell.
    Returns (outer_solid, ear_bottom_z, body_dims).

    The bracket wraps ±X, ±Y, and -Z (unpowered horn side). The +Z face
    is open (just a clearance hole for the powered horn). The bracket
    extends from ear bottom up to body top + wall.
    """
    from build123d import Align, Box, Location

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall

    ear_bottom_z = _ear_bottom_z(servo, wall)

    # --- Outer box dimensions ---
    outer_x = body_x + 2 * (tol + wall)
    outer_y = body_y + 2 * (tol + wall)
    # Top: body top + wall. The pocket cuts through this to expose the
    # +Z face, but the outer box must extend above the pocket so the
    # solid stays watertight for OCCT boolean operations.
    outer_top_z = body_z / 2 + wall + insertion_clearance
    outer_bottom_z = ear_bottom_z
    outer_z = outer_top_z - outer_bottom_z

    outer_center_z = (outer_top_z + outer_bottom_z) / 2

    outer = Box(
        outer_x,
        outer_y,
        outer_z,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    outer = outer.locate(Location((0, 0, outer_center_z)))
    return outer, ear_bottom_z, (body_x, body_y, body_z)


@lru_cache(maxsize=32)
def bracket_envelope(servo: ServoSpec, spec: BracketSpec | None = None):
    """Return the bracket's outer box extended with an insertion channel.

    Used by the CAD emitter to cut the bracket footprint from the parent
    body shell before unioning the finished bracket in. The insertion
    channel extends 5x the bracket height above it (+Z) so the servo
    has a clear insertion path through the parent body shell.
    """
    if spec is None:
        spec = BracketSpec()
    if servo.name == "SCS0009":
        return _scs0009_bracket_envelope(servo, spec)
    # Insertion clearance must extend well past any parent body shell.
    # Use 5x bracket height — enough for bodies up to ~200mm.
    body_z = servo.effective_body_dims[2]
    ear_bot_z = _ear_bottom_z(servo, spec.wall)
    bracket_height = body_z / 2 + spec.wall - ear_bot_z
    outer, _, _ = _bracket_outer(servo, spec, insertion_clearance=bracket_height * 5)
    return outer


def bracket_envelope_ir(servo: ServoSpec, spec: BracketSpec | None = None) -> ShapeScript:
    """Bracket envelope as ShapeScript IR.

    Mirrors bracket_envelope() but emits ShapeScript ops instead of
    calling build123d directly. Used during migration to unify on IR.
    """
    from botcad.shapescript.program import ShapeScript

    if spec is None:
        spec = BracketSpec()

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


@lru_cache(maxsize=32)
def bracket_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Build a bracket solid for wheel/linear applications.

    Wraps ±X, ±Y, and -Z (unpowered horn side). The +Z face is open
    with a clearance hole for the powered horn disc + shaft boss.
    Fasteners go through the mounting ear holes from the -Z side.
    The -Z face has a wire cutout slot for the connector cable.

    The caller should first subtract bracket_envelope() from the parent
    body, then union this bracket in: ``(shell - envelope) + bracket``.

    Returns a build123d Solid centered on the servo body origin.
    """
    from build123d import Align, Box, Cylinder, Location

    if spec is None:
        spec = BracketSpec()

    if servo.name == "SCS0009":
        return _scs0009_bracket_solid(servo, spec)

    outer, ear_bottom_z, bd = _bracket_outer(servo, spec)
    body_x, body_y, body_z = bd
    tol = spec.tolerance
    wall = spec.wall

    # --- Servo body pocket ---
    # Pocket is the servo body cavity, open on +Z for insertion.
    pocket_x = body_x + 2 * tol
    pocket_y = body_y + 2 * tol
    # Extends from body bottom up through the +Z face (open top)
    pocket_z = body_z + tol + wall + 0.001
    pocket_center_z = -body_z / 2 + pocket_z / 2 - tol

    pocket = Box(
        pocket_x,
        pocket_y,
        pocket_z,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    pocket = pocket.locate(Location((0, 0, pocket_center_z)))

    shell = outer - pocket

    # --- +Z face: horn clearance hole ---
    # Large clearance hole for the powered horn disc + shaft boss.
    horn_radius = 0.011
    params = horn_disc_params(servo)
    if params is not None:
        horn_radius = params.radius + spec.shaft_clearance

    shaft_hole = Cylinder(
        horn_radius,
        wall + 0.002,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    shaft_hole = shaft_hole.locate(
        Location((servo.shaft_offset[0], servo.shaft_offset[1], body_z / 2 - 0.001))
    )
    shell = shell - shaft_hole

    # --- Shaft boss clearance (bearing housing above body top) ---
    if servo.shaft_boss_radius > 0 and servo.shaft_boss_height > 0:
        boss_r = servo.shaft_boss_radius + tol
        boss_h = servo.shaft_boss_height + tol + 0.001
        boss = Cylinder(
            boss_r,
            boss_h,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        boss = boss.locate(
            Location((servo.shaft_offset[0], servo.shaft_offset[1], body_z / 2 - 0.001))
        )
        shell = shell - boss

    # --- Through-holes at each ear position (clearance from catalog) ---
    for ear in servo.mounting_ears:
        shell = _cut_fastener_hole(shell, ear, ear_bottom_z, wall)

    # --- Connector passage (shaped cutout for plug + cable to pass through) ---
    if servo.connector_pos is not None:
        wall_x = -(body_x + 2 * (tol + wall)) / 2
        cut, _housing = _connector_port(
            servo,
            spec,
            wall_center=(wall_x, 0, 0),
            exit_axis=(-1.0, 0.0, 0.0),
        )
        if cut is not None:
            shell = shell - cut
        else:
            # Fallback to rectangular slot if no connector info
            cx, cy, cz = servo.connector_pos
            slot_w, slot_h = _cable_slot_dims(servo, spec)
            slot_depth = wall + tol + 0.002
            slot = Box(
                slot_depth,
                slot_w,
                slot_h,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            slot_x = wall_x + slot_depth / 2 - 0.001
            slot = slot.locate(Location((slot_x, cy, cz)))
            shell = shell - slot

    return _as_solid(shell)


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

    center_z = sz + thickness / 2

    return HornDiscParams(
        radius=radius,
        thickness=thickness,
        center_z=center_z,
        center_xy=(sx, sy),
    )


@lru_cache(maxsize=32)
def servo_solid(servo: ServoSpec):
    """Build a detailed solid representing the physical servo body.

    Dispatches to a form-factor-specific builder based on the servo name.
    Centered on body center (0,0,0) in servo local frame.
    """
    if servo.name == "SCS0009":
        return _scs0009_solid(servo)
    return _sts_series_solid(servo)


@lru_cache(maxsize=32)
def _sts_series_solid(servo: ServoSpec):
    """STS-series servo body (STS3215, STS3250, etc.).

    Aluminum mid-section with plastic top/bottom caps, bottom-mounted
    mounting flanges, dual-axis shaft with rear support bearing.

    Geometry matched to the official STS3215 STEP model (same outer
    shell for all STS variants).
    """
    from build123d import Align, Axis, Box, Cylinder, Location, fillet

    # Exact extents from reference CAD interrogation
    body_x = 0.0454
    body_y = 0.0248
    body_z = 0.0326
    r = 0.0040  # Match the large molded corner radii

    # Z-Planes (Relative to body center at Z=0)
    # Body spans Z = [-16.3, +16.3] mm
    z_top_surface = body_z / 2  # +16.3mm
    z_mid_top = z_top_surface - 0.0026  # +13.7mm
    z_mid_bot = z_mid_top - 0.0288  # -15.1mm
    z_cap_bot = z_mid_bot - 0.0012  # -16.3mm (body bottom)

    # 1. Middle Section (Aluminum)
    mid_h = z_mid_top - z_mid_bot
    middle = Box(
        body_x, body_y, mid_h, align=(Align.CENTER, Align.CENTER, Align.CENTER)
    )
    middle = middle.locate(Location((0, 0, (z_mid_top + z_mid_bot) / 2)))
    middle = _as_solid(fillet(middle.edges().filter_by(Axis.Z), r))

    # 2. Top Cap (Plastic)
    top_h = z_top_surface - z_mid_top
    top_cap = Box(body_x, body_y, top_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    top_cap = top_cap.locate(Location((0, 0, z_mid_top)))
    top_cap = _as_solid(fillet(top_cap.edges().filter_by(Axis.Z), r))

    # Raised pill step
    step_h = 0.0017
    step = Box(0.0200, 0.0200, step_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    step = step.locate(Location((0.0125, 0, z_top_surface)))
    step = _as_solid(fillet(step.edges().filter_by(Axis.Z), 0.002))
    top_cap = _as_solid(top_cap.fuse(step))

    # Output shaft boss
    sx, sy, _sz = servo.shaft_offset
    boss_h = 0.0015
    boss = Cylinder(0.0045, boss_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    boss = boss.locate(Location((sx, sy, z_top_surface + step_h)))
    top_cap = _as_solid(top_cap.fuse(boss))

    # 3. Bottom Cap & Flanges
    bot_h = z_mid_bot - z_cap_bot
    bottom_cap = Box(
        body_x, body_y, bot_h, align=(Align.CENTER, Align.CENTER, Align.MAX)
    )
    bottom_cap = bottom_cap.locate(Location((0, 0, z_mid_bot)))
    bottom_cap = _as_solid(fillet(bottom_cap.edges().filter_by(Axis.Z), r))

    # Mounting Flanges
    f_lx = 0.0404
    f_cx = -0.0005
    f_z_top = z_cap_bot
    f_z_bot = f_z_top - 0.0021
    f_h = f_z_top - f_z_bot

    if servo.mounting_ears:
        for _side, ears in _group_ears_by_y_side(servo.mounting_ears).items():
            f_w = 0.004
            f_cy = (
                (body_y / 2 - f_w / 2)
                if ears[0].pos[1] > 0
                else -(body_y / 2 - f_w / 2)
            )
            flange = Box(f_lx, f_w, f_h, align=(Align.CENTER, Align.CENTER, Align.MAX))
            flange = flange.locate(Location((f_cx, f_cy, f_z_top)))
            bottom_cap = _as_solid(bottom_cap.fuse(flange))

    # Support bearing boss
    rear_boss = Cylinder(0.003, 0.0013, align=(Align.CENTER, Align.CENTER, Align.MAX))
    rear_boss = rear_boss.locate(Location((sx, sy, z_cap_bot)))
    bottom_cap = _as_solid(bottom_cap.fuse(rear_boss))

    # Final Union
    body = _as_solid(middle.fuse(top_cap).fuse(bottom_cap))

    # Mounting Holes
    if servo.mounting_ears:
        for ear in servo.mounting_ears:
            h = Cylinder(
                ear.diameter / 2,
                0.020,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            h = h.locate(Location((ear.pos[0], ear.pos[1], f_z_bot)))
            body = _as_solid(body.cut(h))

    # Connector receptacle on the servo body (where wires plug in)
    if servo.connector_pos is not None:
        body = _fuse_servo_connector(body, servo)

    return body


@lru_cache(maxsize=32)
def _scs0009_solid(servo: ServoSpec):
    """SCS0009 micro servo body (SG90-style form factor).

    Classic micro servo shape: rectangular plastic body with mounting
    ears/tabs protruding from the sides at roughly 2/3 body height.
    Single output shaft on top, offset toward +X end.  No rear shaft.

    Dimensions from Feetech datasheet (SC-0090-C001):
        Body:  23.2 x 12.1 x 22.5 mm (L x W x H, no ears)
        Ears:  extend ±4.65mm beyond body in X, 2.5mm thick in Z
        Ear Z: positioned ~7.75mm below body top (flush with lower body)
        Total: 32.5 x 12.1 x 25.25 mm (with ears + shaft boss)
        Shaft: 20T spline, OD 3.95mm, offset +5.8mm from center in X

    Origin = geometric center of main body (no ears).
    """
    from build123d import Align, Axis, Box, Cylinder, Location, fillet

    body_x, body_y, body_z = servo.effective_body_dims
    r = 0.0010  # small fillet for plastic molding

    # Body center at Z=0, spans ±body_z/2
    z_top = body_z / 2  # +11.25mm

    # ── 1. Main body ──────────────────────────────────────────────
    body = Box(
        body_x,
        body_y,
        body_z,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    body = _as_solid(fillet(body.edges().filter_by(Axis.Z), r))

    # ── 2. Output shaft boss (single axis, no rear shaft) ────────
    sx, sy, _sz = servo.shaft_offset
    boss_r = servo.shaft_boss_radius  # ~1.98mm
    boss_h = servo.shaft_boss_height  # ~2.0mm
    boss = Cylinder(boss_r, boss_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    boss = boss.locate(Location((sx, sy, z_top)))
    body = _as_solid(body.fuse(boss))

    # ── 3. Mounting ears (side tabs, SG90-style) ──────────────────
    # Ears protrude in ±X from body sides, at roughly 2/3 height.
    # Each ear: ~4.65mm extension beyond body, full body_y width,
    # ~2.5mm thick in Z.
    ear_ext = 0.00465  # extension beyond body edge in X
    ear_thick = 0.0025  # thickness in Z
    ear_total_x = body_x + 2 * ear_ext  # ~32.5mm total

    # Ears sit with top aligned to a point ~7.75mm below body top
    # (i.e. about 3.5mm above body center for a 22.5mm body)
    ear_top_z = z_top - 0.00775  # ~3.5mm above center
    ear_bot_z = ear_top_z - ear_thick
    ear_cz = (ear_top_z + ear_bot_z) / 2

    ear = Box(
        ear_total_x,
        body_y,
        ear_thick,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    ear = ear.locate(Location((0, 0, ear_cz)))
    body = _as_solid(body.fuse(ear))

    # ── 4. Mounting holes through ears ────────────────────────────
    if servo.mounting_ears:
        for mp in servo.mounting_ears:
            hole = Cylinder(
                mp.diameter / 2,
                0.010,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            hole = hole.locate(Location((mp.pos[0], mp.pos[1], ear_cz)))
            body = _as_solid(body.cut(hole))

    # Connector receptacle on the servo body
    if servo.connector_pos is not None:
        body = _fuse_servo_connector(body, servo)

    return body


# ── SCS0009 micro servo bracket (simple wrap-around cradle) ────────────
#
# The SCS0009 is small enough that a single-piece bracket suffices.
# The servo drops in from +Z, the side ear tabs rest on ledges, and
# screws go down through the tabs into the bracket.  A shaft clearance
# hole on top lets the horn rotate freely.  Cable exits out the -Z face.
#
# Cross-section looking from +Y, X horizontal, Z vertical:
#
#     WWWW===WWWWWWWWWWWW       <- wall + shaft hole + wall
#     W   SSSSSSSSSSSS  W       <- servo body (shaft boss at +X)
#     W   SSSSSSSSSSSS  W
#     W===SSSSSSSSSSSS==W       <- ear tab ledges (ears sit here)
#     W   SSSSSSSSSSSS  W
#     W   SSSSSSSSSSSS  W
#     WWWWWWW====WWWWWWWW       <- wall + cable slot + wall
#


@lru_cache(maxsize=32)
def _scs0009_bracket_solid(servo: ServoSpec, spec: BracketSpec):
    """Build a bracket for the SCS0009 micro servo.

    The bracket is a U-shaped tray that wraps only the lower portion of
    the servo body — from the bottom up to the ear tabs.  The servo
    drops in from above, the wing tabs overhang the bracket walls, and
    screws through the tabs clamp it in place.  Everything above the
    tabs (upper body, shaft, horn) is fully exposed.

    Cross-section looking from +Y, X horizontal, Z vertical:

              (exposed — shaft + upper body)
        ====TTTTTTTTTTTT====    <- ear tabs rest on bracket walls
        W   SSSSSSSSSSSS   W    <- servo body (lower portion)
        W   SSSSSSSSSSSS   W
        WWWWWWWW====WWWWWWWW    <- bottom wall + cable slot
    """
    from build123d import Align, Box, Cylinder, Location

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall

    # Ear tab geometry (must match _scs0009_solid)
    ear_ext = 0.00465  # tab extension beyond body in ±X
    ear_thick = 0.0025  # tab thickness in Z
    ear_top_z = body_z / 2 - 0.00775  # top of ear tab

    # --- Bracket extent ---
    # X: wide enough for body + ear tabs + wall
    outer_x = body_x + 2 * ear_ext + 2 * wall
    # Y: body width + tolerance + wall
    outer_y = body_y + 2 * (tol + wall)
    # Z: from bottom of body up to just below the ear tabs.
    # The tabs sit on top of the bracket walls and overhang them.
    ear_bot_z = ear_top_z - ear_thick  # bottom face of the ear tab
    outer_top_z = ear_bot_z  # bracket walls stop right where the tabs begin
    outer_bot_z = -body_z / 2 - wall
    outer_z = outer_top_z - outer_bot_z
    outer_cz = (outer_top_z + outer_bot_z) / 2

    outer = Box(
        outer_x, outer_y, outer_z, align=(Align.CENTER, Align.CENTER, Align.CENTER)
    )
    outer = outer.locate(Location((0, 0, outer_cz)))

    # --- Body pocket (open on +Z for servo insertion) ---
    pocket_x = body_x + 2 * tol
    pocket_y = body_y + 2 * tol
    pocket_z = outer_z + 0.002  # through the full height (open top)
    pocket = Box(
        pocket_x, pocket_y, pocket_z, align=(Align.CENTER, Align.CENTER, Align.CENTER)
    )
    pocket = pocket.locate(Location((0, 0, outer_cz)))
    shell = outer - pocket

    # --- Through-holes at each ear position (clearance from catalog) ---
    from botcad.fasteners import resolve_fastener

    for ear in servo.mounting_ears:
        fspec = resolve_fastener(ear)
        hole = Cylinder(
            fspec.clearance_hole / 2,
            outer_z + 0.002,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        hole = hole.locate(Location((ear.pos[0], ear.pos[1], outer_cz)))
        shell = shell - hole

    # --- Connector passage (-Z face) ---
    if servo.connector_pos is not None:
        cut, _housing = _connector_port(
            servo,
            spec,
            wall_center=(0, 0, outer_bot_z),
            exit_axis=(0.0, 0.0, -1.0),
        )
        if cut is not None:
            shell = shell - cut
        else:
            _cx, cy, _cz = servo.connector_pos
            slot_w, slot_h = _cable_slot_dims(servo, spec)
            slot_depth = wall + tol + 0.002
            slot = Box(
                slot_w,
                slot_h,
                slot_depth,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            slot_z = outer_bot_z + slot_depth / 2 - 0.001
            slot = slot.locate(Location((0, cy, slot_z)))
            shell = shell - slot

    return _as_solid(shell)


@lru_cache(maxsize=32)
def _scs0009_bracket_envelope(servo: ServoSpec, spec: BracketSpec):
    """Envelope for cutting the SCS0009 bracket footprint from the parent body.

    Same outer box as the bracket, extended 5x bracket height in +Z for
    insertion clearance.
    """
    from build123d import Align, Box, Location

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall

    ear_ext = 0.00465
    ear_thick = 0.0025
    ear_top_z = body_z / 2 - 0.00775
    ear_bot_z = ear_top_z - ear_thick

    outer_x = body_x + 2 * ear_ext + 2 * wall
    outer_y = body_y + 2 * (tol + wall)
    bracket_height = ear_bot_z - (-body_z / 2 - wall)
    outer_top_z = ear_bot_z + bracket_height * 5  # 5x insertion clearance
    outer_bot_z = -body_z / 2 - wall
    outer_z = outer_top_z - outer_bot_z
    outer_cz = (outer_top_z + outer_bot_z) / 2

    outer = Box(
        outer_x, outer_y, outer_z, align=(Align.CENTER, Align.CENTER, Align.CENTER)
    )
    outer = outer.locate(Location((0, 0, outer_cz)))
    return outer


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


@lru_cache(maxsize=32)
def cradle_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Build a shallow-tray cradle for rotational joint applications.

    The cradle cups the servo from below, gripping the ±Y sides and
    bottom (-Z, below mounting ears). Both the +Z and -Z horn faces
    are fully exposed so the coupler can bridge them. Used as the
    static (parent-body) side of a coupler-style joint.

    Cross-section looking from +X, Y horizontal, Z vertical::

            (front horn +Z — exposed)

        W   SSSSSSSSSS   W    <- side walls grip ±Y
        W   SSSSSSSSSS   W    <- body mid-section
        W===SSSSSSSSSS===W    <- ear shelf ledges
        WWWWWWWWWWWWWWWWWWW    <- bottom wall (below ears)

            (rear horn -Z — exposed)

    Built in servo local frame (origin = body center, Z = shaft axis).
    """
    from build123d import Align, Box, Location

    if spec is None:
        spec = BracketSpec()

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall
    sx = servo.shaft_offset[0]

    ear_bottom_z = _ear_bottom_z(servo, wall)

    # --- Cradle extent in X ---
    # Same as before: wraps from -X body edge up to coupler sweep boundary.
    cradle_min_x = -body_x / 2 - tol - wall
    sweep_r = coupler_sweep_radius(servo, spec)
    if sweep_r > 0:
        cradle_max_x = sx - sweep_r - 0.001
    else:
        cradle_max_x = sx - 0.002
    cradle_lx = cradle_max_x - cradle_min_x
    cradle_cx = (cradle_min_x + cradle_max_x) / 2

    # --- Cradle extent in Y (full body width + wall) ---
    outer_ly = body_y + 2 * (tol + wall)

    # --- Cradle extent in Z (shallow tray) ---
    # Bottom: below mounting ears (ear_bottom_z)
    # Top: a few mm above the connector Z (bottom face of body) to
    # grip the body sides. The side walls extend from ear bottom up
    # to roughly the connector Z level + a grip margin.
    # connector_pos Z is at body bottom (-body_z/2); we go a few mm
    # above that for side wall grip.
    grip_margin = 0.004  # 4mm above body bottom face for side grip
    outer_top_z = -body_z / 2 + grip_margin
    outer_bottom_z = ear_bottom_z
    outer_lz = outer_top_z - outer_bottom_z
    outer_cz = (outer_top_z + outer_bottom_z) / 2

    outer = Box(
        cradle_lx,
        outer_ly,
        outer_lz,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    outer = outer.locate(Location((cradle_cx, 0, outer_cz)))

    # --- Pocket (servo body cavity, open on +X, +Z, and -Z) ---
    # The pocket removes the servo body volume from inside the tray.
    pocket_x = body_x + 2 * tol
    pocket_y = body_y + 2 * tol
    pocket_z = outer_lz + 0.002  # through full height for open top/bottom
    pocket = Box(
        pocket_x,
        pocket_y,
        pocket_z,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    pocket = pocket.locate(Location((0, 0, outer_cz)))
    shell = outer - pocket

    # --- Through-holes at each ear position (clearance from catalog) ---
    for ear in servo.mounting_ears:
        if ear.pos[0] > cradle_max_x + 0.001:
            continue
        shell = _cut_fastener_hole(shell, ear, ear_bottom_z, wall)

    # --- Connector passage (connector is at -X end) ---
    if servo.connector_pos is not None:
        cut, _housing = _connector_port(
            servo,
            spec,
            wall_center=(cradle_min_x, 0, 0),
            exit_axis=(-1.0, 0.0, 0.0),
        )
        if cut is not None:
            shell = shell - cut
        else:
            _cx, cy, cz = servo.connector_pos
            slot_w, slot_h = _cable_slot_dims(servo, spec)
            slot_depth = wall + tol + 0.002
            slot = Box(
                slot_depth,
                slot_w,
                slot_h,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            slot_x = cradle_min_x + slot_depth / 2 - 0.001
            slot = slot.locate(Location((slot_x, cy, cz)))
            shell = shell - slot

    return _as_solid(shell)


@lru_cache(maxsize=32)
def cradle_envelope(servo: ServoSpec, spec: BracketSpec | None = None):
    """Cradle envelope for cutting from parent body shell.

    Covers the cradle footprint plus a 5x insertion channel in +X
    so the servo can slide in from the shaft side.
    """
    from build123d import Align, Box, Location

    if spec is None:
        spec = BracketSpec()

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall
    sx = servo.shaft_offset[0]

    ear_bottom_z = _ear_bottom_z(servo, wall)

    cradle_min_x = -body_x / 2 - tol - wall
    # Extend +X by 5x the cradle X extent for insertion clearance.
    # Must extend well past the parent body to avoid near-tangent
    # OCCT boolean issues.
    cradle_nominal_max_x = sx - 0.002
    cradle_lx_nominal = cradle_nominal_max_x - cradle_min_x
    cradle_max_x = cradle_nominal_max_x + cradle_lx_nominal * 5
    cradle_lx = cradle_max_x - cradle_min_x
    cradle_cx = (cradle_min_x + cradle_max_x) / 2

    outer_ly = body_y + 2 * (tol + wall)
    # Match the shallow-tray Z extent from cradle_solid
    grip_margin = 0.004
    outer_top_z = -body_z / 2 + grip_margin
    outer_bottom_z = ear_bottom_z
    outer_lz = outer_top_z - outer_bottom_z
    outer_cz = (outer_top_z + outer_bottom_z) / 2

    envelope = Box(
        cradle_lx,
        outer_ly,
        outer_lz,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    envelope = envelope.locate(Location((cradle_cx, 0, outer_cz)))
    return envelope


def cradle_envelope_ir(servo: ServoSpec, spec: BracketSpec | None = None) -> ShapeScript:
    """Cradle envelope as ShapeScript IR.

    Mirrors cradle_envelope() but emits ShapeScript ops instead of
    calling build123d directly.
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

    envelope = prog.box(cradle_lx, outer_ly, outer_lz, tag="cradle_envelope")
    envelope = prog.locate(envelope, pos=(cradle_cx, 0, outer_cz))

    prog.output_ref = envelope
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


@lru_cache(maxsize=32)
def coupler_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Build a C-shaped coupler bridging front and rear horn faces.

    Side profile (looking from +Y):

              TTTTTTTTTTTT      <- top plate (front horn face)
              T  hh  hh  W     <- horn holes, W = side wall
              T          W
              T   servo  W     <- servo body lives in this gap
              T          W
              T  hh  hh  W     <- horn holes
              BBBBBBBBBBBB      <- bottom plate (rear horn face)

    The coupler is one printed C-shape:
    - Top plate bolted to front horn (screws down into horn)
    - Bottom plate bolted to rear horn (screws up into horn)
    - Side wall on the +X edge (past shaft, outside servo body) connecting them

    Built in shaft-centered frame (origin = shaft center, Z = shaft axis).
    """
    from build123d import Align, Box, Cylinder, Location

    if spec is None:
        spec = BracketSpec()

    tol = spec.tolerance
    sx, sy, sz = servo.shaft_offset
    body_x, body_y, body_z = servo.effective_body_dims

    # --- Collect horn hole positions in shaft-centered frame ---
    front_holes = []
    if servo.horn_mounting_points:
        for mp in servo.horn_mounting_points:
            front_holes.append((mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz))

    rear_holes = []
    if servo.rear_horn_mounting_points:
        for mp in servo.rear_horn_mounting_points:
            rear_holes.append((mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz))

    if not front_holes or not rear_holes:
        return Box(
            0.001, 0.001, 0.001, align=(Align.CENTER, Align.CENTER, Align.CENTER)
        )

    all_holes = front_holes + rear_holes

    # --- Key Z coordinates (in shaft frame) ---
    front_z = front_holes[0][2]  # front horn face Z
    rear_z = rear_holes[0][2]  # rear horn face Z

    # --- Plate extent in XY: cover all horn holes + material margin ---
    import math

    body_plus_x = body_x / 2 - sx  # body +X edge from shaft
    body_half_y = _body_collision_half_y(servo)
    body_diag = math.sqrt(body_plus_x**2 + body_half_y**2)
    wall_clear_x = body_diag + 0.001  # 1mm clearance for mesh tolerance

    plate_t = spec.coupler_thickness
    hole_margin = 0.002  # 2mm material around each hole
    plate_min_x = min(h[0] for h in all_holes) - hole_margin - plate_t
    plate_max_x = max(
        max(h[0] for h in all_holes) + hole_margin + plate_t,
        wall_clear_x + plate_t,  # side wall outer edge
    )
    plate_min_y = min(h[1] for h in all_holes) - hole_margin - plate_t
    plate_max_y = max(h[1] for h in all_holes) + hole_margin + plate_t
    plate_lx = plate_max_x - plate_min_x
    plate_ly = plate_max_y - plate_min_y
    plate_cx = (plate_min_x + plate_max_x) / 2
    plate_cy = (plate_min_y + plate_max_y) / 2

    # --- Shaft boss clearance ---
    boss_clear_r = (
        servo.shaft_boss_radius + tol + 0.001 if servo.shaft_boss_radius > 0 else 0
    )

    horn_clip_r = _horn_clip_radius(servo, spec)

    def _clip_plate_to_horn_circle(plate, clip_r, z_base, z_align_max):
        """Clip a plate to a D-shape: semicircle (-X) + rectangle (+X).

        The semicircle matches the horn bolt circle. The rectangle
        extends to the wall at constant width = disc diameter. No
        overlap zone where rectangle corners stick past the circle.
        """
        clip_h = plate_t + 0.002
        if z_align_max:
            clip_z = z_base + 0.001
            align_z = Align.MAX
        else:
            clip_z = z_base - 0.001
            align_z = Align.MIN
        # Big cylinder covering entire plate (what we subtract from)
        clip_outer = Cylinder(
            plate_lx,
            clip_h,
            align=(Align.CENTER, Align.CENTER, align_z),
        ).locate(Location((0, 0, clip_z)))
        # D-shape keep zone:
        # 1) Semicircle on -X side (rounded horn end)
        keep_circle = Cylinder(
            clip_r,
            clip_h + 0.002,
            align=(Align.CENTER, Align.CENTER, align_z),
        ).locate(Location((0, 0, clip_z)))
        # Cut away the +X half to get just the -X semicircle
        cut_pos_x = Box(
            clip_r + 0.001,
            clip_r * 2 + 0.004,
            clip_h + 0.004,
            align=(Align.MIN, Align.CENTER, align_z),
        ).locate(Location((0, plate_cy, clip_z)))
        keep_semi = keep_circle - cut_pos_x
        # 2) Rectangle from X=0 to plate_max_x, width = 2*clip_r
        keep_rect = Box(
            plate_max_x + 0.001,
            clip_r * 2 + 0.002,
            clip_h + 0.002,
            align=(Align.MIN, Align.CENTER, align_z),
        ).locate(Location((0, plate_cy, clip_z)))
        # Union: seamless at X=0 (both are ±clip_r wide there)
        keep = keep_semi.fuse(keep_rect)
        clip_cut = clip_outer - keep
        return plate - clip_cut

    # --- Build top plate (front horn face) ---
    front_plate = Box(
        plate_lx,
        plate_ly,
        plate_t,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    front_plate = front_plate.locate(Location((plate_cx, plate_cy, front_z)))
    if boss_clear_r > 0:
        boss_hole = Cylinder(
            boss_clear_r,
            plate_t + 0.002,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        boss_hole = boss_hole.locate(Location((0, 0, front_z - 0.001)))
        front_plate = front_plate - boss_hole
    front_plate = _clip_plate_to_horn_circle(
        front_plate, horn_clip_r, front_z, z_align_max=False
    )

    # --- Build bottom plate (rear horn face) ---
    rear_plate = Box(
        plate_lx,
        plate_ly,
        plate_t,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    rear_plate = rear_plate.locate(Location((plate_cx, plate_cy, rear_z)))
    if boss_clear_r > 0:
        boss_hole = Cylinder(
            boss_clear_r,
            plate_t + 0.002,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )
        boss_hole = boss_hole.locate(Location((0, 0, rear_z + 0.001)))
        rear_plate = rear_plate - boss_hole
    # Ear clearance already folded into horn_clip_r above
    rear_plate = _clip_plate_to_horn_circle(
        rear_plate, horn_clip_r, rear_z, z_align_max=True
    )

    # --- Side wall connecting both plates on +X edge ---
    # Same width as the disc (2 * horn_clip_r) for consistent profile.
    # Spans from bottom of rear plate to top of front plate.
    wall_z_lo = rear_z - plate_t
    wall_z_hi = front_z + plate_t
    wall_lz = wall_z_hi - wall_z_lo
    wall_cz = (wall_z_lo + wall_z_hi) / 2
    wall_cx = plate_max_x - plate_t / 2
    wall_width = horn_clip_r * 2

    side_wall = Box(
        plate_t,
        wall_width,
        wall_lz,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    side_wall = side_wall.locate(Location((wall_cx, plate_cy, wall_cz)))

    # Fuse all three pieces (wall overlaps both plates at their +X edges)
    shell = front_plate.fuse(side_wall).fuse(rear_plate)

    # --- Through-holes for front horn mounting ---
    from botcad.fasteners import resolve_fastener

    if servo.horn_mounting_points:
        fspec = resolve_fastener(servo.horn_mounting_points[0])
        horn_hole_r = fspec.clearance_hole / 2
    else:
        horn_hole_r = 0.00125  # fallback

    for hx, hy, _hz in front_holes:
        hole = Cylinder(
            horn_hole_r,
            plate_t + 0.002,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        hole = hole.locate(Location((hx, hy, front_z - 0.001)))
        shell = shell - hole

    # --- Through-holes for rear horn mounting ---
    if servo.rear_horn_mounting_points:
        rear_fspec = resolve_fastener(servo.rear_horn_mounting_points[0])
        rear_hole_r = rear_fspec.clearance_hole / 2
    else:
        rear_hole_r = horn_hole_r

    for hx, hy, _hz in rear_holes:
        hole = Cylinder(
            rear_hole_r,
            plate_t + 0.002,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )
        hole = hole.locate(Location((hx, hy, rear_z + 0.001)))
        shell = shell - hole

    return _as_solid(shell)
