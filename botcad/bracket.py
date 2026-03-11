"""Parametric servo bracket geometry.

Generates a build123d Solid for a 3D-printed bracket that holds a servo
motor inside a parent body segment. The bracket provides:

- A pocket for the servo body (with FDM tolerance clearance)
- A circular shaft opening on the +Z face for the horn
- Ear ledge shelves at mounting ear Z positions
- M3 through-holes from the -Z face for screw mounting
- A cable exit slot on the -X/-Z corner for the PA2.0 connector

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


def _as_solid(shape):
    """Extract a single Solid from a boolean result or shape list.

    build123d boolean ops (cut, union) and fillets can return Compound
    or ShapeList even when the result is a single solid. This extracts
    the Solid so further operations (like filleting or more booleans)
    work correctly.
    """
    from build123d import Compound, ShapeList, Solid

    # 1. Handle ShapeList (returned by fillet and some boolean ops)
    if isinstance(shape, ShapeList):
        if len(shape) == 1:
            return _as_solid(shape[0])
        # If it's a list of multiple solids, try to compound them
        if all(isinstance(s, Solid) for s in shape):
            return _as_solid(Compound(list(shape)))
        return shape

    # 2. Handle Solid
    if isinstance(shape, Solid):
        return shape

    # 3. Handle Compound
    if isinstance(shape, Compound):
        solids = shape.solids()
        if len(solids) == 1:
            return solids[0]
        return shape

    return shape


@dataclass(frozen=True)
class BracketSpec:
    """Parameters controlling bracket geometry around a servo."""

    wall: float = 0.0025  # 2.5mm wall thickness
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
    insertion_clearance extends the box upward (+Z) to cut a clear servo
    insertion path through the parent body shell.
    Returns (outer_solid, ear_bottom_z, body_dims).
    """
    from build123d import Align, Box, Location

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall

    ear_bottom_z = _ear_bottom_z(servo, wall)

    # --- Outer box dimensions ---
    # Boss clearance extends the top face so the bracket wall covers the
    # bearing housing cylinder that protrudes above the servo body.
    outer_x = body_x + 2 * (tol + wall)
    outer_y = body_y + 2 * (tol + wall)
    boss_clearance = (
        (servo.shaft_boss_height + tol) if servo.shaft_boss_height > 0 else 0.0
    )
    outer_top_z = body_z / 2 + boss_clearance + wall + insertion_clearance
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


def bracket_envelope(servo: ServoSpec, spec: BracketSpec | None = None):
    """Return the bracket's outer box extended with an insertion channel.

    Used by the CAD emitter to cut the bracket footprint from the parent
    body shell before unioning the finished bracket in. The insertion
    channel extends 200mm above the bracket so the servo has a clear path
    through the parent body shell (the CSG only removes material where
    shell exists).
    """
    if spec is None:
        spec = BracketSpec()
    outer, _, _ = _bracket_outer(servo, spec, insertion_clearance=0.200)
    return outer


def bracket_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Build a bracket solid in servo local frame.

    The bracket is an outer shell with the servo pocket, shaft clearance,
    screw holes, and cable exit already subtracted. The caller should
    first subtract bracket_envelope() from the parent body, then union
    this bracket in: ``(shell - envelope) + bracket``.

    Returns a build123d Solid centered on the servo body origin.
    """
    from build123d import Align, Box, Cylinder, Location

    if spec is None:
        spec = BracketSpec()

    outer, ear_bottom_z, bd = _bracket_outer(servo, spec)
    body_x, body_y, body_z = bd
    tol = spec.tolerance
    wall = spec.wall
    outer_x = body_x + 2 * (tol + wall)

    # --- Servo body pocket (open on +Z for insertion) ---
    pocket_x = body_x + 2 * tol
    pocket_y = body_y + 2 * tol
    # Pocket goes from body bottom up through the top face (open top)
    pocket_z = body_z + tol + wall + 0.001  # extends past top for clean cut
    pocket_center_z = -body_z / 2 + pocket_z / 2 - tol

    pocket = Box(
        pocket_x,
        pocket_y,
        pocket_z,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    pocket = pocket.locate(Location((0, 0, pocket_center_z)))

    shell = outer - pocket

    # --- Shaft clearance hole (circular, through +Z face) ---
    # Use horn_disc_params to get the horn radius (same loop as used for
    # the horn disc solid, avoiding duplication).
    horn_radius = 0.011  # ~22mm diameter default for STS3215 horn
    params = horn_disc_params(servo)
    if params is not None:
        horn_radius = params.radius + spec.shaft_clearance

    shaft_hole = Cylinder(
        horn_radius,
        wall + 0.002,  # through the top cap
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    shaft_hole = shaft_hole.locate(
        Location((servo.shaft_offset[0], servo.shaft_offset[1], body_z / 2 - 0.001))
    )
    shell = shell - shaft_hole

    # --- Shaft boss clearance (bearing housing cylinder above body top) ---
    if servo.shaft_boss_radius > 0 and servo.shaft_boss_height > 0:
        boss_r = servo.shaft_boss_radius + tol
        boss_h = servo.shaft_boss_height + tol + 0.001  # extra for clean cut
        boss = Cylinder(
            boss_r,
            boss_h,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        boss = boss.locate(
            Location((servo.shaft_offset[0], servo.shaft_offset[1], body_z / 2 - 0.001))
        )
        shell = shell - boss

    # --- Ear ledge shelves ---
    # The pocket cuts away the body area; we need thin shelves at the ear
    # positions to support the servo flanges. The shelves span the full
    # bracket width at the ear Z height.
    if servo.mounting_ears:
        for _side, ears in _group_ears_by_y_side(servo.mounting_ears).items():
            # Shelf spans from body bottom to just below the ear holes
            shelf_top = -body_z / 2  # body bottom face
            shelf_bottom = ear_bottom_z
            shelf_z = shelf_top - shelf_bottom

            if shelf_z <= 0:
                continue

            # Shelf spans full X width, narrow in Y (just the ear overhang)
            ear_y = abs(ears[0].pos[1])
            shelf_y_inner = body_y / 2  # inner edge = body side
            shelf_y_outer = ear_y + ears[0].diameter / 2 + wall
            shelf_width_y = shelf_y_outer - shelf_y_inner

            if shelf_width_y <= 0:
                continue

            shelf_center_y = (shelf_y_inner + shelf_y_outer) / 2
            if ears[0].pos[1] < 0:
                shelf_center_y = -shelf_center_y

            shelf_center_z = (shelf_top + shelf_bottom) / 2

            shelf = Box(
                outer_x,
                shelf_width_y,
                shelf_z,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            shelf = shelf.locate(Location((0, shelf_center_y, shelf_center_z)))
            # This shelf area is already part of the outer box below the
            # pocket, so it exists naturally. No union needed.

    # --- M3 through-holes at each ear position ---
    for ear in servo.mounting_ears:
        hole_r = ear.diameter / 2
        # Hole goes from bracket bottom through to above ear position
        hole_depth = abs(ear.pos[2] - ear_bottom_z) + wall + 0.002
        hole = Cylinder(
            hole_r,
            hole_depth,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        hole = hole.locate(Location((ear.pos[0], ear.pos[1], ear_bottom_z - 0.001)))
        shell = shell - hole

    # --- Cable exit slot ---
    # Slot on the -X, -Z corner for the PA2.0 connector + cable
    if servo.connector_pos is not None:
        cx, cy, cz = servo.connector_pos
        slot_w = spec.cable_slot_width
        slot_h = spec.cable_slot_height
        # Slot goes from connector position through the bracket wall on -X face
        slot_depth = wall + tol + 0.002  # through the wall
        slot = Box(
            slot_depth,
            slot_w,
            slot_h,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        # Position: at the -X wall, centered on connector Y, at connector Z
        slot_x = -outer_x / 2 + slot_depth / 2 - 0.001
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

    # Thickness: Z span from shaft face to top of horn screws
    max_z = max(mp.pos[2] for mp in servo.horn_mounting_points)
    thickness = max_z - sz

    center_z = sz + thickness / 2

    return HornDiscParams(
        radius=radius,
        thickness=thickness,
        center_z=center_z,
        center_xy=(sx, sy),
    )


def servo_solid(servo: ServoSpec):
    """Build a detailed solid representing the physical servo body.

    Pass 11: Absolute precision matching. Using the exact bounding planes
    measured from the official reference CAD. Centered on body center (0,0,0).
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

    return body


# ── Coupler-style bracket: cradle (static) + coupler (moving) ──────────
#
# Cross-section looking from +Y (front), X horizontal, Z vertical:
#
#              CCCCCCCCCC      <- coupler top plate (+Z horn face)
#  ---------    HHHHH  CC     <- front horn
#  -SSSSSSSSSSSSSSSSSS CC     <- servo body
#  -SSSSSSSSSSSSSSSSSS CC     <- servo body (shaft at +X end)
#  -SSSSSSSSSSSSSSSSSS CC     <- servo body
#  ---------    HHHHH  CC     <- rear horn
#              CCCCCCCCCC      <- coupler bottom plate (-Z horn face)
#
# Cradle (dashes) wraps the -X end (back, ears, connector).
# Coupler (C) wraps the +X end (shaft side), bridging both horns.


def cradle_solid(servo: ServoSpec, spec: BracketSpec | None = None):
    """Build a cradle wrapping the -X (back) end of the servo body.

    The cradle holds the servo by its mounting ears on the side opposite
    the shaft. It is open on the +X side so the coupler can reach the
    horns. Used as the static (parent-body) side of a coupler-style joint.

    Built in servo local frame (origin = body center, Z = shaft axis).
    """
    from build123d import Align, Box, Cylinder, Location

    if spec is None:
        spec = BracketSpec()

    body_x, body_y, body_z = servo.effective_body_dims
    tol = spec.tolerance
    wall = spec.wall
    sx, sy, _sz = servo.shaft_offset

    ear_bottom_z = _ear_bottom_z(servo, wall)

    # --- Cradle extent in X ---
    # Wraps from the -X body edge (with wall) up to a safe boundary
    # that won't intersect the coupler's swept volume. The coupler
    # rotates around Z at the shaft center; its plates extend inward
    # from the horn holes. The swept circle radius determines the
    # maximum +X the cradle can reach.
    cradle_min_x = -body_x / 2 - tol - wall

    # Use the coupler's actual sweep radius (plates + side wall) to
    # determine where the cradle must stop.
    sweep_r = coupler_sweep_radius(servo, spec)
    if sweep_r > 0:
        cradle_max_x = sx - sweep_r - 0.001  # 1mm clearance
    else:
        cradle_max_x = sx - 0.002  # fallback
    cradle_lx = cradle_max_x - cradle_min_x
    cradle_cx = (cradle_min_x + cradle_max_x) / 2

    # --- Cradle extent in Y (full body width + wall) ---
    outer_ly = body_y + 2 * (tol + wall)

    # --- Cradle extent in Z (full body height + ears) ---
    outer_top_z = body_z / 2 + tol + wall
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

    # --- Pocket (servo body cavity, open on +X side) ---
    # Extends past cradle_max_x to create the open side
    pocket_x = body_x + 2 * tol
    pocket_y = body_y + 2 * tol
    pocket_z = body_z + 2 * tol
    pocket = Box(
        pocket_x,
        pocket_y,
        pocket_z,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    shell = outer - pocket

    # --- M3 through-holes at each ear position ---
    for ear in servo.mounting_ears:
        # Only include ears within the cradle's X range
        if ear.pos[0] > cradle_max_x + 0.001:
            continue
        hole_r = ear.diameter / 2
        hole_depth = abs(ear.pos[2] - ear_bottom_z) + wall + 0.002
        hole = Cylinder(
            hole_r,
            hole_depth,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        hole = hole.locate(Location((ear.pos[0], ear.pos[1], ear_bottom_z - 0.001)))
        shell = shell - hole

    # --- Cable exit slot (connector is at -X end) ---
    if servo.connector_pos is not None:
        _cx, cy, cz = servo.connector_pos
        slot_w = spec.cable_slot_width
        slot_h = spec.cable_slot_height
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


def cradle_envelope(servo: ServoSpec, spec: BracketSpec | None = None):
    """Cradle envelope for cutting from parent body shell.

    Covers the cradle footprint plus a 200mm insertion channel in +X
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
    # Extend +X 200mm for insertion clearance through parent shell
    cradle_max_x = sx - 0.002 + 0.200
    cradle_lx = cradle_max_x - cradle_min_x
    cradle_cx = (cradle_min_x + cradle_max_x) / 2

    outer_ly = body_y + 2 * (tol + wall)
    outer_top_z = body_z / 2 + tol + wall
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

    # --- Through-holes for front horn mounting (M2.5) ---
    for hx, hy, _hz in front_holes:
        hole = Cylinder(
            0.00125,  # M2.5 clearance
            plate_t + 0.002,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        hole = hole.locate(Location((hx, hy, front_z - 0.001)))
        shell = shell - hole

    # --- Through-holes for rear horn mounting (M2.5) ---
    for hx, hy, _hz in rear_holes:
        hole = Cylinder(
            0.00125,  # M2.5 clearance
            plate_t + 0.002,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )
        hole = hole.locate(Location((hx, hy, rear_z + 0.001)))
        shell = shell - hole

    return _as_solid(shell)
