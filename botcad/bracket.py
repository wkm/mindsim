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


@dataclass(frozen=True)
class BracketSpec:
    """Parameters controlling bracket geometry around a servo."""

    wall: float = 0.0025  # 2.5mm wall thickness
    tolerance: float = 0.0003  # 0.3mm clearance per side for FDM
    shaft_clearance: float = 0.001  # 1mm extra radius around horn
    cable_slot_width: float = 0.010  # 10mm wide cable exit
    cable_slot_height: float = 0.006  # 6mm tall cable exit


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

    bd = servo.body_dimensions
    if bd[0] == 0.0:
        bd = servo.dimensions

    body_x, body_y, body_z = bd
    tol = spec.tolerance
    wall = spec.wall

    # --- Ear geometry ---
    ear_bottom_z = -body_z / 2
    if servo.mounting_ears:
        min_ear_z = min(ear.pos[2] for ear in servo.mounting_ears)
        max_hole_d = max(ear.diameter for ear in servo.mounting_ears)
        ear_bottom_z = min_ear_z - max_hole_d / 2 - wall

    # --- Shaft boss extra height ---
    boss_extra = 0.0
    if servo.shaft_boss_height > 0:
        boss_extra = servo.shaft_boss_height + tol

    # --- Outer box dimensions ---
    outer_x = body_x + 2 * (tol + wall)
    outer_y = body_y + 2 * (tol + wall)
    outer_top_z = body_z / 2 + boss_extra + wall + insertion_clearance
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
    return outer, ear_bottom_z, bd


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
    # Horn diameter: approximate from horn mounting points spread
    horn_radius = 0.011  # ~22mm diameter default for STS3215 horn
    if servo.horn_mounting_points:
        # Find max radial distance from shaft center
        sx, sy, sz = servo.shaft_offset
        max_r = 0.0
        for mp in servo.horn_mounting_points:
            dx = mp.pos[0] - sx
            dy = mp.pos[1] - sy
            r = (dx * dx + dy * dy) ** 0.5 + mp.diameter / 2
            if r > max_r:
                max_r = r
        horn_radius = max_r + spec.shaft_clearance

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
        # Ears are in two Y-rows (±Y). Build one shelf per Y-side.
        ear_y_groups: dict[str, list] = {}
        for ear in servo.mounting_ears:
            y_key = "pos" if ear.pos[1] > 0 else "neg"
            ear_y_groups.setdefault(y_key, []).append(ear)

        for _side, ears in ear_y_groups.items():
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

    return shell


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
