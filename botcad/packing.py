"""Packing solver — body sizing, component placement, mass/inertia computation.

For each body in the kinematic tree:
1. Collect all mounted components and child joint servos
2. Compute minimum bounding volume with clearance padding
3. Place components using heuristic positions (center, bottom, front, etc.)
4. Check for AABB collisions between components
5. Compute composite mass, center of mass, and inertia tensor
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from botcad.component import Vec3

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot

log = logging.getLogger(__name__)

_POSITION_AXES: dict[str, Vec3] = {
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "front": (0.0, 1.0, 0.0),
    "back": (0.0, -1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "center": (0.0, 0.0, 1.0),
}


def solve_packing(bot: Bot) -> None:
    """Solve packing for all bodies in the bot."""
    for body in bot.all_bodies:
        _solve_body(body)


def _solve_body(body: Body) -> None:
    """Solve packing for a single body."""
    # Internal components that need to fit inside this body
    internal_items: list[tuple[str, Vec3, float]] = []
    for mount in body.mounts:
        internal_items.append(
            (mount.label, mount.placed_dimensions, mount.component.mass)
        )

    # Servo center positions — the body needs structural material reaching
    # to each servo center (not the shaft/joint position). The bracket
    # protrudes from the body edge to hold the servo, so the body only
    # needs to reach where the bracket starts.
    from botcad.geometry import servo_placement

    joint_positions: list[Vec3] = []
    for j in body.joints:
        center, quat = servo_placement(
            j.servo.shaft_offset, j.servo.shaft_axis, j.axis, j.pos
        )
        j.solved_servo_center = center
        j.solved_servo_quat = quat
        joint_positions.append(center)

    if not internal_items and not joint_positions and body.explicit_dimensions is None:
        _compute_mass_inertia(body)
        return

    if body.explicit_dimensions is None:
        # Start from shape-derived minimum (respects outer_r, length, etc.)
        shape_dims = body._shape_default_dimensions()
        solved = _compute_bounding_box(internal_items, body.padding, joint_positions)
        # Take the max of shape-derived and solver-computed per axis
        body.solved_dimensions = (
            max(shape_dims[0], solved[0]),
            max(shape_dims[1], solved[1]),
            max(shape_dims[2], solved[2]),
        )

    dims = body.dimensions
    for mount in body.mounts:
        mount.resolved_pos = _resolve_position(
            mount.position, mount.placed_dimensions, dims
        )
        mount.resolved_insertion_axis = _resolve_insertion_axis(
            mount.position, mount.insertion_axis
        )

    _check_internal_overlaps(body)
    _compute_mass_inertia(body)


def _compute_bounding_box(
    items: list[tuple[str, Vec3, float]],
    padding: float,
    joint_positions: list[Vec3] | None = None,
) -> Vec3:
    """Compute minimum bounding box for a body.

    The box must:
    - Fit all internal components (side-by-side, not stacked)
    - Extend to reach all child joint attachment points (the chassis
      needs structural material from center to each joint)
    """
    p2 = padding * 2

    # Start with internal component extents (half-extents from center)
    if items:
        half_x = max(d[0] for _, d, _ in items) / 2
        half_y = max(d[1] for _, d, _ in items) / 2
        half_z = max(d[2] for _, d, _ in items) / 2
    else:
        half_x = half_y = half_z = 0.005

    # Track min/max extents from joint positions — the body must
    # have structural material reaching to each joint
    min_x = -half_x
    max_x = half_x
    min_y = -half_y
    max_y = half_y
    min_z = -half_z
    max_z = half_z

    if joint_positions:
        for jx, jy, jz in joint_positions:
            min_x = min(min_x, jx)
            max_x = max(max_x, jx)
            min_y = min(min_y, jy)
            max_y = max(max_y, jy)
            min_z = min(min_z, jz)
            max_z = max(max_z, jz)

    return (
        (max_x - min_x) + p2,
        (max_y - min_y) + p2,
        (max_z - min_z) + p2,
    )


def _resolve_position(
    position: str | Vec3,
    component_dims: Vec3,
    body_dims: Vec3,
) -> Vec3:
    """Resolve a heuristic position to concrete coordinates."""
    if isinstance(position, tuple):
        return position

    bx, by, bz = body_dims
    cx, cy, cz = component_dims

    match position:
        case "center":
            return (0.0, 0.0, 0.0)
        case "bottom":
            return (0.0, 0.0, -bz / 2 + cz / 2)
        case "top":
            return (0.0, 0.0, bz / 2 - cz / 2)
        case "front":
            return (0.0, by / 2 - cy / 2, 0.0)
        case "back":
            return (0.0, -by / 2 + cy / 2, 0.0)
        case "left":
            return (-bx / 2 + cx / 2, 0.0, 0.0)
        case "right":
            return (bx / 2 - cx / 2, 0.0, 0.0)
        case _:
            return (0.0, 0.0, 0.0)


def _resolve_insertion_axis(
    position: str | Vec3,
    explicit_axis: Vec3 | None,
) -> Vec3:
    """Resolve component insertion axis from mount position heuristic.

    Explicit axis always wins. Otherwise derive from the position keyword.
    """
    if explicit_axis is not None:
        return explicit_axis

    if isinstance(position, tuple):
        return (0.0, 0.0, 1.0)

    return _POSITION_AXES.get(position, (0.0, 0.0, 1.0))


def _compute_mass_inertia(body: Body) -> None:
    """Compute composite mass, CoM, and inertia tensor for a body.

    Uses parallel axis theorem to combine component inertias.
    """
    total_mass = 0.0
    com_x, com_y, com_z = 0.0, 0.0, 0.0

    # Collect mass contributions from mounts
    mass_items: list[tuple[Vec3, float, Vec3]] = []  # (pos, mass, dims)

    for mount in body.mounts:
        m = mount.component.mass
        p = mount.resolved_pos
        d = mount.placed_dimensions
        mass_items.append((p, m, d))
        total_mass += m
        com_x += p[0] * m
        com_y += p[1] * m
        com_z += p[2] * m

    # Add servo masses from child joints
    for joint in body.joints:
        m = joint.servo.mass
        p = joint.pos
        d = joint.servo.dimensions
        mass_items.append((p, m, d))
        total_mass += m
        com_x += p[0] * m
        com_y += p[1] * m
        com_z += p[2] * m

    # Add structural mass (shell/frame) based on body dimensions
    dims = body.dimensions
    # Estimate structural mass: thin-walled box, ~1mm wall, density ~1200 kg/m³ (PLA)
    wall_thickness = 0.001
    density = 1200.0
    surface_area = 2 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2])
    structural_mass = surface_area * wall_thickness * density

    total_mass += structural_mass
    # Structural mass centered at body origin
    # (com_x, com_y, com_z already accumulated, structural is at origin)

    if total_mass > 0:
        com_x /= total_mass
        com_y /= total_mass
        com_z /= total_mass
    else:
        total_mass = 0.01  # minimum mass

    body.solved_mass = total_mass
    body.solved_com = (com_x, com_y, com_z)

    # Provisional — overwritten by CAD geometry in build_cad()
    # Include structural body as a mass item (centered at origin)
    from botcad.geometry import parallel_axis_inertia

    all_mass_items = mass_items + [((0.0, 0.0, 0.0), structural_mass, dims)]
    ixx, iyy, izz, ixy, ixz, iyz = parallel_axis_inertia(
        all_mass_items, (com_x, com_y, com_z)
    )

    # Ensure minimum inertia (MuJoCo needs positive values)
    min_i = 1e-8
    ixx = max(ixx, min_i)
    iyy = max(iyy, min_i)
    izz = max(izz, min_i)

    # fullinertia format: Ixx Iyy Izz Ixy Ixz Iyz
    body.solved_inertia = (ixx, iyy, izz, -ixy, -ixz, -iyz)


# ── Internal overlap detection ──


def _check_internal_overlaps(body: Body) -> None:
    """Check for AABB overlaps between all items inside a body and log warnings."""
    for a, b, overlap in find_internal_overlaps(body):
        log.warning(
            "Packing overlap in body '%s': %s and %s overlap by "
            "%.1fmm x %.1fmm x %.1fmm",
            body.name,
            a,
            b,
            overlap[0] * 1000,
            overlap[1] * 1000,
            overlap[2] * 1000,
        )


def find_internal_overlaps(body: Body) -> list[tuple[str, str, Vec3]]:
    """Find AABB overlaps between all items inside a body.

    Items include mounted components (Pi, battery, camera) and servo bodies
    from child joints.  An overlap means the design can't be physically
    assembled — components would occupy the same space.

    Returns list of (label_a, label_b, overlap_extent) tuples.
    """
    from botcad.geometry import rotate_vec

    # Build list of (label, center, half_extents) for every internal item
    items: list[tuple[str, Vec3, Vec3]] = []

    # Mounted components — resolved_pos is the center, dims are axis-aligned
    for mount in body.mounts:
        d = mount.placed_dimensions
        half = (d[0] / 2, d[1] / 2, d[2] / 2)
        items.append((mount.label, mount.resolved_pos, half))

    # Servo bodies from child joints
    for joint in body.joints:
        servo = joint.servo
        center = joint.solved_servo_center
        quat = joint.solved_servo_quat
        # Compute AABB of the rotated servo box by rotating the 8 corners
        # and taking min/max per axis.
        bd = servo.effective_body_dims
        hx, hy, hz = bd[0] / 2, bd[1] / 2, bd[2] / 2
        corners = [
            (sx * hx, sy * hy, sz * hz)
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        rotated = [rotate_vec(quat, c) for c in corners]
        aabb_half = (
            max(abs(r[0]) for r in rotated),
            max(abs(r[1]) for r in rotated),
            max(abs(r[2]) for r in rotated),
        )
        items.append((f"{joint.name} servo", center, aabb_half))

    # Pairwise AABB overlap check
    overlaps: list[tuple[str, str, Vec3]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            label_a, pos_a, half_a = items[i]
            label_b, pos_b, half_b = items[j]
            if _aabb_overlap(pos_a, half_a, pos_b, half_b):
                extent = _overlap_extent(pos_a, half_a, pos_b, half_b)
                overlaps.append((label_a, label_b, extent))
    return overlaps


def _aabb_overlap(pos_a: Vec3, half_a: Vec3, pos_b: Vec3, half_b: Vec3) -> bool:
    """Check if two axis-aligned bounding boxes overlap."""
    for i in range(3):
        if abs(pos_a[i] - pos_b[i]) >= half_a[i] + half_b[i]:
            return False
    return True


def _overlap_extent(pos_a: Vec3, half_a: Vec3, pos_b: Vec3, half_b: Vec3) -> Vec3:
    """Compute overlap extent per axis (assumes boxes do overlap)."""
    return tuple(  # type: ignore[return-value]
        max(0.0, (half_a[i] + half_b[i]) - abs(pos_a[i] - pos_b[i])) for i in range(3)
    )
