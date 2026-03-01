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
            (mount.label, mount.component.dimensions, mount.component.mass)
        )

    # Child joint positions — the body needs structural material reaching
    # to each joint, but the servos themselves sit at the joint boundary,
    # not packed inside the body.
    joint_positions: list[Vec3] = [j.pos for j in body.joints]

    if not internal_items and not joint_positions and body.explicit_dimensions is None:
        _compute_mass_inertia(body)
        return

    if body.explicit_dimensions is None:
        body.solved_dimensions = _compute_bounding_box(
            internal_items, body.padding, joint_positions
        )

    dims = body.dimensions
    for mount in body.mounts:
        mount.resolved_pos = _resolve_position(
            mount.position, mount.component.dimensions, dims
        )

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
        d = mount.component.dimensions
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

    # Compute inertia tensor about CoM using parallel axis theorem
    ixx, iyy, izz = 0.0, 0.0, 0.0
    ixy, ixz, iyz = 0.0, 0.0, 0.0

    for pos, mass, dims in mass_items:
        if mass <= 0:
            continue
        dx, dy, dz = dims
        # Box inertia about own center
        ix = mass * (dy**2 + dz**2) / 12.0
        iy = mass * (dx**2 + dz**2) / 12.0
        iz = mass * (dx**2 + dy**2) / 12.0

        # Parallel axis: shift to body CoM
        rx = pos[0] - com_x
        ry = pos[1] - com_y
        rz = pos[2] - com_z
        ixx += ix + mass * (ry**2 + rz**2)
        iyy += iy + mass * (rx**2 + rz**2)
        izz += iz + mass * (rx**2 + ry**2)
        ixy += mass * rx * ry
        ixz += mass * rx * rz
        iyz += mass * ry * rz

    # Add structural inertia (thin-walled box about center)
    sx, sy, sz = dims
    ixx += structural_mass * (sy**2 + sz**2) / 12.0
    iyy += structural_mass * (sx**2 + sz**2) / 12.0
    izz += structural_mass * (sx**2 + sy**2) / 12.0

    # Ensure minimum inertia (MuJoCo needs positive values)
    min_i = 1e-8
    ixx = max(ixx, min_i)
    iyy = max(iyy, min_i)
    izz = max(izz, min_i)

    # fullinertia format: Ixx Iyy Izz Ixy Ixz Iyz
    body.solved_inertia = (ixx, iyy, izz, -ixy, -ixz, -iyz)
