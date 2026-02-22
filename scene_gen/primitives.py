"""Primitive geometry types for scene generation.

These map directly to MuJoCo geom types. Concepts compose primitives
to build furniture and other objects.

Coordinate convention:
    - Z-up (MuJoCo default)
    - Object origin at center of footprint on the floor (z=0 is bottom)
    - A table top at 0.75m height has pos=(0, 0, 0.75)

Size convention (matches MuJoCo):
    - BOX: (half_x, half_y, half_z)
    - CYLINDER: (radius, half_height, 0)  -- aligned along Z axis
    - SPHERE: (radius, 0, 0)
    - CAPSULE: (radius, half_height, 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import mujoco
import numpy as np


class Placement(Enum):
    """Where a concept prefers to be placed in a room."""

    WALL = "wall"  # Against a wall, back facing wall
    CENTER = "center"  # Interior of room
    CORNER = "corner"  # Near a corner


class GeomType(IntEnum):
    """MuJoCo geom types for primitive shapes."""

    BOX = mujoco.mjtGeom.mjGEOM_BOX
    CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
    SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
    CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
    ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID


@dataclass(frozen=True)
class Prim:
    """A single primitive shape positioned relative to an object's origin.

    Attributes:
        geom_type: Shape type (box, cylinder, sphere, etc.)
        size: Size parameters — meaning depends on geom_type (see module doc)
        pos: Position relative to object origin (x, y, z)
        rgba: Color and opacity (r, g, b, a), values in [0, 1]
        euler: Rotation in radians (roll, pitch, yaw), default (0, 0, 0)
    """

    geom_type: GeomType
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Common material colors (use in concepts for consistent look)
# ---------------------------------------------------------------------------

WOOD_LIGHT = (0.76, 0.60, 0.42, 1.0)
WOOD_MEDIUM = (0.60, 0.45, 0.30, 1.0)
WOOD_DARK = (0.45, 0.30, 0.15, 1.0)
WOOD_BIRCH = (0.85, 0.75, 0.60, 1.0)

METAL_GRAY = (0.55, 0.55, 0.55, 1.0)
METAL_DARK = (0.30, 0.30, 0.30, 1.0)
METAL_CHROME = (0.75, 0.75, 0.78, 1.0)

FABRIC_BLUE = (0.25, 0.35, 0.55, 1.0)
FABRIC_RED = (0.65, 0.20, 0.15, 1.0)
FABRIC_GREEN = (0.20, 0.45, 0.25, 1.0)
FABRIC_GRAY = (0.45, 0.45, 0.45, 1.0)

PLASTIC_WHITE = (0.90, 0.90, 0.88, 1.0)
PLASTIC_BLACK = (0.15, 0.15, 0.15, 1.0)

FABRIC_BEIGE = (0.78, 0.72, 0.62, 1.0)
FABRIC_BROWN = (0.50, 0.35, 0.22, 1.0)
FABRIC_WHITE = (0.92, 0.90, 0.87, 1.0)
FABRIC_CREAM = (0.95, 0.92, 0.84, 1.0)

RUG_BURGUNDY = (0.55, 0.12, 0.15, 1.0)
RUG_NAVY = (0.12, 0.15, 0.30, 1.0)
RUG_SAGE = (0.55, 0.62, 0.50, 1.0)
RUG_TAN = (0.72, 0.65, 0.52, 1.0)

LAMP_WARM = (0.95, 0.90, 0.70, 0.9)
LAMP_WHITE = (0.95, 0.95, 0.90, 0.85)

# ---------------------------------------------------------------------------
# Quaternion utilities (for placing rotated primitives)
# ---------------------------------------------------------------------------


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler angles (XYZ extrinsic) to quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def rotate_vec_z(v: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rotate a 3D vector around the Z axis by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = v
    return np.array([x * c - y * s, x * s + y * c, z])


# ---------------------------------------------------------------------------
# Footprint / bounding box utilities
# ---------------------------------------------------------------------------


def footprint(prims: tuple[Prim, ...]) -> tuple[float, float]:
    """Compute XY half-extents of the bounding box around a set of prims.

    Returns (half_x, half_y) — the smallest axis-aligned rectangle (before
    world rotation) that contains all prims.  Used for overlap checking.
    """
    if not prims:
        return (0.0, 0.0)

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for p in prims:
        # XY extent depends on geom type
        if p.geom_type == GeomType.BOX:
            hx, hy = p.size[0], p.size[1]
        elif p.geom_type in (GeomType.CYLINDER, GeomType.CAPSULE, GeomType.SPHERE):
            hx = hy = p.size[0]  # radius
        elif p.geom_type == GeomType.ELLIPSOID:
            hx, hy = p.size[0], p.size[1]
        else:
            hx = hy = p.size[0]

        min_x = min(min_x, p.pos[0] - hx)
        max_x = max(max_x, p.pos[0] + hx)
        min_y = min(min_y, p.pos[1] - hy)
        max_y = max(max_y, p.pos[1] + hy)

    return ((max_x - min_x) / 2, (max_y - min_y) / 2)


def obb_overlaps(
    cx_a: float,
    cy_a: float,
    hx_a: float,
    hy_a: float,
    rot_a: float,
    cx_b: float,
    cy_b: float,
    hx_b: float,
    hy_b: float,
    rot_b: float,
    margin: float = 0.1,
) -> bool:
    """2D oriented-bounding-box overlap test (Separating Axis Theorem).

    Each box is defined by center (cx, cy), half-extents (hx, hy), and
    rotation angle (radians, around Z).  *margin* is added to every
    half-extent so objects don't sit flush against each other.
    """
    hx_a += margin
    hy_a += margin
    hx_b += margin
    hy_b += margin

    ca, sa = np.cos(rot_a), np.sin(rot_a)
    cb, sb = np.cos(rot_b), np.sin(rot_b)

    # Local axes for each box
    ax_a = np.array([ca, sa])
    ay_a = np.array([-sa, ca])
    ax_b = np.array([cb, sb])
    ay_b = np.array([-sb, cb])

    d = np.array([cx_b - cx_a, cy_b - cy_a])

    # Test each of the 4 potential separating axes
    for axis in (ax_a, ay_a, ax_b, ay_b):
        proj_d = abs(np.dot(d, axis))
        proj_a = hx_a * abs(np.dot(ax_a, axis)) + hy_a * abs(np.dot(ay_a, axis))
        proj_b = hx_b * abs(np.dot(ax_b, axis)) + hy_b * abs(np.dot(ay_b, axis))
        if proj_d > proj_a + proj_b:
            return False  # separating axis found — no overlap

    return True  # no separator — boxes overlap
