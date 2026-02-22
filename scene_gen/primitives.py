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
from enum import IntEnum

import mujoco
import numpy as np


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
