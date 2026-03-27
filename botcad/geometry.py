"""Shared geometry utilities for servo placement.

Pure math module — no MuJoCo or build123d dependencies.
Used by both the MuJoCo emitter (green servo boxes) and the CAD emitter
(servo pocket cutouts + mounting standoffs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from botcad.component import Vec3

Quat = tuple[float, float, float, float]  # (w, x, y, z)


@dataclass(frozen=True)
class MountRotation:
    """Design-time rotation of a component on its mounting surface."""

    yaw: float = 0.0  # degrees around component local Z


MOUNT_NO_ROTATION = MountRotation()
MOUNT_YAW_90 = MountRotation(yaw=90.0)

# Named Euler rotations (degrees) — used by face rotation, camera orientation
EULER_RX_NEG90: tuple[float, float, float] = (-90.0, 0.0, 0.0)
EULER_RX_POS90: tuple[float, float, float] = (90.0, 0.0, 0.0)
EULER_RX_180: tuple[float, float, float] = (180.0, 0.0, 0.0)
EULER_RY_NEG90: tuple[float, float, float] = (0.0, -90.0, 0.0)
EULER_RY_POS90: tuple[float, float, float] = (0.0, 90.0, 0.0)
EULER_IDENTITY: tuple[float, float, float] = (0.0, 0.0, 0.0)

# Named direction vectors — mount face normals
DIR_POS_X: tuple[float, float, float] = (1.0, 0.0, 0.0)
DIR_NEG_X: tuple[float, float, float] = (-1.0, 0.0, 0.0)
DIR_POS_Y: tuple[float, float, float] = (0.0, 1.0, 0.0)
DIR_NEG_Y: tuple[float, float, float] = (0.0, -1.0, 0.0)
DIR_POS_Z: tuple[float, float, float] = (0.0, 0.0, 1.0)
DIR_NEG_Z: tuple[float, float, float] = (0.0, 0.0, -1.0)


def add_vec3(a: Vec3, b: Vec3) -> Vec3:
    """Element-wise addition of two 3-tuples."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def dist_vec3(a: Vec3, b: Vec3) -> float:
    """Euclidean distance between two 3D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def rotation_between(v_from: Vec3, v_to: Vec3) -> Quat:
    """Quaternion rotating unit vector v_from to v_to.

    Returns (w, x, y, z) in MuJoCo convention.
    """
    fx, fy, fz = _normalize(v_from)
    tx, ty, tz = _normalize(v_to)

    dot = fx * tx + fy * ty + fz * tz

    if dot > 0.9999:
        return (1.0, 0.0, 0.0, 0.0)  # identity

    if dot < -0.9999:
        # 180-degree rotation — pick an arbitrary perpendicular axis
        # Try cross with X first, fall back to Y if parallel
        ax, ay, az = (0.0, -fz, fy)
        if ax * ax + ay * ay + az * az < 1e-6:
            ax, ay, az = (fz, 0.0, -fx)
        ax, ay, az = _normalize((ax, ay, az))
        return (0.0, ax, ay, az)

    # Cross product = rotation axis (unnormalized, magnitude = sin(angle))
    cx = fy * tz - fz * ty
    cy = fz * tx - fx * tz
    cz = fx * ty - fy * tx

    # q = (1 + dot, cross) then normalize — avoids computing angle explicitly
    w = 1.0 + dot
    mag = math.sqrt(w * w + cx * cx + cy * cy + cz * cz)
    return (w / mag, cx / mag, cy / mag, cz / mag)


def rotate_vec(q: Quat, v: Vec3) -> Vec3:
    """Rotate vector v by quaternion q = (w, x, y, z)."""
    w, qx, qy, qz = q
    vx, vy, vz = v

    # q * v * q^-1 via the expanded formula
    # t = 2 * (q_xyz x v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return (
        vx + w * tx + (qy * tz - qz * ty),
        vy + w * ty + (qz * tx - qx * tz),
        vz + w * tz + (qx * ty - qy * tx),
    )


def quat_multiply(a: Quat, b: Quat) -> Quat:
    """Hamilton product of two quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def servo_placement(
    servo_shaft_offset: Vec3,
    servo_shaft_axis: Vec3,
    joint_axis: Vec3,
    joint_pos: Vec3,
) -> tuple[Vec3, Quat]:
    """Compute servo body center position and orientation at a joint.

    The servo's output shaft must coincide with the joint position, and
    the servo's shaft axis must align with the joint's rotation axis.

    For opposite axis signs (e.g. left wheel axis=-X, right wheel axis=+X),
    produces symmetric placement: the servo is rotated 180° around the
    vertical axis to face the other way, NOT flipped upside down.  This
    matches how you'd physically mount identical servos on opposite sides.

    Returns (center_pos, orientation_quat) where:
    - center_pos: where the servo geometric center goes in the parent body frame
    - orientation_quat: rotation to apply to the servo (w, x, y, z)
    """
    ax, ay, az = _normalize(joint_axis)

    # Detect negative principal axis — compute rotation for the positive
    # direction, then compose with a 180° flip that keeps the servo
    # right-side up (rotate around Z for horizontal axes, around Y for
    # vertical axes).
    is_negative = False
    if abs(ax) > 0.5 and ax < 0:
        is_negative = True
        positive_axis = (-ax, -ay, -az)
    elif abs(ay) > 0.5 and ay < 0:
        is_negative = True
        positive_axis = (-ax, -ay, -az)
    elif abs(az) > 0.5 and az < 0:
        is_negative = True
        positive_axis = (-ax, -ay, -az)
    else:
        positive_axis = (ax, ay, az)

    # Rotation that takes servo shaft_axis -> positive joint axis direction
    q = rotation_between(servo_shaft_axis, positive_axis)

    if is_negative:
        # Flip 180° around a perpendicular axis to reverse shaft direction
        # while keeping the servo body "upright" (same vertical orientation).
        if abs(az) > 0.5:
            # Vertical joint: flip around Y
            q_flip: Quat = (0.0, 0.0, 1.0, 0.0)
        else:
            # Horizontal joint (X or Y): flip around Z
            q_flip = (0.0, 0.0, 0.0, 1.0)
        q = quat_multiply(q_flip, q)

    # Rotate the shaft offset into the joint frame
    rotated_offset = rotate_vec(q, servo_shaft_offset)

    # servo_center = joint_pos - rotated_offset
    # (because joint_pos = servo_center + rotated_offset)
    center = (
        joint_pos[0] - rotated_offset[0],
        joint_pos[1] - rotated_offset[1],
        joint_pos[2] - rotated_offset[2],
    )

    return center, q


def quat_to_euler(q: Quat) -> tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to Euler angles (rx, ry, rz) in degrees.

    Uses intrinsic XYZ convention matching build123d's Location(pos, euler).
    Handles gimbal lock at pitch = +/-90 degrees.
    """
    w, x, y, z = q

    # Pitch (Y) — check first for gimbal lock
    # Intrinsic XYZ: R = Rx(rx) · Ry(ry) · Rz(rz)
    # sin(ry) = R[0][2] = 2(xz + wy)
    sinp = 2.0 * (w * y + x * z)
    sinp = max(-1.0, min(1.0, sinp))

    if abs(sinp) > 0.9999:
        # Gimbal lock: only rx ± rz is determined; set rx = 0.
        ry = math.copysign(math.pi / 2, sinp)
        rx = 0.0
        # R[1][0] = 2(xy + wz), R[1][1] = 1 - 2(x² + z²)
        if sinp > 0:
            rz = math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z))
        else:
            rz = math.atan2(-2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z))
    else:
        ry = math.asin(sinp)
        # rx from R[1][2] = -sin(rx)cos(ry), R[2][2] = cos(rx)cos(ry)
        sinr_cosp = 2.0 * (w * x - y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        rx = math.atan2(sinr_cosp, cosr_cosp)
        # rz from R[0][1] = -cos(ry)sin(rz), R[0][0] = cos(ry)cos(rz)
        siny_cosp = 2.0 * (w * z - x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        rz = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def parallel_axis_inertia(
    mass_items: list[tuple[Vec3, float, Vec3]],
    com: Vec3,
) -> tuple[float, float, float, float, float, float]:
    """Combine box-approximated component inertias about a common center of mass.

    Each item is (position, mass, box_dims). Returns (Ixx, Iyy, Izz, Ixy, Ixz, Iyz).
    """
    ixx = iyy = izz = 0.0
    ixy = ixz = iyz = 0.0

    for pos, mass, dims in mass_items:
        if mass <= 0:
            continue
        dx, dy, dz = dims
        # Box inertia about own center
        ix = mass * (dy**2 + dz**2) / 12.0
        iy = mass * (dx**2 + dz**2) / 12.0
        iz = mass * (dx**2 + dy**2) / 12.0
        # Parallel axis shift to COM
        rx = pos[0] - com[0]
        ry = pos[1] - com[1]
        rz = pos[2] - com[2]
        ixx += ix + mass * (ry**2 + rz**2)
        iyy += iy + mass * (rx**2 + rz**2)
        izz += iz + mass * (rx**2 + ry**2)
        ixy += mass * rx * ry
        ixz += mass * rx * rz
        iyz += mass * ry * rz

    return (ixx, iyy, izz, ixy, ixz, iyz)


def _normalize(v: Vec3) -> Vec3:
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if mag < 1e-12:
        import logging

        logging.getLogger(__name__).warning(
            "Normalizing near-zero vector %s, defaulting to Z-up", v
        )
        return (0.0, 0.0, 1.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)
