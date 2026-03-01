"""Shared geometry utilities for servo placement.

Pure math module — no MuJoCo or build123d dependencies.
Used by both the MuJoCo emitter (green servo boxes) and the CAD emitter
(servo pocket cutouts + mounting standoffs).
"""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z)


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

    Returns (center_pos, orientation_quat) where:
    - center_pos: where the servo geometric center goes in the parent body frame
    - orientation_quat: rotation to apply to the servo (w, x, y, z)
    """
    # Rotation that takes servo shaft_axis -> joint axis
    q = rotation_between(servo_shaft_axis, joint_axis)

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


def _normalize(v: Vec3) -> Vec3:
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if mag < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)
