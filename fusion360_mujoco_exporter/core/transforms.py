"""
Coordinate transformation utilities for Fusion 360 to MuJoCo conversion.

Handles:
- Unit conversions (cm to m, kg*cm² to kg*m²)
- Matrix to quaternion conversion
- Coordinate frame transformations
"""

import math
from dataclasses import dataclass
from typing import Tuple, List

# Conversion factors
CM_TO_M = 0.01
MM_TO_M = 0.001
KG_CM2_TO_KG_M2 = 0.0001  # 1 kg*cm² = 0.0001 kg*m²


@dataclass
class Vector3:
    """3D vector."""
    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def scaled(self, factor: float) -> "Vector3":
        return Vector3(self.x * factor, self.y * factor, self.z * factor)

    def to_mjcf_string(self) -> str:
        return f"{self.x:.6g} {self.y:.6g} {self.z:.6g}"


@dataclass
class Quaternion:
    """Quaternion for rotation (w, x, y, z format - MuJoCo convention)."""
    w: float
    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.w, self.x, self.y, self.z)

    def to_mjcf_string(self) -> str:
        return f"{self.w:.6g} {self.x:.6g} {self.y:.6g} {self.z:.6g}"

    def is_identity(self, tolerance: float = 1e-6) -> bool:
        """Check if this quaternion represents no rotation."""
        return (
            abs(self.w - 1.0) < tolerance
            and abs(self.x) < tolerance
            and abs(self.y) < tolerance
            and abs(self.z) < tolerance
        )


@dataclass
class Transform:
    """Rigid body transform (position + orientation)."""
    position: Vector3
    rotation: Quaternion

    @classmethod
    def identity(cls) -> "Transform":
        return cls(
            position=Vector3(0.0, 0.0, 0.0),
            rotation=Quaternion(1.0, 0.0, 0.0, 0.0),
        )


def matrix3d_to_quaternion(matrix: List[float]) -> Quaternion:
    """
    Convert a Fusion 360 Matrix3D (as 16-element list) to quaternion.

    Fusion 360 Matrix3D is a 4x4 transformation matrix in row-major order:
    [m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33]

    The rotation is in the upper-left 3x3 submatrix.
    """
    # Extract rotation matrix elements
    m00, m01, m02 = matrix[0], matrix[1], matrix[2]
    m10, m11, m12 = matrix[4], matrix[5], matrix[6]
    m20, m21, m22 = matrix[8], matrix[9], matrix[10]

    # Convert rotation matrix to quaternion using Shepperd's method
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    # Normalize
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm > 0:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return Quaternion(w, x, y, z)


def matrix3d_to_position(matrix: List[float], scale: float = CM_TO_M) -> Vector3:
    """
    Extract position from Fusion 360 Matrix3D.

    The translation is in elements [3, 7, 11] (column 4, rows 1-3).
    """
    return Vector3(
        matrix[3] * scale,
        matrix[7] * scale,
        matrix[11] * scale,
    )


def matrix3d_to_transform(matrix: List[float], scale: float = CM_TO_M) -> Transform:
    """Convert a Fusion 360 Matrix3D to a Transform."""
    return Transform(
        position=matrix3d_to_position(matrix, scale),
        rotation=matrix3d_to_quaternion(matrix),
    )


def transform_point(point: Vector3, matrix: List[float]) -> Vector3:
    """Apply a 4x4 transformation matrix to a point."""
    x = matrix[0] * point.x + matrix[1] * point.y + matrix[2] * point.z + matrix[3]
    y = matrix[4] * point.x + matrix[5] * point.y + matrix[6] * point.z + matrix[7]
    z = matrix[8] * point.x + matrix[9] * point.y + matrix[10] * point.z + matrix[11]
    return Vector3(x, y, z)


def transform_vector(vector: Vector3, matrix: List[float]) -> Vector3:
    """Apply a 4x4 transformation matrix to a direction vector (ignores translation)."""
    x = matrix[0] * vector.x + matrix[1] * vector.y + matrix[2] * vector.z
    y = matrix[4] * vector.x + matrix[5] * vector.y + matrix[6] * vector.z
    z = matrix[8] * vector.x + matrix[9] * vector.y + matrix[10] * vector.z
    return Vector3(x, y, z)


def invert_matrix3d(matrix: List[float]) -> List[float]:
    """
    Invert a 4x4 transformation matrix (assuming it's a rigid body transform).

    For rigid transforms: R^-1 = R^T, t^-1 = -R^T * t
    """
    # Transpose the rotation part
    r00, r01, r02 = matrix[0], matrix[4], matrix[8]
    r10, r11, r12 = matrix[1], matrix[5], matrix[9]
    r20, r21, r22 = matrix[2], matrix[6], matrix[10]

    # Original translation
    tx, ty, tz = matrix[3], matrix[7], matrix[11]

    # Inverted translation: -R^T * t
    itx = -(r00 * tx + r01 * ty + r02 * tz)
    ity = -(r10 * tx + r11 * ty + r12 * tz)
    itz = -(r20 * tx + r21 * ty + r22 * tz)

    return [
        r00, r01, r02, itx,
        r10, r11, r12, ity,
        r20, r21, r22, itz,
        0.0, 0.0, 0.0, 1.0,
    ]


def multiply_matrices(a: List[float], b: List[float]) -> List[float]:
    """Multiply two 4x4 matrices."""
    result = [0.0] * 16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j]
    return result


def convert_inertia_to_body_frame(
    ixx: float,
    iyy: float,
    izz: float,
    ixy: float,
    iyz: float,
    ixz: float,
    com: Vector3,
    mass: float,
) -> Tuple[float, float, float]:
    """
    Convert inertia tensor from world frame to body-local frame at center of mass.

    Uses the parallel axis theorem: I_com = I_world - m * d²

    Returns diagonal inertia (assumes principal axes align with body frame).
    """
    # For simplicity, return diagonal inertia
    # Full implementation would compute principal axes
    d_sq_x = com.y ** 2 + com.z ** 2
    d_sq_y = com.x ** 2 + com.z ** 2
    d_sq_z = com.x ** 2 + com.y ** 2

    ixx_local = ixx - mass * d_sq_x
    iyy_local = iyy - mass * d_sq_y
    izz_local = izz - mass * d_sq_z

    # Ensure positive (numerical precision)
    ixx_local = max(ixx_local, 1e-10)
    iyy_local = max(iyy_local, 1e-10)
    izz_local = max(izz_local, 1e-10)

    return (ixx_local, iyy_local, izz_local)


def sanitize_name(name: str) -> str:
    """
    Sanitize a Fusion 360 component/joint name for use in MJCF.

    Removes or replaces special characters that aren't valid in MJCF names.
    """
    import re
    # Replace spaces, colons, parentheses with underscores
    sanitized = re.sub(r"[ :()/<>]", "_", name)
    # Remove version suffix like ":1" or "_v1"
    sanitized = re.sub(r"_\d+$", "", sanitized)
    sanitized = re.sub(r":\d+$", "", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized
