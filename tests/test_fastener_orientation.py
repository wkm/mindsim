"""Tests for rotation_between, quat_multiply, and fastener quaternion orientation.

Validates that the pure-math quaternion helpers produce correct results
for the fastener positioning pipeline (mounting ears, horn screws, etc.).
"""

from __future__ import annotations

import math

from botcad.geometry import quat_multiply, rotate_vec, rotation_between


def _approx_vec(v, expected, *, atol=1e-6):
    """Assert two 3-vectors are approximately equal."""
    for i in range(3):
        assert abs(v[i] - expected[i]) < atol, (
            f"Component {i}: {v[i]} != {expected[i]} (diff={abs(v[i] - expected[i])})"
        )


def _approx_quat(q, expected, *, atol=1e-6):
    """Assert two quaternions are approximately equal (up to sign flip)."""
    # q and -q represent the same rotation
    dot = sum(a * b for a, b in zip(q, expected))
    if dot < 0:
        q = tuple(-x for x in q)
    for i in range(4):
        assert abs(q[i] - expected[i]) < atol, (
            f"Quat component {i}: {q[i]} != {expected[i]} (diff={abs(q[i] - expected[i])})"
        )


def _quat_norm(q):
    return math.sqrt(sum(x * x for x in q))


# ---------------------------------------------------------------------------
# rotation_between
# ---------------------------------------------------------------------------


class TestRotationBetween:
    def test_rotation_between_identity(self):
        """Same vectors -> identity quaternion."""
        q = rotation_between((0, 0, 1), (0, 0, 1))
        _approx_quat(q, (1.0, 0.0, 0.0, 0.0))

    def test_rotation_between_identity_x(self):
        """Same X vectors -> identity quaternion."""
        q = rotation_between((1, 0, 0), (1, 0, 0))
        _approx_quat(q, (1.0, 0.0, 0.0, 0.0))

    def test_rotation_between_180_flip(self):
        """Opposite vectors -> 180 degree rotation."""
        q = rotation_between((0, 0, 1), (0, 0, -1))
        # Should be a unit quaternion
        assert abs(_quat_norm(q) - 1.0) < 1e-6
        # w should be ~0 for 180-degree rotation
        assert abs(q[0]) < 1e-6
        # Applying it to (0,0,1) should give (0,0,-1)
        result = rotate_vec(q, (0, 0, 1))
        _approx_vec(result, (0.0, 0.0, -1.0))

    def test_rotation_between_180_flip_x(self):
        """Opposite X vectors -> 180 degree rotation."""
        q = rotation_between((1, 0, 0), (-1, 0, 0))
        assert abs(_quat_norm(q) - 1.0) < 1e-6
        result = rotate_vec(q, (1, 0, 0))
        _approx_vec(result, (-1.0, 0.0, 0.0))

    def test_rotation_between_90_degrees(self):
        """Perpendicular vectors -> 90 degree rotation."""
        q = rotation_between((0, 0, 1), (1, 0, 0))
        assert abs(_quat_norm(q) - 1.0) < 1e-6
        result = rotate_vec(q, (0, 0, 1))
        _approx_vec(result, (1.0, 0.0, 0.0))

    def test_rotation_between_90_y_to_z(self):
        """Y axis to Z axis."""
        q = rotation_between((0, 1, 0), (0, 0, 1))
        result = rotate_vec(q, (0, 1, 0))
        _approx_vec(result, (0.0, 0.0, 1.0))

    def test_result_is_unit_quaternion(self):
        """All rotation_between results should be unit quaternions."""
        test_cases = [
            ((1, 0, 0), (0, 1, 0)),
            ((0, 0, 1), (0, 0, -1)),
            ((1, 1, 0), (0, 0, 1)),
            ((1, 1, 1), (-1, -1, -1)),
        ]
        for v_from, v_to in test_cases:
            q = rotation_between(v_from, v_to)
            assert abs(_quat_norm(q) - 1.0) < 1e-6, (
                f"Non-unit quaternion for {v_from} -> {v_to}: norm={_quat_norm(q)}"
            )


# ---------------------------------------------------------------------------
# quat_multiply
# ---------------------------------------------------------------------------


class TestQuatMultiply:
    def test_identity_left(self):
        """Identity * q = q."""
        q = (0.5, 0.5, 0.5, 0.5)
        result = quat_multiply((1, 0, 0, 0), q)
        _approx_quat(result, q)

    def test_identity_right(self):
        """q * identity = q."""
        q = (0.5, 0.5, 0.5, 0.5)
        result = quat_multiply(q, (1, 0, 0, 0))
        _approx_quat(result, q)

    def test_inverse(self):
        """q * q_conjugate = identity."""
        q = rotation_between((0, 0, 1), (1, 0, 0))
        q_conj = (q[0], -q[1], -q[2], -q[3])
        result = quat_multiply(q, q_conj)
        _approx_quat(result, (1.0, 0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Fastener orientation integration tests
# ---------------------------------------------------------------------------


class TestFastenerOrientation:
    def test_fastener_mounting_ear_orientation(self):
        """MountingEar axis (0,0,1) = insertion direction (shank goes +Z in local).

        With left_wheel servo quat (shaft along -X), fastener_pose() should
        produce the same result as the old manual computation.
        """
        from botcad.component import MountPoint
        from botcad.geometry import Pose, fastener_pose

        servo_quat = (0.707107, 0.0, -0.707107, 0.0)
        servo_pose = Pose((0.0, 0.0, 0.0), servo_quat)

        # New convention: insertion axis = +Z (shank goes up in servo-local)
        ear = MountPoint(
            "ear", pos=(0.0, 0.0, 0.0), diameter=0.003, axis=(0.0, 0.0, 1.0)
        )
        fp = fastener_pose(servo_pose, ear)

        assert abs(_quat_norm(fp.quat) - 1.0) < 1e-6

        # Head direction = +Z after fastener quat rotation
        head_dir = rotate_vec(fp.quat, (0.0, 0.0, 1.0))

        # For left_wheel servo, head should point roughly along ±X
        assert abs(head_dir[0] - 1.0) < 0.1 or abs(head_dir[0] - (-1.0)) < 0.1, (
            f"Expected head along +/-X, got {head_dir}"
        )

    def test_fastener_horn_orientation(self):
        """Horn mount axis (0,0,1) -> no flip needed, head faces along servo shaft.

        For a horn mounting point with axis (0,0,1), the axis_align is identity,
        so the fastener quaternion equals the servo quaternion.
        """
        servo_quat = (0.707107, 0.0, -0.707107, 0.0)
        horn_axis = (0.0, 0.0, 1.0)

        axis_align = rotation_between((0.0, 0.0, 1.0), horn_axis)
        final = quat_multiply(servo_quat, axis_align)

        # axis_align should be identity since source == target
        _approx_quat(axis_align, (1.0, 0.0, 0.0, 0.0))
        # So final should equal servo_quat
        _approx_quat(final, servo_quat)

        # Head direction = servo_quat applied to +Z
        head_dir = rotate_vec(final, (0.0, 0.0, 1.0))
        # For a -Y rotation servo, +Z maps to -X (shaft direction)
        _approx_vec(head_dir, (-1.0, 0.0, 0.0), atol=1e-4)

    def test_axis_align_compose_roundtrip(self):
        """Composing axis_align with its inverse gives identity rotation on the axis."""
        axis = (0.0, 0.0, -1.0)
        q_align = rotation_between((0.0, 0.0, 1.0), axis)
        q_reverse = rotation_between(axis, (0.0, 0.0, 1.0))

        roundtrip = quat_multiply(q_reverse, q_align)
        _approx_quat(roundtrip, (1.0, 0.0, 0.0, 0.0))
