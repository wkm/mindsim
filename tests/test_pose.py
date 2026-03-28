"""Tests for Pose type, euler_to_quat, fastener_pose, and pose helpers."""

from __future__ import annotations

import math

import pytest

from botcad.component import MountPoint
from botcad.geometry import (
    POSE_IDENTITY,
    Pose,
    euler_to_quat,
    fastener_pose,
    pose_compose,
    pose_transform_dir,
    pose_transform_point,
    quat_to_euler,
)


def _approx_vec3(v, expected, *, abs=1e-6):
    assert v[0] == pytest.approx(expected[0], abs=abs)
    assert v[1] == pytest.approx(expected[1], abs=abs)
    assert v[2] == pytest.approx(expected[2], abs=abs)


def _approx_quat(q, expected, *, abs=1e-6):
    """Compare quaternions, handling sign ambiguity (q and -q represent same rotation)."""
    # If dot product is negative, flip one before comparing
    dot = (
        q[0] * expected[0]
        + q[1] * expected[1]
        + q[2] * expected[2]
        + q[3] * expected[3]
    )
    sign = 1.0 if dot >= 0 else -1.0
    for i in range(4):
        assert sign * q[i] == pytest.approx(expected[i], abs=abs), (
            f"quat component {i}: {q} != {expected}"
        )


# ---------------------------------------------------------------------------
# euler_to_quat
# ---------------------------------------------------------------------------


class TestEulerToQuat:
    def test_identity(self):
        q = euler_to_quat((0.0, 0.0, 0.0))
        _approx_quat(q, (1.0, 0.0, 0.0, 0.0))

    @pytest.mark.parametrize(
        "euler",
        [
            (-90.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (0.0, -90.0, 0.0),
            (0.0, 90.0, 0.0),
            (180.0, 0.0, 0.0),
            (0.0, 0.0, 90.0),
        ],
    )
    def test_round_trip(self, euler):
        """euler_to_quat -> quat_to_euler should recover the original angles."""
        q = euler_to_quat(euler)
        recovered = quat_to_euler(q)
        _approx_vec3(recovered, euler, abs=1e-4)

    def test_known_value_rx90(self):
        """90 deg about X: q = (cos(45), sin(45), 0, 0)."""
        q = euler_to_quat((90.0, 0.0, 0.0))
        s = math.sqrt(2) / 2
        _approx_quat(q, (s, s, 0.0, 0.0))

    def test_known_value_rz90(self):
        """90 deg about Z: q = (cos(45), 0, 0, sin(45))."""
        q = euler_to_quat((0.0, 0.0, 90.0))
        s = math.sqrt(2) / 2
        _approx_quat(q, (s, 0.0, 0.0, s))


# ---------------------------------------------------------------------------
# Pose dataclass
# ---------------------------------------------------------------------------


class TestPose:
    def test_frozen(self):
        p = Pose((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
        with pytest.raises(AttributeError):
            p.pos = (0.0, 0.0, 0.0)  # type: ignore[misc]

    def test_identity(self):
        assert POSE_IDENTITY.pos == (0.0, 0.0, 0.0)
        assert POSE_IDENTITY.quat == (1.0, 0.0, 0.0, 0.0)

    def test_identity_transform(self):
        pt = (1.0, 2.0, 3.0)
        result = pose_transform_point(POSE_IDENTITY, pt)
        _approx_vec3(result, pt)

    def test_translation_only(self):
        p = Pose((10.0, 20.0, 30.0), (1.0, 0.0, 0.0, 0.0))
        result = pose_transform_point(p, (1.0, 2.0, 3.0))
        _approx_vec3(result, (11.0, 22.0, 33.0))

    def test_rotation_only(self):
        """90 deg around Z: (1,0,0) -> (0,1,0)."""
        q = euler_to_quat((0.0, 0.0, 90.0))
        p = Pose((0.0, 0.0, 0.0), q)
        result = pose_transform_point(p, (1.0, 0.0, 0.0))
        _approx_vec3(result, (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# pose_transform_point
# ---------------------------------------------------------------------------


class TestPoseTransformPoint:
    def test_rotation_plus_translation(self):
        """90 deg Z rotation then translate by (5,0,0)."""
        q = euler_to_quat((0.0, 0.0, 90.0))
        p = Pose((5.0, 0.0, 0.0), q)
        result = pose_transform_point(p, (1.0, 0.0, 0.0))
        # rotated (1,0,0) -> (0,1,0), then + (5,0,0) -> (5,1,0)
        _approx_vec3(result, (5.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# pose_transform_dir
# ---------------------------------------------------------------------------


class TestPoseTransformDir:
    def test_ignores_translation(self):
        """Direction transform should ignore the position component."""
        q = euler_to_quat((0.0, 0.0, 90.0))
        p = Pose((100.0, 200.0, 300.0), q)
        result = pose_transform_dir(p, (1.0, 0.0, 0.0))
        _approx_vec3(result, (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# pose_compose
# ---------------------------------------------------------------------------


class TestPoseCompose:
    def test_pure_translations(self):
        a = Pose((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        b = Pose((0.0, 2.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        result = pose_compose(a, b)
        _approx_vec3(result.pos, (1.0, 2.0, 0.0))
        _approx_quat(result.quat, (1.0, 0.0, 0.0, 0.0))

    def test_rotation_then_translation(self):
        """Parent rotates 90 Z, child translates (1,0,0) in local frame.
        In world: child pos is parent_rot * (1,0,0) = (0,1,0)."""
        q = euler_to_quat((0.0, 0.0, 90.0))
        parent = Pose((0.0, 0.0, 0.0), q)
        child = Pose((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        result = pose_compose(parent, child)
        _approx_vec3(result.pos, (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# fastener_pose
# ---------------------------------------------------------------------------


class TestFastenerPose:
    def test_insertion_neg_z_head_faces_pos_z(self):
        """axis=(0,0,-1) means screw inserts downward, head faces +Z."""
        mp = MountPoint(
            label="m1", pos=(0.01, 0.02, 0.0), diameter=0.002, axis=(0.0, 0.0, -1.0)
        )
        result = fastener_pose(POSE_IDENTITY, mp)
        _approx_vec3(result.pos, (0.01, 0.02, 0.0))
        # head_dir = -axis = (0,0,1) = +Z, so rotation_between(+Z, +Z) = identity
        _approx_quat(result.quat, (1.0, 0.0, 0.0, 0.0))

    def test_insertion_pos_z_head_faces_neg_z(self):
        """axis=(0,0,1) means screw inserts upward, head faces -Z (flipped)."""
        mp = MountPoint(
            label="m2", pos=(0.0, 0.0, 0.0), diameter=0.002, axis=(0.0, 0.0, 1.0)
        )
        result = fastener_pose(POSE_IDENTITY, mp)
        # head_dir = (0,0,-1), rotation_between(+Z, -Z) = 180 deg
        # Check that +Z is rotated to -Z
        from botcad.geometry import rotate_vec

        z_out = rotate_vec(result.quat, (0.0, 0.0, 1.0))
        _approx_vec3(z_out, (0.0, 0.0, -1.0))

    def test_position_from_parent(self):
        """Parent offset should be applied to mount point position."""
        parent = Pose((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
        mp = MountPoint(
            label="m3", pos=(0.1, 0.0, 0.0), diameter=0.002, axis=(0.0, 0.0, -1.0)
        )
        result = fastener_pose(parent, mp)
        _approx_vec3(result.pos, (1.1, 2.0, 3.0))

    def test_rotated_parent(self):
        """Parent rotated 90 about Z: mount at (1,0,0) -> world (0,1,0)."""
        q = euler_to_quat((0.0, 0.0, 90.0))
        parent = Pose((0.0, 0.0, 0.0), q)
        mp = MountPoint(
            label="m4", pos=(1.0, 0.0, 0.0), diameter=0.002, axis=(0.0, 0.0, -1.0)
        )
        result = fastener_pose(parent, mp)
        _approx_vec3(result.pos, (0.0, 1.0, 0.0))
