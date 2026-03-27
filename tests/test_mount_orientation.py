"""Verify new mount quaternions match old lambda outputs, and packing completeness."""

from __future__ import annotations

import importlib
import sys

import pytest

from botcad.geometry import euler_to_quat, rotate_vec


def _vec_close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


class TestFaceQuatMatchesLambda:
    """Each euler_to_quat rotation must produce the same vector transform
    as the old _FACE_ROTATION lambda table in skeleton.py."""

    def test_front(self):
        # Old lambda: (x, y, z) -> (x, z, -y)
        q = euler_to_quat((-90.0, 0.0, 0.0))
        result = rotate_vec(q, (1.0, 2.0, 3.0))
        assert _vec_close(result, (1.0, 3.0, -2.0))

    def test_back(self):
        # Old lambda: (x, y, z) -> (x, -z, y)
        q = euler_to_quat((90.0, 0.0, 0.0))
        result = rotate_vec(q, (1.0, 2.0, 3.0))
        assert _vec_close(result, (1.0, -3.0, 2.0))

    def test_left(self):
        # Old lambda: (x, y, z) -> (-z, y, x)
        q = euler_to_quat((0.0, -90.0, 0.0))
        result = rotate_vec(q, (1.0, 2.0, 3.0))
        assert _vec_close(result, (-3.0, 2.0, 1.0))

    def test_right(self):
        # Old lambda: (x, y, z) -> (z, y, -x)
        q = euler_to_quat((0.0, 90.0, 0.0))
        result = rotate_vec(q, (1.0, 2.0, 3.0))
        assert _vec_close(result, (3.0, 2.0, -1.0))

    def test_bottom(self):
        # Old lambda: (x, y, z) -> (x, -y, -z)
        q = euler_to_quat((180.0, 0.0, 0.0))
        result = rotate_vec(q, (1.0, 2.0, 3.0))
        assert _vec_close(result, (1.0, -2.0, -3.0))

    def test_yaw_90_matches_rotate_z(self):
        q = euler_to_quat((0.0, 0.0, 90.0))
        result = rotate_vec(q, (1.0, 2.0, 3.0))
        assert _vec_close(result, (-2.0, 1.0, 3.0))


# ── Packing completeness tests ──


def _load_bot(design_path: str):
    """Load a bot from its design.py file."""
    mod_name = design_path.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(mod_name, design_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    bot = mod.build()
    bot.solve()
    return bot


@pytest.fixture(params=["wheeler_arm", "wheeler_base"])
def bot(request):
    return _load_bot(f"bots/{request.param}/design.py")


class TestPackingCompleteness:
    def test_every_mount_has_placement(self, bot):
        pr = bot.packing_result
        assert pr is not None
        for body in bot.all_bodies:
            for mount in body.mounts:
                assert mount in pr.placements, (
                    f"Missing placement for {mount.label} on {body.name}"
                )

    def test_every_joint_has_placement(self, bot):
        pr = bot.packing_result
        assert pr is not None
        for body in bot.all_bodies:
            for joint in body.joints:
                assert joint in pr.placements, f"Missing placement for {joint.name}"

    def test_placement_pose_matches_old_fields(self, bot):
        """Verify dual-write: PackingResult poses match legacy fields."""
        pr = bot.packing_result
        assert pr is not None
        for body in bot.all_bodies:
            for joint in body.joints:
                p = pr.placements[joint]
                assert _vec_close(p.pose.pos, joint.solved_servo_center), (
                    f"Joint {joint.name}: pose.pos={p.pose.pos} != "
                    f"solved_servo_center={joint.solved_servo_center}"
                )
                assert _vec_close(p.pose.quat, joint.solved_servo_quat), (
                    f"Joint {joint.name}: pose.quat={p.pose.quat} != "
                    f"solved_servo_quat={joint.solved_servo_quat}"
                )

            for mount in body.mounts:
                p = pr.placements[mount]
                assert _vec_close(p.pose.pos, mount.resolved_pos), (
                    f"Mount {mount.label}: pose.pos={p.pose.pos} != "
                    f"resolved_pos={mount.resolved_pos}"
                )

    def test_placement_bbox_positive(self, bot):
        """All bbox dimensions should be positive."""
        pr = bot.packing_result
        assert pr is not None
        for key, placement in pr.placements.items():
            label = getattr(key, "name", getattr(key, "label", str(key)))
            for i, val in enumerate(placement.bbox):
                assert val >= 0.0, f"{label} bbox[{i}]={val} is negative"
