"""Unit tests for the parametric bot domain model.

Tests MountPoint, MountingEar, Mount, insertion axis resolution,
packing solver integration, component catalog, and bot builds.

No MuJoCo or CAD dependencies — pure domain model tests.
"""

from __future__ import annotations

import math

import pytest

from botcad.component import CameraSpec, Component, MountingEar, MountPoint, Vec3

# ── TestMountPoint ──


class TestMountPoint:
    def test_defaults(self):
        mp = MountPoint("m1", pos=(0.0, 0.0, 0.0), diameter=0.003)
        assert mp.axis == (0.0, 0.0, 1.0)
        assert mp.fastener_type == ""

    def test_explicit_axis(self):
        mp = MountPoint(
            "m1",
            pos=(0.0, 0.0, 0.0),
            diameter=0.003,
            axis=(0.0, 0.0, -1.0),
            fastener_type="M3",
        )
        assert mp.axis == (0.0, 0.0, -1.0)
        assert mp.fastener_type == "M3"

    def test_frozen_immutability(self):
        mp = MountPoint("m1", pos=(0.0, 0.0, 0.0), diameter=0.003)
        with pytest.raises(AttributeError):
            mp.diameter = 0.005

    def test_mounting_ear_factory_returns_mount_point(self):
        ear = MountingEar("ear_1", pos=(0.01, 0.02, -0.015), hole_diameter=0.0042)
        assert isinstance(ear, MountPoint)

    def test_mounting_ear_field_mapping(self):
        ear = MountingEar("ear_1", pos=(0.01, 0.02, -0.015), hole_diameter=0.0042)
        assert ear.label == "ear_1"
        assert ear.pos == (0.01, 0.02, -0.015)
        assert ear.diameter == 0.0042  # hole_diameter → diameter
        assert ear.axis == (0.0, 0.0, -1.0)  # default for ears
        assert ear.fastener_type == "M3"  # default for ears

    def test_mounting_ear_custom_axis(self):
        ear = MountingEar(
            "ear_1",
            pos=(0.0, 0.0, 0.0),
            hole_diameter=0.004,
            axis=(1.0, 0.0, 0.0),
            fastener_type="M2.5",
        )
        assert ear.axis == (1.0, 0.0, 0.0)
        assert ear.fastener_type == "M2.5"


# ── TestMount ──


class TestMount:
    def test_default_insertion_axis_is_none(self):
        from botcad.skeleton import Mount

        comp = Component("test", dimensions=(0.01, 0.01, 0.01), mass=0.001)
        m = Mount(component=comp, label="test", position="center")
        assert m.insertion_axis is None

    def test_explicit_insertion_axis_preserved(self):
        from botcad.skeleton import Mount

        comp = Component("test", dimensions=(0.01, 0.01, 0.01), mass=0.001)
        m = Mount(
            component=comp,
            label="test",
            position="bottom",
            insertion_axis=(0.0, 0.0, -1.0),
        )
        assert m.insertion_axis == (0.0, 0.0, -1.0)

    def test_resolved_insertion_axis_default(self):
        from botcad.skeleton import Mount

        comp = Component("test", dimensions=(0.01, 0.01, 0.01), mass=0.001)
        m = Mount(component=comp, label="test", position="center")
        assert m.resolved_insertion_axis == (0.0, 0.0, 1.0)


# ── TestInsertionAxisResolution ──


class TestInsertionAxisResolution:
    @pytest.fixture
    def resolve(self):
        from botcad.packing import _resolve_insertion_axis

        return _resolve_insertion_axis

    def test_top(self, resolve):
        assert resolve("top", None) == (0.0, 0.0, 1.0)

    def test_bottom(self, resolve):
        assert resolve("bottom", None) == (0.0, 0.0, -1.0)

    def test_front(self, resolve):
        assert resolve("front", None) == (0.0, 1.0, 0.0)

    def test_back(self, resolve):
        assert resolve("back", None) == (0.0, -1.0, 0.0)

    def test_left(self, resolve):
        assert resolve("left", None) == (-1.0, 0.0, 0.0)

    def test_right(self, resolve):
        assert resolve("right", None) == (1.0, 0.0, 0.0)

    def test_center(self, resolve):
        assert resolve("center", None) == (0.0, 0.0, 1.0)

    def test_vec3_position_defaults_to_plus_z(self, resolve):
        assert resolve((0.01, 0.02, 0.03), None) == (0.0, 0.0, 1.0)

    def test_explicit_override_wins(self, resolve):
        assert resolve("bottom", (1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)

    def test_explicit_override_wins_for_vec3(self, resolve):
        assert resolve((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)) == (0.0, -1.0, 0.0)


# ── TestPackingSolver ──


class TestPackingSolver:
    def _make_bot_with_mount(self, position="center", insertion_axis=None):
        from botcad.skeleton import Bot

        comp = Component("widget", dimensions=(0.02, 0.02, 0.01), mass=0.01)
        bot = Bot("test_bot")
        body = bot.body("base", padding=0.005)
        body.mount(comp, position=position, insertion_axis=insertion_axis)
        bot.solve()
        return bot

    def test_solve_fills_resolved_insertion_axis(self):
        bot = self._make_bot_with_mount(position="top")
        mount = bot.all_bodies[0].mounts[0]
        assert mount.resolved_insertion_axis == (0.0, 0.0, 1.0)

    def test_bottom_resolved_axis(self):
        bot = self._make_bot_with_mount(position="bottom")
        mount = bot.all_bodies[0].mounts[0]
        assert mount.resolved_insertion_axis == (0.0, 0.0, -1.0)

    def test_explicit_axis_preserved_through_solve(self):
        bot = self._make_bot_with_mount(position="top", insertion_axis=(0.0, 1.0, 0.0))
        mount = bot.all_bodies[0].mounts[0]
        assert mount.resolved_insertion_axis == (0.0, 1.0, 0.0)

    def test_resolved_pos_center_at_origin(self):
        bot = self._make_bot_with_mount(position="center")
        mount = bot.all_bodies[0].mounts[0]
        assert mount.resolved_pos == (0.0, 0.0, 0.0)

    def test_resolved_pos_bottom_negative_z(self):
        bot = self._make_bot_with_mount(position="bottom")
        mount = bot.all_bodies[0].mounts[0]
        assert mount.resolved_pos[2] < 0.0


# ── TestComponentCatalog ──


class TestComponentCatalog:
    def test_sts3215_ears_are_mount_points(self):
        from botcad.components.servo import STS3215

        servo = STS3215()
        for ear in servo.mounting_ears:
            assert isinstance(ear, MountPoint)
            assert ear.fastener_type == "M3"
            assert ear.axis == (0.0, 0.0, -1.0)

    def test_all_components_have_fastener_type(self):
        from botcad.components.camera import OV5647
        from botcad.components.compute import RaspberryPiZero2W
        from botcad.components.wheel import PololuWheel90mm

        for factory in [RaspberryPiZero2W, OV5647, PololuWheel90mm]:
            comp = factory()
            for mp in comp.mounting_points:
                assert mp.fastener_type != "", (
                    f"{comp.name} mount {mp.label} has no fastener_type"
                )

    def test_horn_mounting_points_axis(self):
        from botcad.components.servo import STS3215

        servo = STS3215()
        for mp in servo.horn_mounting_points:
            assert mp.axis == (0.0, 0.0, 1.0)
            assert mp.fastener_type == "M2"

    def test_sts3215_specs_match_datasheet(self):
        from botcad.components.servo import STS3215

        servo = STS3215()
        assert math.isclose(servo.stall_torque, 2.942, rel_tol=1e-3)
        assert math.isclose(servo.no_load_speed, (math.pi / 3) / 0.222, rel_tol=1e-6)
        assert servo.gear_ratio == 345.0
        assert math.isclose(servo.typical_current, 0.18, rel_tol=1e-3)
        lo, hi = servo.range_rad
        assert math.isclose(hi - lo, 2 * math.pi, rel_tol=1e-6)

    def test_rear_horn_mounting_points_axis(self):
        from botcad.components.servo import STS3215

        servo = STS3215()
        for mp in servo.rear_horn_mounting_points:
            assert mp.axis == (0.0, 0.0, -1.0)
            assert mp.fastener_type == "M2"

    def test_pi_mount_points(self):
        from botcad.components.compute import RaspberryPiZero2W

        pi = RaspberryPiZero2W()
        for mp in pi.mounting_points:
            assert mp.axis == (0.0, 0.0, -1.0)
            assert mp.fastener_type == "M2"

    def test_camera_mount_points(self):
        from botcad.components.camera import OV5647

        cam = OV5647()
        for mp in cam.mounting_points:
            assert mp.axis == (0.0, 0.0, -1.0)
            assert mp.fastener_type == "M2"

    def test_wheel_mount_points(self):
        from botcad.components.wheel import PololuWheel90mm

        wheel = PololuWheel90mm()
        for mp in wheel.mounting_points:
            assert mp.axis == (0.0, 0.0, 1.0)
            assert mp.fastener_type == "M3"


# ── TestBotBuild ──


def _is_unit_vector(v: Vec3, tol: float = 1e-6) -> bool:
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return abs(mag - 1.0) < tol


class TestBotBuild:
    def test_wheeler_arm_builds(self):
        import importlib
        import sys

        spec = importlib.util.spec_from_file_location(
            "wheeler_arm_design", "bots/wheeler_arm/design.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # Prevent the module from running emit() or writing files
        sys.modules["wheeler_arm_design"] = mod
        spec.loader.exec_module(mod)

        bot = mod.build()
        bot.solve()

        assert len(bot.all_bodies) > 0
        for body in bot.all_bodies:
            for mount in body.mounts:
                assert _is_unit_vector(mount.resolved_insertion_axis), (
                    f"{body.name}/{mount.label} axis not unit: {mount.resolved_insertion_axis}"
                )

    def test_wheeler_base_builds(self):
        import importlib
        import sys

        spec = importlib.util.spec_from_file_location(
            "wheeler_base_design", "bots/wheeler_base/design.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["wheeler_base_design"] = mod
        spec.loader.exec_module(mod)

        bot = mod.build()
        bot.solve()

        assert len(bot.all_bodies) > 0
        for body in bot.all_bodies:
            for mount in body.mounts:
                assert _is_unit_vector(mount.resolved_insertion_axis), (
                    f"{body.name}/{mount.label} axis not unit: {mount.resolved_insertion_axis}"
                )

    def test_so101_arm_builds(self):
        import importlib
        import sys

        spec = importlib.util.spec_from_file_location(
            "so101_arm_design", "bots/so101_arm/design.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["so101_arm_design"] = mod
        spec.loader.exec_module(mod)

        bot = mod.build()
        bot.solve()

        # 7 bodies: base, turntable, upper_arm, forearm, wrist, wrist_roll, jaw
        assert len(bot.all_bodies) == 7
        # 6 joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        assert len(bot.all_joints) == 6

        for body in bot.all_bodies:
            for mount in body.mounts:
                assert _is_unit_vector(mount.resolved_insertion_axis), (
                    f"{body.name}/{mount.label} axis not unit: {mount.resolved_insertion_axis}"
                )


class TestPackingOverlaps:
    """Verify no internal component/servo overlaps in any bot design."""

    @staticmethod
    def _build_bot(design_path: str, module_name: str):
        import importlib
        import sys

        spec = importlib.util.spec_from_file_location(module_name, design_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        bot = mod.build()
        bot.solve()
        return bot

    @pytest.mark.parametrize(
        "design_path,module_name",
        [
            ("bots/wheeler_base/design.py", "wb_overlap_test"),
        ],
    )
    def test_no_packing_overlaps(self, design_path, module_name):
        from botcad.packing import find_internal_overlaps

        bot = self._build_bot(design_path, module_name)
        all_overlaps = []
        for body in bot.all_bodies:
            for a, b, extent in find_internal_overlaps(body):
                all_overlaps.append(
                    f"{body.name}: {a} vs {b} "
                    f"({extent[0] * 1000:.1f}x{extent[1] * 1000:.1f}x{extent[2] * 1000:.1f}mm)"
                )
        assert not all_overlaps, "Packing overlaps found:\n" + "\n".join(all_overlaps)

    @pytest.mark.xfail(
        reason="Known packing overlaps — designs need rework", strict=True
    )
    @pytest.mark.parametrize(
        "design_path,module_name",
        [
            ("bots/wheeler_arm/design.py", "wa_overlap_test"),
            ("bots/so101_arm/design.py", "so_overlap_test"),
        ],
    )
    def test_no_packing_overlaps_xfail(self, design_path, module_name):
        from botcad.packing import find_internal_overlaps

        bot = self._build_bot(design_path, module_name)
        all_overlaps = []
        for body in bot.all_bodies:
            for a, b, extent in find_internal_overlaps(body):
                all_overlaps.append(
                    f"{body.name}: {a} vs {b} "
                    f"({extent[0] * 1000:.1f}x{extent[1] * 1000:.1f}x{extent[2] * 1000:.1f}mm)"
                )
        assert not all_overlaps, "Packing overlaps found:\n" + "\n".join(all_overlaps)


# ── TestCameraSpec ──


class TestCameraSpec:
    def test_ov5647_is_camera_spec(self):
        from botcad.components.camera import OV5647

        cam = OV5647()
        assert isinstance(cam, CameraSpec)
        assert cam.fov_deg == 72.0

    def test_picamera2_is_camera_spec(self):
        from botcad.components.camera import PiCamera2

        cam = PiCamera2()
        assert isinstance(cam, CameraSpec)
        assert cam.fov_deg == 62.2
        assert cam.resolution == (3280, 2464)

    def test_camera_spec_inherits_component(self):
        from botcad.components.camera import PiCamera2

        cam = PiCamera2()
        assert isinstance(cam, Component)
        assert cam.name == "PiCamera2"


# ── TestGripJoint ──


class TestGripJoint:
    def test_grip_defaults_false(self):
        from botcad.components.servo import STS3215
        from botcad.skeleton import Joint

        j = Joint(
            name="test",
            servo=STS3215(),
            axis=(1.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        )
        assert j.grip is False

    def test_grip_set_true(self):
        from botcad.components.servo import STS3215
        from botcad.skeleton import Bot

        bot = Bot("test_grip")
        base = bot.body("base")
        j = base.joint("gripper", servo=STS3215(), grip=True)
        assert j.grip is True


# ── TestJawShape ──


class TestJawShape:
    def test_jaw_dimensions(self):
        from botcad.skeleton import Body, BodyShape

        b = Body(
            name="jaw",
            shape=BodyShape.JAW,
            jaw_length=0.04,
            jaw_width=0.03,
            jaw_thickness=0.005,
        )
        dims = b.dimensions
        assert dims == (0.03, 0.005, 0.04)  # X=width, Y=thickness, Z=length

    def test_jaw_defaults(self):
        from botcad.skeleton import Body, BodyShape

        b = Body(name="jaw", shape=BodyShape.JAW)
        dims = b.dimensions
        assert dims[0] == 0.03  # default width
        assert dims[1] == 0.005  # default thickness
        assert dims[2] == 0.04  # default length
