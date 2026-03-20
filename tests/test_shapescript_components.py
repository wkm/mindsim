"""Roundtrip volume tests: ShapeScript component emitters vs direct build123d.

Each test builds a component solid via both paths and asserts volumes match
within 0.1% (rel=0.001).
"""

from __future__ import annotations

import pytest


def _execute(prog):
    """Execute a ShapeScript program and return the output solid."""
    from botcad.shapescript.backend_occt import OcctBackend

    backend = OcctBackend()
    result = backend.execute(prog)
    assert prog.output_ref is not None, "Program has no output_ref"
    return result.shapes[prog.output_ref.id]


class TestCameraScript:
    """camera_script() vs camera_solid() roundtrip."""

    def test_ov5647_volume(self):
        from botcad.components.camera import OV5647, camera_solid
        from botcad.shapescript.emit_components import camera_script

        spec = OV5647()
        direct = camera_solid(spec)
        ir_solid = _execute(camera_script(spec))

        direct_vol = abs(direct.volume)
        ir_vol = abs(ir_solid.volume)
        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"camera volume mismatch: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )

    def test_picamera2_volume(self):
        from botcad.components.camera import PiCamera2, camera_solid
        from botcad.shapescript.emit_components import camera_script

        spec = PiCamera2()
        direct = camera_solid(spec)
        ir_solid = _execute(camera_script(spec))

        direct_vol = abs(direct.volume)
        ir_vol = abs(ir_solid.volume)
        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"camera volume mismatch: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )


class TestBatteryScript:
    """battery_script() vs battery_solid() roundtrip."""

    def test_lipo2s_1000_volume(self):
        from botcad.components.battery import LiPo2S, battery_solid
        from botcad.shapescript.emit_components import battery_script

        spec = LiPo2S(1000)
        direct = battery_solid(spec)
        ir_solid = _execute(battery_script(spec))

        direct_vol = abs(direct.volume)
        ir_vol = abs(ir_solid.volume)
        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"battery volume mismatch: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )

    def test_lipo2s_500_volume(self):
        from botcad.components.battery import LiPo2S, battery_solid
        from botcad.shapescript.emit_components import battery_script

        spec = LiPo2S(500)
        direct = battery_solid(spec)
        ir_solid = _execute(battery_script(spec))

        direct_vol = abs(direct.volume)
        ir_vol = abs(ir_solid.volume)
        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"battery volume mismatch: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )


class TestBearingScript:
    """bearing_script() vs _make_bearing_solid() roundtrip."""

    def test_608zz_volume(self):
        from botcad.component import BearingSpec
        from botcad.emit.cad import _make_bearing_solid
        from botcad.shapescript.emit_components import bearing_script

        spec = BearingSpec(
            name="608zz",
            dimensions=(0.022, 0.022, 0.007),
            mass=0.012,
            od=0.022,
            id=0.008,
            width=0.007,
        )
        direct = _make_bearing_solid(spec)
        ir_solid = _execute(bearing_script(spec))

        direct_vol = abs(direct.volume)
        ir_vol = abs(ir_solid.volume)
        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"bearing volume mismatch: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )

    def test_mr128zz_volume(self):
        """Smaller bearing: MR128ZZ (OD=12mm, ID=8mm, W=3.5mm)."""
        from botcad.component import BearingSpec
        from botcad.emit.cad import _make_bearing_solid
        from botcad.shapescript.emit_components import bearing_script

        spec = BearingSpec(
            name="MR128ZZ",
            dimensions=(0.012, 0.012, 0.0035),
            mass=0.005,
            od=0.012,
            id=0.008,
            width=0.0035,
        )
        direct = _make_bearing_solid(spec)
        ir_solid = _execute(bearing_script(spec))

        direct_vol = abs(direct.volume)
        ir_vol = abs(ir_solid.volume)
        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"bearing volume mismatch: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )


class TestHornScript:
    """horn_script() vs _horn_solid() roundtrip."""

    def test_sts3215_volume(self):
        from botcad.components.servo import STS3215
        from botcad.emit.cad import _horn_solid
        from botcad.shapescript.emit_components import horn_script

        servo = STS3215()
        direct = _horn_solid(servo)
        prog = horn_script(servo)

        # Both should produce a solid (STS3215 has horn mounting points)
        assert direct is not None, "direct _horn_solid returned None"
        assert prog is not None, "horn_script returned None"

        ir_solid = _execute(prog)

        direct_vol = abs(direct.volume)
        ir_vol = abs(ir_solid.volume)
        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"horn volume mismatch: direct={direct_vol:.6e}, IR={ir_vol:.6e}"
        )

    def test_no_horn_returns_none(self):
        """Servo without horn mounting points returns None."""
        from botcad.component import ServoSpec
        from botcad.shapescript.emit_components import horn_script

        # Minimal servo with no horn_mounting_points
        servo = ServoSpec(name="dummy", dimensions=(0.01, 0.01, 0.01), mass=0.01)
        assert horn_script(servo) is None
