"""Roundtrip volume tests for bracket/cradle/coupler ShapeScript emitters.

Verifies that executing the ShapeScript program through the OCCT backend
produces a solid whose volume matches the direct build123d solid.
"""

from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")

from botcad.shapescript.backend_occt import OcctBackend
from botcad.shapescript.emit_bracket import (
    bracket_envelope_script,
    coupler_solid_script,
    cradle_envelope_script,
    cradle_solid_script,
)


def _servo():
    from botcad.components.servo import STS3215

    return STS3215()


def _total_volume(solid):
    """Get absolute volume, handling both Solid and Compound."""
    return abs(solid.volume)


def _exec(prog):
    """Execute a ShapeScript through the OCCT backend."""
    return OcctBackend().execute(prog)


class TestBracketEnvelopeRoundtrip:
    """bracket_envelope via ShapeScript must match direct build123d volume."""

    def test_volume_matches(self):
        from botcad.bracket import BracketSpec, _bracket_envelope_b3d as bracket_envelope

        servo = _servo()
        spec = BracketSpec()

        direct_solid = bracket_envelope(servo, spec)
        direct_vol = abs(direct_solid.volume)

        prog = bracket_envelope_script(servo, spec)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        ir_vol = abs(ir_solid.volume)

        assert ir_vol == pytest.approx(direct_vol, rel=1e-6), (
            f"bracket_envelope volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_has_native_ops(self):
        """STS3215 bracket_envelope should use native Box + LocateOp, not PrebuiltOp."""
        from botcad.shapescript.ops import BoxOp, LocateOp

        servo = _servo()
        prog = bracket_envelope_script(servo)
        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert LocateOp in op_types

    def test_output_ref_set(self):
        servo = _servo()
        prog = bracket_envelope_script(servo)
        assert prog.output_ref is not None


class TestSCS0009BracketEnvelopeRoundtrip:
    """SCS0009 bracket_envelope via ShapeScript must match direct build123d volume."""

    def test_volume_matches(self):
        from botcad.bracket import BracketSpec, _bracket_envelope_b3d as bracket_envelope
        from botcad.components.servo import SCS0009

        servo = SCS0009()
        spec = BracketSpec()

        direct_solid = bracket_envelope(servo, spec)
        direct_vol = abs(direct_solid.volume)

        prog = bracket_envelope_script(servo, spec)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        ir_vol = abs(ir_solid.volume)

        assert ir_vol == pytest.approx(direct_vol, rel=1e-6), (
            f"SCS0009 bracket_envelope volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_has_native_ops(self):
        """SCS0009 bracket_envelope should use native Box + LocateOp, not PrebuiltOp."""
        from botcad.components.servo import SCS0009
        from botcad.shapescript.ops import BoxOp, LocateOp

        servo = SCS0009()
        prog = bracket_envelope_script(servo)
        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert LocateOp in op_types

    def test_output_ref_set(self):
        from botcad.components.servo import SCS0009

        servo = SCS0009()
        prog = bracket_envelope_script(servo)
        assert prog.output_ref is not None


class TestCradleEnvelopeRoundtrip:
    """cradle_envelope via ShapeScript must match direct build123d volume."""

    def test_volume_matches(self):
        from botcad.bracket import BracketSpec, _cradle_envelope_b3d as cradle_envelope

        servo = _servo()
        spec = BracketSpec()

        direct_solid = cradle_envelope(servo, spec)
        direct_vol = abs(direct_solid.volume)

        prog = cradle_envelope_script(servo, spec)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        ir_vol = abs(ir_solid.volume)

        assert ir_vol == pytest.approx(direct_vol, rel=1e-6), (
            f"cradle_envelope volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_has_native_ops(self):
        """cradle_envelope should use native Box + LocateOp, not PrebuiltOp."""
        from botcad.shapescript.ops import BoxOp, LocateOp

        servo = _servo()
        prog = cradle_envelope_script(servo)
        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert LocateOp in op_types

    def test_output_ref_set(self):
        servo = _servo()
        prog = cradle_envelope_script(servo)
        assert prog.output_ref is not None


class TestCradleSolidRoundtrip:
    """cradle_solid via ShapeScript must match direct build123d volume."""

    def test_volume_matches(self):
        from botcad.bracket import BracketSpec, _cradle_solid_b3d as cradle_solid

        servo = _servo()
        spec = BracketSpec()

        direct_solid = cradle_solid(servo, spec)
        direct_vol = _total_volume(direct_solid)

        prog = cradle_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"cradle_solid volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_has_native_ops(self):
        """cradle_solid should use native Box/Cylinder/Cut ops."""
        from botcad.shapescript.ops import BoxOp, CutOp

        servo = _servo()
        prog = cradle_solid_script(servo)
        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert CutOp in op_types
        assert len(prog.ops) > 5

    def test_output_ref_set(self):
        servo = _servo()
        prog = cradle_solid_script(servo)
        assert prog.output_ref is not None


class TestCouplerSolidRoundtrip:
    """coupler_solid via ShapeScript must match direct build123d volume."""

    def test_volume_matches(self):
        from botcad.bracket import BracketSpec, _coupler_solid_b3d as coupler_solid

        servo = _servo()
        spec = BracketSpec()

        direct_solid = coupler_solid(servo, spec)
        direct_vol = _total_volume(direct_solid)

        prog = coupler_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"coupler_solid volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_output_ref_set(self):
        servo = _servo()
        prog = coupler_solid_script(servo)
        assert prog.output_ref is not None


class TestBracketSolidScript:
    """bracket_solid via ShapeScript must match direct build123d volume."""

    def test_sts3215_volume_matches(self):
        from botcad.bracket import BracketSpec, _bracket_solid_b3d as bracket_solid
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_bracket import bracket_solid_script

        servo = STS3215()
        spec = BracketSpec()

        direct = bracket_solid(servo, spec)
        direct_vol = _total_volume(direct)

        prog = bracket_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"STS3215 bracket_solid volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_scs0009_volume_matches(self):
        from botcad.bracket import BracketSpec, _bracket_solid_b3d as bracket_solid
        from botcad.components.servo import SCS0009
        from botcad.shapescript.emit_bracket import bracket_solid_script

        servo = SCS0009()
        spec = BracketSpec()

        direct = bracket_solid(servo, spec)
        direct_vol = _total_volume(direct)

        prog = bracket_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"SCS0009 bracket_solid volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_sts3215_has_shapescript_ops(self):
        """STS3215 bracket should use native ShapeScript ops, not just PrebuiltOp."""
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_bracket import bracket_solid_script
        from botcad.shapescript.ops import BoxOp, CutOp, CylinderOp

        servo = STS3215()
        prog = bracket_solid_script(servo)

        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert CylinderOp in op_types
        assert CutOp in op_types
        assert len(prog.ops) > 10

    def test_scs0009_has_native_ops(self):
        """SCS0009 bracket should use native Box/Cylinder/Cut ops."""
        from botcad.components.servo import SCS0009
        from botcad.shapescript.emit_bracket import bracket_solid_script
        from botcad.shapescript.ops import BoxOp, CutOp, CylinderOp

        servo = SCS0009()
        prog = bracket_solid_script(servo)

        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert CylinderOp in op_types
        assert CutOp in op_types
        assert len(prog.ops) > 5

    def test_output_ref_set(self):
        from botcad.shapescript.emit_bracket import bracket_solid_script

        servo = _servo()
        prog = bracket_solid_script(servo)
        assert prog.output_ref is not None


class TestConnectorPortNative:
    """Connector port should use native Box+locate ops, not PrebuiltOp."""

    def test_bracket_sts3215_no_prebuilt_connector(self):
        """STS3215 bracket connector port should not use PrebuiltOp."""
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_bracket import bracket_solid_script
        from botcad.shapescript.ops import PrebuiltOp

        servo = STS3215()
        prog = bracket_solid_script(servo)

        prebuilt_ops = [op for op in prog.ops if isinstance(op, PrebuiltOp)]
        assert len(prebuilt_ops) == 0, (
            f"Expected no PrebuiltOp but found {len(prebuilt_ops)}: "
            f"{[op.tag for op in prebuilt_ops]}"
        )

    def test_cradle_sts3215_no_prebuilt_connector(self):
        """STS3215 cradle connector port should not use PrebuiltOp."""
        from botcad.components.servo import STS3215
        from botcad.shapescript.ops import PrebuiltOp

        servo = STS3215()
        prog = cradle_solid_script(servo)

        prebuilt_ops = [op for op in prog.ops if isinstance(op, PrebuiltOp)]
        assert len(prebuilt_ops) == 0, (
            f"Expected no PrebuiltOp but found {len(prebuilt_ops)}: "
            f"{[op.tag for op in prebuilt_ops]}"
        )

    def test_bracket_sts3215_connector_volume_matches(self):
        """STS3215 bracket with connector port: IR volume matches direct."""
        from botcad.bracket import BracketSpec, _bracket_solid_b3d as bracket_solid
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_bracket import bracket_solid_script

        servo = STS3215()
        spec = BracketSpec()

        direct = bracket_solid(servo, spec)
        direct_vol = _total_volume(direct)

        prog = bracket_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"STS3215 bracket connector port volume mismatch: "
            f"IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_cradle_sts3215_connector_volume_matches(self):
        """STS3215 cradle with connector port: IR volume matches direct."""
        from botcad.bracket import BracketSpec, _cradle_solid_b3d as cradle_solid
        from botcad.components.servo import STS3215

        servo = STS3215()
        spec = BracketSpec()

        direct = cradle_solid(servo, spec)
        direct_vol = _total_volume(direct)

        prog = cradle_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"STS3215 cradle connector port volume mismatch: "
            f"IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )
