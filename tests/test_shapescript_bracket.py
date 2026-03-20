"""Roundtrip volume tests for bracket/cradle envelope ShapeScript wrappers.

Verifies that executing the ShapeScript program through the OCCT backend
produces a solid whose volume matches the direct build123d solid.
"""

from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")

from botcad.shapescript.backend_occt import OcctBackend
from botcad.shapescript.emit_bracket import (
    bracket_envelope_script,
    cradle_envelope_script,
)


def _servo():
    from botcad.components.servo import STS3215

    return STS3215()


class TestBracketEnvelopeRoundtrip:
    """bracket_envelope via ShapeScript must match direct build123d volume."""

    def test_volume_matches(self):
        from botcad.bracket import BracketSpec, bracket_envelope

        servo = _servo()
        spec = BracketSpec()

        # Direct path
        direct_solid = bracket_envelope(servo, spec)
        direct_vol = abs(direct_solid.volume)

        # ShapeScript path
        prog = bracket_envelope_script(servo, spec)
        result = OcctBackend().execute(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        ir_vol = abs(ir_solid.volume)

        assert ir_vol == pytest.approx(direct_vol, rel=1e-6), (
            f"bracket_envelope volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_has_prebuilt_op(self):
        from botcad.shapescript.ops import PrebuiltOp

        servo = _servo()
        prog = bracket_envelope_script(servo)
        assert len(prog.ops) == 1
        assert isinstance(prog.ops[0], PrebuiltOp)
        assert prog.ops[0].tag == "bracket_envelope"

    def test_output_ref_set(self):
        servo = _servo()
        prog = bracket_envelope_script(servo)
        assert prog.output_ref is not None
        assert prog.output_ref == prog.ops[0].ref


class TestCradleEnvelopeRoundtrip:
    """cradle_envelope via ShapeScript must match direct build123d volume."""

    def test_volume_matches(self):
        from botcad.bracket import BracketSpec, cradle_envelope

        servo = _servo()
        spec = BracketSpec()

        # Direct path
        direct_solid = cradle_envelope(servo, spec)
        direct_vol = abs(direct_solid.volume)

        # ShapeScript path
        prog = cradle_envelope_script(servo, spec)
        result = OcctBackend().execute(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        ir_vol = abs(ir_solid.volume)

        assert ir_vol == pytest.approx(direct_vol, rel=1e-6), (
            f"cradle_envelope volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_has_prebuilt_op(self):
        from botcad.shapescript.ops import PrebuiltOp

        servo = _servo()
        prog = cradle_envelope_script(servo)
        assert len(prog.ops) == 1
        assert isinstance(prog.ops[0], PrebuiltOp)
        assert prog.ops[0].tag == "cradle_envelope"

    def test_output_ref_set(self):
        servo = _servo()
        prog = cradle_envelope_script(servo)
        assert prog.output_ref is not None
        assert prog.output_ref == prog.ops[0].ref


def _total_volume(solid):
    """Get absolute volume, handling both Solid and Compound."""
    return abs(solid.volume)


def _exec(prog):
    """Execute a ShapeScript through the OCCT backend."""
    return OcctBackend().execute(prog)


class TestBracketSolidScript:
    """bracket_solid via ShapeScript must match direct build123d volume."""

    def test_sts3215_volume_matches(self):
        from botcad.bracket import BracketSpec, bracket_solid
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_bracket import bracket_solid_script

        servo = STS3215()
        spec = BracketSpec()

        # Direct path
        direct = bracket_solid(servo, spec)
        direct_vol = _total_volume(direct)

        # ShapeScript path
        prog = bracket_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])

        assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
            f"STS3215 bracket_solid volume mismatch: IR={ir_vol:.10e} vs direct={direct_vol:.10e}"
        )

    def test_scs0009_volume_matches(self):
        from botcad.bracket import BracketSpec, bracket_solid
        from botcad.components.servo import SCS0009
        from botcad.shapescript.emit_bracket import bracket_solid_script

        servo = SCS0009()
        spec = BracketSpec()

        # Direct path
        direct = bracket_solid(servo, spec)
        direct_vol = _total_volume(direct)

        # ShapeScript path
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
        # Should have native box, cylinder, cut ops (not just a single PrebuiltOp)
        assert BoxOp in op_types
        assert CylinderOp in op_types
        assert CutOp in op_types
        # Should have at least several ops (outer + pocket + cuts)
        assert len(prog.ops) > 10

    def test_scs0009_uses_prebuilt(self):
        """SCS0009 bracket should be a single PrebuiltOp."""
        from botcad.components.servo import SCS0009
        from botcad.shapescript.emit_bracket import bracket_solid_script
        from botcad.shapescript.ops import PrebuiltOp

        servo = SCS0009()
        prog = bracket_solid_script(servo)

        assert len(prog.ops) == 1
        assert isinstance(prog.ops[0], PrebuiltOp)
        assert prog.ops[0].tag == "bracket_solid"

    def test_output_ref_set(self):
        from botcad.shapescript.emit_bracket import bracket_solid_script

        servo = _servo()
        prog = bracket_solid_script(servo)
        assert prog.output_ref is not None
