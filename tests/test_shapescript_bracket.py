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
