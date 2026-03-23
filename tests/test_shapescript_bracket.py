"""ShapeScript IR tests for bracket/cradle/coupler emitters.

Verifies that executing the ShapeScript program through the OCCT backend
produces valid solids with expected properties.
"""

from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")

from botcad.bracket import bracket_envelope as bracket_envelope_script  # noqa: E402
from botcad.bracket import coupler_solid as coupler_solid_script  # noqa: E402
from botcad.bracket import cradle_envelope as cradle_envelope_script  # noqa: E402
from botcad.bracket import cradle_solid as cradle_solid_script  # noqa: E402
from botcad.shapescript.backend_occt import OcctBackend  # noqa: E402


def _servo():
    from botcad.components.servo import STS3215

    return STS3215()


def _total_volume(solid):
    """Get absolute volume, handling both Solid and Compound."""
    return abs(solid.volume)


def _exec(prog):
    """Execute a ShapeScript through the OCCT backend."""
    return OcctBackend().execute(prog)


class TestBracketEnvelopeIR:
    """bracket_envelope ShapeScript IR tests."""

    def test_produces_solid(self):
        from botcad.bracket import BracketSpec

        servo = _servo()
        spec = BracketSpec()

        prog = bracket_envelope_script(servo, spec)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        assert abs(ir_solid.volume) > 1e-9

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


class TestSCS0009BracketEnvelopeIR:
    """SCS0009 bracket_envelope ShapeScript IR tests."""

    def test_produces_solid(self):
        from botcad.bracket import BracketSpec
        from botcad.components.servo import SCS0009

        servo = SCS0009()
        spec = BracketSpec()

        prog = bracket_envelope_script(servo, spec)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        assert abs(ir_solid.volume) > 1e-9

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


class TestCradleEnvelopeIR:
    """cradle_envelope ShapeScript IR tests."""

    def test_produces_solid(self):
        from botcad.bracket import BracketSpec

        servo = _servo()
        spec = BracketSpec()

        prog = cradle_envelope_script(servo, spec)
        result = _exec(prog)
        ir_solid = result.shapes[prog.output_ref.id]
        assert abs(ir_solid.volume) > 1e-9

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


class TestCradleSolidIR:
    """cradle_solid ShapeScript IR tests."""

    def test_produces_solid(self):
        from botcad.bracket import BracketSpec

        servo = _servo()
        spec = BracketSpec()

        prog = cradle_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])
        assert ir_vol > 1e-9

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


class TestCouplerSolidIR:
    """coupler_solid ShapeScript IR tests."""

    def test_produces_solid(self):
        from botcad.bracket import BracketSpec

        servo = _servo()
        spec = BracketSpec()

        prog = coupler_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])
        assert ir_vol > 1e-9

    def test_output_ref_set(self):
        servo = _servo()
        prog = coupler_solid_script(servo)
        assert prog.output_ref is not None


class TestBracketSolidScript:
    """bracket_solid ShapeScript IR tests."""

    def test_sts3215_produces_solid(self):
        from botcad.bracket import BracketSpec
        from botcad.bracket import bracket_solid as bracket_solid_script
        from botcad.components.servo import STS3215

        servo = STS3215()
        spec = BracketSpec()

        prog = bracket_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])
        assert ir_vol > 1e-9

    def test_scs0009_produces_solid(self):
        from botcad.bracket import BracketSpec
        from botcad.bracket import bracket_solid as bracket_solid_script
        from botcad.components.servo import SCS0009

        servo = SCS0009()
        spec = BracketSpec()

        prog = bracket_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])
        assert ir_vol > 1e-9

    def test_sts3215_has_shapescript_ops(self):
        """STS3215 bracket should use native ShapeScript ops, not just PrebuiltOp."""
        from botcad.bracket import bracket_solid as bracket_solid_script
        from botcad.components.servo import STS3215
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
        from botcad.bracket import bracket_solid as bracket_solid_script
        from botcad.components.servo import SCS0009
        from botcad.shapescript.ops import BoxOp, CutOp, CylinderOp

        servo = SCS0009()
        prog = bracket_solid_script(servo)

        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert CylinderOp in op_types
        assert CutOp in op_types
        assert len(prog.ops) > 5

    def test_output_ref_set(self):
        from botcad.bracket import bracket_solid as bracket_solid_script

        servo = _servo()
        prog = bracket_solid_script(servo)
        assert prog.output_ref is not None


class TestConnectorPortNative:
    """Connector port should use native Box+locate ops, not PrebuiltOp."""

    def test_bracket_sts3215_no_prebuilt_connector(self):
        """STS3215 bracket connector port should not use PrebuiltOp."""
        from botcad.bracket import bracket_solid as bracket_solid_script
        from botcad.components.servo import STS3215
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

    def test_bracket_sts3215_connector_produces_solid(self):
        """STS3215 bracket with connector port: IR produces valid solid."""
        from botcad.bracket import BracketSpec
        from botcad.bracket import bracket_solid as bracket_solid_script
        from botcad.components.servo import STS3215

        servo = STS3215()
        spec = BracketSpec()

        prog = bracket_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])
        assert ir_vol > 1e-9

    def test_cradle_sts3215_connector_produces_solid(self):
        """STS3215 cradle with connector port: IR produces valid solid."""
        from botcad.bracket import BracketSpec
        from botcad.components.servo import STS3215

        servo = STS3215()
        spec = BracketSpec()

        prog = cradle_solid_script(servo, spec)
        result = _exec(prog)
        ir_vol = _total_volume(result.shapes[prog.output_ref.id])
        assert ir_vol > 1e-9
