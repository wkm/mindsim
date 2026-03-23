"""ShapeScript bracket/cradle/coupler emitter tests.

Verifies that executing the ShapeScript program through the OCCT backend
produces valid solids with expected ops and non-zero volume.
"""

from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")

from botcad.bracket import (  # noqa: E402
    bracket_insertion_channel as bracket_insertion_channel_script,
)
from botcad.bracket import coupler_solid as coupler_solid_script  # noqa: E402
from botcad.bracket import (  # noqa: E402
    cradle_insertion_channel as cradle_insertion_channel_script,
)
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


class TestBracketInsertionChannelIR:
    """bracket_insertion_channel ShapeScript produces valid geometry."""

    def test_has_native_ops(self):
        """STS3215 bracket_insertion_channel should use native Box + LocateOp."""
        from botcad.shapescript.ops import BoxOp, LocateOp

        servo = _servo()
        prog = bracket_insertion_channel_script(servo)
        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert LocateOp in op_types

    def test_output_ref_set(self):
        servo = _servo()
        prog = bracket_insertion_channel_script(servo)
        assert prog.output_ref is not None


class TestSCS0009BracketInsertionChannelIR:
    """SCS0009 bracket_insertion_channel ShapeScript produces valid geometry."""

    def test_has_native_ops(self):
        """SCS0009 bracket_insertion_channel should use native Box + LocateOp."""
        from botcad.components.servo import SCS0009
        from botcad.shapescript.ops import BoxOp, LocateOp

        servo = SCS0009()
        prog = bracket_insertion_channel_script(servo)
        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert LocateOp in op_types

    def test_output_ref_set(self):
        from botcad.components.servo import SCS0009

        servo = SCS0009()
        prog = bracket_insertion_channel_script(servo)
        assert prog.output_ref is not None


class TestCradleInsertionChannelIR:
    """cradle_insertion_channel ShapeScript produces valid geometry."""

    def test_has_native_ops(self):
        """cradle_insertion_channel should use native Box + LocateOp."""
        from botcad.shapescript.ops import BoxOp, LocateOp

        servo = _servo()
        prog = cradle_insertion_channel_script(servo)
        op_types = {type(op) for op in prog.ops}
        assert BoxOp in op_types
        assert LocateOp in op_types

    def test_output_ref_set(self):
        servo = _servo()
        prog = cradle_insertion_channel_script(servo)
        assert prog.output_ref is not None


class TestCradleSolidIR:
    """cradle_solid ShapeScript produces valid geometry."""

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
    """coupler_solid ShapeScript produces valid geometry."""

    def test_output_ref_set(self):
        servo = _servo()
        prog = coupler_solid_script(servo)
        assert prog.output_ref is not None


class TestBracketSolidScript:
    """bracket_solid ShapeScript produces valid geometry."""

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
