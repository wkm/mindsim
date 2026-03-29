"""Tests for RegularPolygonExtrudeOp and ChamferByFaceOp."""

from __future__ import annotations

import math

import pytest

b3d = pytest.importorskip("build123d")

from botcad.shapescript.backend_occt import ExecutionResult, OcctBackend  # noqa: E402
from botcad.shapescript.ops import ShapeRef  # noqa: E402
from botcad.shapescript.program import ShapeScriptBuilder  # noqa: E402
from botcad.units import Meters  # noqa: E402


def _exec(prog: ShapeScriptBuilder) -> ExecutionResult:
    return OcctBackend().execute(prog)


# -- RegularPolygonExtrudeOp Tests --


class TestRegularPolygonExtrudeOp:
    def test_hex_prism_volume(self):
        """6-sided polygon extruded to 5mm should produce a hex prism.

        Volume = (3√3/2) * r² * h where r is circumradius.
        """
        r = 1.0
        h = 5.0
        prog = ShapeScriptBuilder()
        ref = prog.regular_polygon_extrude(r, 6, h)
        prog.query_volume(ref)
        result = _exec(prog)
        expected = (3 * math.sqrt(3) / 2) * r**2 * h
        assert result.queries[0] == pytest.approx(expected, rel=1e-3)

    def test_square_prism_volume(self):
        """4-sided polygon extruded = square prism.

        For circumradius r, side = r*√2, area = 2*r², volume = 2*r²*h.
        """
        r = 1.0
        h = 3.0
        prog = ShapeScriptBuilder()
        ref = prog.regular_polygon_extrude(r, 4, h)
        prog.query_volume(ref)
        result = _exec(prog)
        expected = 2 * r**2 * h
        assert result.queries[0] == pytest.approx(expected, rel=1e-3)

    def test_min_z_align(self):
        """MIN_Z aligned polygon extrude has bottom at z=0."""
        from botcad.shapescript.ops import Align3

        prog = ShapeScriptBuilder()
        ref = prog.regular_polygon_extrude(1.0, 6, 2.0, align=Align3(z="min"))
        prog.query_bbox(ref)
        result = _exec(prog)
        (mn, mx) = result.queries[0]
        assert mn[2] == pytest.approx(0.0, abs=1e-6)
        assert mx[2] == pytest.approx(2.0, abs=1e-6)

    def test_tag_is_declared(self):
        """Tag should be registered in the tag registry."""
        prog = ShapeScriptBuilder()
        ref = prog.regular_polygon_extrude(1.0, 6, 1.0, tag="hex_recess")
        result = _exec(prog)
        assert "hex_recess" in result.tags.tags_on(ref)

    def test_op_dataclass_frozen(self):
        """Op should be frozen."""
        from botcad.shapescript.ops import RegularPolygonExtrudeOp

        op = RegularPolygonExtrudeOp(
            ref=ShapeRef("rpe_0"), radius=1.0, sides=6, height=5.0
        )
        with pytest.raises(AttributeError):
            op.radius = 2.0

    def test_can_be_used_in_cut(self):
        """Hex prism can be used as a boolean cut tool."""
        prog = ShapeScriptBuilder()
        box = prog.box(10.0, 10.0, 10.0)
        hex_tool = prog.regular_polygon_extrude(1.0, 6, 20.0)
        result = prog.cut(box, hex_tool)
        prog.query_volume(box)
        prog.query_volume(result)
        r = _exec(prog)
        assert r.queries[1] < r.queries[0]


# -- ChamferByFaceOp Tests --


class TestChamferByFaceOp:
    def test_chamfer_top_face_reduces_volume(self):
        """Chamfering top face edges of a cylinder should reduce volume."""
        prog = ShapeScriptBuilder()
        cyl = prog.cylinder(1.0, 2.0)
        chamfered = prog.chamfer_by_face(cyl, axis="z", end="max", size=0.1)
        prog.query_volume(cyl)
        prog.query_volume(chamfered)
        r = _exec(prog)
        assert r.queries[1] < r.queries[0]

    def test_chamfer_bottom_face_reduces_volume(self):
        """Chamfering bottom face edges should also reduce volume."""
        prog = ShapeScriptBuilder()
        cyl = prog.cylinder(1.0, 2.0)
        chamfered = prog.chamfer_by_face(cyl, axis="z", end="min", size=0.1)
        prog.query_volume(cyl)
        prog.query_volume(chamfered)
        r = _exec(prog)
        assert r.queries[1] < r.queries[0]

    def test_chamfer_box_top(self):
        """Chamfering top face of a box should reduce volume."""
        prog = ShapeScriptBuilder()
        box = prog.box(2.0, 2.0, 2.0)
        chamfered = prog.chamfer_by_face(box, axis="z", end="max", size=0.1)
        prog.query_volume(box)
        prog.query_volume(chamfered)
        r = _exec(prog)
        assert r.queries[1] < r.queries[0]

    def test_op_dataclass_frozen(self):
        """Op should be frozen."""
        from botcad.shapescript.ops import ChamferByFaceOp

        op = ChamferByFaceOp(
            ref=ShapeRef("cbf_0"),
            target=ShapeRef("a"),
            axis="z",
            end="max",
            size=0.1,
        )
        with pytest.raises(AttributeError):
            op.size = 0.2

    def test_passthrough_on_failure(self):
        """If chamfer fails (e.g. size too large), pass through without error."""
        prog = ShapeScriptBuilder()
        box = prog.box(1.0, 1.0, 1.0)
        # Size larger than half the box edge — should fail gracefully
        chamfered = prog.chamfer_by_face(box, axis="z", end="max", size=0.6)
        prog.query_volume(chamfered)
        r = _exec(prog)
        assert r.queries[0] > 0  # didn't crash


# -- Fastener without PrebuiltOp --


class TestFastenerNoPrebuilt:
    def test_fastener_script_has_no_prebuilt_ops(self):
        """fastener_script should not use any PrebuiltOp."""
        from botcad.fasteners import FastenerSpec, HeadType
        from botcad.shapescript.emit_components import fastener_script
        from botcad.shapescript.ops import PrebuiltOp

        spec = FastenerSpec(
            designation="M2",
            thread_diameter=Meters(0.002),
            thread_pitch=Meters(0.0004),
            head_type=HeadType.SOCKET_HEAD_CAP,
            head_diameter=Meters(0.0038),
            head_height=Meters(0.002),
            socket_size=Meters(0.0015),
            clearance_hole=Meters(0.0024),
            close_fit_hole=Meters(0.0022),
        )
        prog = fastener_script(spec, length=0.008)
        for op in prog.ops:
            assert not isinstance(op, PrebuiltOp), f"Found PrebuiltOp: {op}"

    def test_phillips_fastener_no_prebuilt(self):
        """Phillips fastener should also have no PrebuiltOp."""
        from botcad.fasteners import FastenerSpec, HeadType
        from botcad.shapescript.emit_components import fastener_script
        from botcad.shapescript.ops import PrebuiltOp

        spec = FastenerSpec(
            designation="M2",
            thread_diameter=Meters(0.002),
            thread_pitch=Meters(0.0004),
            head_type=HeadType.PAN_HEAD_PHILLIPS,
            head_diameter=Meters(0.004),
            head_height=Meters(0.0016),
            socket_size=Meters(0.0),
            clearance_hole=Meters(0.0024),
            close_fit_hole=Meters(0.0022),
        )
        prog = fastener_script(spec, length=0.008)
        for op in prog.ops:
            assert not isinstance(op, PrebuiltOp), f"Found PrebuiltOp: {op}"

    def test_socket_head_fastener_volume(self):
        """Socket head cap screw should have reasonable volume."""
        from botcad.fasteners import FastenerSpec, HeadType
        from botcad.shapescript.emit_components import fastener_script

        spec = FastenerSpec(
            designation="M2",
            thread_diameter=Meters(0.002),
            thread_pitch=Meters(0.0004),
            head_type=HeadType.SOCKET_HEAD_CAP,
            head_diameter=Meters(0.0038),
            head_height=Meters(0.002),
            socket_size=Meters(0.0015),
            clearance_hole=Meters(0.0024),
            close_fit_hole=Meters(0.0022),
        )
        prog = fastener_script(spec, length=0.008)
        prog.query_volume(prog.output_ref)
        r = _exec(prog)
        # Should have positive volume
        assert r.queries[0] > 0
