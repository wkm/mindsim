"""Tests for the ShapeScript → CadStep conversion pipeline.

Validates that every op type can be converted to CadStep objects
and formatted via format_op() without crashing. This catches missing
imports in cad_steps.py.
"""

from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")

from botcad.shapescript.backend_occt import OcctBackend  # noqa: E402
from botcad.shapescript.cad_steps import (  # noqa: E402
    format_op,
    shapescript_to_cad_steps,
)
from botcad.shapescript.program import ShapeScriptBuilder  # noqa: E402


def _exec(prog):
    return OcctBackend().execute(prog)


class TestAllOpTypesConvertToSteps:
    """Every op type must survive shapescript_to_cad_steps + format_op."""

    def test_box(self):
        prog = ShapeScriptBuilder()
        prog.output_ref = prog.box(1, 1, 1, tag="test")
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert len(steps) >= 1
        assert all(s.script for s in steps)

    def test_cylinder(self):
        prog = ShapeScriptBuilder()
        prog.output_ref = prog.cylinder(0.5, 1, tag="test")
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert len(steps) >= 1

    def test_sphere(self):
        prog = ShapeScriptBuilder()
        prog.output_ref = prog.sphere(0.5)
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert len(steps) >= 1

    def test_copy(self):
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1, tag="proto")
        c = prog.copy(b, tag="clone")
        prog.output_ref = c
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert any("Copy" in s.script for s in steps)

    def test_cut(self):
        prog = ShapeScriptBuilder()
        a = prog.box(1, 1, 1)
        b = prog.cylinder(0.1, 2)
        prog.output_ref = prog.cut(a, b)
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert any("Cut" in s.script for s in steps)

    def test_fuse(self):
        prog = ShapeScriptBuilder()
        a = prog.box(1, 1, 1)
        b = prog.box(1, 1, 1)
        b = prog.locate(b, pos=(2, 0, 0))
        prog.output_ref = prog.fuse(a, b)
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert any("Fuse" in s.script for s in steps)

    def test_locate(self):
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1)
        prog.output_ref = prog.locate(b, pos=(5, 0, 0))
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert any("Locate" in s.script for s in steps)

    def test_fillet_all(self):
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1)
        prog.output_ref = prog.fillet_all(b, 0.05)
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert any("FilletAll" in s.script for s in steps)

    def test_fillet_by_axis(self):
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1)
        prog.output_ref = prog.fillet_by_axis(b, "z", 0.05)
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert any("FilletAxis" in s.script for s in steps)

    def test_radial_array(self):
        prog = ShapeScriptBuilder()
        b = prog.box(0.01, 0.002, 0.01)
        b = prog.locate(b, pos=(0.05, 0, 0))
        prog.output_ref = prog.radial_array(b, 6, axis="z", tag="spokes")
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert any("RadialArray" in s.script for s in steps)

    def test_call(self):
        sub = ShapeScriptBuilder()
        sub.output_ref = sub.box(1, 1, 1)
        prog = ShapeScriptBuilder()
        prog.sub_programs["test_sub"] = sub
        prog.output_ref = prog.call("test_sub", tag="called")
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert any("Call" in s.script for s in steps)

    def test_prebuilt(self):
        from build123d import Box

        solid = Box(1, 1, 1)
        prog = ShapeScriptBuilder()
        prog.output_ref = prog.prebuilt(solid, tag="external")
        result = _exec(prog)
        steps = shapescript_to_cad_steps(prog, result)
        assert any("Prebuilt" in s.script for s in steps)


class TestFormatOpAllTypes:
    """format_op must handle every op type without NameError."""

    def test_all_ops(self):
        from botcad.shapescript.ops import (
            BoxOp,
            CallOp,
            ChamferOp,
            CopyOp,
            CutOp,
            CylinderOp,
            FilletAllEdgesOp,
            FilletByAxisOp,
            FilletOp,
            FuseOp,
            LocateOp,
            PrebuiltOp,
            RadialArrayOp,
            ShapeRef,
            SphereOp,
        )

        ops = [
            BoxOp(ref=ShapeRef("b"), width=1, length=1, height=1),
            CylinderOp(ref=ShapeRef("c"), radius=0.5, height=1),
            SphereOp(ref=ShapeRef("s"), radius=0.5),
            PrebuiltOp(ref=ShapeRef("p"), solid_hash="abc"),
            CallOp(ref=ShapeRef("call"), sub_program_key="test"),
            CopyOp(ref=ShapeRef("cp"), source=ShapeRef("b")),
            RadialArrayOp(ref=ShapeRef("arr"), source=ShapeRef("b"), count=6),
            FuseOp(ref=ShapeRef("f"), target=ShapeRef("a"), tool=ShapeRef("b")),
            CutOp(ref=ShapeRef("ct"), target=ShapeRef("a"), tool=ShapeRef("b")),
            LocateOp(ref=ShapeRef("l"), target=ShapeRef("a"), pos=(1, 0, 0)),
            FilletOp(ref=ShapeRef("fi"), target=ShapeRef("a"), tags=("e",), radius=0.1),
            FilletAllEdgesOp(ref=ShapeRef("fa"), target=ShapeRef("a"), radius=0.1),
            FilletByAxisOp(
                ref=ShapeRef("fb"), target=ShapeRef("a"), axis="z", radius=0.1
            ),
            ChamferOp(ref=ShapeRef("ch"), target=ShapeRef("a"), tags=("e",), size=0.1),
        ]

        for op in ops:
            result = format_op(op)
            assert isinstance(result, str), (
                f"format_op({type(op).__name__}) returned {type(result)}"
            )
            assert len(result) > 0, (
                f"format_op({type(op).__name__}) returned empty string"
            )
