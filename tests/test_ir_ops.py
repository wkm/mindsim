"""Tests for CadIR operation dataclasses, CadProgram, and TagRegistry."""
from __future__ import annotations

import pytest
from botcad.ir.ops import (
    Align3,
    BoxOp,
    ChamferOp,
    CutOp,
    CylinderOp,
    ExportSTEPOp,
    ExportSTLOp,
    FilletOp,
    FuseOp,
    LocateOp,
    PrebuiltOp,
    QueryAreaOp,
    QueryBBoxOp,
    QueryCentroidOp,
    QueryInertiaOp,
    QueryVolumeOp,
    ShapeRef,
    SphereOp,
)


class TestShapeRef:
    def test_equality(self):
        assert ShapeRef("a") == ShapeRef("a")

    def test_hashable(self):
        s = {ShapeRef("a"), ShapeRef("b"), ShapeRef("a")}
        assert len(s) == 2

    def test_str(self):
        assert str(ShapeRef("box_0")) == "box_0"


class TestPrimitiveOps:
    def test_box_is_frozen(self):
        op = BoxOp(ref=ShapeRef("b0"), width=0.06, length=0.04, height=0.02)
        with pytest.raises(AttributeError):
            op.width = 0.1

    def test_box_defaults(self):
        op = BoxOp(ref=ShapeRef("b0"), width=1, length=1, height=1)
        assert op.align == Align3.CENTER
        assert op.tag is None

    def test_box_with_tag(self):
        op = BoxOp(ref=ShapeRef("b0"), width=1, length=1, height=1, tag="shell")
        assert op.tag == "shell"

    def test_cylinder_fields(self):
        op = CylinderOp(ref=ShapeRef("c0"), radius=0.01, height=0.05, tag="pocket")
        assert op.radius == 0.01
        assert op.height == 0.05
        assert op.tag == "pocket"

    def test_sphere_fields(self):
        op = SphereOp(ref=ShapeRef("s0"), radius=0.02)
        assert op.radius == 0.02

    def test_prebuilt_fields(self):
        op = PrebuiltOp(ref=ShapeRef("p0"), solid_hash="abc123", tag="bracket")
        assert op.solid_hash == "abc123"
        assert op.tag == "bracket"

    def test_prebuilt_is_frozen(self):
        op = PrebuiltOp(ref=ShapeRef("p0"), solid_hash="abc123")
        with pytest.raises(AttributeError):
            op.solid_hash = "xyz"


class TestBooleanOps:
    def test_fuse_refs(self):
        op = FuseOp(ref=ShapeRef("f0"), target=ShapeRef("a"), tool=ShapeRef("b"))
        assert op.target == ShapeRef("a")
        assert op.tool == ShapeRef("b")

    def test_cut_refs(self):
        op = CutOp(ref=ShapeRef("c0"), target=ShapeRef("a"), tool=ShapeRef("b"))
        assert op.target == ShapeRef("a")


class TestTransformOps:
    def test_locate_fields(self):
        op = LocateOp(
            ref=ShapeRef("l0"),
            target=ShapeRef("a"),
            pos=(0.01, 0.0, 0.0),
            euler_deg=(0.0, 0.0, 0.0),
        )
        assert op.pos == (0.01, 0.0, 0.0)


class TestModificationOps:
    def test_fillet_tags(self):
        op = FilletOp(
            ref=ShapeRef("f0"),
            target=ShapeRef("a"),
            tags=("pocket",),
            radius=0.002,
        )
        assert op.tags == ("pocket",)
        assert op.radius == 0.002

    def test_chamfer_tags(self):
        op = ChamferOp(
            ref=ShapeRef("ch0"),
            target=ShapeRef("a"),
            tags=("edge1",),
            size=0.001,
        )
        assert op.size == 0.001


class TestQueryOps:
    def test_query_volume(self):
        op = QueryVolumeOp(target=ShapeRef("a"))
        assert op.target == ShapeRef("a")

    def test_query_centroid(self):
        op = QueryCentroidOp(target=ShapeRef("a"))
        assert op.target == ShapeRef("a")

    def test_query_inertia(self):
        op = QueryInertiaOp(target=ShapeRef("a"))
        assert op.target == ShapeRef("a")

    def test_query_bbox(self):
        op = QueryBBoxOp(target=ShapeRef("a"))
        assert op.target == ShapeRef("a")

    def test_query_area(self):
        op = QueryAreaOp(target=ShapeRef("a"))
        assert op.target == ShapeRef("a")


class TestExportOps:
    def test_export_stl(self):
        op = ExportSTLOp(target=ShapeRef("a"), path="meshes/base.stl")
        assert op.path == "meshes/base.stl"

    def test_export_step(self):
        op = ExportSTEPOp(targets=(ShapeRef("a"), ShapeRef("b")), path="assembly.step")
        assert len(op.targets) == 2


class TestOpHashing:
    def test_identical_ops_same_hash(self):
        a = BoxOp(ref=ShapeRef("b0"), width=1, length=1, height=1)
        b = BoxOp(ref=ShapeRef("b0"), width=1, length=1, height=1)
        assert hash(a) == hash(b)

    def test_different_ops_different_hash(self):
        a = BoxOp(ref=ShapeRef("b0"), width=1, length=1, height=1)
        b = BoxOp(ref=ShapeRef("b0"), width=2, length=1, height=1)
        assert hash(a) != hash(b)


# ── Task 2: CadProgram tests ──

from botcad.ir.program import CadProgram


class TestCadProgram:
    def test_box_returns_shape_ref(self):
        prog = CadProgram()
        ref = prog.box(0.06, 0.04, 0.02)
        assert isinstance(ref, ShapeRef)

    def test_ops_are_recorded(self):
        prog = CadProgram()
        prog.box(1, 1, 1)
        assert len(prog.ops) == 1
        assert isinstance(prog.ops[0], BoxOp)

    def test_cut_records_both_refs(self):
        prog = CadProgram()
        a = prog.box(1, 1, 1)
        b = prog.cylinder(0.1, 2)
        c = prog.cut(a, b)
        assert len(prog.ops) == 3
        cut_op = prog.ops[2]
        assert isinstance(cut_op, CutOp)
        assert cut_op.target == a
        assert cut_op.tool == b
        assert cut_op.ref == c

    def test_content_hash_deterministic(self):
        p1 = CadProgram()
        a = p1.box(1, 1, 1)
        b = p1.cylinder(0.1, 2)
        p1.cut(a, b)

        p2 = CadProgram()
        a = p2.box(1, 1, 1)
        b = p2.cylinder(0.1, 2)
        p2.cut(a, b)

        assert p1.content_hash() == p2.content_hash()

    def test_content_hash_changes_with_params(self):
        p1 = CadProgram()
        p1.box(1, 1, 1)

        p2 = CadProgram()
        p2.box(2, 1, 1)

        assert p1.content_hash() != p2.content_hash()

    def test_locate_with_defaults(self):
        prog = CadProgram()
        a = prog.box(1, 1, 1)
        b = prog.locate(a, pos=(0.01, 0, 0))
        assert isinstance(prog.ops[1], LocateOp)
        assert prog.ops[1].euler_deg == (0.0, 0.0, 0.0)

    def test_fillet_with_tags(self):
        prog = CadProgram()
        a = prog.box(1, 1, 1, tag="edges")
        b = prog.fillet(a, tags=("edges",), radius=0.05)
        assert isinstance(prog.ops[1], FilletOp)
        assert prog.ops[1].tags == ("edges",)

    def test_query_volume(self):
        prog = CadProgram()
        a = prog.box(1, 1, 1)
        prog.query_volume(a)
        assert len(prog.ops) == 2
        assert isinstance(prog.ops[1], QueryVolumeOp)

    def test_to_json_roundtrip(self):
        prog = CadProgram()
        a = prog.box(1, 1, 1, tag="shell")
        b = prog.cylinder(0.1, 2, tag="pocket")
        prog.cut(a, b)
        prog.query_volume(prog.ops[2].ref)

        json_str = prog.to_json()
        prog2 = CadProgram.from_json(json_str)
        assert prog.content_hash() == prog2.content_hash()
        assert len(prog2.ops) == len(prog.ops)

    def test_output_ref_field(self):
        prog = CadProgram()
        assert prog.output_ref is None
        ref = prog.box(1, 1, 1)
        prog.output_ref = ref
        assert prog.output_ref == ref

    def test_prebuilt_solids_field(self):
        prog = CadProgram()
        assert prog.prebuilt_solids == {}
        prog.prebuilt_solids["pre_0"] = object()
        assert "pre_0" in prog.prebuilt_solids

    def test_content_hash_includes_prebuilt_hashes(self):
        """Programs with different prebuilt solid hashes produce different content hashes."""
        p1 = CadProgram()
        p1.ops.append(PrebuiltOp(ref=ShapeRef("pre_0"), solid_hash="hash_a"))
        p1._counter = 1

        p2 = CadProgram()
        p2.ops.append(PrebuiltOp(ref=ShapeRef("pre_0"), solid_hash="hash_b"))
        p2._counter = 1

        assert p1.content_hash() != p2.content_hash()


# ── Task 3: TagRegistry tests ──

from botcad.ir.tags import TagRegistry


class TestTagRegistry:
    def test_declare_tag(self):
        reg = TagRegistry()
        reg.declare("pocket", ShapeRef("c0"))
        assert reg.source_ref("pocket") == ShapeRef("c0")

    def test_unknown_tag_raises(self):
        reg = TagRegistry()
        with pytest.raises(KeyError, match="pocket"):
            reg.source_ref("pocket")

    def test_propagate_through_cut(self):
        """When shape A is cut by tagged shape B, result inherits B's tags."""
        reg = TagRegistry()
        reg.declare("pocket", ShapeRef("tool"))
        reg.propagate_boolean(
            result_ref=ShapeRef("cut_0"),
            target_ref=ShapeRef("box"),
            tool_ref=ShapeRef("tool"),
        )
        assert "pocket" in reg.tags_on(ShapeRef("cut_0"))

    def test_propagate_through_fuse(self):
        """Union inherits tags from both operands."""
        reg = TagRegistry()
        reg.declare("shell", ShapeRef("a"))
        reg.declare("bracket", ShapeRef("b"))
        reg.propagate_boolean(
            result_ref=ShapeRef("fuse_0"),
            target_ref=ShapeRef("a"),
            tool_ref=ShapeRef("b"),
        )
        tags = reg.tags_on(ShapeRef("fuse_0"))
        assert "shell" in tags
        assert "bracket" in tags

    def test_propagate_through_transform(self):
        """Locate/move inherits tags from source."""
        reg = TagRegistry()
        reg.declare("pocket", ShapeRef("c0"))
        reg.propagate_transform(
            result_ref=ShapeRef("loc_0"),
            source_ref=ShapeRef("c0"),
        )
        assert "pocket" in reg.tags_on(ShapeRef("loc_0"))

    def test_tags_on_untagged_shape(self):
        reg = TagRegistry()
        assert reg.tags_on(ShapeRef("unknown")) == frozenset()
