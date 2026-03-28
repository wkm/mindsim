"""Tests for the OCCT backend -- executes ShapeScript against build123d.

Property tests validate geometric invariants that must hold regardless
of backend implementation. These tests double as a backend conformance
suite for future backends (truck, mesh-CSG, etc.).
"""

from __future__ import annotations

import math

import pytest

b3d = pytest.importorskip("build123d")

from botcad.shapescript.backend_occt import ExecutionResult, OcctBackend  # noqa: E402
from botcad.shapescript.program import ShapeScriptBuilder  # noqa: E402

# -- Helpers --


def _exec(prog: ShapeScriptBuilder) -> ExecutionResult:
    return OcctBackend().execute(prog)


# -- Primitive Volume Tests --


class TestPrimitiveVolume:
    def test_box_volume(self):
        prog = ShapeScriptBuilder()
        b = prog.box(1.0, 2.0, 3.0)
        prog.query_volume(b)
        result = _exec(prog)
        assert result.queries[0] == pytest.approx(6.0, rel=1e-6)

    def test_cylinder_volume(self):
        prog = ShapeScriptBuilder()
        c = prog.cylinder(1.0, 2.0)
        prog.query_volume(c)
        result = _exec(prog)
        expected = math.pi * 1.0**2 * 2.0
        assert result.queries[0] == pytest.approx(expected, rel=1e-4)

    def test_sphere_volume(self):
        prog = ShapeScriptBuilder()
        s = prog.sphere(1.0)
        prog.query_volume(s)
        result = _exec(prog)
        expected = (4 / 3) * math.pi * 1.0**3
        assert result.queries[0] == pytest.approx(expected, rel=1e-4)

    def test_box_min_z_align(self):
        """MIN_Z aligned box has its bottom at z=0."""
        from botcad.shapescript.ops import Align3

        prog = ShapeScriptBuilder()
        b = prog.box(1.0, 1.0, 2.0, align=Align3(z="min"))
        prog.query_bbox(b)
        result = _exec(prog)
        (mn, mx) = result.queries[0]
        assert mn[2] == pytest.approx(0.0, abs=1e-9)
        assert mx[2] == pytest.approx(2.0, abs=1e-9)

    def test_box_min_x_align(self):
        """MIN X aligned box has its left face at x=0."""
        from botcad.shapescript.ops import Align3

        prog = ShapeScriptBuilder()
        b = prog.box(2.0, 1.0, 1.0, align=Align3(x="min"))
        prog.query_bbox(b)
        result = _exec(prog)
        (mn, mx) = result.queries[0]
        assert mn[0] == pytest.approx(0.0, abs=1e-9)
        assert mx[0] == pytest.approx(2.0, abs=1e-9)

    def test_box_max_y_align(self):
        """MAX Y aligned box has its top face at y=0."""
        from botcad.shapescript.ops import Align3

        prog = ShapeScriptBuilder()
        b = prog.box(1.0, 3.0, 1.0, align=Align3(y="max"))
        prog.query_bbox(b)
        result = _exec(prog)
        (mn, mx) = result.queries[0]
        assert mn[1] == pytest.approx(-3.0, abs=1e-9)
        assert mx[1] == pytest.approx(0.0, abs=1e-9)

    def test_cylinder_min_z_align(self):
        """MIN Z cylinder has bottom at z=0."""
        from botcad.shapescript.ops import Align3

        prog = ShapeScriptBuilder()
        c = prog.cylinder(0.5, 2.0, align=Align3(z="min"))
        prog.query_bbox(c)
        result = _exec(prog)
        (mn, mx) = result.queries[0]
        assert mn[2] == pytest.approx(0.0, abs=1e-9)
        assert mx[2] == pytest.approx(2.0, abs=1e-9)


# -- Boolean Invariants --


class TestBooleanInvariants:
    def test_cut_reduces_volume(self):
        prog = ShapeScriptBuilder()
        box = prog.box(1, 1, 1)
        hole = prog.cylinder(0.1, 2)
        result = prog.cut(box, hole)
        prog.query_volume(box)
        prog.query_volume(result)
        r = _exec(prog)
        assert r.queries[1] < r.queries[0]

    def test_cut_exact_volume(self):
        prog = ShapeScriptBuilder()
        box = prog.box(1, 1, 1)
        hole = prog.cylinder(0.1, 2)  # through-hole
        result = prog.cut(box, hole)
        prog.query_volume(result)
        r = _exec(prog)
        expected = 1.0 - math.pi * 0.1**2 * 1.0
        assert r.queries[0] == pytest.approx(expected, rel=0.01)

    def test_fuse_nonoverlapping_is_sum(self):
        prog = ShapeScriptBuilder()
        a = prog.box(1, 1, 1)
        b = prog.box(1, 1, 1)
        b = prog.locate(b, pos=(5.0, 0, 0))
        c = prog.fuse(a, b)
        prog.query_volume(c)
        r = _exec(prog)
        assert r.queries[0] == pytest.approx(2.0, rel=0.001)

    def test_fuse_overlapping_less_than_sum(self):
        prog = ShapeScriptBuilder()
        a = prog.box(1, 1, 1)
        b = prog.box(1, 1, 1)
        b = prog.locate(b, pos=(0.5, 0, 0))  # 50% overlap
        c = prog.fuse(a, b)
        prog.query_volume(a)
        prog.query_volume(c)
        r = _exec(prog)
        assert r.queries[1] < 2 * r.queries[0]
        assert r.queries[1] > r.queries[0]

    def test_cut_nonoverlapping_preserves_target(self):
        """Cut with non-overlapping tool should preserve target volume."""
        prog = ShapeScriptBuilder()
        box = prog.box(1, 1, 1)
        tool = prog.box(0.5, 0.5, 0.5)
        tool = prog.locate(tool, pos=(10, 10, 10))  # far away
        result = prog.cut(box, tool)
        prog.query_volume(box)
        prog.query_volume(result)
        r = _exec(prog)
        assert r.queries[0] == pytest.approx(r.queries[1], rel=1e-9)


# -- Transform Tests --


class TestTransforms:
    def test_locate_preserves_volume(self):
        prog = ShapeScriptBuilder()
        a = prog.box(1, 1, 1)
        b = prog.locate(a, pos=(10, 20, 30))
        prog.query_volume(a)
        prog.query_volume(b)
        r = _exec(prog)
        assert r.queries[0] == pytest.approx(r.queries[1], rel=1e-9)

    def test_locate_moves_centroid(self):
        prog = ShapeScriptBuilder()
        a = prog.box(1, 1, 1)
        b = prog.locate(a, pos=(10, 0, 0))
        prog.query_centroid(b)
        r = _exec(prog)
        cx, _cy, _cz = r.queries[0]
        assert cx == pytest.approx(10.0, abs=0.01)

    def test_locate_rotation(self):
        """90-degree rotation around Z swaps X and Y centroids."""
        prog = ShapeScriptBuilder()
        a = prog.box(2, 1, 1)
        a = prog.locate(a, pos=(5, 0, 0))
        b = prog.locate(a, euler_deg=(0, 0, 90))
        prog.query_centroid(b)
        r = _exec(prog)
        cx, cy, _cz = r.queries[0]
        # After 90 deg Z rotation: x=5 -> y=5, y=0 -> x=0
        assert abs(cx) < 0.1
        assert cy == pytest.approx(5.0, abs=0.1)

    def test_locate_preserves_bbox_size(self):
        """Translation should not change bounding box dimensions."""
        prog = ShapeScriptBuilder()
        a = prog.box(2, 3, 4)
        b = prog.locate(a, pos=(100, 200, 300))
        prog.query_bbox(a)
        prog.query_bbox(b)
        r = _exec(prog)
        (mn_a, mx_a) = r.queries[0]
        (mn_b, mx_b) = r.queries[1]
        for i in range(3):
            size_a = mx_a[i] - mn_a[i]
            size_b = mx_b[i] - mn_b[i]
            assert size_a == pytest.approx(size_b, rel=1e-9)


# -- Query Tests --


class TestQueries:
    def test_query_bbox(self):
        prog = ShapeScriptBuilder()
        b = prog.box(2, 4, 6)
        prog.query_bbox(b)
        r = _exec(prog)
        (mn, mx) = r.queries[0]
        assert mn[0] == pytest.approx(-1, abs=0.01)
        assert mx[0] == pytest.approx(1, abs=0.01)
        assert mn[1] == pytest.approx(-2, abs=0.01)
        assert mx[1] == pytest.approx(2, abs=0.01)

    def test_query_area(self):
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1)
        prog.query_area(b)
        r = _exec(prog)
        assert r.queries[0] == pytest.approx(6.0, rel=1e-6)

    def test_query_inertia_returns_3x3(self):
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1)
        prog.query_inertia(b)
        r = _exec(prog)
        mat = r.queries[0]
        assert len(mat) == 3
        assert len(mat[0]) == 3

    def test_multiple_queries(self):
        prog = ShapeScriptBuilder()
        b = prog.box(2, 3, 4)
        prog.query_volume(b)
        prog.query_area(b)
        prog.query_centroid(b)
        r = _exec(prog)
        assert len(r.queries) == 3
        assert r.queries[0] == pytest.approx(24.0, rel=1e-6)  # volume

    def test_query_centroid_centered_box(self):
        """Centered box should have centroid at origin."""
        prog = ShapeScriptBuilder()
        b = prog.box(2, 4, 6)
        prog.query_centroid(b)
        r = _exec(prog)
        cx, cy, cz = r.queries[0]
        assert cx == pytest.approx(0.0, abs=1e-9)
        assert cy == pytest.approx(0.0, abs=1e-9)
        assert cz == pytest.approx(0.0, abs=1e-9)

    def test_sphere_area(self):
        """Sphere surface area = 4*pi*r^2."""
        prog = ShapeScriptBuilder()
        s = prog.sphere(1.0)
        prog.query_area(s)
        r = _exec(prog)
        expected = 4 * math.pi * 1.0**2
        assert r.queries[0] == pytest.approx(expected, rel=1e-3)


# -- Tag-Based Fillet Tests --


class TestFilletWithTags:
    def test_fillet_reduces_volume_slightly(self):
        """Fillet on a box removes corner material."""
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1, tag="edges")
        f = prog.fillet(b, tags=("edges",), radius=0.05)
        prog.query_volume(b)
        prog.query_volume(f)
        r = _exec(prog)
        assert r.queries[1] < r.queries[0]  # fillet removes material
        assert r.queries[1] > r.queries[0] * 0.90  # but not much

    def test_fillet_on_cut_edges(self):
        """Tags from a cylinder propagate through cut, fillet resolves them."""
        prog = ShapeScriptBuilder()
        box = prog.box(1, 1, 1)
        hole = prog.cylinder(0.2, 2, tag="hole")
        result = prog.cut(box, hole)
        filleted = prog.fillet(result, tags=("hole",), radius=0.02)
        prog.query_volume(filleted)
        r = _exec(prog)
        assert r.queries[0] > 0  # didn't crash, produced valid geometry


# -- PrebuiltOp Tests --


class TestPrebuiltOp:
    def test_prebuilt_solid_injection(self):
        """PrebuiltOp correctly injects a pre-built solid."""
        from build123d import Box

        from botcad.shapescript.ops import PrebuiltOp

        prog = ShapeScriptBuilder()
        prebuilt_box = Box(1, 1, 1)
        ref = prog._next_ref("pre")
        prog.ops.append(PrebuiltOp(ref=ref, solid_hash="abc123", tag="bracket"))
        prog.prebuilt_solids[ref.id] = prebuilt_box
        prog.query_volume(ref)

        r = _exec(prog)
        assert r.queries[0] == pytest.approx(1.0, rel=1e-6)

    def test_prebuilt_missing_raises(self):
        """PrebuiltOp without associated solid raises ValueError."""
        from botcad.shapescript.ops import PrebuiltOp, ShapeRef

        prog = ShapeScriptBuilder()
        ref = ShapeRef("missing_0")
        prog.ops.append(PrebuiltOp(ref=ref, solid_hash="abc123"))

        with pytest.raises(ValueError, match="no associated solid"):
            _exec(prog)


# -- Tag Propagation Integration --


class TestTagPropagation:
    def test_tags_propagate_through_cut(self):
        """Tags from tool propagate through cut op."""
        prog = ShapeScriptBuilder()
        box = prog.box(1, 1, 1, tag="shell")
        hole = prog.cylinder(0.1, 2, tag="pocket")
        result = prog.cut(box, hole)
        r = _exec(prog)
        # Result should have both tags
        assert "shell" in r.tags.tags_on(result)
        assert "pocket" in r.tags.tags_on(result)

    def test_tags_propagate_through_fuse(self):
        """Tags from both operands propagate through fuse."""
        prog = ShapeScriptBuilder()
        a = prog.box(1, 1, 1, tag="base")
        b = prog.box(0.5, 0.5, 0.5, tag="mount")
        b = prog.locate(b, pos=(2, 0, 0))
        c = prog.fuse(a, b)
        r = _exec(prog)
        assert "base" in r.tags.tags_on(c)
        assert "mount" in r.tags.tags_on(c)

    def test_tags_propagate_through_locate(self):
        """Tags survive a locate transform."""
        prog = ShapeScriptBuilder()
        a = prog.cylinder(0.5, 1.0, tag="shaft")
        b = prog.locate(a, pos=(5, 5, 5))
        r = _exec(prog)
        assert "shaft" in r.tags.tags_on(b)


# -- Shape Table Tests --


class TestShapeTable:
    def test_all_shape_producing_ops_in_table(self):
        """Every shape-producing op's ref should be in result.shapes."""
        prog = ShapeScriptBuilder()
        a = prog.box(1, 1, 1)
        b = prog.cylinder(0.5, 2)
        c = prog.fuse(a, b)
        d = prog.locate(c, pos=(1, 0, 0))
        r = _exec(prog)
        for ref in [a, b, c, d]:
            assert ref.id in r.shapes

    def test_execution_result_has_tags(self):
        """ExecutionResult includes a populated TagRegistry."""
        prog = ShapeScriptBuilder()
        prog.box(1, 1, 1, tag="shell")
        r = _exec(prog)
        assert isinstance(r.tags, type(r.tags))
        assert "shell" in r.tags.tags_on(prog.ops[0].ref)


# -- CallOp Backend Tests --


class TestCallOp:
    def test_call_produces_solid(self):
        """Sub-program box -> query volume = 1.0."""
        sub = ShapeScriptBuilder()
        b = sub.box(1.0, 1.0, 1.0)
        sub.output_ref = b

        prog = ShapeScriptBuilder()
        prog.sub_programs["unit_box"] = sub
        ref = prog.call("unit_box")
        prog.query_volume(ref)
        r = _exec(prog)
        assert r.queries[0] == pytest.approx(1.0, rel=1e-6)

    def test_call_then_cut(self):
        """Call result usable in subsequent boolean ops."""
        sub = ShapeScriptBuilder()
        b = sub.box(2.0, 2.0, 2.0)
        sub.output_ref = b

        prog = ShapeScriptBuilder()
        prog.sub_programs["big_box"] = sub
        base = prog.call("big_box")
        hole = prog.cylinder(0.5, 10.0)
        result = prog.cut(base, hole)
        prog.query_volume(result)
        r = _exec(prog)
        expected = 8.0 - math.pi * 0.5**2 * 2.0
        assert r.queries[0] == pytest.approx(expected, rel=0.01)

    def test_call_with_locate(self):
        """Call result can be moved."""
        sub = ShapeScriptBuilder()
        b = sub.box(1.0, 1.0, 1.0)
        sub.output_ref = b

        prog = ShapeScriptBuilder()
        prog.sub_programs["unit_box"] = sub
        ref = prog.call("unit_box")
        moved = prog.locate(ref, pos=(10.0, 0, 0))
        prog.query_centroid(moved)
        r = _exec(prog)
        cx, _cy, _cz = r.queries[0]
        assert cx == pytest.approx(10.0, abs=0.01)

    def test_same_sub_program_called_twice(self):
        """Two calls produce independent shapes, fused volume = 2.0."""
        sub = ShapeScriptBuilder()
        b = sub.box(1.0, 1.0, 1.0)
        sub.output_ref = b

        prog = ShapeScriptBuilder()
        prog.sub_programs["unit_box"] = sub
        a = prog.call("unit_box")
        b = prog.call("unit_box")
        b = prog.locate(b, pos=(5.0, 0, 0))  # non-overlapping
        fused = prog.fuse(a, b)
        prog.query_volume(fused)
        r = _exec(prog)
        assert r.queries[0] == pytest.approx(2.0, rel=0.001)

    def test_call_missing_sub_program_raises(self):
        """CallOp with missing sub-program raises ValueError."""
        prog = ShapeScriptBuilder()
        prog.call("nonexistent")
        with pytest.raises(ValueError, match="sub-program 'nonexistent' not found"):
            _exec(prog)

    def test_call_with_tag(self):
        """CallOp tag is declared in the tag registry."""
        sub = ShapeScriptBuilder()
        b = sub.box(1.0, 1.0, 1.0)
        sub.output_ref = b

        prog = ShapeScriptBuilder()
        prog.sub_programs["unit_box"] = sub
        ref = prog.call("unit_box", tag="bracket")
        r = _exec(prog)
        assert "bracket" in r.tags.tags_on(ref)


# -- FilletAllEdgesOp / FilletByAxisOp Tests --


class TestFilletOps:
    def test_fillet_all_reduces_volume(self):
        """FilletAllEdgesOp removes corner material from a box."""
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1)
        f = prog.fillet_all(b, 0.05)
        prog.query_volume(b)
        prog.query_volume(f)
        r = _exec(prog)
        assert r.queries[1] < r.queries[0]  # fillet removes material

    def test_fillet_by_axis_z(self):
        """FilletByAxisOp on Z-aligned edges produces valid geometry."""
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1)
        f = prog.fillet_by_axis(b, "z", 0.05)
        prog.query_volume(f)
        r = _exec(prog)
        assert r.queries[0] > 0

    def test_fillet_by_axis_removes_less_than_fillet_all(self):
        """Axis-filtered fillet removes less material than fillet-all."""
        prog = ShapeScriptBuilder()
        b = prog.box(1, 1, 1)
        fa = prog.fillet_all(b, 0.05)
        fz = prog.fillet_by_axis(b, "z", 0.05)
        prog.query_volume(fa)
        prog.query_volume(fz)
        r = _exec(prog)
        assert r.queries[0] < r.queries[1]  # all < axis (more edges filleted)
