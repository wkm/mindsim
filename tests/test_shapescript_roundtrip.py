"""Golden-file roundtrip tests: ShapeScript path vs direct build123d path.

Tests build bodies through both paths and compare volumes/bounding boxes.
Uses synthetic bodies first (fast), then real bots (slower, marked).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

BOTS_DIR = Path(__file__).parent.parent / "bots"
BOT_DESIGNS = sorted(BOTS_DIR.glob("*/design.py"))


def _clear_all_caches():
    """Clear lru_caches that hold build123d shapes.

    The direct path uses .locate() which mutates cached shapes in-place.
    We must clear caches between runs to avoid cross-contamination.
    """
    from botcad.bracket import (
        bracket_envelope,
        bracket_solid,
        coupler_solid,
        cradle_envelope,
        cradle_solid,
    )
    from botcad.emit.cad import _make_body_solid, _make_wheel_solid

    for fn in [
        bracket_envelope,
        bracket_solid,
        coupler_solid,
        cradle_envelope,
        cradle_solid,
        _make_body_solid,
        _make_wheel_solid,
    ]:
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()


def _build_direct(body, pj=None, wire_segs=None):
    """Build via direct build123d path."""
    from botcad.emit.cad import _make_body_solid

    return _make_body_solid(body, pj, wire_segs)


def _build_ir(body, pj=None, wire_segs=None):
    """Build via ShapeScript path."""
    from botcad.shapescript.backend_occt import OcctBackend
    from botcad.shapescript.emit_body import emit_body_ir

    prog = emit_body_ir(body, pj, wire_segs)
    backend = OcctBackend()
    result = backend.execute(prog)
    if prog.output_ref is None:
        return None
    return result.shapes.get(prog.output_ref.id)


def _get_volume(solid):
    return abs(solid.volume)


def _get_bbox(solid):
    bb = solid.bounding_box()
    return (
        (bb.min.X, bb.min.Y, bb.min.Z),
        (bb.max.X, bb.max.Y, bb.max.Z),
    )


def _assert_solids_match(ir_solid, direct_solid, label: str):
    """Assert volume and bbox match between two solids."""
    if direct_solid is None:
        assert ir_solid is None, f"{label}: direct=None but IR produced a solid"
        return
    assert ir_solid is not None, f"{label}: direct produced solid but IR=None"

    direct_vol = _get_volume(direct_solid)
    ir_vol = _get_volume(ir_solid)
    assert ir_vol == pytest.approx(direct_vol, rel=0.001), (
        f"{label}: volume mismatch — direct={direct_vol:.6e}, IR={ir_vol:.6e}"
    )

    direct_bb = _get_bbox(direct_solid)
    ir_bb = _get_bbox(ir_solid)
    for i, axis in enumerate(("X", "Y", "Z")):
        assert ir_bb[0][i] == pytest.approx(direct_bb[0][i], abs=0.0001), (
            f"{label}: bbox min {axis} — direct={direct_bb[0][i]:.6f}, "
            f"IR={ir_bb[0][i]:.6f}"
        )
        assert ir_bb[1][i] == pytest.approx(direct_bb[1][i], abs=0.0001), (
            f"{label}: bbox max {axis} — direct={direct_bb[1][i]:.6f}, "
            f"IR={ir_bb[1][i]:.6f}"
        )


# ── Synthetic body tests (fast, no bracket booleans) ──


class TestIREmitBasicShapes:
    """Test IR emission for basic body shapes without joints."""

    def test_box_body(self):
        from botcad.skeleton import Body, BodyShape

        body = Body(
            name="test_box", shape=BodyShape.BOX, explicit_dimensions=(0.06, 0.04, 0.02)
        )

        _clear_all_caches()
        ir = _build_ir(body)
        _clear_all_caches()
        direct = _build_direct(body)
        _assert_solids_match(ir, direct, "box")

    def test_cylinder_body(self):
        from botcad.skeleton import Body, BodyShape

        body = Body(name="test_cyl", shape=BodyShape.CYLINDER, radius=0.02, width=0.03)

        _clear_all_caches()
        ir = _build_ir(body)
        _clear_all_caches()
        direct = _build_direct(body)
        _assert_solids_match(ir, direct, "cylinder")

    def test_sphere_body(self):
        from botcad.skeleton import Body, BodyShape

        body = Body(name="test_sph", shape=BodyShape.SPHERE, radius=0.015)

        _clear_all_caches()
        ir = _build_ir(body)
        _clear_all_caches()
        direct = _build_direct(body)
        _assert_solids_match(ir, direct, "sphere")

    def test_tube_body(self):
        from botcad.skeleton import Body, BodyShape

        body = Body(name="test_tube", shape=BodyShape.TUBE, outer_r=0.01, length=0.08)

        _clear_all_caches()
        ir = _build_ir(body)
        _clear_all_caches()
        direct = _build_direct(body)
        _assert_solids_match(ir, direct, "tube")

    def test_jaw_body(self):
        from botcad.skeleton import Body, BodyShape

        body = Body(
            name="test_jaw",
            shape=BodyShape.JAW,
            jaw_width=0.03,
            jaw_thickness=0.005,
            jaw_length=0.04,
        )

        _clear_all_caches()
        ir = _build_ir(body)
        _clear_all_caches()
        direct = _build_direct(body)
        _assert_solids_match(ir, direct, "jaw")


class TestIREmitComponentPockets:
    """Test component pocket cutting."""

    def test_box_pocket(self):
        from botcad.component import Component
        from botcad.skeleton import Body, BodyShape

        body = Body(
            name="test_pocket",
            shape=BodyShape.BOX,
            explicit_dimensions=(0.06, 0.04, 0.03),
        )
        comp = Component(name="sensor", dimensions=(0.02, 0.01, 0.005), mass=0.01)
        body.mount(comp, position="center")
        # Manually resolve mount position (normally done by packing solver)
        body.mounts[0].resolved_pos = (0.0, 0.0, 0.0)

        _clear_all_caches()
        ir = _build_ir(body)
        _clear_all_caches()
        direct = _build_direct(body)
        _assert_solids_match(ir, direct, "box_pocket")

    def test_bearing_pocket(self):
        from botcad.component import BearingSpec
        from botcad.skeleton import Body, BodyShape

        body = Body(
            name="test_bearing",
            shape=BodyShape.BOX,
            explicit_dimensions=(0.06, 0.04, 0.03),
        )
        bearing = BearingSpec(
            name="608zz",
            dimensions=(0.022, 0.022, 0.007),
            mass=0.012,
            od=0.022,
            id=0.008,
            width=0.007,
        )
        body.mount(bearing, position="center")
        body.mounts[0].resolved_pos = (0.0, 0.0, 0.0)

        _clear_all_caches()
        ir = _build_ir(body)
        _clear_all_caches()
        direct = _build_direct(body)
        _assert_solids_match(ir, direct, "bearing_pocket")


class TestIREmitCylinderOrientation:
    """Test cylinder body orientation when attached to a joint."""

    def test_cylinder_child_z_axis(self):
        """Child cylinder on Z axis — identity orientation."""
        from botcad.components.servo import STS3215
        from botcad.skeleton import Body, BodyShape, Joint

        servo = STS3215()
        joint = Joint(name="j1", servo=servo, axis=(0.0, 0.0, 1.0), pos=(0.0, 0.0, 0.0))
        child = Body(
            name="child_cyl", shape=BodyShape.CYLINDER, radius=0.015, width=0.02
        )
        joint.child = child

        _clear_all_caches()
        ir = _build_ir(child, pj=joint)
        _clear_all_caches()
        direct = _build_direct(child, pj=joint)
        _assert_solids_match(ir, direct, "cyl_z_axis")

    def test_cylinder_child_x_axis(self):
        """Child cylinder on X axis — 90 rotation."""
        from botcad.components.servo import STS3215
        from botcad.geometry import rotation_between
        from botcad.skeleton import Body, BodyShape, Joint

        servo = STS3215()
        joint = Joint(name="j2", servo=servo, axis=(1.0, 0.0, 0.0), pos=(0.0, 0.0, 0.0))
        child = Body(
            name="child_cyl_x", shape=BodyShape.CYLINDER, radius=0.015, width=0.02
        )
        child.frame_quat = rotation_between((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
        joint.child = child

        _clear_all_caches()
        ir = _build_ir(child, pj=joint)
        _clear_all_caches()
        direct = _build_direct(child, pj=joint)
        _assert_solids_match(ir, direct, "cyl_x_axis")


# ── Real bot tests (slower, per-body with timeout) ──


def _load_bot(design_path: Path):
    bot_name = design_path.parent.name
    module_name = f"bots.{bot_name}.design"
    spec = importlib.util.spec_from_file_location(module_name, design_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    for fn_name in ("design", "build", "make_bot", "create_bot"):
        if hasattr(mod, fn_name):
            bot = getattr(mod, fn_name)()
            bot.solve()
            return bot
    if hasattr(mod, "bot"):
        bot = mod.bot
        if not bot.all_bodies:
            bot.solve()
        return bot
    raise RuntimeError(f"No design()/build()/bot found in {design_path}")


@pytest.fixture(scope="module", params=[p.parent.name for p in BOT_DESIGNS], ids=str)
def bot_fixture(request):
    bot_name = request.param
    return _load_bot(BOTS_DIR / bot_name / "design.py")


@pytest.mark.slow
class TestIRRoundtripBots:
    """Full roundtrip tests for real bot designs."""

    @pytest.mark.timeout(120)
    def test_all_bodies_match(self, bot_fixture):
        bot = bot_fixture
        pj_map = {}
        for joint in bot.all_joints:
            if joint.child is not None:
                pj_map[joint.child.name] = joint
        ws_map = {}
        for route in bot.wire_routes:
            for seg in route.segments:
                ws_map.setdefault(seg.body_name, []).append((seg, route.bus_type))

        for body in bot.all_bodies:
            pj = pj_map.get(body.name)
            wire_segs = tuple(ws_map.get(body.name, [])) or None

            _clear_all_caches()
            ir_solid = _build_ir(body, pj, wire_segs)
            _clear_all_caches()
            direct_solid = _build_direct(body, pj, wire_segs)
            _assert_solids_match(ir_solid, direct_solid, f"bot:{bot.name}/{body.name}")


@pytest.mark.slow
class TestBuildCadIntegration:
    """build_cad() through ShapeScript path produces valid CadModel."""

    @pytest.mark.timeout(300)
    def test_build_cad_via_ir(self, bot_fixture, monkeypatch):
        monkeypatch.setenv("SHAPESCRIPT", "1")
        _clear_all_caches()

        from botcad.emit.cad import build_cad

        cad = build_cad(bot_fixture)

        from botcad.skeleton import BodyKind

        fabricated = [
            b for b in bot_fixture.all_bodies if b.kind == BodyKind.FABRICATED
        ]
        assert len(cad.body_solids) == len(fabricated)
        for name, solid in cad.body_solids.items():
            assert abs(solid.volume) > 1e-9, f"{name} has zero volume"
