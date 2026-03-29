"""Tests for the CAD emission pipeline.

Validates that _make_body_solid produces geometrically correct solids:
bounding boxes match expectations, boolean ops don't silently discard
material, and symmetric designs produce symmetric meshes.

Requires build123d.
"""

from __future__ import annotations

import importlib
from functools import cache
from pathlib import Path

import pytest

from botcad.geometry import servo_placement
from botcad.skeleton import Bot

# build123d is a required dep but guard for clarity
b3d = pytest.importorskip("build123d")


# ── Helpers ──


@cache
def _build_bot(name: str) -> Bot:
    """Build and solve a bot by name (cached across tests)."""
    mod = importlib.import_module(f"bots.{name}.design")
    bot = mod.build()
    bot.solve()
    return bot


def _total_volume(solid) -> float:
    """Sum volume of all solids in a shape (Solid or Compound)."""
    solids = solid.solids() if hasattr(solid, "solids") else [solid]
    return sum(abs(s.volume) for s in solids)


def _parent_joint_map(bot: Bot) -> dict[str, object]:
    """Map child body name → parent Joint."""
    result: dict[str, object] = {}
    for body in bot.all_bodies:
        for joint in body.joints:
            if joint.child is not None:
                result[joint.child.name] = joint
    return result


def _all_bot_designs():
    """Discover all bot design modules."""
    bots_dir = Path(__file__).parent.parent / "bots"
    return [p.parent.name for p in sorted(bots_dir.glob("*/design.py"))]


# ── _ensure_solid ──


class TestEnsureSolid:
    """_ensure_solid must not discard substantial geometry fragments."""

    def test_keeps_single_solid(self):
        from botcad.emit.cad import _ensure_solid

        box = b3d.Box(1, 1, 1)
        result = _ensure_solid(box)
        assert isinstance(result, b3d.Solid)
        assert abs(result.volume - 1.0) < 0.01

    def test_keeps_both_halves_of_split_solid(self):
        """Cutting a box in half should preserve both pieces."""
        from botcad.emit.cad import _ensure_solid

        C = (b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.CENTER)
        box = b3d.Box(1, 1, 1, align=C)
        # Cut a slot through the middle along Y, splitting into two halves
        slot = b3d.Box(1.1, 0.1, 1.1, align=C)
        split = box - slot

        result = _ensure_solid(split)
        # Both halves should survive (total volume ~ 0.9)
        assert _total_volume(result) > 0.85, (
            f"Lost material: {_total_volume(result):.3f}"
        )

    def test_discards_tiny_slivers(self):
        """Genuinely tiny fragments should be cleaned up."""
        from botcad.emit.cad import _ensure_solid

        big = b3d.Box(1, 1, 1)
        tiny = b3d.Box(0.01, 0.01, 0.01)
        tiny = tiny.locate(b3d.Location((2, 0, 0)))
        compound = b3d.Compound(children=[big, tiny])

        result = _ensure_solid(compound)
        solids = result.solids() if hasattr(result, "solids") else [result]
        assert len(solids) == 1, "Tiny sliver should be discarded"
        assert abs(solids[0].volume - 1.0) < 0.01


# ── Body solid bounding box ──


class TestBodySolidBBox:
    """Body solids should span the expected bounding box."""

    def test_wheeler_base_body_spans_full_y(self):
        """The base body must have material on both +Y and -Y sides.

        This is the exact regression that the _ensure_solid bug caused:
        one Y-half was silently discarded.
        """
        from botcad.emit.cad import _make_body_solid

        bot = _build_bot("wheeler_base")
        base = bot.root

        # Build with wire segments (the real code path)
        wire_map: dict[str, list] = {}
        for route in bot.wire_routes:
            for seg in route.segments:
                wire_map.setdefault(seg.body_name, []).append((seg, route.bus_type))

        solid = _make_body_solid(base, None, tuple(wire_map.get(base.name, [])))
        bb = solid.bounding_box()

        dims = base.dimensions
        # Mesh should span most of the body's Y extent on both sides
        assert -dims[1] / 4 > bb.min.Y, (
            f"Missing -Y material: y_min={bb.min.Y:.4f}, expected < {-dims[1] / 4:.4f}"
        )
        assert dims[1] / 4 < bb.max.Y, (
            f"Missing +Y material: y_max={bb.max.Y:.4f}, expected > {dims[1] / 4:.4f}"
        )

    def test_body_solid_volume_is_positive(self):
        """Every body solid should have meaningful volume."""
        from botcad.emit.cad import _make_body_solid

        bot = _build_bot("wheeler_base")
        for body in bot.all_bodies:
            solid = _make_body_solid(body)
            assert solid is not None, f"Body {body.name} produced None"
            assert _total_volume(solid) > 1e-9, f"Body {body.name} has zero volume"


# ── Symmetric designs produce symmetric meshes ──


class TestWheelerSymmetry:
    """Wheeler bots have mirror-symmetric left/right wheel joints."""

    def test_servo_placements_are_mirror_symmetric(self):
        """Left and right wheel servo placements should mirror in X."""
        bot = _build_bot("wheeler_base")
        base = bot.root
        joints = {j.name: j for j in base.joints}

        from botcad.ids import JointId

        left = joints[JointId("left_wheel")]
        right = joints[JointId("right_wheel")]

        lc, _lq = servo_placement(
            left.servo.shaft_offset,
            left.servo.shaft_axis,
            left.axis,
            left.pos,
        )
        rc, _rq = servo_placement(
            right.servo.shaft_offset,
            right.servo.shaft_axis,
            right.axis,
            right.pos,
        )

        # Centers should mirror in X, match in Y and Z
        assert abs(lc[0] + rc[0]) < 1e-6, f"X not mirrored: {lc[0]} vs {rc[0]}"
        assert abs(lc[1] - rc[1]) < 1e-6, f"Y differs: {lc[1]} vs {rc[1]}"
        assert abs(lc[2] - rc[2]) < 1e-6, f"Z differs: {lc[2]} vs {rc[2]}"

    def test_bracket_insertion_channels_mirror_in_body(self):
        """Left and right servo insertion channel cuts should be symmetric in X."""
        from botcad.bracket import BracketSpec
        from botcad.bracket import (
            bracket_insertion_channel_solid as bracket_insertion_channel,
        )
        from botcad.geometry import quat_to_euler

        bot = _build_bot("wheeler_base")
        base = bot.root
        spec = BracketSpec()

        bboxes = {}
        for joint in base.joints:
            servo = joint.servo
            center, quat = servo_placement(
                servo.shaft_offset,
                servo.shaft_axis,
                joint.axis,
                joint.pos,
            )
            euler = quat_to_euler(quat)
            channel = bracket_insertion_channel(servo, spec)
            channel = channel.locate(b3d.Location(center, euler))
            bboxes[joint.name] = channel.bounding_box()

        from botcad.ids import JointId

        lbb = bboxes[JointId("left_wheel")]
        rbb = bboxes[JointId("right_wheel")]

        # Y extents should be identical
        assert abs(lbb.min.Y - rbb.min.Y) < 1e-6
        assert abs(lbb.max.Y - rbb.max.Y) < 1e-6

        # X extents should mirror
        assert abs(lbb.min.X + rbb.max.X) < 1e-4
        assert abs(lbb.max.X + rbb.min.X) < 1e-4


# ── All bots emit valid body solids ──


BOT_DESIGNS = _all_bot_designs()


class TestAllBotBodySolids:
    """Every bot's body solids should pass basic sanity checks."""

    @pytest.mark.parametrize("bot_name", BOT_DESIGNS)
    def test_body_solids_have_volume(self, bot_name):
        """Each body produces a solid with positive volume."""
        from botcad.emit.cad import _make_body_solid

        bot = _build_bot(bot_name)
        pj_map = _parent_joint_map(bot)

        for body in bot.all_bodies:
            pj = pj_map.get(body.name)
            solid = _make_body_solid(body, pj)
            assert solid is not None, (
                f"{bot_name}/{body.name}: _make_body_solid returned None"
            )
            assert _total_volume(solid) > 1e-9, (
                f"{bot_name}/{body.name}: effectively zero volume"
            )

    @pytest.mark.parametrize("bot_name", BOT_DESIGNS)
    def test_body_solid_bbox_within_body_dims(self, bot_name):
        """Body solid bbox shouldn't wildly exceed body dimensions.

        Tested without parent_joint (pre-orientation) since axis rotation
        can legitimately swap dimensions (e.g. a wheel cylinder).
        """
        from botcad.emit.cad import _make_body_solid

        bot = _build_bot(bot_name)
        for body in bot.all_bodies:
            solid = _make_body_solid(body)
            if solid is None:
                continue
            bb = solid.bounding_box()
            dims = body.dimensions
            # Allow 2x margin for brackets/couplers that extend beyond body
            margin = 2.0
            for axis, i, size in [
                ("X", 0, bb.max.X - bb.min.X),
                ("Y", 1, bb.max.Y - bb.min.Y),
                ("Z", 2, bb.max.Z - bb.min.Z),
            ]:
                assert size < dims[i] * margin + 0.02, (
                    f"{bot_name}/{body.name}: {axis} extent {size:.4f} "
                    f"too large vs dim {dims[i]:.4f}"
                )
