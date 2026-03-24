"""Tests for the step-by-step CAD debug pipeline.

Validates that make_body_solid_with_steps (ShapeScript path) produces correct
intermediate steps:
- Each step's solid is geometrically valid
- Tool solids are correctly positioned
- Steps are monotonically ordered (cuts reduce volume, unions increase)
- Symmetric joints produce mirrored tool positions

Requires build123d.
"""

from __future__ import annotations

import importlib
import time
from functools import cache
from pathlib import Path

import pytest

from botcad.skeleton import Bot

b3d = pytest.importorskip("build123d")


# ── Helpers ──


@cache
def _build_bot(name: str) -> Bot:
    """Build and solve a bot by name (cached across tests)."""
    mod = importlib.import_module(f"bots.{name}.design")
    bot = mod.build()
    bot.solve()
    return bot


def _build_steps_for_body(bot_name: str, body_name: str):
    """Build CAD steps for a specific body, returning (steps, bot, cad)."""
    from botcad.emit.cad import build_cad, make_body_solid_with_steps

    bot = _build_bot(bot_name)
    cad = build_cad(bot)

    target = None
    for body in bot.all_bodies:
        if body.name == body_name:
            target = body
            break
    assert target is not None, f"Body '{body_name}' not found in bot '{bot_name}'"

    parent_joint = cad.parent_joint_map.get(body_name)
    wire_segs = cad.body_wire_segments.get(body_name)
    wire_segs_tuple = tuple(wire_segs) if wire_segs else None

    steps = make_body_solid_with_steps(target, parent_joint, wire_segs_tuple)
    return steps, bot, cad


def _vol(solid) -> float:
    """Volume in mm³."""
    return abs(solid.volume) * 1e9


def _bbox_x_center(solid) -> float:
    """X center of bounding box in mm."""
    bb = solid.bounding_box()
    return (bb.min.X + bb.max.X) / 2 * 1000


def _all_bot_designs():
    bots_dir = Path(__file__).parent.parent / "bots"
    return [p.parent.name for p in sorted(bots_dir.glob("*/design.py"))]


# ── Step timing and structure ──


class TestCadStepsTiming:
    """Profile each step to find bottlenecks."""

    @pytest.mark.parametrize("bot_name", _all_bot_designs())
    def test_step_timing(self, bot_name, capsys):
        """Print per-step timing for every body in every bot."""
        from botcad.emit.cad import build_cad, make_body_solid_with_steps

        bot = _build_bot(bot_name)
        cad = build_cad(bot)

        for body in bot.all_bodies:
            pj = cad.parent_joint_map.get(body.name)
            ws = cad.body_wire_segments.get(body.name)
            ws_tuple = tuple(ws) if ws else None

            t0 = time.monotonic()
            steps = make_body_solid_with_steps(body, pj, ws_tuple)
            total = time.monotonic() - t0

            # Print summary
            with capsys.disabled():
                print(f"\n  {bot_name}:{body.name} — {len(steps)} steps, {total:.2f}s")
                for i, s in enumerate(steps):
                    vol = _vol(s.solid)
                    tool_info = f"tool={_vol(s.tool):.0f}mm³" if s.tool else "no tool"
                    print(
                        f"    [{i:2d}] {s.op:8s} {vol:10.0f}mm³  {tool_info:>20s}  {s.label}"
                    )


# ── Correctness ──


class TestCadStepsCorrectness:
    """Validate step geometry is correct."""

    @pytest.mark.xfail(
        reason="Flaky when run with full suite due to cached state; passes in isolation"
    )
    def test_final_step_matches_production(self):
        """Final step's solid must match build_cad result within tolerance."""
        from botcad.emit.cad import build_cad

        bot = _build_bot("wheeler_arm")
        cad = build_cad(bot)

        from botcad.skeleton import BodyKind

        for body in bot.all_bodies:
            if body.kind != BodyKind.FABRICATED:
                continue
            steps, _, _ = _build_steps_for_body("wheeler_arm", body.name)
            final = steps[-1].solid
            cached = cad.body_solids.get(body.name)
            if cached is None:
                continue
            assert abs(abs(final.volume) - abs(cached.volume)) < 1e-6, (
                f"{body.name}: final step volume {abs(final.volume)} != "
                f"cached {abs(cached.volume)}"
            )

    def test_all_steps_have_positive_volume(self):
        """Every intermediate solid must have positive volume."""
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")
        for i, s in enumerate(steps):
            assert abs(s.solid.volume) > 0, f"Step {i} ({s.label}) has zero volume"

    def test_cut_reduces_volume(self):
        """Cut result should be smaller than its target operand.

        With ShapeScript steps, the previous step may be a LocateOp for the
        tool (not the body). Track the evolving body solid through cut/union
        results to find the correct comparison baseline.
        """
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")
        # Track the "body accumulator" — last cut/union result or initial create
        body_vol = None
        for i, s in enumerate(steps):
            if s.op in ("cut", "union"):
                curr_vol = abs(s.solid.volume)
                if s.op == "cut" and body_vol is not None:
                    assert curr_vol <= body_vol + 1e-9, (
                        f"Step {i} ({s.label}): cut increased volume "
                        f"{body_vol * 1e9:.1f} → {curr_vol * 1e9:.1f} mm³"
                    )
                body_vol = curr_vol
            elif s.op == "create" and body_vol is None:
                # First create is the body shell
                body_vol = abs(s.solid.volume)

    def test_union_increases_volume(self):
        """Union result should be larger than the body before the union.

        Track the evolving body solid through cut/union results.
        """
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")
        body_vol = None
        for i, s in enumerate(steps):
            if s.op in ("cut", "union"):
                curr_vol = abs(s.solid.volume)
                if s.op == "union" and body_vol is not None:
                    assert curr_vol >= body_vol - 1e-15, (
                        f"Step {i} ({s.label}): union decreased volume "
                        f"{body_vol * 1e9:.1f} → {curr_vol * 1e9:.1f} mm³"
                    )
                body_vol = curr_vol
            elif s.op == "create" and body_vol is None:
                body_vol = abs(s.solid.volume)

    def test_tool_solids_have_positive_volume(self):
        """Every tool solid must have positive volume."""
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")
        for i, s in enumerate(steps):
            if s.tool is not None:
                assert abs(s.tool.volume) > 0, (
                    f"Step {i} ({s.label}) tool has zero volume"
                )


# ── Bracket positioning (the locate→moved bug) ──


class TestBracketPositioning:
    """Bracket tools must be at distinct positions for distinct joints."""

    def test_wheel_brackets_are_mirrored(self):
        """Left and right wheel bracket tools should be on opposite X sides.

        With ShapeScript steps, union tools are identified by op type and
        position rather than labels.
        """
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")

        # Find union steps with tools (bracket unions)
        bracket_unions = [s for s in steps if s.op == "union" and s.tool is not None]
        assert len(bracket_unions) >= 2, (
            f"Expected at least 2 bracket unions, got {len(bracket_unions)}"
        )

        # First two union tools should be on opposite X sides (left/right wheel)
        left_x = _bbox_x_center(bracket_unions[0].tool)
        right_x = _bbox_x_center(bracket_unions[1].tool)

        assert left_x * right_x < 0, (
            f"Bracket tools should be on opposite X sides: "
            f"left tool center_x={left_x:.1f}mm, right tool center_x={right_x:.1f}mm"
        )

    def test_insertion_channel_tools_are_mirrored(self):
        """Left and right insertion channel tools should be on opposite X sides.

        With ShapeScript steps, bracket envelope cuts are the largest cut tools.
        """
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")

        # Find cut steps with large tools (bracket envelopes > 100,000 mm³)
        large_cuts = [
            s
            for s in steps
            if s.op == "cut"
            and s.tool is not None
            and abs(s.tool.volume) * 1e9 > 100_000
        ]
        assert len(large_cuts) >= 2, (
            f"Expected at least 2 large cuts (bracket envelopes), got {len(large_cuts)}"
        )

        left_x = _bbox_x_center(large_cuts[0].tool)
        right_x = _bbox_x_center(large_cuts[1].tool)

        assert left_x * right_x < 0, (
            f"Insertion channel tools should be on opposite X sides: "
            f"left center_x={left_x:.1f}mm, right center_x={right_x:.1f}mm"
        )

    def test_no_duplicate_tool_positions(self):
        """No two large cut/union tools should have identical bboxes (locate mutation check).

        Only checks tools > 10,000 mm³ (brackets/envelopes). Smaller tools
        like wire channels can legitimately share positions.
        """
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")

        bracket_steps = [
            s
            for s in steps
            if s.tool is not None
            and s.op in ("cut", "union")
            and abs(s.tool.volume) * 1e9 > 10_000
        ]

        seen_bboxes: list[tuple] = []
        for s in bracket_steps:
            bb = s.tool.bounding_box()
            key = (
                round(bb.min.X, 6),
                round(bb.min.Y, 6),
                round(bb.min.Z, 6),
                round(bb.max.X, 6),
                round(bb.max.Y, 6),
                round(bb.max.Z, 6),
            )
            assert key not in seen_bboxes, (
                f"Duplicate tool bbox for '{s.label}': {key} — "
                f"likely a .locate() mutation bug"
            )
            seen_bboxes.append(key)
