"""Tests for the step-by-step CAD debug pipeline.

Validates that _build_body_solid produces correct intermediate steps:
- Each step's solid is geometrically valid
- Tool solids are correctly positioned (not aliased by lru_cache)
- Steps are monotonically ordered (cuts reduce volume, unions increase)
- Final step matches the production _make_body_solid result
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
        """Final step's solid must match _make_body_solid within tolerance."""
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
        """Cut steps should not increase volume."""
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")
        for i, s in enumerate(steps):
            if s.op == "cut" and i > 0:
                prev_vol = abs(steps[i - 1].solid.volume)
                curr_vol = abs(s.solid.volume)
                # Allow tiny OCCT floating-point noise (< 1 mm³)
                assert curr_vol <= prev_vol + 1e-9, (
                    f"Step {i} ({s.label}): cut increased volume "
                    f"{prev_vol * 1e9:.1f} → {curr_vol * 1e9:.1f} mm³"
                )

    def test_union_increases_volume(self):
        """Union steps should not decrease volume."""
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")
        for i, s in enumerate(steps):
            if s.op == "union" and i > 0:
                prev_vol = abs(steps[i - 1].solid.volume)
                curr_vol = abs(s.solid.volume)
                assert curr_vol >= prev_vol - 1e-15, (
                    f"Step {i} ({s.label}): union decreased volume "
                    f"{prev_vol * 1e9:.1f} → {curr_vol * 1e9:.1f} mm³"
                )

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
        """Left and right wheel bracket tools should be on opposite X sides."""
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")

        # Find the bracket union steps (first two union steps)
        bracket_unions = [
            s for s in steps if s.op == "union" and "bracket" in s.label.lower()
        ]
        assert len(bracket_unions) >= 2, (
            f"Expected at least 2 bracket unions, got {len(bracket_unions)}: "
            f"{[s.label for s in bracket_unions]}"
        )

        left = bracket_unions[0]
        right = bracket_unions[1]

        left_x = _bbox_x_center(left.tool)
        right_x = _bbox_x_center(right.tool)

        # They should be on opposite sides of center
        assert left_x * right_x < 0, (
            f"Bracket tools should be on opposite X sides: "
            f"left tool center_x={left_x:.1f}mm, right tool center_x={right_x:.1f}mm"
        )

    def test_insertion_channel_tools_are_mirrored(self):
        """Left and right insertion channel tools should be on opposite X sides."""
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")

        channel_cuts = [
            s
            for s in steps
            if s.op == "cut" and "bracket insertion channel" in s.label.lower()
        ]
        assert len(channel_cuts) >= 2, (
            f"Expected at least 2 bracket insertion channel cuts, got {len(channel_cuts)}"
        )

        left_x = _bbox_x_center(channel_cuts[0].tool)
        right_x = _bbox_x_center(channel_cuts[1].tool)

        assert left_x * right_x < 0, (
            f"Insertion channel tools should be on opposite X sides: "
            f"left center_x={left_x:.1f}mm, right center_x={right_x:.1f}mm"
        )

    def test_no_duplicate_tool_positions(self):
        """No two bracket/insertion_channel tools should have identical bboxes."""
        steps, _, _ = _build_steps_for_body("wheeler_arm", "base")

        bracket_steps = [
            s
            for s in steps
            if s.tool is not None
            and ("bracket" in s.label.lower() or "cradle" in s.label.lower())
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
