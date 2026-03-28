"""Tests for wire bend radius DFM check."""

from __future__ import annotations

import math

from botcad.dfm.checks.wire_bend_radius import WireBendRadius, _compute_bend_radius


def test_check_name():
    check = WireBendRadius()
    assert check.name == "wire_bend_radius"


def test_sharp_bend_detected():
    """Unit test: a 90-degree bend with short segments should be flagged."""
    # 90 degree bend, 5mm segments
    radius = _compute_bend_radius(
        angle_rad=math.pi / 2,  # 90 degrees
        seg_length=0.005,  # 5mm
    )
    # radius = 5mm / (2 * sin(45 deg)) = 5 / (2 * 0.7071) = 3.54mm
    assert radius < 0.004  # less than 4mm
    # This is below the 7.5mm threshold for static bends


def test_gentle_bend_safe():
    """A gentle 10-degree bend with long segments has a large radius."""
    radius = _compute_bend_radius(
        angle_rad=math.radians(10),
        seg_length=0.020,  # 20mm
    )
    # Should be well above any threshold
    assert radius > 0.100  # > 100mm


def test_collinear_segments_infinite_radius():
    """Nearly collinear segments should produce infinite (or very large) radius."""
    radius = _compute_bend_radius(
        angle_rad=0.0,
        seg_length=0.010,
    )
    assert radius == float("inf")


def test_u_turn_small_radius():
    """A 180-degree U-turn should produce the smallest possible radius."""
    radius = _compute_bend_radius(
        angle_rad=math.pi,  # 180 degrees
        seg_length=0.010,  # 10mm
    )
    # radius = 10mm / (2 * sin(90 deg)) = 10 / 2 = 5mm
    assert abs(radius - 0.005) < 1e-6


def test_wheeler_base_bend_radius():
    from botcad.assembly.build import build_assembly_sequence
    from bots.wheeler_base.design import build

    bot = build()
    seq = build_assembly_sequence(bot)
    check = WireBendRadius()
    findings = check.run(bot, seq, {})
    # May or may not have findings depending on route geometry
    # But structure should be valid
    for f in findings:
        assert f.check_name == "wire_bend_radius"
        assert f.measured is not None
        assert f.threshold is not None
        assert (
            f.measured < f.threshold
        )  # measured should be below threshold (that's why it's a finding)
