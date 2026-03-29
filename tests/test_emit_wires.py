"""Tests for wire 3D visualization: world polylines and tube STLs."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


def _load_bot(name: str):
    """Import a bot design module, build, and solve."""
    design_py = Path(__file__).resolve().parent.parent / "bots" / name / "design.py"
    spec = importlib.util.spec_from_file_location(f"bots.{name}.design", design_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bot = mod.build()
    bot.solve()
    return bot


@pytest.fixture(scope="module")
def bot():
    return _load_bot("wheeler_base")


@pytest.fixture(scope="module")
def servo_bus_route(bot):
    routes = {r.label: r for r in bot.wire_routes}
    assert "servo_bus" in routes, "wheeler_base must have a servo_bus route"
    return routes["servo_bus"]


# ---------------------------------------------------------------------------
# Test 1: route_to_world_polyline basic behaviour
# ---------------------------------------------------------------------------


def test_servo_bus_polyline(bot, servo_bus_route):
    """Servo bus route produces a valid world polyline with 2+ points."""
    from botcad.emit.emit_wires import route_to_world_polyline

    polyline = route_to_world_polyline(bot, servo_bus_route)
    assert len(polyline) >= 2, f"Expected 2+ points, got {len(polyline)}"
    for pt in polyline:
        assert isinstance(pt, tuple), f"Expected tuple, got {type(pt)}"
        assert len(pt) == 3, f"Expected 3-tuple, got {len(pt)}-tuple"


# ---------------------------------------------------------------------------
# Test 2: deduplication of shared endpoints
# ---------------------------------------------------------------------------


def test_deduplicates_shared_endpoints(bot, servo_bus_route):
    """No two consecutive polyline points should be within 1mm."""
    from botcad.emit.emit_wires import route_to_world_polyline

    polyline = route_to_world_polyline(bot, servo_bus_route)
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        dist = math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))
        assert dist >= 0.001, (
            f"Points {i} and {i + 1} are only {dist * 1000:.3f}mm apart: {a}, {b}"
        )


# ---------------------------------------------------------------------------
# Test 3: emit_wire_tubes produces STL files
# ---------------------------------------------------------------------------


def test_produces_stl_files(bot, tmp_path):
    """emit_wire_tubes should produce non-empty STL files."""
    from botcad.emit.emit_wires import emit_wire_tubes

    filenames = emit_wire_tubes(bot, tmp_path)
    assert len(filenames) > 0, "Expected at least one STL file"
    for fname in filenames:
        stl_path = tmp_path / fname
        assert stl_path.exists(), f"{fname} was not created"
        assert stl_path.stat().st_size > 0, f"{fname} is empty"


# ---------------------------------------------------------------------------
# Test 4: only routes with matching WireNet produce STLs
# ---------------------------------------------------------------------------


def test_skips_routes_without_wirenet(bot, tmp_path):
    """Every produced STL must correspond to a WireNet label."""
    from botcad.emit.emit_wires import emit_wire_tubes

    net_labels = {net.label for net in bot.wire_nets}
    filenames = emit_wire_tubes(bot, tmp_path)
    for fname in filenames:
        # filenames are "wire_{label}.stl"
        label = fname.removeprefix("wire_").removesuffix(".stl")
        assert label in net_labels, (
            f"STL {fname} has label '{label}' not in wire_nets: {net_labels}"
        )


# ---------------------------------------------------------------------------
# Test 5: STL count matches expected active routes
# ---------------------------------------------------------------------------


def test_stl_count_matches_active_routes(bot, tmp_path):
    """Number of STLs should equal the number of routes that have segments
    and a matching WireNet label."""
    from botcad.emit.emit_wires import emit_wire_tubes, route_to_world_polyline

    net_labels = {net.label for net in bot.wire_nets}
    expected = 0
    for route in bot.wire_routes:
        if route.label not in net_labels:
            continue
        if not route.segments:
            continue
        polyline = route_to_world_polyline(bot, route)
        if len(polyline) >= 2:
            expected += 1

    filenames = emit_wire_tubes(bot, tmp_path)
    assert len(filenames) == expected, (
        f"Expected {expected} STLs, got {len(filenames)}: {filenames}"
    )
