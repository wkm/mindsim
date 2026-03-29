# Wire 3D Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace broken wire stubs/segments with WireNet-filtered continuous tube meshes per route.

**Architecture:** New `botcad/emit/emit_wires.py` transforms body-local WireRoute segments to world coordinates, generates ShapeScript tube meshes, exports as STL. Manifest emits one entry per route parented to root body. Stubs and per-segment emission deleted entirely.

**Tech Stack:** Python, ShapeScript/OCCT, build123d STL export

**Spec:** `docs/superpowers/specs/2026-03-29-wire-3d-visualization-design.md`

---

### Task 1: Create emit_wires.py — route_to_world_polyline

**Files:**
- Create: `botcad/emit/emit_wires.py`
- Test: `tests/test_emit_wires.py`

- [ ] **Step 1: Write test for route_to_world_polyline**

```python
# tests/test_emit_wires.py
from __future__ import annotations
import pytest

def _build_bot(name: str):
    """Build and solve a bot by name."""
    from bots import load_bot
    bot = load_bot(name)
    bot.solve()
    return bot

class TestRouteToWorldPolyline:
    def test_servo_bus_polyline(self):
        bot = _build_bot("wheeler_base")
        from botcad.emit.emit_wires import route_to_world_polyline
        servo_route = next(r for r in bot.wire_routes if r.label == "servo_bus")
        polyline = route_to_world_polyline(bot, servo_route)
        # Should have at least 2 points
        assert len(polyline) >= 2
        # All points are 3-tuples
        for pt in polyline:
            assert len(pt) == 3

    def test_deduplicates_shared_endpoints(self):
        bot = _build_bot("wheeler_base")
        from botcad.emit.emit_wires import route_to_world_polyline
        servo_route = next(r for r in bot.wire_routes if r.label == "servo_bus")
        polyline = route_to_world_polyline(bot, servo_route)
        # No consecutive duplicate points (within 1mm)
        import math
        for i in range(len(polyline) - 1):
            dx = polyline[i+1][0] - polyline[i][0]
            dy = polyline[i+1][1] - polyline[i][1]
            dz = polyline[i+1][2] - polyline[i][2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            assert dist > 0.001, f"Duplicate points at index {i}: {polyline[i]}"
```

- [ ] **Step 2: Run test, verify it fails** (module doesn't exist yet)

Run: `uv run pytest tests/test_emit_wires.py -v`

- [ ] **Step 3: Implement route_to_world_polyline**

```python
# botcad/emit/emit_wires.py
"""Wire tube mesh generation from WireNet-filtered routes.

Transforms body-local WireRoute segments to world coordinates and
generates continuous ShapeScript tube meshes — one STL per route.
"""
from __future__ import annotations

import math
from pathlib import Path

from botcad.component import BusType
from botcad.geometry import add_vec3
from botcad.routing import WireRoute
from botcad.skeleton import Body, Bot
from botcad.units import Meters, Position

# Wire visual radii (meters) — smaller than channel cut radii.
# These represent the actual cable diameter for rendering.
_WIRE_VISUAL_RADIUS: dict[BusType, float] = {
    BusType.UART_HALF_DUPLEX: 0.0009,  # 0.9mm
    BusType.CSI: 0.0018,               # 1.8mm
    BusType.POWER: 0.0012,             # 1.2mm
}
_DEFAULT_WIRE_RADIUS = 0.0015  # 1.5mm fallback


def route_to_world_polyline(bot: Bot, route: WireRoute) -> list[Position]:
    """Transform body-local segments to world coordinates, concatenate."""
    body_by_name: dict[str, Body] = {b.name: b for b in bot.all_bodies}
    polyline: list[Position] = []

    for seg in route.segments:
        body = body_by_name.get(str(seg.body_name))
        if body is None:
            continue
        wp = body.world_pos
        world_start = add_vec3(wp, seg.start)
        world_end = add_vec3(wp, seg.end)

        # Deduplicate shared endpoints between consecutive segments
        if polyline and _dist(polyline[-1], world_start) < 0.001:
            pass  # skip duplicate start
        else:
            polyline.append(world_start)
        polyline.append(world_end)

    return polyline


def _dist(a: Position, b: Position) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_emit_wires.py -v`

- [ ] **Step 5: Run lint**

Run: `make lint`

- [ ] **Step 6: Commit**

```
git add botcad/emit/emit_wires.py tests/test_emit_wires.py
git commit -m "feat: route_to_world_polyline — transform wire segments to world coords"
```

---

### Task 2: emit_wire_tubes — ShapeScript tube generation + STL export

**Files:**
- Modify: `botcad/emit/emit_wires.py`
- Test: `tests/test_emit_wires.py`

- [ ] **Step 1: Write test for emit_wire_tubes**

```python
class TestEmitWireTubes:
    def test_produces_stl_files(self, tmp_path):
        bot = _build_bot("wheeler_base")
        from botcad.emit.emit_wires import emit_wire_tubes
        stl_names = emit_wire_tubes(bot, tmp_path)
        # Should produce at least one STL
        assert len(stl_names) > 0
        # Each file should exist and be non-empty
        for name in stl_names:
            stl_path = tmp_path / name
            assert stl_path.exists(), f"Missing STL: {name}"
            assert stl_path.stat().st_size > 0, f"Empty STL: {name}"

    def test_skips_routes_without_wirenet(self, tmp_path):
        bot = _build_bot("wheeler_base")
        from botcad.emit.emit_wires import emit_wire_tubes
        # Count WireNets
        net_labels = {net.label for net in bot.wire_nets}
        stl_names = emit_wire_tubes(bot, tmp_path)
        # Each STL should correspond to a route with a matching WireNet
        for name in stl_names:
            # name is "wire_{label}.stl"
            label = name.replace("wire_", "").replace(".stl", "")
            assert label in net_labels, f"STL {name} has no matching WireNet"

    def test_stl_count_matches_active_routes(self, tmp_path):
        bot = _build_bot("wheeler_base")
        from botcad.emit.emit_wires import emit_wire_tubes
        net_labels = {net.label for net in bot.wire_nets}
        # Count routes that have both segments and a matching net
        expected = sum(
            1 for r in bot.wire_routes
            if r.label in net_labels and r.segments
        )
        stl_names = emit_wire_tubes(bot, tmp_path)
        assert len(stl_names) == expected
```

- [ ] **Step 2: Run test, verify it fails**

- [ ] **Step 3: Implement emit_wire_tubes**

Add to `botcad/emit/emit_wires.py`. Use ShapeScript to build cylinder segments along the polyline, execute via OcctBackend, export as STL. Reference `emit_wire_channel` in `botcad/shapescript/emit_components.py:606-657` for the cylinder orientation math.

```python
def emit_wire_tubes(bot: Bot, meshes_dir: Path) -> list[str]:
    """Generate one tube STL per WireNet-matched route. Returns filenames."""
    net_labels = {net.label for net in bot.wire_nets}
    stl_names: list[str] = []

    for route in bot.wire_routes:
        if route.label not in net_labels:
            continue
        if not route.segments:
            continue

        polyline = route_to_world_polyline(bot, route)
        if len(polyline) < 2:
            continue

        radius = _WIRE_VISUAL_RADIUS.get(route.bus_type, _DEFAULT_WIRE_RADIUS)
        stl_name = f"wire_{route.label}.stl"

        solid = _build_tube_solid(polyline, radius)
        if solid is not None:
            from botcad.shapescript.backend_occt import export_stl
            export_stl(solid, str(meshes_dir / stl_name))
            stl_names.append(stl_name)

    return stl_names


def _build_tube_solid(polyline: list[Position], radius: float):
    """Build a tube solid along a polyline using build123d cylinders."""
    from build123d import Cylinder, Location, Sphere

    from botcad.emit.cad import _orient_z_to_axis

    result = None
    for i in range(len(polyline) - 1):
        start, end = polyline[i], polyline[i + 1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 0.001:
            continue

        axis = (dx / length, dy / length, dz / length)
        mid = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
            (start[2] + end[2]) / 2,
        )

        cyl = Cylinder(radius, length)
        seg_solid = _orient_z_to_axis(cyl, axis).moved(Location(mid))

        if result is None:
            result = seg_solid
        else:
            result = result.fuse(seg_solid)

    return result
```

Note: `_orient_z_to_axis` and `_wire_segment_solid` already exist in `cad.py`. Check if `_orient_z_to_axis` is a private function — if so, either make it importable or duplicate the logic. The plan agent found that `_axis_angle_to_quat` is already imported by `emit_components.py`, so cross-module imports from `cad.py` are acceptable.

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_emit_wires.py -v`

- [ ] **Step 5: Run lint**

Run: `make lint`

- [ ] **Step 6: Commit**

```
git commit -m "feat: emit_wire_tubes — ShapeScript tube mesh generation per route"
```

---

### Task 3: Delete stubs + replace per-segment emission

**Files:**
- Modify: `botcad/emit/viewer.py` — delete `_emit_wire_stubs`, replace wire manifest entries
- Modify: `botcad/emit/cad.py` — replace per-segment loop with emit_wire_tubes call

- [ ] **Step 1: In cad.py, replace per-segment wire STL loop (lines ~595-613) with:**

```python
# --- Per-wire STLs ---
from botcad.emit.emit_wires import emit_wire_tubes
emit_wire_tubes(bot, meshes_dir)
```

Remove the `bus_radii` dict and the `_wire_segment_solid` call loop.

- [ ] **Step 2: In viewer.py, replace per-segment manifest entries (~lines 459-474) with per-route entries:**

```python
# 4. Wire routes (one entry per route, world coords, parented to root)
net_labels = {net.label for net in bot.wire_nets}
root_body_name = str(bot.root.name) if bot.root else ""
for route in bot.wire_routes:
    if route.label not in net_labels:
        continue
    if not route.segments:
        continue
    bus_color = _BUS_TYPE_COLORS.get(
        str(route.bus_type), [0.53, 0.53, 0.53, 1.0]
    )
    manifest["parts"].append({
        "id": f"wire_{route.label}",
        "name": route.label,
        "kind": "fabricated",
        "category": "wire",
        "bus_type": str(route.bus_type),
        "route_label": route.label,
        "parent_body": root_body_name,
        "mesh": f"wire_{route.label}.stl",
        "color": bus_color,
    })
```

- [ ] **Step 3: Delete _emit_wire_stubs() function and its call site**

Remove the call (around line 477) and the entire function body (~lines 492-689).

- [ ] **Step 4: Run lint**

Run: `make lint`

- [ ] **Step 5: Run full validation**

Run: `make validate`

Fix any test failures. Likely: snapshot baselines need updating, and any tests that check for stub manifest entries need adjustment.

- [ ] **Step 6: Update render baselines if needed**

Run: `uv run pytest tests/test_shapescript_snapshots.py --snapshot-update`

- [ ] **Step 7: Commit**

```
git commit -m "feat: replace wire stubs + per-segment emission with WireNet-driven route tubes"
```

---

### Task 4: Final verification

- [ ] **Step 1: Run full test suite**

Run: `make validate`

- [ ] **Step 2: Manual verification**

Start the server and load wheeler_base in the viewer:
- Verify continuous colored tubes from source to destination per route
- Verify no stubs on unconnected ports
- Verify wire tubes follow channel paths (not colliding with geometry)
- Verify wire visibility toggle works
- Verify wiring diagram tab still works

- [ ] **Step 3: Commit any remaining fixes**
