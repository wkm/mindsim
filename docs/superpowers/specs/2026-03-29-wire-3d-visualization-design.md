# Wire 3D Visualization: WireNet-Driven Continuous Tubes

## Context

The current 3D wire rendering is broken: stubs appear on every port regardless of connectivity, stubs collide with body geometry, and route segments are disconnected cylinders. We're replacing the entire wire visualization pipeline with a WireNet-driven approach that renders continuous tube meshes per route.

This is Phase 2A of the wiring system. Phase 1 (WireNet logical model + diagram view) is merged. Phase 2B (spline/cable curves) and Phase 3 (`solve_routing()` consumes WireNets) follow later.

## Design

### What changes

1. **Delete `_emit_wire_stubs()`** in `botcad/emit/viewer.py` — the function that generates stubs on every port. Removed entirely.

2. **Replace per-segment STL emission** in `botcad/emit/cad.py` — instead of emitting individual STL files per wire segment, emit one continuous tube mesh per `WireRoute`.

3. **Filter by WireNet** — only emit wire meshes for routes that have a matching `WireNet` (by label). Routes without a matching net are silently skipped. Unconnected ports get no wire.

4. **World coordinates** — transform all body-local segment points to world frame and concatenate into a single polyline per route. One mesh per route, ignoring joint motion (wires don't move with joints in this pass).

5. **Manifest changes** — wire parts become one entry per route with polyline metadata, not per-segment or per-port entries.

### What stays unchanged

- `solve_routing()` — still produces physical routes with body-local segments
- Wire channel cuts in `emit_body.py` — still uses per-body segments for channel geometry (these are correct)
- DFM checks (wire bend radius, channel sizing) — still work on `WireRoute` segments
- Wiring diagram (Phase 1) — unaffected

### Data flow

```
WireNet[] (logical connectivity, from derive_wirenets)
  ↓ filter: only nets with matching WireRoute
WireRoute[] (physical segments, from solve_routing)
  ↓ transform body-local segments to world coordinates
  ↓ concatenate segment endpoints into polyline per route
ShapeScript tube emission (one tube per route)
  ↓ export as STL
Manifest entry per route → Viewer renders
```

### Tube mesh generation

New function in `botcad/emit/emit_wires.py`:

```python
def emit_wire_tube(prog: ShapeScript, polyline: list[Position], radius: Meters, tag: str) -> ShapeRef:
    """Emit a tube mesh following a polyline path."""
```

- Input: world-space polyline points from concatenated route segments
- Radius from bus type (wire visual radii from `cad.py`, NOT channel radii): UART 0.9mm, CSI 1.8mm, Power 1.2mm
- Generate as a series of cylinders between consecutive points (matching the current `_wire_segment_solid` approach but in ShapeScript)
- Tag for debugging: `"wire_{route_label}"`

### World coordinate transform

New utility function:

```python
def route_to_world_polyline(bot: Bot, route: WireRoute) -> list[Position]:
    """Transform all body-local segments to world coordinates and concatenate."""
```

For each segment in the route:
1. Look up the body by `segment.body_name`
2. Transform `segment.start` and `segment.end` through `body.world_pos` / `body.world_quat`
3. Append to polyline (deduplicating shared endpoints between consecutive segments)

### Manifest format

Replace current per-segment/per-stub wire entries with per-route entries:

```json
{
  "id": "wire_servo_bus",
  "category": "wire",
  "bus_type": "uart_half_duplex",
  "route_label": "servo_bus",
  "parent_body": "base",
  "mesh": "wire_servo_bus.stl",
  "material": {
    "color": [0.20, 0.60, 0.86, 1.0],
    "metallic": 0.0,
    "roughness": 0.8,
    "opacity": 1.0
  }
}
```

All wire meshes parented to the root body since they're in world coordinates.

### Viewer changes

Minimal — the viewer already loads wire parts from the manifest. The main changes:
- Wire group now contains fewer, larger meshes (one per route vs many per segment+stub)
- Wire visibility toggle still works (category === "wire")
- No stub-related code to maintain

## Files modified

### Deleted code
- `botcad/emit/viewer.py` — remove `_emit_wire_stubs()` and its call site
- `botcad/emit/cad.py` — remove per-segment wire STL emission loop

### New code
- `botcad/emit/emit_wires.py` — `route_to_world_polyline()`, `emit_wire_tubes()` (ShapeScript tube generation + STL export)

### Modified code
- `botcad/emit/cad.py` — call new `emit_wire_tubes()` instead of per-segment emission
- `botcad/emit/viewer.py` — emit per-route manifest entries instead of stubs; remove `_emit_wire_stubs` call

### Unchanged
- `botcad/routing.py` — no changes
- `botcad/shapescript/emit_body.py` — wire channel cuts unchanged
- `botcad/wirenet.py` — no changes
- `botcad/dfm/checks/wire_*.py` — no changes

## Verification

- `make lint` passes
- `make validate` passes (existing tests, render baselines may need update for wire changes)
- Manual: load wheeler_base in viewer, verify continuous colored tubes from source to destination
- Manual: verify no stubs on unconnected ports
- Manual: verify wire tubes don't collide with body geometry (they follow channel paths)
- Manual: verify wire visibility toggle still works
- Compare wire tube paths against wire channel cuts — they should overlap (same route data)
