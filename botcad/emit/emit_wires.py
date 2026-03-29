"""Wire 3D visualization: world-frame polylines and tube STLs.

Transforms body-local wire segments into world coordinates and generates
tube meshes for the viewer's 3D wire overlay.
"""

from __future__ import annotations

import math
from pathlib import Path

from botcad.component import BusType
from botcad.routing import WireRoute
from botcad.skeleton import Bot
from botcad.units import Meters, Position, WireGauge, mm

# Wire gauge per bus type — derives visual radius from AWG + insulation.
# CSI ribbon is flat (not round) but approximated as a thin round cable for now.
_WIRE_GAUGE: dict[BusType, WireGauge] = {
    BusType.UART_HALF_DUPLEX: WireGauge(awg=26),  # standard servo signal wire
    BusType.POWER: WireGauge(awg=22),  # battery/power distribution
}
_DEFAULT_GAUGE = WireGauge(awg=26)

# CSI ribbon: not a round wire. Use a fixed small radius as rough approximation.
_CSI_VISUAL_RADIUS = mm(1.0)


def _visual_radius(bus_type: BusType) -> Meters:
    """Get the visual radius for a bus type."""
    if bus_type == BusType.CSI:
        return _CSI_VISUAL_RADIUS
    gauge = _WIRE_GAUGE.get(bus_type, _DEFAULT_GAUGE)
    return gauge.outer_radius


def route_to_world_polyline(bot: Bot, route: WireRoute) -> list[Position]:
    """Transform body-local segments to world coordinates, concatenate.

    Builds a continuous polyline by converting each segment's start/end
    from body-local to world frame and deduplicating shared endpoints
    (consecutive points within 1 mm are merged).
    """
    body_lookup = {b.name: b for b in bot.all_bodies}
    polyline: list[Position] = []

    for seg in route.segments:
        body = body_lookup.get(seg.body_name)
        if body is None:
            continue

        world_pos = body.world_pos
        start_world: Position = (
            Meters(world_pos[0] + seg.start[0]),
            Meters(world_pos[1] + seg.start[1]),
            Meters(world_pos[2] + seg.start[2]),
        )
        end_world: Position = (
            Meters(world_pos[0] + seg.end[0]),
            Meters(world_pos[1] + seg.end[1]),
            Meters(world_pos[2] + seg.end[2]),
        )

        # Deduplicate: skip start if it's within 1mm of the last point
        if not polyline or _dist(polyline[-1], start_world) >= 0.001:
            polyline.append(start_world)

        # Always append end (next segment's start will be deduped)
        if _dist(polyline[-1], end_world) >= 0.001:
            polyline.append(end_world)

    return polyline


def wire_route_solid(bot: Bot, route: WireRoute):
    """Build a tube solid for a single route. Returns a build123d shape or None."""
    from build123d import Align, Compound, Cylinder, Location, Sphere

    polyline = route_to_world_polyline(bot, route)
    if len(polyline) < 2:
        return None

    radius = _visual_radius(route.bus_type)

    parts = []
    for j in range(len(polyline) - 1):
        p0 = polyline[j]
        p1 = polyline[j + 1]

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-6:
            continue

        axis = (dx / length, dy / length, dz / length)
        mid = (
            (p0[0] + p1[0]) / 2,
            (p0[1] + p1[1]) / 2,
            (p0[2] + p1[2]) / 2,
        )

        cyl = Cylinder(radius, length, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        cap0 = Sphere(radius).moved(Location((0, 0, -length / 2)))
        cap1 = Sphere(radius).moved(Location((0, 0, length / 2)))
        seg_solid = cyl + cap0 + cap1

        seg_solid = _orient_z_to_axis(seg_solid, axis).moved(Location(mid))
        parts.append(seg_solid)

    if not parts:
        return None

    return Compound(children=parts) if len(parts) > 1 else parts[0]


def emit_wire_tubes(bot: Bot, meshes_dir: Path) -> list[str]:
    """Generate one tube STL per WireNet-matched route. Returns filenames."""
    from build123d import export_stl

    net_labels = {net.label for net in bot.wire_nets}
    filenames: list[str] = []

    for route in bot.wire_routes:
        if route.label not in net_labels:
            continue
        if not route.segments:
            continue

        tube = wire_route_solid(bot, route)
        if tube is None:
            continue

        stl_name = f"wire_{route.label}.stl"
        export_stl(tube, str(meshes_dir / stl_name))
        filenames.append(stl_name)

    return filenames


# ── Internal helpers ──


def _dist(a: Position, b: Position) -> float:
    """Euclidean distance between two positions."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _axis_angle_to_quat(
    axis: tuple[float, float, float], angle: float
) -> tuple[float, float, float, float]:
    """Convert axis-angle to quaternion (w, x, y, z)."""
    half = angle / 2
    s = math.sin(half)
    return (math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)


def _axis_to_quat(
    axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Quaternion rotating Z-up to align with the given axis."""
    ax, ay, az = axis
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, az))))
    if angle_deg < 1e-4:
        return (1.0, 0.0, 0.0, 0.0)

    rot_axis_x = -ay
    rot_axis_y = ax
    rot_mag = math.sqrt(rot_axis_x**2 + rot_axis_y**2)
    if rot_mag < 1e-9:
        return _axis_angle_to_quat((1.0, 0.0, 0.0), math.pi)

    rot_axis_x /= rot_mag
    rot_axis_y /= rot_mag
    return _axis_angle_to_quat((rot_axis_x, rot_axis_y, 0.0), math.radians(angle_deg))


def _orient_z_to_axis(solid, axis: tuple[float, float, float]):
    """Rotate a solid from Z-up to align with the given axis direction."""
    from build123d import Location

    from botcad.geometry import quat_to_euler

    q = _axis_to_quat(axis)
    w, x, y, z = q
    if abs(w - 1.0) < 1e-9 and abs(x) + abs(y) + abs(z) < 1e-9:
        return solid
    euler = quat_to_euler(q)
    return solid.moved(Location((0, 0, 0), euler))
