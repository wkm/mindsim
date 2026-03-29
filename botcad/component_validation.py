"""Component-level wire collision validation.

Checks whether wire port connector housings and wire stubs overlap
within a single component. Uses AABB (axis-aligned bounding box)
intersection — fast and conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from botcad.component import Component, WirePort
from botcad.connectors import WIRE_STUB_LENGTH, WIRE_STUB_RADIUS, connector_spec
from botcad.units import Meters


class CollisionKind(StrEnum):
    CONNECTOR = "connector"
    STUB = "stub"


@dataclass(frozen=True)
class WireCollision:
    """A detected overlap between two wire port envelopes."""

    port_a: str  # label of first port
    port_b: str  # label of second port
    kind: CollisionKind
    overlap_mm: (
        float  # min penetration depth across axes, in mm (positive = overlapping)
    )


# Default connector envelope when connector_type is empty (3 mm cube).
_DEFAULT_HALF = Meters(0.0015)  # half of 3 mm


def _aabb_overlap(
    p1: tuple[float, float, float],
    h1: tuple[float, float, float],
    p2: tuple[float, float, float],
    h2: tuple[float, float, float],
) -> float | None:
    """Return overlap (mm) if two AABBs intersect, else None.

    Each AABB is given by center *p* and half-extents *h*.
    Overlap = min penetration depth across all three axes, in mm.
    """
    penetrations: list[float] = []
    for i in range(3):
        pen = h1[i] + h2[i] - abs(p1[i] - p2[i])
        if pen <= 0:
            return None
        penetrations.append(pen)
    # Convert smallest penetration to mm.
    return min(penetrations) * 1000.0


def _connector_aabb(
    port: WirePort,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return (center, half_extents) for the connector housing AABB."""
    px, py, pz = float(port.pos[0]), float(port.pos[1]), float(port.pos[2])
    if port.connector_type:
        try:
            spec = connector_spec(port.connector_type)
        except KeyError:
            # Unknown connector type — use default envelope.
            d = float(_DEFAULT_HALF)
            return (px, py, pz), (d, d, d)
        bx, by, bz = spec.body_dimensions
        return (px, py, pz), (bx / 2, by / 2, bz / 2)
    # Fallback: 3 mm cube.
    d = float(_DEFAULT_HALF)
    return (px, py, pz), (d, d, d)


def _stub_aabb(
    port: WirePort,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return (center, half_extents) for the wire stub AABB, or None."""
    if not port.connector_type:
        return None  # no connector spec → skip stub
    try:
        spec = connector_spec(port.connector_type)
    except KeyError:
        return None  # unknown connector → skip stub
    dx, dy, dz = spec.wire_exit_direction
    ox, oy, oz = spec.wire_exit_offset

    px, py, pz = float(port.pos[0]), float(port.pos[1]), float(port.pos[2])
    length = float(WIRE_STUB_LENGTH)
    radius = float(WIRE_STUB_RADIUS)

    # Stub base starts at port pos + wire_exit_offset.
    bx = px + float(ox)
    by = py + float(oy)
    bz = pz + float(oz)

    # Tip of stub.
    tx = bx + dx * length
    ty = by + dy * length
    tz = bz + dz * length

    # AABB center = midpoint of base and tip.
    cx = (bx + tx) / 2
    cy = (by + ty) / 2
    cz = (bz + tz) / 2

    # Half-extents = half the span along each axis, expanded by radius
    # perpendicular to the cylinder axis.  For a general direction we
    # expand all three axes by the radius (conservative).
    hx = abs(tx - bx) / 2 + radius
    hy = abs(ty - by) / 2 + radius
    hz = abs(tz - bz) / 2 + radius

    return (cx, cy, cz), (hx, hy, hz)


def validate_wire_collisions(component: Component) -> list[WireCollision]:
    """Check all wire-port pairs for connector and stub AABB overlaps."""
    results: list[WireCollision] = []

    ports = component.wire_ports
    if len(ports) < 2:
        return results

    for pa, pb in combinations(ports, 2):
        # --- connector body check ---
        aabb_a = _connector_aabb(pa)
        aabb_b = _connector_aabb(pb)
        if aabb_a is not None and aabb_b is not None:
            overlap = _aabb_overlap(aabb_a[0], aabb_a[1], aabb_b[0], aabb_b[1])
            if overlap is not None:
                results.append(
                    WireCollision(
                        port_a=pa.label,
                        port_b=pb.label,
                        kind=CollisionKind.CONNECTOR,
                        overlap_mm=overlap,
                    )
                )

        # --- wire stub check ---
        stub_a = _stub_aabb(pa)
        stub_b = _stub_aabb(pb)
        if stub_a is not None and stub_b is not None:
            overlap = _aabb_overlap(stub_a[0], stub_a[1], stub_b[0], stub_b[1])
            if overlap is not None:
                results.append(
                    WireCollision(
                        port_a=pa.label,
                        port_b=pb.label,
                        kind=CollisionKind.STUB,
                        overlap_mm=overlap,
                    )
                )

    return results
