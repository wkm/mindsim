"""Fastener catalog with ISO-accurate dimensions and CAD geometry.

Not a Component (no mass budget, no wire ports) — hardware catalog items
that larger parts (servos, brackets, bodies) compose from.

All dimensions in meters (SI), matching the rest of botcad.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING

from botcad.units import Meters, mm

if TYPE_CHECKING:
    from botcad.component import MountPoint


class HeadType(StrEnum):
    SOCKET_HEAD_CAP = "socket_head_cap"  # hex socket (Allen) — default
    BUTTON_HEAD = "button_head"  # low-profile hex socket
    PAN_HEAD_PHILLIPS = "pan_head_phillips"


@dataclass(frozen=True)
class FastenerSpec:
    """ISO metric fastener with accurate head geometry."""

    designation: str  # "M2", "M2.5", "M3"
    thread_diameter: Meters  # meters
    thread_pitch: Meters  # meters (coarse)
    head_type: HeadType
    head_diameter: Meters  # meters
    head_height: Meters  # meters
    socket_size: Meters  # meters (hex AF, or 0 for Phillips)
    clearance_hole: Meters  # meters
    close_fit_hole: Meters  # meters


# ── ISO 4762 Socket Head Cap Screws ──────────────────────────────────

_SOCKET_HEAD_CAP_CATALOG: dict[str, FastenerSpec] = {
    "M2": FastenerSpec(
        designation="M2",
        thread_diameter=mm(2),
        thread_pitch=mm(0.4),
        head_type=HeadType.SOCKET_HEAD_CAP,
        head_diameter=mm(3.8),
        head_height=mm(2),
        socket_size=mm(1.5),  # 1.5mm hex AF
        clearance_hole=mm(2.4),
        close_fit_hole=mm(2.2),
    ),
    "M2.5": FastenerSpec(
        designation="M2.5",
        thread_diameter=mm(2.5),
        thread_pitch=mm(0.45),
        head_type=HeadType.SOCKET_HEAD_CAP,
        head_diameter=mm(4.5),
        head_height=mm(2.5),
        socket_size=mm(2.0),  # 2.0mm hex AF
        clearance_hole=mm(2.9),
        close_fit_hole=mm(2.7),
    ),
    "M3": FastenerSpec(
        designation="M3",
        thread_diameter=mm(3),
        thread_pitch=mm(0.5),
        head_type=HeadType.SOCKET_HEAD_CAP,
        head_diameter=mm(5.5),
        head_height=mm(3),
        socket_size=mm(2.5),  # 2.5mm hex AF
        clearance_hole=mm(3.4),
        close_fit_hole=mm(3.2),
    ),
}

# ── ISO 7045 Pan Head Phillips Screws ────────────────────────────────

_PAN_HEAD_PHILLIPS_CATALOG: dict[str, FastenerSpec] = {
    "M2": FastenerSpec(
        designation="M2",
        thread_diameter=mm(2),
        thread_pitch=mm(0.4),
        head_type=HeadType.PAN_HEAD_PHILLIPS,
        head_diameter=mm(4),
        head_height=mm(1.6),
        socket_size=mm(0),  # Phillips — no hex
        clearance_hole=mm(2.4),
        close_fit_hole=mm(2.2),
    ),
    "M2.5": FastenerSpec(
        designation="M2.5",
        thread_diameter=mm(2.5),
        thread_pitch=mm(0.45),
        head_type=HeadType.PAN_HEAD_PHILLIPS,
        head_diameter=mm(5),
        head_height=mm(2),
        socket_size=mm(0),
        clearance_hole=mm(2.9),
        close_fit_hole=mm(2.7),
    ),
    "M3": FastenerSpec(
        designation="M3",
        thread_diameter=mm(3),
        thread_pitch=mm(0.5),
        head_type=HeadType.PAN_HEAD_PHILLIPS,
        head_diameter=mm(6),
        head_height=mm(2.4),
        socket_size=mm(0),
        clearance_hole=mm(3.4),
        close_fit_hole=mm(3.2),
    ),
}

# ── Combined catalog keyed by (designation, head_type) ───────────────

_CATALOG: dict[tuple[str, HeadType], FastenerSpec] = {}
for _spec in _SOCKET_HEAD_CAP_CATALOG.values():
    _CATALOG[(_spec.designation, _spec.head_type)] = _spec
for _spec in _PAN_HEAD_PHILLIPS_CATALOG.values():
    _CATALOG[(_spec.designation, _spec.head_type)] = _spec


def fastener_spec(
    designation: str,
    head_type: HeadType = HeadType.SOCKET_HEAD_CAP,
) -> FastenerSpec:
    """Look up a fastener by designation and head type.

    Raises KeyError if the combination is not in the catalog.
    """
    return _CATALOG[(designation, head_type)]


def fastener_key(mp: MountPoint) -> tuple[str, str]:
    """Derive the (designation, head_type) key from a MountPoint.

    Used for deduplication in hardware export, BOM, and MuJoCo emitters.
    """
    ft = mp.fastener_type or f"M{mp.diameter * 1000:.0f}"
    ht = mp.head_type or ""
    return (ft, ht)


def fastener_stl_stem(mp: MountPoint) -> str:
    """Filename stem for hardware STL, e.g. 'hardware_M3_shc'."""
    ft, ht = fastener_key(mp)
    return f"hardware_{ft}_{ht or 'shc'}"


def resolve_fastener(mp: MountPoint) -> FastenerSpec:
    """Resolve a MountPoint to a full FastenerSpec.

    Uses mp.fastener_type as the designation (e.g. "M3") and mp.head_type
    to select head style (defaults to socket head cap if empty).
    """
    designation = mp.fastener_type
    if not designation:
        # Derive from diameter as fallback
        d_mm = mp.diameter * 1000
        if d_mm <= 2.2:
            designation = "M2"
        elif d_mm <= 2.7:
            designation = "M2.5"
        else:
            designation = "M3"

    head = HeadType.SOCKET_HEAD_CAP
    if mp.head_type:
        head = HeadType(mp.head_type)

    return fastener_spec(designation, head)


# ── CAD geometry builders ────────────────────────────────────────────


@lru_cache(maxsize=64)
def fastener_solid(spec: FastenerSpec, length: float):
    """Build an accurate screw solid in local frame.

    Head at Z=0 (top face), shank extends in -Z direction.
    Socket head cap gets a hexagonal recess. Phillips gets a cross recess.
    No thread geometry — correct outer envelope only.

    Args:
        spec: Fastener specification from catalog.
        length: Shank length in meters (below head).
    """
    from build123d import (
        Align,
        Axis,
        Cylinder,
        Location,
        RegularPolygon,
        chamfer,
        extrude,
    )

    head_r = spec.head_diameter / 2
    head_h = spec.head_height
    shank_r = spec.thread_diameter / 2

    # Head cylinder (Z=0 at top face, head extends downward to Z=-head_h)
    head = Cylinder(head_r, head_h, align=(Align.CENTER, Align.CENTER, Align.MAX))

    # Chamfer the top edge of the head
    top_face = head.faces().sort_by(Axis.Z)[-1]
    chamfer_size = min(0.2 * head_h, 0.0003)  # 0.3mm max chamfer
    head = chamfer(top_face.edges(), chamfer_size)

    # Socket/recess
    if spec.head_type == HeadType.SOCKET_HEAD_CAP and spec.socket_size > 0:
        # Hexagonal socket recess (inscribed circle = socket_size)
        # RegularPolygon uses circumscribed radius, so r = AF / cos(30°)
        import math

        hex_r = spec.socket_size / 2 / math.cos(math.radians(30))
        recess_depth = spec.head_height * 0.6
        hex_profile = RegularPolygon(hex_r, 6)
        hex_solid = extrude(hex_profile, recess_depth)
        # Position: centered at top of head, extruding downward
        hex_solid = hex_solid.moved(Location((0, 0, -recess_depth)))
        head = head - hex_solid

    elif spec.head_type == HeadType.PAN_HEAD_PHILLIPS:
        # Simplified cross recess: two perpendicular slots
        from build123d import Box

        slot_w = spec.head_diameter * 0.12  # narrow slot
        slot_l = spec.head_diameter * 0.7  # across most of head
        slot_d = spec.head_height * 0.4
        slot1 = Box(
            slot_l, slot_w, slot_d, align=(Align.CENTER, Align.CENTER, Align.MAX)
        )
        slot2 = Box(
            slot_w, slot_l, slot_d, align=(Align.CENTER, Align.CENTER, Align.MAX)
        )
        head = head - slot1 - slot2

    # Shank (extends from Z=-head_h to Z=-(head_h + length))
    shank = Cylinder(shank_r, length, align=(Align.CENTER, Align.CENTER, Align.MAX))
    shank = shank.moved(Location((0, 0, -head_h)))

    return head + shank
