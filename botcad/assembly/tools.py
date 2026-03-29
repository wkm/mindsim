"""Tool library for assembly and DFM clearance checking.

Each tool has physical dimensions (for clearance ray-casting)
and a solid geometry callable (for viewer visualization).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from build123d import Solid

from botcad.units import Meters


class ToolKind(Enum):
    HEX_KEY_2 = "hex_key_2mm"
    HEX_KEY_2_5 = "hex_key_2.5mm"
    HEX_KEY_3 = "hex_key_3mm"
    PHILLIPS_0 = "phillips_#0"
    PHILLIPS_1 = "phillips_#1"
    FINGERS = "fingers"
    TWEEZERS = "tweezers"
    PLIERS = "pliers"


@dataclass(frozen=True)
class ToolSpec:
    kind: ToolKind
    shaft_diameter: Meters
    shaft_length: Meters
    head_diameter: Meters  # clearance envelope
    grip_clearance: Meters  # lateral space needed for hand
    solid: Callable[[], Solid]  # geometry for visualization


# ---------------------------------------------------------------------------
# Solid geometry factories
# ---------------------------------------------------------------------------


def _cylinder_solid(diameter: float, length: float) -> Callable[[], Solid]:
    """Return a callable that builds a cylinder along Z."""

    def make() -> Solid:
        from build123d import Cylinder

        return Cylinder(radius=diameter / 2, height=length).solid()

    return make


def _box_solid(width: float, depth: float, height: float) -> Callable[[], Solid]:
    """Return a callable that builds an axis-aligned box."""

    def make() -> Solid:
        from build123d import Box

        return Box(width, depth, height).solid()

    return make


def _phillips_solid(
    shaft_d: float, shaft_l: float, head_d: float, head_l: float
) -> Callable[[], Solid]:
    """Return a callable that builds a screwdriver: shaft cylinder + head cylinder."""

    def make() -> Solid:
        from build123d import Cylinder, Pos

        shaft = Cylinder(radius=shaft_d / 2, height=shaft_l)
        head = Pos(0, 0, (shaft_l + head_l) / 2) * Cylinder(
            radius=head_d / 2, height=head_l
        )
        return (shaft + head).solid()

    return make


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

TOOL_LIBRARY: dict[ToolKind, ToolSpec] = {
    ToolKind.HEX_KEY_2: ToolSpec(
        kind=ToolKind.HEX_KEY_2,
        shaft_diameter=0.002,
        shaft_length=0.045,
        head_diameter=0.002,
        grip_clearance=0.025,
        solid=_cylinder_solid(0.002, 0.045),
    ),
    ToolKind.HEX_KEY_2_5: ToolSpec(
        kind=ToolKind.HEX_KEY_2_5,
        shaft_diameter=0.0025,
        shaft_length=0.050,
        head_diameter=0.0025,
        grip_clearance=0.025,
        solid=_cylinder_solid(0.0025, 0.050),
    ),
    ToolKind.HEX_KEY_3: ToolSpec(
        kind=ToolKind.HEX_KEY_3,
        shaft_diameter=0.003,
        shaft_length=0.055,
        head_diameter=0.003,
        grip_clearance=0.025,
        solid=_cylinder_solid(0.003, 0.055),
    ),
    ToolKind.PHILLIPS_0: ToolSpec(
        kind=ToolKind.PHILLIPS_0,
        shaft_diameter=0.004,
        shaft_length=0.060,
        head_diameter=0.006,
        grip_clearance=0.030,
        solid=_phillips_solid(0.004, 0.060, 0.006, 0.010),
    ),
    ToolKind.PHILLIPS_1: ToolSpec(
        kind=ToolKind.PHILLIPS_1,
        shaft_diameter=0.005,
        shaft_length=0.070,
        head_diameter=0.008,
        grip_clearance=0.035,
        solid=_phillips_solid(0.005, 0.070, 0.008, 0.012),
    ),
    ToolKind.FINGERS: ToolSpec(
        kind=ToolKind.FINGERS,
        shaft_diameter=0.015,
        shaft_length=0.060,
        head_diameter=0.015,
        grip_clearance=0.040,
        solid=_box_solid(0.015, 0.015, 0.060),
    ),
    ToolKind.TWEEZERS: ToolSpec(
        kind=ToolKind.TWEEZERS,
        shaft_diameter=0.003,
        shaft_length=0.080,
        head_diameter=0.003,
        grip_clearance=0.020,
        solid=_box_solid(0.003, 0.010, 0.080),
    ),
    ToolKind.PLIERS: ToolSpec(
        kind=ToolKind.PLIERS,
        shaft_diameter=0.010,
        shaft_length=0.050,
        head_diameter=0.025,
        grip_clearance=0.050,
        solid=_box_solid(0.025, 0.010, 0.050),
    ),
}
