"""Wheel components."""

from __future__ import annotations

import math

from botcad.colors import COLOR_STRUCTURE_RUBBER
from botcad.component import Appearance, Component, MountPoint


def PololuWheel90mm() -> Component:
    """Pololu 90x10mm wheel with silicone tire.

    Press-fits onto the STS3215's 25T spline output shaft.
    Silicone tire with horizontal treads. Plastic hub.

    Product: https://www.pololu.com/product/4939
    Price: $9.49/pair

    Specs:
        - 90mm diameter, 10mm wide
        - 25T spline center bore (5.8mm diameter)
        - 6x M3 mounting holes:
            - 1 pair at 12.7mm (0.5") spacing
            - 2 pairs at 19.1mm (0.75") spacing
        - Mass: 22g (actual)
    """
    return Component(
        name="Pololu 90x10mm Wheel",
        dimensions=(0.090, 0.090, 0.010),  # diameter x diameter x width
        mass=0.022,
        mounting_points=(
            # 6x M3 holes on 6 spokes (60° intervals)
            # Spoke 0, 3: r = 6.35mm (12.7mm spacing pair)
            # Spoke 1, 2, 4, 5: r = 9.55mm (19.1mm spacing pairs)
            MountPoint(
                "m1",
                pos=(0.00635, 0.0, 0.0),
                diameter=0.003,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m2",
                pos=(
                    0.00955 * math.cos(math.pi / 3),
                    0.00955 * math.sin(math.pi / 3),
                    0.0,
                ),
                diameter=0.003,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m3",
                pos=(
                    0.00955 * math.cos(2 * math.pi / 3),
                    0.00955 * math.sin(2 * math.pi / 3),
                    0.0,
                ),
                diameter=0.003,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m4",
                pos=(-0.00635, 0.0, 0.0),
                diameter=0.003,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m5",
                pos=(
                    0.00955 * math.cos(4 * math.pi / 3),
                    0.00955 * math.sin(4 * math.pi / 3),
                    0.0,
                ),
                diameter=0.003,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m6",
                pos=(
                    0.00955 * math.cos(5 * math.pi / 3),
                    0.00955 * math.sin(5 * math.pi / 3),
                    0.0,
                ),
                diameter=0.003,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
        ),
        appearance=Appearance(color=COLOR_STRUCTURE_RUBBER.rgba),
        is_wheel=True,
    )
