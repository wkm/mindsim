"""Wheel components."""

from __future__ import annotations

import math

from botcad.component import Component, ComponentKind, MountPoint
from botcad.materials import MAT_RUBBER
from botcad.units import Meters, grams, mm, mm3


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
        dimensions=mm3(90, 90, 10),  # diameter x diameter x width
        mass=grams(22),
        mounting_points=(
            # 6x M3 holes on 6 spokes (60° intervals)
            # Spoke 0, 3: r = 6.35mm (12.7mm spacing pair)
            # Spoke 1, 2, 4, 5: r = 9.55mm (19.1mm spacing pairs)
            MountPoint(
                "m1",
                pos=(mm(6.35), Meters(0.0), Meters(0.0)),
                diameter=mm(3),
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m2",
                pos=(
                    Meters(0.00955 * math.cos(math.pi / 3)),
                    Meters(0.00955 * math.sin(math.pi / 3)),
                    Meters(0.0),
                ),
                diameter=mm(3),
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m3",
                pos=(
                    Meters(0.00955 * math.cos(2 * math.pi / 3)),
                    Meters(0.00955 * math.sin(2 * math.pi / 3)),
                    Meters(0.0),
                ),
                diameter=mm(3),
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m4",
                pos=(mm(-6.35), Meters(0.0), Meters(0.0)),
                diameter=mm(3),
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m5",
                pos=(
                    Meters(0.00955 * math.cos(4 * math.pi / 3)),
                    Meters(0.00955 * math.sin(4 * math.pi / 3)),
                    Meters(0.0),
                ),
                diameter=mm(3),
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
            MountPoint(
                "m6",
                pos=(
                    Meters(0.00955 * math.cos(5 * math.pi / 3)),
                    Meters(0.00955 * math.sin(5 * math.pi / 3)),
                    Meters(0.0),
                ),
                diameter=mm(3),
                axis=(0.0, 0.0, 1.0),
                fastener_type="M3",
            ),
        ),
        default_material=MAT_RUBBER,
        kind=ComponentKind.WHEEL,
    )
