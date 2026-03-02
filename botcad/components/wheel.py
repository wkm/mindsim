"""Wheel components."""

from __future__ import annotations

from botcad.component import Component, MountPoint


def PololuWheel90mm() -> Component:
    """Pololu 90x10mm wheel with silicone tire.

    Press-fits onto the STS3215's 25T spline output shaft.
    Silicone tire with horizontal treads. Plastic hub.

    Product: https://www.pololu.com/product/4939
    Price: $9.49/pair

    Specs:
        - 90mm diameter, 10mm wide
        - 25T spline center bore (5.8mm diameter)
        - 6x M3 mounting holes, opposite pairs at 19.1mm spacing
        - Mass: ~15g (estimated)
    """
    return Component(
        name="Pololu 90x10mm Wheel",
        dimensions=(0.090, 0.090, 0.010),  # diameter x diameter x width
        mass=0.015,
        mounting_points=(
            # 6x M3 holes in 3 pairs at 19.1mm spacing
            MountPoint("m1", pos=(0.00955, 0.0, 0.0), diameter=0.003),
            MountPoint("m2", pos=(-0.00955, 0.0, 0.0), diameter=0.003),
            MountPoint("m3", pos=(0.0, 0.00955, 0.0), diameter=0.003),
            MountPoint("m4", pos=(0.0, -0.00955, 0.0), diameter=0.003),
            MountPoint("m5", pos=(0.00675, 0.00675, 0.0), diameter=0.003),
            MountPoint("m6", pos=(-0.00675, -0.00675, 0.0), diameter=0.003),
        ),
        color=(0.2, 0.2, 0.2, 1.0),  # dark rubber
    )
