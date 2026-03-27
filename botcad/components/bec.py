"""Voltage regulator (BEC) components."""

from __future__ import annotations

from botcad.component import BusType, Component, MountPoint, WirePort
from botcad.materials import MAT_PCB_DARK_GREEN


def BEC5V() -> Component:
    """Pololu D24V10F5 5V 1A step-down voltage regulator.

    Converts 6-36V input to regulated 5V output for Pi power.
    Dims from datasheet: 12.7 x 10.2 x 3.0mm, ~1.5g.
    Product: https://www.pololu.com/product/2831
    """
    return Component(
        name="BEC5V",
        dimensions=(0.0127, 0.0102, 0.003),
        mass=0.0015,
        wire_ports=(
            WirePort(
                "power_in",
                pos=(-0.00635, 0.0, 0.0),
                bus_type=BusType.POWER,
                connector_type="xt30",
            ),
            WirePort(
                "power_out",
                pos=(0.00635, 0.0, 0.0),
                bus_type=BusType.POWER,
                connector_type="usb_c",
            ),
        ),
        mounting_points=(
            MountPoint("m1", pos=(-0.004, -0.004, 0.0), diameter=0.002),
            MountPoint("m2", pos=(0.004, -0.004, 0.0), diameter=0.002),
        ),
        default_material=MAT_PCB_DARK_GREEN,
        voltage=5.0,
        typical_current=0.01,
    )
