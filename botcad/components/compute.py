"""Compute board components."""

from __future__ import annotations

from botcad.component import (
    BusType,
    Component,
    ComponentKind,
    MountPoint,
    WirePort,
)
from botcad.materials import MAT_FR4_GREEN


def RaspberryPiZero2W() -> Component:
    """Raspberry Pi Zero 2W — 65 x 30 x 5mm, 10g."""
    return Component(
        name="RaspberryPiZero2W",
        dimensions=(0.065, 0.030, 0.005),
        mass=0.010,
        wire_ports=(
            # GPIO header (40-pin, along one long edge)
            WirePort("gpio", pos=(0.0, 0.013, 0.0025), bus_type=BusType.GPIO),
            # USB micro (power input)
            WirePort(
                "usb_power",
                pos=(-0.0325, 0.0, 0.0),
                bus_type=BusType.USB,
                connector_type="usb_c",
            ),
            # USB micro OTG (data out to controller)
            WirePort(
                "usb_data",
                pos=(-0.020, 0.0, 0.0),
                bus_type=BusType.USB,
                connector_type="usb_c",
            ),
            # CSI camera connector
            WirePort("csi", pos=(0.016, 0.0, 0.0025), bus_type=BusType.CSI),
            # UART (on GPIO pins 8/10)
            WirePort(
                "uart", pos=(-0.01, 0.013, 0.0025), bus_type=BusType.UART_HALF_DUPLEX
            ),
        ),
        mounting_points=(
            # Four M2.5 holes at standard Pi Zero locations
            MountPoint(
                "m1",
                pos=(-0.029, -0.0115, 0.0),
                diameter=0.00275,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            ),
            MountPoint(
                "m2",
                pos=(0.029, -0.0115, 0.0),
                diameter=0.00275,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            ),
            MountPoint(
                "m3",
                pos=(-0.029, 0.0115, 0.0),
                diameter=0.00275,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            ),
            MountPoint(
                "m4",
                pos=(0.029, 0.0115, 0.0),
                diameter=0.00275,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            ),
        ),
        default_material=MAT_FR4_GREEN,
        voltage=5.0,
        typical_current=0.6,  # ~600mA typical under load
        kind=ComponentKind.COMPUTE,
    )
