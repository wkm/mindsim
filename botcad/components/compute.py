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
from botcad.units import Amps, Meters, Volts, grams, mm, mm3


def RaspberryPiZero2W() -> Component:
    """Raspberry Pi Zero 2W — 65 x 30 x 5mm, 10g."""
    return Component(
        name="RaspberryPiZero2W",
        dimensions=mm3(65, 30, 5),
        mass=grams(10),
        wire_ports=(
            # GPIO header (40-pin, along one long edge)
            WirePort(
                "gpio",
                pos=(Meters(0.0), mm(13), mm(2.5)),
                bus_type=BusType.GPIO,
            ),
            # USB micro (power input)
            WirePort(
                "usb_power",
                pos=(mm(-32.5), Meters(0.0), Meters(0.0)),
                bus_type=BusType.USB,
                connector_type="usb_c",
            ),
            # USB micro OTG (data out to controller)
            WirePort(
                "usb_data",
                pos=(mm(-20), Meters(0.0), Meters(0.0)),
                bus_type=BusType.USB,
                connector_type="usb_c",
            ),
            # CSI camera connector
            WirePort(
                "csi",
                pos=(mm(16), Meters(0.0), mm(2.5)),
                bus_type=BusType.CSI,
            ),
            # UART (on GPIO pins 8/10)
            WirePort(
                "uart",
                pos=(mm(-10), mm(13), mm(2.5)),
                bus_type=BusType.UART_HALF_DUPLEX,
            ),
        ),
        mounting_points=(
            # Four M2.5 holes at standard Pi Zero locations
            MountPoint(
                "m1",
                pos=(mm(-29), mm(-11.5), Meters(0.0)),
                diameter=mm(2.75),
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            ),
            MountPoint(
                "m2",
                pos=(mm(29), mm(-11.5), Meters(0.0)),
                diameter=mm(2.75),
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            ),
            MountPoint(
                "m3",
                pos=(mm(-29), mm(11.5), Meters(0.0)),
                diameter=mm(2.75),
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            ),
            MountPoint(
                "m4",
                pos=(mm(29), mm(11.5), Meters(0.0)),
                diameter=mm(2.75),
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            ),
        ),
        default_material=MAT_FR4_GREEN,
        voltage=Volts(5.0),
        typical_current=Amps(0.6),  # ~600mA typical under load
        kind=ComponentKind.COMPUTE,
    )
