"""Electronic Speed Controller (ESC) components.

ESC local frame convention:
    X = length axis
    Y = width axis
    Z = height axis
    Origin = geometric center
"""

from __future__ import annotations

from botcad.component import BusType, Component, ComponentKind, WirePort
from botcad.materials import MAT_ABS_DARK
from botcad.units import Amps, Kg, Meters, Volts, mm


def SimonK30A() -> Component:
    """Generic 30A brushless ESC (SimonK firmware compatible).

    Common lightweight ESC for small fixed-wing planes.
    Dimensions: ~45 x 24 x 11mm (without heatshrink)
    Weight: ~25g
    Continuous current: 30A
    Battery: 2S-4S LiPo
    BEC output: 5V / 2A
    """
    length = mm(45.0)
    width = mm(24.0)
    height = mm(11.0)

    return Component(
        name="ESC-30A",
        dimensions=(length, width, height),
        mass=Kg(0.025),
        kind=ComponentKind.GENERIC,
        wire_ports=(
            # Power input (from battery)
            WirePort(
                "batt_pos",
                pos=(Meters(length / 2), Meters(0.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="xt30",
            ),
            # Motor output (3 phase wires)
            WirePort(
                "motor_out",
                pos=(Meters(-length / 2), Meters(0.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="bullet_3.5mm",
                permanent=True,
            ),
            # Signal input (from flight controller)
            WirePort(
                "signal",
                pos=(Meters(length / 2), mm(8.0), Meters(0.0)),
                bus_type=BusType.PWM,
                connector_type="servo_3pin",
            ),
        ),
        default_material=MAT_ABS_DARK,
        voltage=Volts(11.1),  # 3S nominal
        typical_current=Amps(15.0),
    )
