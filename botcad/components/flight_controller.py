"""Flight controller components.

Flight controller local frame convention:
    X = forward axis (arrow direction on board)
    Y = left axis
    Z = up axis
    Origin = geometric center
"""

from __future__ import annotations

from botcad.component import BusType, Component, ComponentKind, MountPoint, WirePort
from botcad.materials import MAT_FR4_GREEN
from botcad.units import Amps, Kg, Meters, Volts, mm


def MatekF405Wing() -> Component:
    """Matek F405-Wing flight controller.

    Popular flight controller for fixed-wing planes with iNav/ArduPilot.
    Built-in OSD, barometer, current sensor.

    Dimensions: 36 x 36 x 8mm (standard 30.5mm mounting pattern)
    Weight: ~10g
    MCU: STM32F405
    IMU: ICM20602
    Baro: BMP280
    Mounting: 30.5 x 30.5mm, M3
    """
    size = mm(36.0)
    height = mm(8.0)
    mount_spacing = mm(30.5 / 2)

    return Component(
        name="MatekF405Wing",
        dimensions=(size, size, height),
        mass=Kg(0.010),
        kind=ComponentKind.GENERIC,
        wire_ports=(
            WirePort(
                "uart1",
                pos=(Meters(size / 2), Meters(0.0), Meters(0.0)),
                bus_type=BusType.UART_HALF_DUPLEX,
                connector_type="jst_sh_4pin",
            ),
            WirePort(
                "pwm_out",
                pos=(Meters(-size / 2), Meters(0.0), Meters(0.0)),
                bus_type=BusType.PWM,
                connector_type="pin_header",
            ),
            WirePort(
                "power",
                pos=(mm(12.0), Meters(size / 2), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="solder_pad",
            ),
        ),
        mounting_points=(
            MountPoint(
                "fc_1",
                pos=(mount_spacing, mount_spacing, Meters(0.0)),
                diameter=mm(3.0),
            ),
            MountPoint(
                "fc_2",
                pos=(mount_spacing, Meters(-mount_spacing), Meters(0.0)),
                diameter=mm(3.0),
            ),
            MountPoint(
                "fc_3",
                pos=(Meters(-mount_spacing), mount_spacing, Meters(0.0)),
                diameter=mm(3.0),
            ),
            MountPoint(
                "fc_4",
                pos=(Meters(-mount_spacing), Meters(-mount_spacing), Meters(0.0)),
                diameter=mm(3.0),
            ),
        ),
        default_material=MAT_FR4_GREEN,
        voltage=Volts(11.1),
        typical_current=Amps(0.3),
    )
