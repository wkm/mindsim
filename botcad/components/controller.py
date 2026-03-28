"""Motor controller board components."""

from __future__ import annotations

from botcad.component import BusType, Component, MountPoint, WirePort
from botcad.materials import MAT_PCB_DARK_GREEN
from botcad.units import Amps, Meters, Volts, grams, mm


def WaveshareSerialBus() -> Component:
    """Waveshare Serial Bus Servo Driver Board.

    USB-C host interface, drives up to 253 STS/SCS servos via UART
    half-duplex bus. Powers servos from external 6-12V supply.

    Product: https://www.waveshare.com/bus-servo-adapter-a.htm
    Dims from wiki spec sheet:
      PCB: 42 x 33mm, 16g.
      Mounting holes: 4x M2.5, 37 x 28mm rectangular pattern.
      Profile height ~12mm (DC barrel jack is tallest at ~9mm above PCB).
    Connectors:
      - DC5521 barrel jack: left edge, lower area (9-12.6V input)
      - 2-pin screw terminal: bottom edge, left of center (DC+/DC-)
      - USB-C: bottom edge, right of center (host)
      - 2x 3-pin servo headers: top edge (D/V/G)
      - 3-pin UART header: top-right (TX/RX/GND)
    """
    pcb_x = mm(42)  # 42mm
    pcb_y = mm(33)  # 33mm
    profile_z = mm(12)  # 12mm total height (DC jack dominates)

    # Mounting holes: 37 x 28mm pattern, M2.5 (2.5mm diameter)
    hole_dx = mm(37 / 2)  # ±18.5mm
    hole_dy = mm(28 / 2)  # ±14mm

    return Component(
        name="WaveshareSerialBus",
        dimensions=(pcb_x, pcb_y, profile_z),
        mass=grams(16),
        wire_ports=(
            WirePort(
                "usb_c",
                pos=(mm(5), Meters(-pcb_y / 2), Meters(0.0)),
                bus_type=BusType.USB,
                connector_type="usb_c",
            ),
            WirePort(
                "servo_bus",
                pos=(Meters(0.0), Meters(pcb_y / 2), Meters(0.0)),
                bus_type=BusType.UART_HALF_DUPLEX,
                connector_type="5264_3pin",
            ),
            WirePort(
                "power_in",
                pos=(Meters(-pcb_x / 2), mm(-5), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="dc5521",
            ),
        ),
        mounting_points=(
            MountPoint(
                "m1",
                pos=(Meters(-hole_dx), Meters(-hole_dy), Meters(0.0)),
                diameter=mm(2.5),
            ),
            MountPoint(
                "m2",
                pos=(Meters(hole_dx), Meters(-hole_dy), Meters(0.0)),
                diameter=mm(2.5),
            ),
            MountPoint(
                "m3",
                pos=(Meters(-hole_dx), Meters(hole_dy), Meters(0.0)),
                diameter=mm(2.5),
            ),
            MountPoint(
                "m4",
                pos=(Meters(hole_dx), Meters(hole_dy), Meters(0.0)),
                diameter=mm(2.5),
            ),
        ),
        default_material=MAT_PCB_DARK_GREEN,
        voltage=Volts(5.0),
        typical_current=Amps(0.1),  # board logic only, servo power is separate
    )
