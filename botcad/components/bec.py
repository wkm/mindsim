"""Voltage regulator (BEC) components."""

from __future__ import annotations

from botcad.component import BusType, Component, WirePort
from botcad.materials import MAT_PCB_DARK_GREEN
from botcad.units import Amps, Meters, Volts, grams, mm


def BEC5V() -> Component:
    """Pololu D24V10F5 5V 1A step-down voltage regulator.

    Converts 6-36V input to regulated 5V output for Pi power.
    Dims from datasheet dimension drawing (pololu.com/product/2831):
      PCB: 12.7 x 17.8mm, profile height 2.8mm (inductor is tallest part).
      5-pin header along one short edge, 2.54mm pitch.
      Drill tolerance ±0.1mm, board edge ±0.3mm.
    Mass ~1.5g.
    """
    pcb_x = mm(12.7)  # 12.7mm (500 mil)
    pcb_y = mm(17.8)  # 17.8mm (700 mil)
    profile_z = mm(2.8)  # 2.8mm total profile height

    # Pin header along -Y short edge: 5 pins at 2.54mm pitch,
    # inset 1.27mm from edges.  VIN/GND/VOUT + PG/SHDN.
    pin_inset_y = mm(1.27)  # 1.27mm from bottom edge
    pin_y = Meters(-pcb_y / 2 + pin_inset_y)

    return Component(
        name="BEC5V",
        dimensions=(pcb_x, pcb_y, profile_z),
        mass=grams(1.5),
        wire_ports=(
            WirePort(
                "power_in",
                pos=(Meters(-pcb_x / 4), pin_y, Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="pin_header",
            ),
            WirePort(
                "power_out",
                pos=(Meters(pcb_x / 4), pin_y, Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="pin_header",
            ),
        ),
        mounting_points=(
            # Pololu D24V10F5 has no mounting holes — soldered via pin header
        ),
        default_material=MAT_PCB_DARK_GREEN,
        voltage=Volts(5.0),
        typical_current=Amps(0.01),
    )
