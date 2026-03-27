"""Voltage regulator (BEC) components."""

from __future__ import annotations

from botcad.component import BusType, Component, WirePort
from botcad.materials import MAT_PCB_DARK_GREEN


def BEC5V() -> Component:
    """Pololu D24V10F5 5V 1A step-down voltage regulator.

    Converts 6-36V input to regulated 5V output for Pi power.
    Dims from datasheet dimension drawing (pololu.com/product/2831):
      PCB: 12.7 x 17.8mm, profile height 2.8mm (inductor is tallest part).
      5-pin header along one short edge, 2.54mm pitch.
      Drill tolerance ±0.1mm, board edge ±0.3mm.
    Mass ~1.5g.
    """
    pcb_x = 0.0127  # 12.7mm (500 mil)
    pcb_y = 0.0178  # 17.8mm (700 mil)
    profile_z = 0.0028  # 2.8mm total profile height

    # Pin header along -Y short edge: 5 pins at 2.54mm pitch,
    # inset 1.27mm from edges.  VIN/GND/VOUT + PG/SHDN.
    pin_inset_y = 0.00127  # 1.27mm from bottom edge
    pin_y = -pcb_y / 2 + pin_inset_y

    return Component(
        name="BEC5V",
        dimensions=(pcb_x, pcb_y, profile_z),
        mass=0.0015,
        wire_ports=(
            WirePort(
                "power_in",
                pos=(-pcb_x / 4, pin_y, 0.0),
                bus_type=BusType.POWER,
                connector_type="pin_header",
            ),
            WirePort(
                "power_out",
                pos=(pcb_x / 4, pin_y, 0.0),
                bus_type=BusType.POWER,
                connector_type="pin_header",
            ),
        ),
        mounting_points=(
            # Pololu D24V10F5 has no mounting holes — soldered via pin header
        ),
        default_material=MAT_PCB_DARK_GREEN,
        voltage=5.0,
        typical_current=0.01,
    )
