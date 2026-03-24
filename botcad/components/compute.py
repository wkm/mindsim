"""Compute board components."""

from __future__ import annotations

from botcad.colors import COLOR_ELECTRONICS_PCB
from botcad.component import (
    Appearance,
    BusType,
    Component,
    ComponentKind,
    MountPoint,
    WirePort,
)


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
            WirePort("usb_power", pos=(-0.0325, 0.0, 0.0), bus_type=BusType.USB),
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
        appearance=Appearance(color=COLOR_ELECTRONICS_PCB.rgba),
        voltage=5.0,
        typical_current=0.6,  # ~600mA typical under load
        kind=ComponentKind.COMPUTE,
    )


def raspberry_pi_zero_solid():
    """Parametric solid for Raspberry Pi Zero.

    Dimensions per official mechanical drawings:
    - PCB: 65mm x 30mm x 1.5mm
    - Corner radii: 3mm
    - Mounting holes: 4x 2.75mm diameter, located 3.5mm from edges
      (centered at x = +/- 29mm, y = +/- 11.5mm)
    - Envelope Z height (including components): 5mm
    """
    from build123d import Align, Box, Cylinder, Location

    from botcad.cad_utils import as_solid as _as_solid

    C = (Align.CENTER, Align.CENTER, Align.CENTER)

    # Base PCB Plate (at Z=0 roughly)
    # Using 1.5mm thickness for the PCB itself
    pcb = Box(0.065, 0.030, 0.0015, align=C)

    # Add corner fillets
    edges_to_fillet = [
        e for e in pcb.edges() if e.geom_type == "LINE" and abs(e.direction.Z) > 0.9
    ]
    if len(edges_to_fillet) == 4:
        pcb = pcb.fillet(0.003, edges_to_fillet)

    # Cut mounting holes
    hole_radius = 0.00275 / 2.0
    hole_depth = 0.002  # Cut clear through the 1.5mm PCB

    # Holes are exactly 3.5mm from edges on a 65x30 board -> +/-29 in X, +/-11.5 in Y
    holes = [
        Location((-0.029, -0.0115, 0)),
        Location((0.029, -0.0115, 0)),
        Location((-0.029, 0.0115, 0)),
        Location((0.029, 0.0115, 0)),
    ]

    for loc in holes:
        cyl = Cylinder(hole_radius, hole_depth, align=C).moved(loc)
        pcb = pcb - cyl

    # Add component volumes to reach the 5mm bounding box height and provide geometric context
    # HDMI/Power/USB ports on the bottom edge (approximate)
    ports_block = Box(0.040, 0.007, 0.003, align=C)
    ports_block = ports_block.moved(Location((0, -0.0115, 0.002)))

    # GPIO header on the top edge
    gpio_block = Box(0.051, 0.005, 0.003, align=C)
    gpio_block = gpio_block.moved(Location((0, 0.0125, 0.002)))

    # Main SoC
    soc_block = Box(0.012, 0.012, 0.0015, align=C)
    soc_block = soc_block.moved(Location((0, 0, 0.0015)))

    # Union all the parts
    solid = pcb + ports_block + gpio_block + soc_block

    # Wrap to ensure we return a Solid object proper
    return _as_solid(solid)
