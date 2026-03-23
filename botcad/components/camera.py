"""Camera module components."""

from __future__ import annotations

from botcad.colors import COLOR_ELECTRONICS_DARK, COLOR_STRUCTURE_DARK
from botcad.component import Appearance, BusType, CameraSpec, MountPoint, WirePort


def OV5647() -> CameraSpec:
    """OV5647 camera module (Raspberry Pi Camera v1 compatible).

    25 x 24 x 9mm, 3g. CSI ribbon cable connection. 72° diagonal FOV.
    Reference: https://www.raspberrypi.com/documentation/accessories/camera.html
    """
    return CameraSpec(
        name="OV5647",
        dimensions=(0.025, 0.024, 0.009),
        mass=0.003,
        fov_deg=72.0,
        resolution=(2592, 1944),
        wire_ports=(
            # CSI ribbon cable at bottom edge (24mm height -> Y=-12mm)
            WirePort(
                "csi",
                pos=(0.0, -0.012, 0.0),
                bus_type=BusType.CSI,
                connector_type="csi_15pin",
            ),
        ),
        mounting_points=(
            # 4x M2 mounting holes (21mm x 12.5mm pitch)
            # Bottom holes 2.5mm from bottom edge (Y = -12 + 2.5 = -9.5)
            # Top holes 12.5mm above bottom holes (Y = -9.5 + 12.5 = 3.0)
            MountPoint(
                "m1",
                pos=(-0.0105, -0.0095, 0.0),
                diameter=0.0022,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            MountPoint(
                "m2",
                pos=(0.0105, -0.0095, 0.0),
                diameter=0.0022,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            MountPoint(
                "m3",
                pos=(-0.0105, 0.0030, 0.0),
                diameter=0.0022,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            MountPoint(
                "m4",
                pos=(0.0105, 0.0030, 0.0),
                diameter=0.0022,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
        ),
        appearance=Appearance(color=COLOR_ELECTRONICS_DARK.rgba),
    )


def camera_solid(spec: CameraSpec):
    """Build a detailed parametric solid for a camera module.

    Models the PCB, lens housing, lens barrel, and CSI connector.
    """
    from build123d import Align, Box, Cylinder, Location

    C = (Align.CENTER, Align.CENTER, Align.CENTER)

    # 1. PCB (The foundation)
    pcb_w, pcb_h, _pcb_t = spec.dimensions
    pcb_thick = 0.0016  # standard 1.6mm FR4
    pcb = Box(pcb_w, pcb_h, pcb_thick, align=C)

    # 2. Mounting Holes (Subtract from PCB)
    for mp in spec.mounting_points:
        hole = Cylinder(mp.diameter / 2, pcb_thick + 0.001, align=C)
        hole = hole.moved(Location(mp.pos))
        pcb = pcb - hole

    # 3. Lens Module (Centered at (0, 2.5mm) on the PCB)
    # Most v1.3 modules have the lens centered horizontally but offset toward the top.
    # Lens base: 8.5 x 8.5 mm
    base_size = 0.0085
    base_height = 0.0050
    lens_y_offset = 0.0025  # 2.5mm from PCB center
    lens_base = Box(
        base_size, base_size, base_height, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    lens_base = lens_base.moved(Location((0, lens_y_offset, pcb_thick / 2)))

    # Lens barrel
    barrel_r = 0.00325
    barrel_h = 0.0025
    lens_barrel = Cylinder(
        barrel_r, barrel_h, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    lens_barrel = lens_barrel.moved(
        Location((0, lens_y_offset, pcb_thick / 2 + base_height))
    )

    # 4. CSI Connector (Bottom edge)
    conn_w = 0.016
    conn_h = 0.005
    conn_t = 0.002
    connector = Box(conn_w, conn_t, conn_h, align=(Align.CENTER, Align.MAX, Align.MIN))
    connector = connector.moved(Location((0, -pcb_h / 2, pcb_thick / 2)))

    # 5. Composite Solid
    pcb.color = COLOR_ELECTRONICS_DARK.rgb
    lens_base.color = COLOR_STRUCTURE_DARK.rgb
    lens_barrel.color = (0.1, 0.1, 0.1)  # lens glass — near black
    connector.color = (0.8, 0.8, 0.8)  # metal connector

    return pcb.fuse(lens_base).fuse(lens_barrel).fuse(connector)


def PiCamera2() -> CameraSpec:
    """Raspberry Pi Camera Module 2 (IMX219). 8MP, 62.2° diagonal FOV.

    25 x 24 x 9mm, 3g. CSI ribbon cable. Same form factor as OV5647.
    """
    return CameraSpec(
        name="PiCamera2",
        dimensions=(0.025, 0.024, 0.009),
        mass=0.003,
        fov_deg=62.2,
        resolution=(3280, 2464),
        wire_ports=(
            WirePort(
                "csi",
                pos=(0.0, -0.012, 0.0),
                bus_type=BusType.CSI,
                connector_type="csi_15pin",
            ),
        ),
        mounting_points=(
            # 4x M2 mounting holes (21mm x 12.5mm pitch)
            MountPoint(
                "m1",
                pos=(-0.0105, -0.0095, 0.0),
                diameter=0.0022,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            MountPoint(
                "m2",
                pos=(0.0105, -0.0095, 0.0),
                diameter=0.0022,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            MountPoint(
                "m3",
                pos=(-0.0105, 0.0030, 0.0),
                diameter=0.0022,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            MountPoint(
                "m4",
                pos=(0.0105, 0.0030, 0.0),
                diameter=0.0022,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
        ),
        appearance=Appearance(color=COLOR_ELECTRONICS_DARK.rgba),
    )
