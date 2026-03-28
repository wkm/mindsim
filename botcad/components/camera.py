"""Camera module components."""

from __future__ import annotations

from botcad.component import BusType, CameraSpec, MountPoint, WirePort
from botcad.materials import MAT_ABS_DARK
from botcad.units import Degrees, Meters, grams, mm

# V1/V2 share the same PCB: 23.862 x 25mm.
# Holes at (2, 2), (20.8, 2), (2, 23), (20.8, 23) from bottom-left.
# 18.8mm H x 21mm V pitch, asymmetric X inset (2mm left, 3.062mm right).
_PICAMERA_V1V2_MOUNTING_POINTS = (
    MountPoint(
        "m1",
        pos=(Meters(-0.009931), mm(-10.5), Meters(0.0)),
        diameter=mm(2.2),
        axis=(0.0, 0.0, -1.0),
        fastener_type="M2",
    ),
    MountPoint(
        "m2",
        pos=(Meters(0.008869), mm(-10.5), Meters(0.0)),
        diameter=mm(2.2),
        axis=(0.0, 0.0, -1.0),
        fastener_type="M2",
    ),
    MountPoint(
        "m3",
        pos=(Meters(-0.009931), mm(10.5), Meters(0.0)),
        diameter=mm(2.2),
        axis=(0.0, 0.0, -1.0),
        fastener_type="M2",
    ),
    MountPoint(
        "m4",
        pos=(Meters(0.008869), mm(10.5), Meters(0.0)),
        diameter=mm(2.2),
        axis=(0.0, 0.0, -1.0),
        fastener_type="M2",
    ),
)

_PICAMERA_V1V2_WIRE_PORTS = (
    WirePort(
        "csi",
        pos=(Meters(0.0), mm(-12.5), Meters(0.0)),
        bus_type=BusType.CSI,
        connector_type="csi_15pin",
    ),
)


def OV5647() -> CameraSpec:
    """OV5647 camera module (Raspberry Pi Camera Module v1.3).

    PCB: 23.862 x 25mm, ~9mm total thickness, 3g. 72° diagonal FOV.
    Same PCB form factor and mounting holes as Camera Module V2.
    Ref: datasheets.raspberrypi.com/camera/camera-module-2-mechanical-drawing.pdf
    """
    return CameraSpec(
        name="OV5647",
        dimensions=(Meters(0.023862), mm(25), mm(9)),
        mass=grams(3),
        fov=Degrees(72.0),
        resolution=(2592, 1944),
        wire_ports=_PICAMERA_V1V2_WIRE_PORTS,
        mounting_points=_PICAMERA_V1V2_MOUNTING_POINTS,
        default_material=MAT_ABS_DARK,
    )


def PiCamera2() -> CameraSpec:
    """Raspberry Pi Camera Module 2 (IMX219). 8MP, 62.2° H-FOV.

    PCB: 23.862 x 25mm, ~9mm total thickness, 3g.
    CSI-2 ribbon cable (15-pin, 1mm pitch).
    Ref: datasheets.raspberrypi.com/camera/camera-module-2-mechanical-drawing.pdf
    """
    return CameraSpec(
        name="PiCamera2",
        dimensions=(Meters(0.023862), mm(25), mm(9)),
        mass=grams(3),
        fov=Degrees(62.2),
        resolution=(3280, 2464),
        wire_ports=_PICAMERA_V1V2_WIRE_PORTS,
        mounting_points=_PICAMERA_V1V2_MOUNTING_POINTS,
        default_material=MAT_ABS_DARK,
    )


# V3 PCB: 25 x 23.862mm. Holes: 4x M2 (ø2.2mm, ø4.75mm pad).
# 21mm H x 12.5mm V pitch. 2mm inset from left/right/bottom edges.
# Positions from bottom-left: (2, 2), (23, 2), (2, 14.5), (23, 14.5).
# PCB center at (12.5, 11.931).
_PICAMERA3_MOUNTING_POINTS = (
    MountPoint(
        "m1",
        pos=(mm(-10.5), Meters(-0.009931), Meters(0.0)),
        diameter=mm(2.2),
        axis=(0.0, 0.0, -1.0),
        fastener_type="M2",
    ),
    MountPoint(
        "m2",
        pos=(mm(10.5), Meters(-0.009931), Meters(0.0)),
        diameter=mm(2.2),
        axis=(0.0, 0.0, -1.0),
        fastener_type="M2",
    ),
    MountPoint(
        "m3",
        pos=(mm(-10.5), Meters(0.002569), Meters(0.0)),
        diameter=mm(2.2),
        axis=(0.0, 0.0, -1.0),
        fastener_type="M2",
    ),
    MountPoint(
        "m4",
        pos=(mm(10.5), Meters(0.002569), Meters(0.0)),
        diameter=mm(2.2),
        axis=(0.0, 0.0, -1.0),
        fastener_type="M2",
    ),
)


_PICAMERA3_WIRE_PORTS = (
    WirePort(
        "csi",
        pos=(Meters(0.0), Meters(-0.011931), Meters(0.0)),
        bus_type=BusType.CSI,
        connector_type="csi_15pin",
    ),
)


def PiCamera3() -> CameraSpec:
    """Raspberry Pi Camera Module 3 — standard variant (IMX708).

    12MP (4608x2592), 66° H-FOV / 75° diagonal, PDAF autofocus, f/1.8.
    PCB: 25 x 23.862 x 1.12mm, total height 11.3mm, ~4g.
    Ref: datasheets.raspberrypi.com/camera/camera-module-3-standard-mechanical-drawing.pdf
    """
    return CameraSpec(
        name="PiCamera3",
        dimensions=(mm(25), Meters(0.023862), mm(11.3)),
        mass=grams(4),
        fov=Degrees(75.0),
        resolution=(4608, 2592),
        wire_ports=_PICAMERA3_WIRE_PORTS,
        mounting_points=_PICAMERA3_MOUNTING_POINTS,
        default_material=MAT_ABS_DARK,
    )


def PiCamera3Wide() -> CameraSpec:
    """Raspberry Pi Camera Module 3 — wide-angle variant (IMX708).

    12MP (4608x2592), 102° H-FOV / 120° diagonal, PDAF autofocus, f/2.2.
    PCB: 25 x 23.862 x 1.12mm, total height 12.0mm, ~4g.
    Same sensor/PCB as PiCamera3, wider lens (ø6.95mm vs ø5.75mm).
    Ref: datasheets.raspberrypi.com/camera/camera-module-3-wide-mechanical-drawing.pdf
    """
    return CameraSpec(
        name="PiCamera3Wide",
        dimensions=(mm(25), Meters(0.023862), mm(12)),
        mass=grams(4),
        fov=Degrees(120.0),
        resolution=(4608, 2592),
        wire_ports=_PICAMERA3_WIRE_PORTS,
        mounting_points=_PICAMERA3_MOUNTING_POINTS,
        default_material=MAT_ABS_DARK,
    )
