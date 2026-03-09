"""Camera module components."""

from __future__ import annotations

from botcad.component import BusType, CameraSpec, MountPoint, WirePort


def OV5647() -> CameraSpec:
    """OV5647 camera module (Raspberry Pi Camera v1 compatible).

    25 x 24 x 9mm, 3g. CSI ribbon cable connection. 72° diagonal FOV.
    """
    return CameraSpec(
        name="OV5647",
        dimensions=(0.025, 0.024, 0.009),
        mass=0.003,
        fov_deg=72.0,
        resolution=(2592, 1944),
        wire_ports=(
            # CSI ribbon cable at bottom edge
            WirePort("csi", pos=(0.0, -0.012, 0.0), bus_type=BusType.CSI),
        ),
        mounting_points=(
            # Two M2 mounting holes
            MountPoint(
                "m1",
                pos=(-0.0105, -0.0095, 0.0),
                diameter=0.002,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            MountPoint(
                "m2",
                pos=(0.0105, -0.0095, 0.0),
                diameter=0.002,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
        ),
        color=(0.85, 0.75, 0.1, 1.0),  # yellow — distinct from PCB green
    )


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
        wire_ports=(WirePort("csi", pos=(0.0, -0.012, 0.0), bus_type=BusType.CSI),),
        mounting_points=(
            MountPoint(
                "m1",
                pos=(-0.0105, -0.0095, 0.0),
                diameter=0.002,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            MountPoint(
                "m2",
                pos=(0.0105, -0.0095, 0.0),
                diameter=0.002,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
        ),
        color=(0.85, 0.75, 0.1, 1.0),  # yellow — distinct from PCB green
    )
