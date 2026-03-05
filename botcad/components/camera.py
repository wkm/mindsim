"""Camera module components."""

from __future__ import annotations

from botcad.component import Component, MountPoint, WirePort


def OV5647() -> Component:
    """OV5647 camera module (Raspberry Pi Camera v1 compatible).

    25 x 24 x 9mm, 3g. CSI ribbon cable connection. 72° diagonal FOV.
    """
    return Component(
        name="OV5647",
        dimensions=(0.025, 0.024, 0.009),
        mass=0.003,
        wire_ports=(
            # CSI ribbon cable at bottom edge
            WirePort("csi", pos=(0.0, -0.012, 0.0), bus_type="csi"),
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
        color=(0.1, 0.6, 0.1, 1.0),  # PCB green
    )
