"""Ball bearing components."""

from __future__ import annotations

from botcad.component import BearingSpec


def Bearing6x3x3() -> BearingSpec:
    """Standard 6x3x3 ball bearing (OD 6mm, ID 3mm, width 3mm).

    Used for small pivots and links in 3D printed designs.
    """
    return BearingSpec(
        name="Bearing 6x3x3",
        dimensions=(0.006, 0.006, 0.003),
        mass=0.001,  # ~1g
        od=0.006,
        id=0.003,
        width=0.003,
        color=(0.8, 0.8, 0.85, 1.0),  # steel silver
    )
