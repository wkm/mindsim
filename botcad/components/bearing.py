"""Ball bearing components."""

from __future__ import annotations

from botcad.component import BearingSpec
from botcad.materials import MAT_BEARING_STEEL
from botcad.units import grams, mm, mm3


def Bearing6x3x3() -> BearingSpec:
    """Standard 6x3x3 ball bearing (OD 6mm, ID 3mm, width 3mm).

    Used for small pivots and links in 3D printed designs.
    """
    return BearingSpec(
        name="Bearing 6x3x3",
        dimensions=mm3(6, 6, 3),
        mass=grams(1),
        od=mm(6),
        id=mm(3),
        width=mm(3),
        default_material=MAT_BEARING_STEEL,
    )
