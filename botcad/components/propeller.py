"""Propeller components with real-world specs.

Propeller local frame convention:
    Z = thrust axis (thrust in +Z direction)
    X, Y = blade plane
    Origin = hub center
"""

from __future__ import annotations

from botcad.component import PropellerSpec
from botcad.materials import MAT_ABS_DARK
from botcad.units import Kg, mm


def Propeller9x45() -> PropellerSpec:
    """9x4.5 inch propeller (common for small fixed-wing planes).

    APC-style 9x4.5 — standard for 800-1000mm wingspan planes
    with 2212-class motors on 3S LiPo.
    Diameter: 228.6mm (9 inches)
    Pitch: 114.3mm (4.5 inches)
    Weight: ~10g
    """
    return PropellerSpec(
        name="Propeller-9x4.5",
        dimensions=(
            mm(228.6),  # diameter
            mm(228.6),  # diameter
            mm(8.0),  # hub thickness
        ),
        mass=Kg(0.010),
        default_material=MAT_ABS_DARK,
        diameter=mm(228.6),
        pitch=mm(114.3),
        blades=2,
    )
