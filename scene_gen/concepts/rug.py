"""Rug — a flat textile on the floor.

Sims 1 style: bold colors, distinct shapes. Includes rectangular area rugs,
long narrow runners, round rugs (flat cylinder), and small doormats.

Parameters:
    width:     Width (X) in meters
    depth:     Depth (Y) in meters
    thickness: Thickness in meters (very thin)
    is_round:  Use a flat cylinder instead of a box
    color:     RGBA for the rug surface
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    RUG_BURGUNDY,
    RUG_GOLD,
    RUG_NAVY,
    RUG_PLUM,
    RUG_ROSE,
    RUG_SAGE,
    RUG_TAN,
    RUG_TEAL,
    GeomType,
    Placement,
    Prim,
)

GROUND_COVER = True
PLACEMENT = Placement.CENTER


@dataclass(frozen=True)
class Params:
    width: float = 1.60
    depth: float = 1.20
    thickness: float = 0.012
    is_round: bool = False
    color: tuple[float, float, float, float] = RUG_BURGUNDY


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a rug (1 prim: flat box or flat cylinder)."""
    ht = params.thickness / 2

    if params.is_round:
        # Round rug: flat cylinder, radius = average of width/depth halves
        radius = (params.width + params.depth) / 4
        return (Prim(GeomType.CYLINDER, (radius, ht, 0), (0, 0, ht), params.color),)
    else:
        # Rectangular rug: flat box
        hw = params.width / 2
        hd = params.depth / 2
        return (Prim(GeomType.BOX, (hw, hd, ht), (0, 0, ht), params.color),)


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "area rug": Params(
        width=1.60,
        depth=1.20,
        color=RUG_BURGUNDY,
    ),
    "runner": Params(
        width=0.65,
        depth=2.40,
        color=RUG_NAVY,
    ),
    "round small": Params(
        width=1.00,
        depth=1.00,
        is_round=True,
        color=RUG_ROSE,
    ),
    "round large": Params(
        width=1.80,
        depth=1.80,
        is_round=True,
        color=RUG_TEAL,
    ),
    "doormat": Params(
        width=0.60,
        depth=0.40,
        thickness=0.015,
        color=RUG_TAN,
    ),
    "large area": Params(
        width=2.40,
        depth=1.80,
        color=RUG_SAGE,
    ),
    "accent": Params(
        width=0.90,
        depth=0.60,
        color=RUG_GOLD,
    ),
    "hall runner": Params(
        width=0.55,
        depth=3.00,
        color=RUG_PLUM,
    ),
}
