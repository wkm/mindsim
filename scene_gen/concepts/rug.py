"""Rug — a flat textile on the floor.

Parameters:
    width:     Width (X) in meters
    depth:     Depth (Y) in meters
    thickness: Thickness in meters (very thin)
    color:     RGBA for the rug surface
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    FABRIC_BEIGE,
    FABRIC_GRAY,
    RUG_BURGUNDY,
    RUG_NAVY,
    RUG_SAGE,
    RUG_TAN,
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
    color: tuple[float, float, float, float] = RUG_BURGUNDY


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a rug (1 prim)."""
    hw = params.width / 2
    hd = params.depth / 2
    ht = params.thickness / 2
    return (Prim(GeomType.BOX, (hw, hd, ht), (0, 0, ht), params.color),)


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "area rug": Params(),
    "runner": Params(width=0.70, depth=2.40, color=RUG_NAVY),
    "doormat": Params(width=0.60, depth=0.40, thickness=0.015, color=RUG_TAN),
    "large": Params(width=2.40, depth=1.80, color=RUG_SAGE),
    "accent": Params(width=0.90, depth=0.60, color=FABRIC_BEIGE),
    "square": Params(width=1.50, depth=1.50, color=FABRIC_GRAY),
}
