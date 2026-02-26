"""Kitchen counter — a low cabinet with a countertop slab on top.

Common kitchen furniture providing both a visual landmark and obstacle.
Cabinet body + thin countertop = 2 prims.

Parameters:
    width:           Half-extent X (meters)
    depth:           Half-extent Y (meters)
    cabinet_height:  Half-extent Z of the cabinet body
    top_thickness:   Half-extent Z of the countertop slab
    cabinet_color:   RGBA for the cabinet body
    top_color:       RGBA for the countertop surface
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_GRAY,
    WOOD_DARK,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Counter-specific colors
GRANITE_GRAY = (0.45, 0.44, 0.43, 1.0)
GRANITE_BLACK = (0.20, 0.20, 0.20, 1.0)
MARBLE_WHITE = (0.90, 0.88, 0.85, 1.0)
BUTCHER_BLOCK = (0.70, 0.52, 0.32, 1.0)
CABINET_WHITE = (0.92, 0.91, 0.89, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.60
    depth: float = 0.30
    cabinet_height: float = 0.43
    top_thickness: float = 0.02
    cabinet_color: tuple[float, float, float, float] = CABINET_WHITE
    top_color: tuple[float, float, float, float] = GRANITE_GRAY


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a kitchen counter (cabinet + countertop = 2 prims)."""
    # Cabinet: box sitting on floor
    cabinet = Prim(
        GeomType.BOX,
        (params.width, params.depth, params.cabinet_height),
        (0, 0, params.cabinet_height),
        params.cabinet_color,
    )

    # Countertop: thin slab on top of cabinet, slightly wider/deeper for overhang
    top_z = params.cabinet_height * 2 + params.top_thickness
    countertop = Prim(
        GeomType.BOX,
        (params.width + 0.02, params.depth + 0.01, params.top_thickness),
        (0, 0, top_z),
        params.top_color,
    )

    return (cabinet, countertop)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "standard counter": Params(),
    "short counter": Params(
        width=0.40,
        depth=0.30,
        cabinet_height=0.43,
    ),
    "long counter": Params(
        width=0.90,
        depth=0.30,
        cabinet_height=0.43,
        top_color=MARBLE_WHITE,
    ),
    "island": Params(
        width=0.60,
        depth=0.50,
        cabinet_height=0.45,
        cabinet_color=WOOD_DARK,
        top_color=GRANITE_BLACK,
    ),
    "butcher block": Params(
        width=0.50,
        depth=0.30,
        cabinet_height=0.43,
        cabinet_color=WOOD_MEDIUM,
        top_color=BUTCHER_BLOCK,
    ),
    "modern": Params(
        width=0.70,
        depth=0.32,
        cabinet_height=0.45,
        cabinet_color=METAL_GRAY,
        top_color=MARBLE_WHITE,
    ),
}
