"""Kitchen counter -- a low cabinet with countertop, optional backsplash and details.

Sims 1 style: chunky cabinets with clearly distinct countertop slab, optional
backsplash strip at the back wall, varied proportions for islands and bars.

Parameters:
    width:            Half-extent X (meters)
    depth:            Half-extent Y (meters)
    cabinet_height:   Half-extent Z of the cabinet body
    top_thickness:    Half-extent Z of the countertop slab
    cabinet_color:    RGBA for the cabinet body
    top_color:        RGBA for the countertop surface
    has_backsplash:   Whether to add a thin tall strip at the back
    backsplash_color: RGBA for the backsplash
    has_drawer_line:  Whether to show a drawer divider line on the front
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
CABINET_BLUE = (0.30, 0.42, 0.58, 1.0)
CABINET_GREEN = (0.35, 0.52, 0.40, 1.0)
TILE_WHITE = (0.88, 0.87, 0.85, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.60
    depth: float = 0.30
    cabinet_height: float = 0.43
    top_thickness: float = 0.025
    cabinet_color: tuple[float, float, float, float] = CABINET_WHITE
    top_color: tuple[float, float, float, float] = GRANITE_GRAY
    has_backsplash: bool = False
    backsplash_color: tuple[float, float, float, float] = TILE_WHITE
    has_drawer_line: bool = True


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a kitchen counter (cabinet + countertop + details, 2-5 prims)."""
    prims: list[Prim] = []

    # Cabinet: box sitting on floor
    cabinet = Prim(
        GeomType.BOX,
        (params.width, params.depth, params.cabinet_height),
        (0, 0, params.cabinet_height),
        params.cabinet_color,
    )
    prims.append(cabinet)

    # Countertop: thin slab on top of cabinet, slightly wider/deeper for overhang
    top_z = params.cabinet_height * 2 + params.top_thickness
    countertop = Prim(
        GeomType.BOX,
        (params.width + 0.02, params.depth + 0.015, params.top_thickness),
        (0, 0, top_z),
        params.top_color,
    )
    prims.append(countertop)

    # Drawer divider line: thin horizontal strip on the front face
    if params.has_drawer_line:
        divider_z = params.cabinet_height * 1.0  # mid-height of cabinet
        divider = Prim(
            GeomType.BOX,
            (params.width * 0.90, 0.004, 0.005),
            (0, params.depth + 0.004, divider_z),
            params.top_color,  # contrast line using countertop color
        )
        prims.append(divider)

    # Backsplash: thin vertical strip at the back wall
    if params.has_backsplash:
        splash_height = 0.15
        splash_z = (
            params.cabinet_height * 2 + params.top_thickness * 2 + splash_height / 2
        )
        splash = Prim(
            GeomType.BOX,
            (params.width, 0.01, splash_height / 2),
            (0, -params.depth - 0.01, splash_z),
            params.backsplash_color,
        )
        prims.append(splash)

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "standard counter": Params(),
    "with backsplash": Params(
        has_backsplash=True,
    ),
    "long counter": Params(
        width=0.90,
        depth=0.30,
        cabinet_height=0.43,
        top_color=MARBLE_WHITE,
        has_backsplash=True,
    ),
    "island": Params(
        width=0.60,
        depth=0.50,
        cabinet_height=0.45,
        cabinet_color=WOOD_DARK,
        top_color=GRANITE_BLACK,
        has_drawer_line=False,
    ),
    "butcher block": Params(
        width=0.50,
        depth=0.30,
        cabinet_height=0.43,
        cabinet_color=WOOD_MEDIUM,
        top_color=BUTCHER_BLOCK,
    ),
    "bar counter": Params(
        width=0.70,
        depth=0.25,
        cabinet_height=0.55,
        top_thickness=0.03,
        cabinet_color=WOOD_DARK,
        top_color=GRANITE_BLACK,
        has_drawer_line=False,
    ),
    "blue shaker": Params(
        width=0.60,
        depth=0.30,
        cabinet_color=CABINET_BLUE,
        top_color=MARBLE_WHITE,
        has_backsplash=True,
        backsplash_color=TILE_WHITE,
    ),
    "modern dark": Params(
        width=0.70,
        depth=0.32,
        cabinet_height=0.45,
        cabinet_color=METAL_GRAY,
        top_color=MARBLE_WHITE,
        has_backsplash=True,
        backsplash_color=(0.85, 0.83, 0.80, 1.0),
    ),
}
