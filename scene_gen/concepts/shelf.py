"""Shelf — a vertical frame with horizontal shelves.

Parameters:
    width:          Overall width (X) in meters
    depth:          Overall depth (Y) in meters
    height:         Overall height (Z) in meters
    n_shelves:      Number of shelf boards (including top and bottom)
    board_thickness: Thickness of each board in meters
    side_thickness:  Thickness of side panels in meters
    board_color:     RGBA for shelf boards
    side_color:      RGBA for side panels
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_DARK,
    METAL_GRAY,
    WOOD_BIRCH,
    WOOD_DARK,
    WOOD_LIGHT,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL


@dataclass(frozen=True)
class Params:
    width: float = 0.80
    depth: float = 0.30
    height: float = 1.20
    n_shelves: int = 4
    board_thickness: float = 0.02
    side_thickness: float = 0.02
    board_color: tuple[float, float, float, float] = WOOD_BIRCH
    side_color: tuple[float, float, float, float] = WOOD_MEDIUM


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a shelf (2 sides + N boards)."""
    hw = params.width / 2
    hd = params.depth / 2
    hbt = params.board_thickness / 2
    hst = params.side_thickness / 2
    hh = params.height / 2

    bc = params.board_color
    sc = params.side_color

    prims: list[Prim] = []

    # Two side panels (full height)
    side_x = hw - hst
    prims.append(Prim(GeomType.BOX, (hst, hd, hh), (-side_x, 0, hh), sc))
    prims.append(Prim(GeomType.BOX, (hst, hd, hh), (side_x, 0, hh), sc))

    # Shelf boards evenly spaced from bottom to top
    inner_hw = hw - params.side_thickness  # boards fit between sides
    for i in range(params.n_shelves):
        if params.n_shelves > 1:
            frac = i / (params.n_shelves - 1)
        else:
            frac = 0.0
        z = hbt + frac * (params.height - params.board_thickness)
        prims.append(Prim(GeomType.BOX, (inner_hw, hd, hbt), (0, 0, z), bc))

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "bookcase": Params(),
    "tall narrow": Params(
        width=0.45,
        depth=0.25,
        height=1.80,
        n_shelves=6,
        board_color=WOOD_DARK,
        side_color=WOOD_DARK,
    ),
    "wide low": Params(
        width=1.20,
        depth=0.35,
        height=0.70,
        n_shelves=2,
        board_color=WOOD_LIGHT,
        side_color=WOOD_MEDIUM,
    ),
    "cube": Params(
        width=0.50,
        depth=0.50,
        height=0.50,
        n_shelves=2,
        board_color=WOOD_BIRCH,
        side_color=WOOD_BIRCH,
    ),
    "industrial": Params(
        width=0.90,
        depth=0.35,
        height=1.40,
        n_shelves=4,
        board_thickness=0.035,
        side_thickness=0.035,
        board_color=METAL_GRAY,
        side_color=METAL_DARK,
    ),
    "minimal": Params(
        width=0.70,
        depth=0.25,
        height=1.10,
        n_shelves=3,
        board_thickness=0.015,
        side_thickness=0.015,
        board_color=WOOD_LIGHT,
        side_color=WOOD_LIGHT,
    ),
}
