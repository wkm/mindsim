"""Bookcase — a tall open-front shelving unit with visible books.

Sims 1 style: chunky side panels, saturated wood tones, optional colored
book blocks on shelves and cabinet doors (opaque bottom half).

Parameters:
    width:           Overall width (X) in meters
    depth:           Overall depth (Y) in meters
    height:          Overall height (Z) in meters
    n_shelves:       Number of shelf boards (including top and bottom)
    board_thickness: Thickness of each shelf board in meters
    side_thickness:  Thickness of side panels in meters
    has_cabinet:     Opaque panel covering the bottom half (cabinet doors)
    has_books:       Add colored book blocks on one shelf
    book_color:      RGBA for the book blocks
    board_color:     RGBA for shelf boards
    side_color:      RGBA for side panels
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    FABRIC_BLUE,
    FABRIC_GREEN,
    FABRIC_RED,
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
    width: float = 0.45
    depth: float = 0.18
    height: float = 0.90
    n_shelves: int = 4
    board_thickness: float = 0.025
    side_thickness: float = 0.025
    has_cabinet: bool = False
    has_books: bool = False
    book_color: tuple[float, float, float, float] = FABRIC_RED
    board_color: tuple[float, float, float, float] = WOOD_MEDIUM
    side_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a bookcase (2 sides + N shelves + optional cabinet/books, max 8)."""
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

    inner_hw = hw - params.side_thickness  # boards fit between sides

    # Cabinet door: opaque panel covering the bottom half, on the front face
    if params.has_cabinet and len(prims) < 7:
        cabinet_h = params.height * 0.40
        cabinet_hh = cabinet_h / 2
        door_t = 0.015
        prims.append(
            Prim(
                GeomType.BOX,
                (inner_hw, door_t / 2, cabinet_hh),
                (0, -hd + door_t / 2, cabinet_hh + params.board_thickness),
                sc,
            )
        )

    # Shelf boards evenly spaced from bottom to top
    remaining = 8 - len(prims)
    # Reserve 1 slot for books if needed
    if params.has_books:
        remaining -= 1
    n = min(params.n_shelves, remaining)
    for i in range(n):
        if n > 1:
            frac = i / (n - 1)
        else:
            frac = 0.0
        z = hbt + frac * (params.height - params.board_thickness)
        prims.append(Prim(GeomType.BOX, (inner_hw, hd, hbt), (0, 0, z), bc))

    # Books: a colored box block sitting on the second shelf from bottom
    if params.has_books and len(prims) < 8 and n >= 3:
        # Place books on the second shelf (index 1)
        shelf_1_z = hbt + (1 / (n - 1)) * (params.height - params.board_thickness)
        # Shelf spacing for book height
        shelf_2_z = hbt + (2 / (n - 1)) * (params.height - params.board_thickness)
        book_h = (shelf_2_z - shelf_1_z) * 0.70
        book_hh = book_h / 2
        book_w = inner_hw * 0.75
        book_d = hd * 0.80
        book_z = shelf_1_z + hbt + book_hh
        prims.append(
            Prim(
                GeomType.BOX,
                (book_w, book_d, book_hh),
                (0, 0, book_z),
                params.book_color,
            )
        )

    return tuple(prims[:8])


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "bookcase": Params(),
    "tall w/ books": Params(
        height=1.10,
        n_shelves=5,
        has_books=True,
        book_color=FABRIC_BLUE,
        board_color=WOOD_DARK,
        side_color=WOOD_DARK,
    ),
    "wide low": Params(
        width=0.60,
        height=0.65,
        n_shelves=3,
        has_books=True,
        book_color=FABRIC_GREEN,
        board_color=WOOD_BIRCH,
        side_color=WOOD_MEDIUM,
    ),
    "cabinet bottom": Params(
        width=0.50,
        height=1.00,
        n_shelves=4,
        has_cabinet=True,
        board_color=WOOD_DARK,
        side_color=WOOD_DARK,
    ),
    "display cabinet": Params(
        width=0.50,
        depth=0.22,
        height=1.00,
        n_shelves=4,
        has_cabinet=True,
        has_books=True,
        book_color=FABRIC_RED,
        board_color=WOOD_LIGHT,
        side_color=WOOD_LIGHT,
    ),
    "narrow deep": Params(
        width=0.35,
        depth=0.25,
        height=0.90,
        n_shelves=4,
        board_color=WOOD_MEDIUM,
        side_color=WOOD_DARK,
    ),
    "small": Params(
        n_shelves=3,
        height=0.65,
        has_books=True,
        book_color=FABRIC_RED,
        board_color=WOOD_LIGHT,
        side_color=WOOD_LIGHT,
    ),
    "industrial": Params(
        width=0.50,
        depth=0.22,
        height=1.00,
        n_shelves=4,
        board_thickness=0.03,
        side_thickness=0.03,
        board_color=METAL_GRAY,
        side_color=METAL_DARK,
    ),
}
