"""Bookstack — a small pile of stacked books on the floor.

Provides small, low-profile clutter with varied colors. Each book is a thin
box, stacked vertically with slight size and color variation.

Parameters:
    book_width:    Half-extent X for each book (meters)
    book_depth:    Half-extent Y for each book (meters)
    book_heights:  Tuple of half-heights for each book (bottom to top)
    colors:        Tuple of RGBA colors for each book (bottom to top)
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    FABRIC_BLUE,
    FABRIC_RED,
    WOOD_DARK,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CENTER

# Book cover colors
BOOK_GREEN = (0.18, 0.35, 0.22, 1.0)
BOOK_MAROON = (0.45, 0.12, 0.15, 1.0)
BOOK_NAVY = (0.15, 0.18, 0.35, 1.0)
BOOK_TAN = (0.70, 0.62, 0.48, 1.0)
BOOK_CREAM = (0.92, 0.88, 0.78, 1.0)


@dataclass(frozen=True)
class Params:
    book_width: float = 0.06
    book_depth: float = 0.04
    book_heights: tuple[float, ...] = (0.010, 0.008, 0.012)
    colors: tuple[tuple[float, float, float, float], ...] = (
        FABRIC_RED,
        FABRIC_BLUE,
        WOOD_DARK,
    )


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a stack of books (1 box prim per book)."""
    prims = []
    z_cursor = 0.0

    for i, half_h in enumerate(params.book_heights):
        color = params.colors[i % len(params.colors)]
        # Slight offset per book for realism
        offset_x = 0.003 * (i % 3 - 1)
        offset_y = 0.002 * ((i + 1) % 3 - 1)

        prims.append(
            Prim(
                GeomType.BOX,
                (params.book_width, params.book_depth, half_h),
                (offset_x, offset_y, z_cursor + half_h),
                color,
            )
        )
        z_cursor += half_h * 2

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "book stack": Params(),
    "small stack": Params(
        book_width=0.055,
        book_depth=0.038,
        book_heights=(0.008, 0.010),
        colors=(BOOK_NAVY, BOOK_TAN),
    ),
    "tall stack": Params(
        book_width=0.06,
        book_depth=0.04,
        book_heights=(0.012, 0.010, 0.008, 0.015),
        colors=(FABRIC_RED, BOOK_GREEN, FABRIC_BLUE, BOOK_MAROON),
    ),
    "textbooks": Params(
        book_width=0.08,
        book_depth=0.06,
        book_heights=(0.018, 0.015, 0.020),
        colors=(BOOK_NAVY, BOOK_MAROON, WOOD_DARK),
    ),
    "paperbacks": Params(
        book_width=0.045,
        book_depth=0.032,
        book_heights=(0.007, 0.008, 0.006),
        colors=(BOOK_CREAM, BOOK_TAN, BOOK_GREEN),
    ),
    "magazines": Params(
        book_width=0.08,
        book_depth=0.055,
        book_heights=(0.003, 0.004, 0.003),
        colors=(FABRIC_RED, FABRIC_BLUE, BOOK_TAN),
    ),
}
