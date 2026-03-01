"""Bookstack — a small pile of stacked books on the floor.

Provides small, low-profile clutter with varied colors. Sims 1 style:
chunky books with saturated cover colors, recognizable silhouettes.
Variations include tall stacks, short stacks, leaning stacks (offset
books), magazine piles (wide + flat), and stacks with bookends.

Parameters:
    book_width:    Half-extent X for each book (meters)
    book_depth:    Half-extent Y for each book (meters)
    book_heights:  Tuple of half-heights for each book (bottom to top)
    book_offsets:  Optional X offsets per book for lean effect (meters)
    colors:        Tuple of RGBA colors for each book (bottom to top)
    has_bookend:   Include a bookend (small upright slab) on one side
    bookend_color: RGBA for bookend
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    FABRIC_BLUE,
    FABRIC_ORANGE,
    FABRIC_PURPLE,
    FABRIC_RED,
    FABRIC_TEAL,
    FABRIC_YELLOW,
    METAL_DARK,
    WOOD_DARK,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CENTER

# Book cover colors — saturated Sims 1 palette
BOOK_GREEN = (0.15, 0.50, 0.22, 1.0)
BOOK_MAROON = (0.55, 0.10, 0.12, 1.0)
BOOK_NAVY = (0.12, 0.15, 0.45, 1.0)
BOOK_TAN = (0.72, 0.62, 0.42, 1.0)
BOOK_CREAM = (0.92, 0.88, 0.75, 1.0)
BOOK_CRIMSON = (0.78, 0.12, 0.18, 1.0)
BOOK_TEAL = (0.12, 0.52, 0.48, 1.0)
BOOK_PURPLE = (0.42, 0.18, 0.55, 1.0)
BOOK_GOLD = (0.80, 0.68, 0.18, 1.0)
BOOK_ORANGE = (0.85, 0.45, 0.12, 1.0)


@dataclass(frozen=True)
class Params:
    book_width: float = 0.06
    book_depth: float = 0.04
    book_heights: tuple[float, ...] = (0.010, 0.008, 0.012)
    book_offsets: tuple[float, ...] | None = None
    colors: tuple[tuple[float, float, float, float], ...] = (
        FABRIC_RED,
        FABRIC_BLUE,
        WOOD_DARK,
    )
    has_bookend: bool = False
    bookend_color: tuple[float, float, float, float] = METAL_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a stack of books (1 box per book + optional bookend).

    Max 8 prims: up to 7 books + 1 bookend.
    """
    prims: list[Prim] = []
    z_cursor = 0.0

    for i, half_h in enumerate(params.book_heights):
        color = params.colors[i % len(params.colors)]
        # Use explicit offsets if provided, else slight jitter
        if params.book_offsets is not None:
            offset_x = params.book_offsets[i % len(params.book_offsets)]
        else:
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

    # Optional bookend — a small upright slab next to the stack
    if params.has_bookend:
        bookend_h = z_cursor * 0.7  # 70% of stack height
        prims.append(
            Prim(
                GeomType.BOX,
                (0.005, params.book_depth * 0.8, bookend_h / 2),
                (params.book_width + 0.012, 0, bookend_h / 2),
                params.bookend_color,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations — Sims 1 style: chunky, colorful, varied
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "tall stack": Params(
        book_width=0.06,
        book_depth=0.04,
        book_heights=(0.012, 0.010, 0.008, 0.015, 0.010),
        colors=(BOOK_CRIMSON, BOOK_NAVY, BOOK_GREEN, BOOK_GOLD, BOOK_TEAL),
    ),
    "short stack": Params(
        book_width=0.055,
        book_depth=0.038,
        book_heights=(0.010, 0.008),
        colors=(BOOK_PURPLE, BOOK_TAN),
    ),
    "leaning stack": Params(
        book_width=0.06,
        book_depth=0.04,
        book_heights=(0.012, 0.010, 0.009, 0.011),
        book_offsets=(0.0, 0.005, 0.010, 0.015),
        colors=(FABRIC_RED, BOOK_NAVY, FABRIC_TEAL, BOOK_ORANGE),
    ),
    "magazine pile": Params(
        book_width=0.085,
        book_depth=0.058,
        book_heights=(0.003, 0.003, 0.004, 0.003),
        colors=(FABRIC_ORANGE, FABRIC_BLUE, BOOK_CRIMSON, FABRIC_YELLOW),
    ),
    "with bookend": Params(
        book_width=0.06,
        book_depth=0.04,
        book_heights=(0.010, 0.012, 0.008),
        colors=(BOOK_GREEN, BOOK_MAROON, BOOK_NAVY),
        has_bookend=True,
        bookend_color=METAL_DARK,
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
        book_heights=(0.007, 0.008, 0.006, 0.007),
        colors=(BOOK_CREAM, BOOK_TEAL, BOOK_GOLD, FABRIC_PURPLE),
    ),
    "colorful pile": Params(
        book_width=0.065,
        book_depth=0.042,
        book_heights=(0.010, 0.009, 0.011, 0.008, 0.010),
        colors=(FABRIC_RED, FABRIC_YELLOW, FABRIC_BLUE, FABRIC_PURPLE, FABRIC_ORANGE),
    ),
}
