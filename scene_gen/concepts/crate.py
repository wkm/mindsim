"""Crate — a simple box obstacle of variable size.

The simplest possible concept (1 prim). Provides small-to-medium clutter
that a navigation robot must detect and avoid. Great for training diversity.

Parameters:
    width:   Half-extent X (meters)
    depth:   Half-extent Y (meters)
    height:  Half-extent Z (meters)
    color:   RGBA
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    PLASTIC_BLACK,
    WOOD_DARK,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CENTER

# Cardboard color
CARDBOARD = (0.72, 0.60, 0.45, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.25
    depth: float = 0.20
    height: float = 0.20
    color: tuple[float, float, float, float] = CARDBOARD


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a crate (1 box prim)."""
    return (
        Prim(
            GeomType.BOX,
            (params.width, params.depth, params.height),
            (0, 0, params.height),
            params.color,
        ),
    )


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "cardboard box": Params(),
    "small box": Params(width=0.15, depth=0.12, height=0.12),
    "shipping crate": Params(
        width=0.40,
        depth=0.30,
        height=0.30,
        color=WOOD_MEDIUM,
    ),
    "large crate": Params(
        width=0.50,
        depth=0.40,
        height=0.40,
        color=WOOD_DARK,
    ),
    "storage bin": Params(
        width=0.30,
        depth=0.22,
        height=0.18,
        color=PLASTIC_BLACK,
    ),
    "moving box": Params(
        width=0.30,
        depth=0.25,
        height=0.30,
        color=CARDBOARD,
    ),
}
