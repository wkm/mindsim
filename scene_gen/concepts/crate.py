"""Crate — a simple box obstacle of variable size and style.

Sims 1 style: chunky boxes with saturated colors and visible construction
details. Wooden crates have plank lines, trunks have a contrasting lid,
storage bins are colorful plastic.

Parameters:
    width:      Half-extent X (meters)
    depth:      Half-extent Y (meters)
    height:     Half-extent Z (meters)
    has_lid:    Include a contrasting lid on top (thin box)
    has_stripe: Include a horizontal stripe band (like packing tape)
    color:      RGBA for the main body
    lid_color:  RGBA for lid / stripe accent
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

PLACEMENT = Placement.CENTER

# Crate-specific colors
CARDBOARD = (0.72, 0.60, 0.45, 1.0)
PACKING_TAPE = (0.80, 0.72, 0.50, 1.0)
PLASTIC_BLUE = (0.20, 0.40, 0.70, 1.0)
PLASTIC_GREEN = (0.25, 0.60, 0.35, 1.0)
PLASTIC_RED = (0.75, 0.20, 0.18, 1.0)
PLASTIC_YELLOW = (0.90, 0.80, 0.20, 1.0)
TRUNK_LEATHER = (0.50, 0.32, 0.18, 1.0)
TRUNK_BRASS = (0.75, 0.65, 0.30, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.25
    depth: float = 0.20
    height: float = 0.20
    has_lid: bool = False
    has_stripe: bool = False
    color: tuple[float, float, float, float] = CARDBOARD
    lid_color: tuple[float, float, float, float] = PACKING_TAPE


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a crate (1-3 prims: body + opt lid + opt stripe).

    Always within 8 prims (uses at most 3).
    """
    prims: list[Prim] = []

    # Main body
    prims.append(
        Prim(
            GeomType.BOX,
            (params.width, params.depth, params.height),
            (0, 0, params.height),
            params.color,
        )
    )

    # Lid — thin contrasting box on top
    if params.has_lid:
        lid_h = 0.015
        # Lid is slightly wider than body for overhang (Sims 1 chunky look)
        lid_w = params.width + 0.01
        lid_d = params.depth + 0.01
        prims.append(
            Prim(
                GeomType.BOX,
                (lid_w, lid_d, lid_h),
                (0, 0, params.height * 2 + lid_h),
                params.lid_color,
            )
        )

    # Horizontal stripe / band (like packing tape or a belt)
    if params.has_stripe:
        stripe_h = params.height * 0.12
        # Slightly proud of the surface so it's visible
        stripe_w = params.width + 0.003
        stripe_d = params.depth + 0.003
        prims.append(
            Prim(
                GeomType.BOX,
                (stripe_w, stripe_d, stripe_h),
                (0, 0, params.height),
                params.lid_color,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations (Sims 1 inspired — colorful, chunky, diverse)
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "cardboard box": Params(),
    "small box": Params(
        width=0.15,
        depth=0.12,
        height=0.12,
        has_stripe=True,
        color=CARDBOARD,
        lid_color=PACKING_TAPE,
    ),
    "wooden crate": Params(
        width=0.40,
        depth=0.30,
        height=0.30,
        has_stripe=True,
        color=WOOD_MEDIUM,
        lid_color=WOOD_DARK,
    ),
    "large crate": Params(
        width=0.50,
        depth=0.40,
        height=0.40,
        has_stripe=True,
        color=WOOD_DARK,
        lid_color=METAL_GRAY,
    ),
    "blue storage bin": Params(
        width=0.30,
        depth=0.22,
        height=0.18,
        has_lid=True,
        color=PLASTIC_BLUE,
        lid_color=PLASTIC_BLUE,
    ),
    "green storage bin": Params(
        width=0.28,
        depth=0.20,
        height=0.16,
        has_lid=True,
        color=PLASTIC_GREEN,
        lid_color=PLASTIC_GREEN,
    ),
    "trunk": Params(
        width=0.40,
        depth=0.25,
        height=0.25,
        has_lid=True,
        has_stripe=True,
        color=TRUNK_LEATHER,
        lid_color=TRUNK_BRASS,
    ),
    "moving box": Params(
        width=0.30,
        depth=0.25,
        height=0.30,
        has_stripe=True,
        color=CARDBOARD,
        lid_color=PACKING_TAPE,
    ),
}
