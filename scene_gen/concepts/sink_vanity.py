"""Sink vanity -- cabinet base + countertop + optional mirror.

A bathroom vanity placed against a wall. The cabinet provides the bulk,
a thin countertop sits on top, and an optional mirror floats above.

Parameters:
    cab_width:    Half-extent X of the cabinet
    cab_depth:    Half-extent Y of the cabinet
    cab_height:   Half-extent Z of the cabinet
    top_width:    Half-extent X of the countertop
    top_depth:    Half-extent Y of the countertop
    top_thick:    Half-extent Z (thickness) of the countertop
    mirror_width: Half-extent X of the mirror (0 to omit)
    mirror_height: Half-extent Z of the mirror
    cab_color:    RGBA for the cabinet
    top_color:    RGBA for the countertop
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    PLASTIC_WHITE,
    WOOD_LIGHT,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Mirror tint (slight blue-gray)
MIRROR_TINT = (0.70, 0.72, 0.78, 1.0)


@dataclass(frozen=True)
class Params:
    cab_width: float = 0.30
    cab_depth: float = 0.25
    cab_height: float = 0.40
    top_width: float = 0.32
    top_depth: float = 0.27
    top_thick: float = 0.02
    mirror_width: float = 0.25
    mirror_height: float = 0.20
    cab_color: tuple[float, float, float, float] = WOOD_MEDIUM
    top_color: tuple[float, float, float, float] = PLASTIC_WHITE


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a sink vanity (cabinet + countertop + optional mirror)."""
    # Cabinet: box sitting on floor
    cabinet = Prim(
        GeomType.BOX,
        (params.cab_width, params.cab_depth, params.cab_height),
        (0, 0, params.cab_height),
        params.cab_color,
    )

    # Countertop: thin box on top of cabinet
    top_z = params.cab_height * 2 + params.top_thick
    countertop = Prim(
        GeomType.BOX,
        (params.top_width, params.top_depth, params.top_thick),
        (0, 0, top_z),
        params.top_color,
    )

    prims = [cabinet, countertop]

    # Mirror: thin box floating above countertop (omit if width is 0)
    if params.mirror_width > 0:
        mirror_z = (
            params.cab_height * 2 + params.top_thick * 2 + 0.10 + params.mirror_height
        )
        mirror = Prim(
            GeomType.BOX,
            (params.mirror_width, 0.01, params.mirror_height),
            (0, -(params.cab_depth - 0.01), mirror_z),
            MIRROR_TINT,
        )
        prims.append(mirror)

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "single vanity": Params(),
    "double vanity": Params(
        cab_width=0.55,
        cab_depth=0.27,
        cab_height=0.42,
        top_width=0.58,
        top_depth=0.29,
        mirror_width=0.50,
        mirror_height=0.25,
    ),
    "pedestal sink": Params(
        cab_width=0.12,
        cab_depth=0.12,
        cab_height=0.38,
        top_width=0.22,
        top_depth=0.20,
        top_thick=0.03,
        mirror_width=0.20,
        mirror_height=0.20,
        cab_color=PLASTIC_WHITE,
        top_color=PLASTIC_WHITE,
    ),
    "modern vanity": Params(
        cab_width=0.35,
        cab_depth=0.25,
        cab_height=0.35,
        top_width=0.37,
        top_depth=0.27,
        mirror_width=0.30,
        mirror_height=0.22,
        cab_color=PLASTIC_WHITE,
        top_color=METAL_CHROME,
    ),
    "rustic vanity": Params(
        cab_width=0.32,
        cab_depth=0.26,
        cab_height=0.42,
        top_width=0.34,
        top_depth=0.28,
        mirror_width=0.28,
        mirror_height=0.22,
        cab_color=WOOD_LIGHT,
        top_color=PLASTIC_WHITE,
    ),
}
