"""Toilet -- box pedestal + cylinder bowl + box tank.

A bathroom fixture placed against a wall. Uses 3 prims for a recognizable
toilet silhouette at low geometric cost.

Parameters:
    base_width:   Half-extent X of the pedestal base
    base_depth:   Half-extent Y of the pedestal base
    base_height:  Half-extent Z of the pedestal base
    bowl_radius:  Radius of the bowl cylinder
    bowl_half_h:  Half-height of the bowl cylinder
    tank_width:   Half-extent X of the tank
    tank_depth:   Half-extent Y of the tank
    tank_height:  Half-extent Z of the tank
    color:        RGBA for the entire toilet
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    PLASTIC_WHITE,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Ceramic white (slightly warmer than plastic white)
CERAMIC_WHITE = (0.92, 0.90, 0.87, 1.0)


@dataclass(frozen=True)
class Params:
    base_width: float = 0.20
    base_depth: float = 0.20
    base_height: float = 0.20
    bowl_radius: float = 0.18
    bowl_half_h: float = 0.08
    tank_width: float = 0.18
    tank_depth: float = 0.08
    tank_height: float = 0.20
    color: tuple[float, float, float, float] = CERAMIC_WHITE


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a toilet (base box + bowl cylinder + tank box = 3 prims)."""
    # Base pedestal: box sitting on floor
    base = Prim(
        GeomType.BOX,
        (params.base_width, params.base_depth, params.base_height),
        (0, 0, params.base_height),
        params.color,
    )

    # Bowl: cylinder sitting on top of base
    bowl_z = params.base_height * 2 + params.bowl_half_h
    bowl = Prim(
        GeomType.CYLINDER,
        (params.bowl_radius, params.bowl_half_h, 0),
        (0, 0, bowl_z),
        params.color,
    )

    # Tank: box behind the bowl, sitting on base
    tank_z = params.base_height * 2 + params.tank_height
    tank_y = -(params.base_depth + params.tank_depth)
    tank = Prim(
        GeomType.BOX,
        (params.tank_width, params.tank_depth, params.tank_height),
        (0, tank_y, tank_z),
        params.color,
    )

    return (base, bowl, tank)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "standard toilet": Params(),
    "compact toilet": Params(
        base_width=0.17,
        base_depth=0.17,
        base_height=0.18,
        bowl_radius=0.15,
        bowl_half_h=0.06,
        tank_width=0.15,
        tank_depth=0.07,
        tank_height=0.18,
    ),
    "elongated toilet": Params(
        base_width=0.20,
        base_depth=0.25,
        base_height=0.20,
        bowl_radius=0.18,
        bowl_half_h=0.09,
        tank_depth=0.09,
        tank_height=0.22,
    ),
    "modern toilet": Params(
        base_width=0.22,
        base_depth=0.22,
        base_height=0.22,
        bowl_radius=0.20,
        bowl_half_h=0.07,
        tank_width=0.20,
        tank_depth=0.06,
        tank_height=0.14,
        color=PLASTIC_WHITE,
    ),
    "round bowl toilet": Params(
        base_width=0.18,
        base_depth=0.18,
        base_height=0.19,
        bowl_radius=0.16,
        bowl_half_h=0.07,
        tank_width=0.16,
        tank_depth=0.08,
        tank_height=0.19,
    ),
}
