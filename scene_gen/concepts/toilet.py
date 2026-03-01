"""Toilet -- chunky Sims 1-style bathroom fixture.

Recognizable toilet silhouette with visible tank, bowl, seat, and pedestal.
Variations range from standard to vintage pull-chain to modern low-profile.

Parameters:
    base_width:   Half-extent X of the pedestal base
    base_depth:   Half-extent Y of the pedestal base
    base_height:  Half-extent Z of the pedestal base
    bowl_radius:  Radius of the bowl cylinder
    bowl_half_h:  Half-height of the bowl cylinder
    seat_radius:  Radius of the seat ring (0 to omit)
    tank_width:   Half-extent X of the tank
    tank_depth:   Half-extent Y of the tank
    tank_height:  Half-extent Z of the tank
    tank_cap_h:   Half-height of the tank lid cap (0 to omit)
    has_lid:      Whether to show an angled lid on the bowl
    color:        RGBA for the ceramic body
    accent_color: RGBA for seat/lid/accents
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
# Warm cream ceramic
CERAMIC_CREAM = (0.94, 0.91, 0.82, 1.0)
# Vintage pale green
VINTAGE_GREEN = (0.78, 0.88, 0.78, 1.0)
# Vintage pale pink
VINTAGE_PINK = (0.90, 0.80, 0.82, 1.0)


@dataclass(frozen=True)
class Params:
    base_width: float = 0.18
    base_depth: float = 0.18
    base_height: float = 0.18
    bowl_radius: float = 0.17
    bowl_half_h: float = 0.07
    seat_radius: float = 0.18
    tank_width: float = 0.17
    tank_depth: float = 0.08
    tank_height: float = 0.20
    tank_cap_h: float = 0.015
    has_lid: bool = True
    color: tuple[float, float, float, float] = CERAMIC_WHITE
    accent_color: tuple[float, float, float, float] = PLASTIC_WHITE


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a toilet (up to 6 prims: base + bowl + seat + lid + tank + tank cap)."""
    prims: list[Prim] = []

    # 1. Base pedestal: box sitting on floor
    base = Prim(
        GeomType.BOX,
        (params.base_width, params.base_depth, params.base_height),
        (0, 0, params.base_height),
        params.color,
    )
    prims.append(base)

    # 2. Bowl: cylinder sitting on top of base, extends forward
    bowl_z = params.base_height * 2 + params.bowl_half_h
    bowl = Prim(
        GeomType.CYLINDER,
        (params.bowl_radius, params.bowl_half_h, 0),
        (0, params.base_depth * 0.3, bowl_z),
        params.color,
    )
    prims.append(bowl)

    # 3. Seat ring: thin cylinder on top of bowl (visible rim)
    if params.seat_radius > 0:
        seat_z = bowl_z + params.bowl_half_h + 0.008
        seat = Prim(
            GeomType.CYLINDER,
            (params.seat_radius, 0.008, 0),
            (0, params.base_depth * 0.3, seat_z),
            params.accent_color,
        )
        prims.append(seat)

    # 4. Lid: tilted ellipsoid resting on top of bowl (if shown)
    if params.has_lid:
        lid_z = bowl_z + params.bowl_half_h + 0.02
        lid = Prim(
            GeomType.ELLIPSOID,
            (params.bowl_radius * 0.9, params.bowl_radius * 0.85, 0.015),
            (0, params.base_depth * 0.3, lid_z),
            params.accent_color,
        )
        prims.append(lid)

    # 5. Tank: box behind the bowl, sitting on base
    tank_z = params.base_height * 2 + params.tank_height
    tank_y = -(params.base_depth + params.tank_depth)
    tank = Prim(
        GeomType.BOX,
        (params.tank_width, params.tank_depth, params.tank_height),
        (0, tank_y, tank_z),
        params.color,
    )
    prims.append(tank)

    # 6. Tank cap: thin box on top of tank (the lid)
    if params.tank_cap_h > 0:
        cap_z = tank_z + params.tank_height + params.tank_cap_h
        tank_cap = Prim(
            GeomType.BOX,
            (params.tank_width + 0.01, params.tank_depth + 0.01, params.tank_cap_h),
            (0, tank_y, cap_z),
            params.accent_color,
        )
        prims.append(tank_cap)

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "standard toilet": Params(),
    "compact toilet": Params(
        base_width=0.15,
        base_depth=0.15,
        base_height=0.16,
        bowl_radius=0.14,
        bowl_half_h=0.06,
        seat_radius=0.15,
        tank_width=0.14,
        tank_depth=0.06,
        tank_height=0.16,
        tank_cap_h=0.012,
    ),
    "modern low-profile": Params(
        base_width=0.20,
        base_depth=0.22,
        base_height=0.14,
        bowl_radius=0.19,
        bowl_half_h=0.06,
        seat_radius=0.20,
        tank_width=0.19,
        tank_depth=0.05,
        tank_height=0.12,
        tank_cap_h=0.01,
        color=PLASTIC_WHITE,
        accent_color=PLASTIC_WHITE,
    ),
    "vintage green": Params(
        base_width=0.17,
        base_depth=0.17,
        base_height=0.20,
        bowl_radius=0.16,
        bowl_half_h=0.08,
        seat_radius=0.17,
        tank_width=0.16,
        tank_depth=0.10,
        tank_height=0.24,
        tank_cap_h=0.018,
        color=VINTAGE_GREEN,
        accent_color=VINTAGE_GREEN,
    ),
    "vintage pink": Params(
        base_width=0.17,
        base_depth=0.17,
        base_height=0.20,
        bowl_radius=0.16,
        bowl_half_h=0.08,
        seat_radius=0.17,
        tank_width=0.16,
        tank_depth=0.10,
        tank_height=0.24,
        tank_cap_h=0.018,
        color=VINTAGE_PINK,
        accent_color=VINTAGE_PINK,
    ),
    "elongated bowl": Params(
        base_width=0.18,
        base_depth=0.22,
        base_height=0.18,
        bowl_radius=0.18,
        bowl_half_h=0.09,
        seat_radius=0.19,
        tank_width=0.17,
        tank_depth=0.09,
        tank_height=0.22,
    ),
    "round bowl": Params(
        base_width=0.16,
        base_depth=0.16,
        base_height=0.18,
        bowl_radius=0.16,
        bowl_half_h=0.07,
        seat_radius=0.17,
        tank_width=0.15,
        tank_depth=0.08,
        tank_height=0.18,
    ),
    "cream ceramic": Params(
        base_width=0.18,
        base_depth=0.19,
        base_height=0.19,
        bowl_radius=0.17,
        bowl_half_h=0.08,
        seat_radius=0.18,
        tank_width=0.17,
        tank_depth=0.09,
        tank_height=0.21,
        color=CERAMIC_CREAM,
        accent_color=CERAMIC_CREAM,
    ),
}
