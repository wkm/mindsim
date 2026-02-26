"""Sink vanity -- chunky Sims 1-style bathroom sink.

Variations include pedestal sinks, double vanities, modern floating,
and farmhouse styles. The basin/bowl is always visible as a distinct prim.

Parameters:
    cab_width:     Half-extent X of the cabinet/pedestal
    cab_depth:     Half-extent Y of the cabinet/pedestal
    cab_height:    Half-extent Z of the cabinet/pedestal
    top_width:     Half-extent X of the countertop
    top_depth:     Half-extent Y of the countertop
    top_thick:     Half-extent Z (thickness) of the countertop
    basin_radius:  Radius of the basin bowl (0 to omit)
    basin_depth:   Half-height of the basin bowl
    mirror_width:  Half-extent X of the mirror (0 to omit)
    mirror_height: Half-extent Z of the mirror
    has_faucet:    Whether to add a faucet cylinder
    cab_color:     RGBA for the cabinet
    top_color:     RGBA for the countertop
    basin_color:   RGBA for the basin interior
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
# Basin interior -- slightly blue-white to suggest porcelain
BASIN_WHITE = (0.88, 0.90, 0.94, 1.0)
# Farmhouse cream
FARMHOUSE_CREAM = (0.94, 0.91, 0.82, 1.0)
# Sage green cabinet
SAGE_GREEN = (0.55, 0.65, 0.52, 1.0)
# Navy blue cabinet
NAVY_CAB = (0.18, 0.22, 0.38, 1.0)


@dataclass(frozen=True)
class Params:
    cab_width: float = 0.30
    cab_depth: float = 0.25
    cab_height: float = 0.40
    top_width: float = 0.32
    top_depth: float = 0.27
    top_thick: float = 0.02
    basin_radius: float = 0.12
    basin_depth: float = 0.04
    mirror_width: float = 0.25
    mirror_height: float = 0.20
    has_faucet: bool = True
    cab_color: tuple[float, float, float, float] = WOOD_MEDIUM
    top_color: tuple[float, float, float, float] = PLASTIC_WHITE
    basin_color: tuple[float, float, float, float] = BASIN_WHITE


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a sink vanity (up to 6 prims: cabinet + top + basin + faucet + mirror + backsplash)."""
    prims: list[Prim] = []

    # 1. Cabinet: box sitting on floor
    cabinet = Prim(
        GeomType.BOX,
        (params.cab_width, params.cab_depth, params.cab_height),
        (0, 0, params.cab_height),
        params.cab_color,
    )
    prims.append(cabinet)

    # 2. Countertop: thin box on top of cabinet
    top_z = params.cab_height * 2 + params.top_thick
    countertop = Prim(
        GeomType.BOX,
        (params.top_width, params.top_depth, params.top_thick),
        (0, 0, top_z),
        params.top_color,
    )
    prims.append(countertop)

    # 3. Basin: cylinder sunk into countertop (visible bowl)
    if params.basin_radius > 0:
        basin_z = top_z + params.top_thick - params.basin_depth * 0.5
        basin = Prim(
            GeomType.CYLINDER,
            (params.basin_radius, params.basin_depth, 0),
            (0, params.cab_depth * 0.15, basin_z),
            params.basin_color,
        )
        prims.append(basin)

    # 4. Faucet: small cylinder behind basin
    if params.has_faucet:
        faucet_z = top_z + params.top_thick + 0.04
        faucet = Prim(
            GeomType.CYLINDER,
            (0.012, 0.04, 0),
            (0, -(params.cab_depth * 0.4), faucet_z),
            METAL_CHROME,
        )
        prims.append(faucet)

    # 5. Mirror: thin box floating above countertop (omit if width is 0)
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
        basin_radius=0.10,
        basin_depth=0.035,
        mirror_width=0.50,
        mirror_height=0.25,
    ),
    "pedestal sink": Params(
        cab_width=0.08,
        cab_depth=0.08,
        cab_height=0.38,
        top_width=0.20,
        top_depth=0.18,
        top_thick=0.03,
        basin_radius=0.14,
        basin_depth=0.05,
        mirror_width=0.18,
        mirror_height=0.18,
        cab_color=PLASTIC_WHITE,
        top_color=PLASTIC_WHITE,
        basin_color=BASIN_WHITE,
    ),
    "modern floating": Params(
        cab_width=0.35,
        cab_depth=0.22,
        cab_height=0.12,
        top_width=0.37,
        top_depth=0.24,
        basin_radius=0.13,
        basin_depth=0.04,
        mirror_width=0.30,
        mirror_height=0.22,
        cab_color=PLASTIC_WHITE,
        top_color=METAL_CHROME,
        basin_color=BASIN_WHITE,
    ),
    "farmhouse apron": Params(
        cab_width=0.35,
        cab_depth=0.28,
        cab_height=0.42,
        top_width=0.37,
        top_depth=0.30,
        top_thick=0.03,
        basin_radius=0.14,
        basin_depth=0.06,
        mirror_width=0.28,
        mirror_height=0.22,
        cab_color=FARMHOUSE_CREAM,
        top_color=FARMHOUSE_CREAM,
        basin_color=BASIN_WHITE,
    ),
    "sage green vanity": Params(
        cab_width=0.32,
        cab_depth=0.26,
        cab_height=0.40,
        top_width=0.34,
        top_depth=0.28,
        basin_radius=0.12,
        basin_depth=0.04,
        mirror_width=0.28,
        mirror_height=0.22,
        cab_color=SAGE_GREEN,
        top_color=PLASTIC_WHITE,
    ),
    "navy vanity": Params(
        cab_width=0.32,
        cab_depth=0.26,
        cab_height=0.40,
        top_width=0.34,
        top_depth=0.28,
        basin_radius=0.12,
        basin_depth=0.04,
        mirror_width=0.28,
        mirror_height=0.22,
        cab_color=NAVY_CAB,
        top_color=PLASTIC_WHITE,
    ),
    "rustic wood": Params(
        cab_width=0.32,
        cab_depth=0.26,
        cab_height=0.42,
        top_width=0.34,
        top_depth=0.28,
        basin_radius=0.13,
        basin_depth=0.05,
        mirror_width=0.28,
        mirror_height=0.22,
        cab_color=WOOD_LIGHT,
        top_color=PLASTIC_WHITE,
    ),
}
