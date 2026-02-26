"""Bathtub -- chunky Sims 1-style bathroom tub.

Distinct silhouettes for standard built-in, clawfoot (with visible feet),
corner tub, and modern freestanding oval. The outer shell + inner basin
create the illusion of a hollow vessel; extra prims add feet, rims, or faucets.

Parameters:
    width:        Half-extent X (across the tub)
    depth:        Half-extent Y (length of the tub)
    height:       Half-extent Z (wall height)
    wall_thick:   Wall thickness (outer minus inner half-extents)
    has_rim:      Whether to add a visible rim along the top edge
    rim_thick:    Half-height of the rim
    has_feet:     Whether to add 4 clawfoot-style feet
    foot_radius:  Radius of each foot sphere
    foot_height:  Height the tub is raised on feet
    has_faucet:   Whether to add a faucet cylinder at one end
    shell_color:  RGBA for the outer shell
    basin_color:  RGBA for the inner basin
    accent_color: RGBA for rim/feet/faucet accents
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    PLASTIC_WHITE,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Basin interior -- slightly blue-white to suggest water/porcelain
BASIN_INTERIOR = (0.85, 0.88, 0.92, 1.0)
# Vintage cream for clawfoot tubs
CLAWFOOT_CREAM = (0.94, 0.91, 0.84, 1.0)
# Cast iron dark exterior
CAST_IRON = (0.25, 0.25, 0.28, 1.0)
# Brass/gold accent for feet and fixtures
BRASS_GOLD = (0.78, 0.65, 0.30, 1.0)
# Pale green retro tub
RETRO_GREEN = (0.75, 0.86, 0.75, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.40
    depth: float = 0.80
    height: float = 0.30
    wall_thick: float = 0.04
    has_rim: bool = False
    rim_thick: float = 0.012
    has_feet: bool = False
    foot_radius: float = 0.03
    foot_height: float = 0.06
    has_faucet: bool = True
    shell_color: tuple[float, float, float, float] = PLASTIC_WHITE
    basin_color: tuple[float, float, float, float] = BASIN_INTERIOR
    accent_color: tuple[float, float, float, float] = METAL_CHROME


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a bathtub (up to 8 prims: shell + basin + rim + faucet + 4 feet)."""
    prims: list[Prim] = []

    # Base elevation: raised if tub has feet
    base_z = params.foot_height if params.has_feet else 0.0

    # 1. Outer shell: box sitting on floor (or on feet)
    shell = Prim(
        GeomType.BOX,
        (params.width, params.depth, params.height),
        (0, 0, base_z + params.height),
        params.shell_color,
    )
    prims.append(shell)

    # 2. Inner basin: slightly smaller box for hollow look
    inner_w = params.width - params.wall_thick
    inner_d = params.depth - params.wall_thick
    inner_h = params.height - params.wall_thick
    basin_z = base_z + params.wall_thick + inner_h
    basin = Prim(
        GeomType.BOX,
        (inner_w, inner_d, inner_h),
        (0, 0, basin_z),
        params.basin_color,
    )
    prims.append(basin)

    # 3. Rim: thin box around the top edge for a visible lip
    if params.has_rim:
        rim_z = base_z + params.height * 2 + params.rim_thick
        rim = Prim(
            GeomType.BOX,
            (params.width + 0.01, params.depth + 0.01, params.rim_thick),
            (0, 0, rim_z),
            params.accent_color,
        )
        prims.append(rim)

    # 4. Faucet: small cylinder at one end of the tub
    if params.has_faucet:
        faucet_z = base_z + params.height * 2 + 0.04
        faucet = Prim(
            GeomType.CYLINDER,
            (0.015, 0.04, 0),
            (0, -(params.depth - 0.06), faucet_z),
            params.accent_color,
        )
        prims.append(faucet)

    # 5-8. Clawfoot feet: 4 spheres at corners
    if params.has_feet:
        fx = params.width - 0.04
        fy = params.depth - 0.06
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            foot = Prim(
                GeomType.SPHERE,
                (params.foot_radius, 0, 0),
                (sx * fx, sy * fy, params.foot_radius),
                params.accent_color,
            )
            prims.append(foot)

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "standard bathtub": Params(),
    "clawfoot cream": Params(
        width=0.38,
        depth=0.82,
        height=0.35,
        wall_thick=0.03,
        has_rim=True,
        rim_thick=0.015,
        has_feet=True,
        foot_radius=0.035,
        foot_height=0.07,
        shell_color=CLAWFOOT_CREAM,
        accent_color=BRASS_GOLD,
    ),
    "clawfoot cast iron": Params(
        width=0.38,
        depth=0.85,
        height=0.38,
        wall_thick=0.035,
        has_rim=True,
        rim_thick=0.012,
        has_feet=True,
        foot_radius=0.03,
        foot_height=0.06,
        shell_color=CAST_IRON,
        basin_color=BASIN_INTERIOR,
        accent_color=METAL_CHROME,
    ),
    "corner tub": Params(
        width=0.60,
        depth=0.60,
        height=0.30,
        wall_thick=0.05,
        has_rim=True,
        rim_thick=0.012,
        has_faucet=True,
    ),
    "modern freestanding": Params(
        width=0.36,
        depth=0.78,
        height=0.34,
        wall_thick=0.035,
        has_rim=True,
        rim_thick=0.01,
        has_feet=False,
        shell_color=PLASTIC_WHITE,
        accent_color=METAL_CHROME,
    ),
    "deep soaking": Params(
        width=0.35,
        depth=0.70,
        height=0.42,
        wall_thick=0.04,
        has_rim=True,
        rim_thick=0.015,
    ),
    "retro green": Params(
        width=0.40,
        depth=0.80,
        height=0.32,
        wall_thick=0.04,
        has_rim=True,
        rim_thick=0.012,
        shell_color=RETRO_GREEN,
        basin_color=BASIN_INTERIOR,
        accent_color=METAL_CHROME,
    ),
    "compact alcove": Params(
        width=0.36,
        depth=0.72,
        height=0.28,
        wall_thick=0.04,
        has_rim=False,
        has_faucet=True,
    ),
}
