"""Bathtub -- chunky Sims 1-style bathroom tub.

Built from wall panels around a basin floor so the hollow interior is
visible from the catalog camera angle.  MuJoCo renders solid volumes,
so a single-box approach just looks like a block -- separate walls
create actual negative space.

Parameters:
    width:        Half-extent X (across the tub)
    depth:        Half-extent Y (length of the tub)
    height:       Half-extent Z (wall height)
    wall_thick:   Wall thickness for side/end panels
    has_feet:     Whether to add 4 clawfoot-style feet
    foot_radius:  Radius of each foot sphere
    foot_height:  Height the tub is raised on feet
    has_faucet:   Whether to add a faucet cylinder at one end
    shell_color:  RGBA for the tub walls
    basin_color:  RGBA for the basin floor (contrasting interior tone)
    accent_color: RGBA for feet/faucet accents
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
    wall_thick: float = 0.05
    has_feet: bool = False
    foot_radius: float = 0.035
    foot_height: float = 0.07
    has_faucet: bool = True
    shell_color: tuple[float, float, float, float] = PLASTIC_WHITE
    basin_color: tuple[float, float, float, float] = BASIN_INTERIOR
    accent_color: tuple[float, float, float, float] = METAL_CHROME


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a bathtub from wall panels around a basin floor.

    Built from separate panels so the hollow interior is visible:
      - Basin floor (basin_color) — the bottom of the tub
      - 2 long side walls (left/right along Y)
      - End walls as budget allows (front visible to camera, back against room wall)
      - Optional faucet, feet

    Prim budget (max 8):
      Standard:  floor + 2 sides + front + back + faucet = 6
      Clawfoot:  floor + 2 sides + front + 4 feet = 8
    """
    prims: list[Prim] = []

    base_z = params.foot_height if params.has_feet else 0.0
    w = params.width
    d = params.depth
    full_h = params.height * 2
    wt = params.wall_thick
    hwt = wt / 2

    # Floor: thin slab at the bottom, basin_color to read as interior
    floor_ht = 0.015
    prims.append(
        Prim(
            GeomType.BOX,
            (w, d, floor_ht),
            (0, 0, base_z + floor_ht),
            params.basin_color,
        )
    )

    # Wall height: from floor top to tub rim
    floor_top = base_z + floor_ht * 2
    wall_hh = (full_h - floor_ht * 2) / 2
    wall_z = floor_top + wall_hh

    # Left side wall (-X)
    prims.append(
        Prim(GeomType.BOX, (hwt, d, wall_hh), (-w + hwt, 0, wall_z), params.shell_color)
    )

    # Right side wall (+X)
    prims.append(
        Prim(GeomType.BOX, (hwt, d, wall_hh), (w - hwt, 0, wall_z), params.shell_color)
    )

    # Front end wall (-Y, camera-facing)
    inner_w = w - wt  # avoid overlap with side walls
    prims.append(
        Prim(
            GeomType.BOX,
            (inner_w, hwt, wall_hh),
            (0, -d + hwt, wall_z),
            params.shell_color,
        )
    )

    # Back end wall (+Y, against room wall) — skip for clawfoot to save budget
    if not params.has_feet:
        prims.append(
            Prim(
                GeomType.BOX,
                (inner_w, hwt, wall_hh),
                (0, d - hwt, wall_z),
                params.shell_color,
            )
        )

    # Faucet: chunky cylinder at the back end
    if params.has_faucet and len(prims) < 8:
        faucet_hh = 0.05
        faucet_z = base_z + full_h + faucet_hh
        prims.append(
            Prim(
                GeomType.CYLINDER,
                (0.02, faucet_hh, 0),
                (0, d - wt * 2, faucet_z),
                params.accent_color,
            )
        )

    # Clawfoot feet: 4 spheres at corners
    if params.has_feet:
        fx = w * 0.75
        fy = d * 0.80
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            if len(prims) >= 8:
                break
            prims.append(
                Prim(
                    GeomType.SPHERE,
                    (params.foot_radius, 0, 0),
                    (sx * fx, sy * fy, params.foot_radius),
                    params.accent_color,
                )
            )

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
        wall_thick=0.05,
        has_feet=True,
        foot_radius=0.04,
        foot_height=0.08,
        has_faucet=False,
        shell_color=CLAWFOOT_CREAM,
        accent_color=BRASS_GOLD,
    ),
    "clawfoot cast iron": Params(
        width=0.38,
        depth=0.85,
        height=0.38,
        wall_thick=0.05,
        has_feet=True,
        foot_radius=0.035,
        foot_height=0.07,
        has_faucet=False,
        shell_color=CAST_IRON,
        basin_color=BASIN_INTERIOR,
        accent_color=METAL_CHROME,
    ),
    "corner tub": Params(
        width=0.60,
        depth=0.60,
        height=0.30,
        wall_thick=0.06,
    ),
    "modern freestanding": Params(
        width=0.36,
        depth=0.78,
        height=0.34,
        wall_thick=0.05,
    ),
    "deep soaking": Params(
        width=0.35,
        depth=0.70,
        height=0.42,
        wall_thick=0.05,
    ),
    "retro green": Params(
        width=0.40,
        depth=0.80,
        height=0.32,
        wall_thick=0.05,
        shell_color=RETRO_GREEN,
        basin_color=BASIN_INTERIOR,
        accent_color=METAL_CHROME,
    ),
    "compact alcove": Params(
        width=0.36,
        depth=0.72,
        height=0.28,
    ),
}
