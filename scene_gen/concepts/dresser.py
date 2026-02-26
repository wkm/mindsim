"""Dresser — a wide, low cabinet with visible drawer fronts.

Parameters:
    width:           Total width (X) in meters
    depth:           Total depth (Y) in meters
    height:          Total height (Z) in meters
    n_drawers:       Number of drawer rows (visual only)
    drawer_gap:      Gap between drawers in meters
    panel_thickness: Side panel / top thickness in meters
    body_color:      RGBA for body (sides, top, back)
    drawer_color:    RGBA for drawer fronts
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_GRAY,
    WOOD_BIRCH,
    WOOD_DARK,
    WOOD_LIGHT,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL


@dataclass(frozen=True)
class Params:
    width: float = 1.20
    depth: float = 0.50
    height: float = 0.80
    n_drawers: int = 3
    drawer_gap: float = 0.015
    panel_thickness: float = 0.025
    body_color: tuple[float, float, float, float] = WOOD_MEDIUM
    drawer_color: tuple[float, float, float, float] = WOOD_LIGHT


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a dresser (top + 2 sides + back + drawer fronts).

    Stays within 8 geoms: top + 2 sides + back + up to 4 drawer fronts.
    """
    hw = params.width / 2
    hd = params.depth / 2
    t = params.panel_thickness
    ht = t / 2
    bc = params.body_color
    dc = params.drawer_color

    prims: list[Prim] = []

    # Top panel
    prims.append(Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.height - ht), bc))

    # Left side panel
    side_h = (params.height - t) / 2
    prims.append(Prim(GeomType.BOX, (ht, hd, side_h), (-hw + ht, 0, side_h), bc))

    # Right side panel
    prims.append(Prim(GeomType.BOX, (ht, hd, side_h), (hw - ht, 0, side_h), bc))

    # Back panel (thin)
    back_t = 0.012
    prims.append(
        Prim(
            GeomType.BOX,
            (hw - t, back_t / 2, side_h),
            (0, hd - back_t / 2, side_h),
            bc,
        )
    )

    # Drawer fronts — evenly spaced vertically
    n = min(params.n_drawers, 4)  # cap at 4 to stay within 8 geoms total
    usable_h = params.height - t  # below the top
    drawer_h = (usable_h - (n + 1) * params.drawer_gap) / n
    drawer_hh = drawer_h / 2
    drawer_hw = hw - t - params.drawer_gap
    drawer_front_t = 0.015

    for i in range(n):
        z = params.drawer_gap + i * (drawer_h + params.drawer_gap) + drawer_hh
        prims.append(
            Prim(
                GeomType.BOX,
                (drawer_hw, drawer_front_t / 2, drawer_hh),
                (0, -hd + drawer_front_t / 2, z),
                dc,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "dresser": Params(),
    "wide dresser": Params(width=1.50, n_drawers=3),
    "tall chest": Params(
        width=0.80,
        height=1.20,
        n_drawers=4,
        body_color=WOOD_DARK,
        drawer_color=WOOD_MEDIUM,
    ),
    "low credenza": Params(
        width=1.60,
        height=0.55,
        depth=0.45,
        n_drawers=2,
        body_color=WOOD_DARK,
        drawer_color=WOOD_DARK,
    ),
    "birch dresser": Params(
        body_color=WOOD_BIRCH,
        drawer_color=WOOD_BIRCH,
    ),
    "industrial": Params(
        body_color=METAL_GRAY,
        drawer_color=WOOD_MEDIUM,
    ),
}
