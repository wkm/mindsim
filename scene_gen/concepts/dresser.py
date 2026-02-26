"""Dresser — a wide, low cabinet with visible drawer fronts.

Sims 1 style: chunky proportions, saturated colors, visible drawer lines
made from thin contrasting boxes.

Variations: low wide dresser, tall chest of drawers, vanity (with mirror),
mid-century modern, colorful nursery, industrial.

Parameters:
    width:           Total width (X) in meters
    depth:           Total depth (Y) in meters
    height:          Total height (Z) in meters
    n_drawers:       Number of drawer rows (visual only)
    drawer_gap:      Gap between drawers in meters
    panel_thickness: Side panel / top thickness in meters
    has_mirror:      Include a vanity mirror on top (thin tall box)
    body_color:      RGBA for body (sides, top, back)
    drawer_color:    RGBA for drawer fronts
    accent_color:    RGBA for drawer line accents / mirror frame
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    METAL_GRAY,
    PLASTIC_BLACK,
    WOOD_BIRCH,
    WOOD_DARK,
    WOOD_LIGHT,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Sims 1 inspired colors
SIMS_PINK = (0.85, 0.45, 0.55, 1.0)
SIMS_TEAL = (0.30, 0.65, 0.60, 1.0)
SIMS_YELLOW = (0.90, 0.80, 0.35, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 1.20
    depth: float = 0.50
    height: float = 0.80
    n_drawers: int = 3
    drawer_gap: float = 0.015
    panel_thickness: float = 0.025
    has_mirror: bool = False
    body_color: tuple[float, float, float, float] = WOOD_MEDIUM
    drawer_color: tuple[float, float, float, float] = WOOD_LIGHT
    accent_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a dresser (top + 2 sides + back + drawer fronts + opt mirror).

    Stays within 8 geoms:
    - Without mirror: top + 2 sides + back + up to 4 drawer fronts = 8
    - With mirror: top + 2 sides + mirror + up to 4 drawer fronts = 8
      (drops back panel to make room for mirror)
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

    if params.has_mirror:
        # Vanity mirror — tall thin panel centered on top
        mirror_w = params.width * 0.5
        mirror_h = params.height * 0.5
        mirror_t = 0.02
        prims.append(
            Prim(
                GeomType.BOX,
                (mirror_w / 2, mirror_t / 2, mirror_h / 2),
                (0, hd * 0.3, params.height + mirror_h / 2),
                params.accent_color,
            )
        )
    else:
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

    # Drawer fronts — evenly spaced vertically, with thin accent lines
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
# Named variations (Sims 1 inspired — chunky, colorful, recognizable)
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "dresser": Params(),
    "wide dresser": Params(
        width=1.50,
        height=0.75,
        n_drawers=3,
        body_color=WOOD_BIRCH,
        drawer_color=WOOD_LIGHT,
        accent_color=WOOD_MEDIUM,
    ),
    "tall chest": Params(
        width=0.80,
        height=1.20,
        n_drawers=4,
        body_color=WOOD_DARK,
        drawer_color=WOOD_MEDIUM,
        accent_color=WOOD_LIGHT,
    ),
    "vanity": Params(
        width=1.10,
        height=0.75,
        n_drawers=3,
        has_mirror=True,
        body_color=WOOD_LIGHT,
        drawer_color=WOOD_BIRCH,
        accent_color=METAL_CHROME,
    ),
    "mid-century": Params(
        width=1.30,
        height=0.70,
        depth=0.45,
        n_drawers=2,
        drawer_gap=0.02,
        panel_thickness=0.03,
        body_color=SIMS_TEAL,
        drawer_color=WOOD_BIRCH,
        accent_color=METAL_CHROME,
    ),
    "nursery pink": Params(
        width=1.00,
        height=0.80,
        n_drawers=3,
        body_color=SIMS_PINK,
        drawer_color=(0.95, 0.85, 0.88, 1.0),
        accent_color=WOOD_LIGHT,
    ),
    "industrial": Params(
        width=1.20,
        height=0.85,
        n_drawers=3,
        body_color=METAL_GRAY,
        drawer_color=WOOD_MEDIUM,
        accent_color=PLASTIC_BLACK,
    ),
    "low credenza": Params(
        width=1.60,
        height=0.55,
        depth=0.45,
        n_drawers=2,
        body_color=WOOD_DARK,
        drawer_color=WOOD_DARK,
        accent_color=METAL_GRAY,
    ),
}
