"""Nightstand — small bedside table with optional drawer and shelf.

Sims 1 style: chunky, small, recognizable silhouettes. These are compact
pieces that sit next to a bed — visible drawer fronts, optional open shelf.

Variations: classic with drawer, modern cube, lamp shelf, rustic,
colorful, round-leg.

Parameters:
    width:           Width (X) in meters
    depth:           Depth (Y) in meters
    height:          Height (Z) in meters
    top_thickness:   Top slab thickness in meters
    leg_width:       Leg cross-section width
    has_drawer:      Include a visible drawer front
    has_shelf:       Include an open shelf (mid-height panel)
    is_cube:         Solid cube style (no legs, panel sides instead)
    body_color:      RGBA for top and legs
    drawer_color:    RGBA for drawer front
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    METAL_DARK,
    PLASTIC_WHITE,
    WOOD_BIRCH,
    WOOD_DARK,
    WOOD_LIGHT,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Sims 1 accent colors
SIMS_MINT = (0.55, 0.82, 0.70, 1.0)
SIMS_CORAL = (0.90, 0.50, 0.40, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.45
    depth: float = 0.40
    height: float = 0.55
    top_thickness: float = 0.025
    leg_width: float = 0.035
    has_drawer: bool = True
    has_shelf: bool = False
    is_cube: bool = False
    body_color: tuple[float, float, float, float] = WOOD_MEDIUM
    drawer_color: tuple[float, float, float, float] = WOOD_LIGHT


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a nightstand (up to 8 prims).

    Cube style: top + 2 sides + back + opt shelf + opt drawer = 4-6
    Leg style:  top + 4 legs + opt shelf + opt drawer = 5-7
    """
    hw = params.width / 2
    hd = params.depth / 2
    ht = params.top_thickness / 2
    bc = params.body_color

    prims: list[Prim] = []

    # Top
    prims.append(Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.height - ht), bc))

    if params.is_cube:
        # Cube / enclosed style: side panels + back instead of legs
        panel_h = (params.height - params.top_thickness) / 2
        panel_t = params.top_thickness / 2

        # Left side
        prims.append(
            Prim(GeomType.BOX, (panel_t, hd, panel_h), (-hw + panel_t, 0, panel_h), bc)
        )
        # Right side
        prims.append(
            Prim(GeomType.BOX, (panel_t, hd, panel_h), (hw - panel_t, 0, panel_h), bc)
        )
        # Back panel
        back_t = 0.012
        prims.append(
            Prim(
                GeomType.BOX,
                (hw - params.top_thickness, back_t / 2, panel_h),
                (0, hd - back_t / 2, panel_h),
                bc,
            )
        )
    else:
        # Leg style
        hlw = params.leg_width / 2
        leg_full = params.height - params.top_thickness
        leg_half = leg_full / 2
        lx = hw - hlw - 0.005
        ly = hd - hlw - 0.005

        prims.extend(
            [
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, -ly, leg_half), bc),
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, -ly, leg_half), bc),
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, ly, leg_half), bc),
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, ly, leg_half), bc),
            ]
        )

    # Open shelf (mid-height horizontal panel)
    if params.has_shelf:
        shelf_z = params.height * 0.35
        shelf_hw = hw - params.leg_width - 0.005
        shelf_hd = hd - 0.01
        prims.append(
            Prim(GeomType.BOX, (shelf_hw, shelf_hd, 0.01), (0, 0, shelf_z), bc)
        )

    # Drawer front (centered vertically in the upper half)
    if params.has_drawer:
        drawer_h = (params.height - params.top_thickness) * 0.35
        drawer_hh = drawer_h / 2
        drawer_z = params.height * 0.55
        drawer_hw = hw - params.leg_width - 0.01
        drawer_t = 0.012
        prims.append(
            Prim(
                GeomType.BOX,
                (drawer_hw, drawer_t / 2, drawer_hh),
                (0, -hd + drawer_t / 2, drawer_z),
                params.drawer_color,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations (Sims 1 inspired — chunky, small, colorful)
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "nightstand": Params(),
    "modern cube": Params(
        width=0.42,
        depth=0.40,
        height=0.50,
        is_cube=True,
        has_drawer=True,
        body_color=WOOD_DARK,
        drawer_color=PLASTIC_WHITE,
    ),
    "with shelf": Params(
        width=0.48,
        depth=0.42,
        height=0.58,
        has_drawer=True,
        has_shelf=True,
        body_color=WOOD_BIRCH,
        drawer_color=WOOD_LIGHT,
    ),
    "rustic": Params(
        width=0.50,
        depth=0.42,
        height=0.55,
        leg_width=0.045,
        top_thickness=0.035,
        has_drawer=True,
        body_color=WOOD_DARK,
        drawer_color=WOOD_MEDIUM,
    ),
    "tall": Params(
        height=0.65,
        has_drawer=True,
        body_color=WOOD_DARK,
        drawer_color=WOOD_DARK,
    ),
    "mint cube": Params(
        width=0.40,
        depth=0.38,
        height=0.48,
        is_cube=True,
        has_drawer=False,
        body_color=SIMS_MINT,
        drawer_color=SIMS_MINT,
    ),
    "coral compact": Params(
        width=0.38,
        depth=0.35,
        height=0.48,
        has_drawer=True,
        body_color=SIMS_CORAL,
        drawer_color=(0.95, 0.80, 0.75, 1.0),
    ),
    "industrial": Params(
        width=0.45,
        depth=0.40,
        height=0.52,
        leg_width=0.025,
        body_color=METAL_DARK,
        drawer_color=METAL_CHROME,
    ),
}
