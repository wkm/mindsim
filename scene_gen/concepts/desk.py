"""Desk — a work surface with legs, optional modesty panel, drawer, and monitor shelf.

Sims 1 style: chunky, toylike proportions with saturated colors.

Parameters:
    width:           Desktop width (X) in meters
    depth:           Desktop depth (Y) in meters
    height:          Desktop height from floor
    top_thickness:   Desktop slab thickness
    leg_width:       Leg cross-section width
    has_panel:       Include modesty panel (back panel between legs)
    panel_thickness: Modesty panel thickness
    has_drawer:      Include a drawer box inset under the desktop
    has_shelf:       Include a raised monitor shelf on the back half
    shelf_height:    Height of the monitor shelf above desktop
    color:           RGBA for the whole desk
    panel_color:     RGBA for the modesty panel (if different)
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_GRAY,
    PLASTIC_BLACK,
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


@dataclass(frozen=True)
class Params:
    width: float = 1.20
    depth: float = 0.60
    height: float = 0.74
    top_thickness: float = 0.035
    leg_width: float = 0.05
    has_panel: bool = True
    panel_thickness: float = 0.018
    has_drawer: bool = False
    has_shelf: bool = False
    shelf_height: float = 0.12
    color: tuple[float, float, float, float] = WOOD_MEDIUM
    panel_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a desk (top + 4 legs + optional panel/drawer/shelf, max 8 prims)."""
    hw = params.width / 2
    hd = params.depth / 2
    ht = params.top_thickness / 2
    hlw = params.leg_width / 2
    c = params.color

    prims: list[Prim] = []

    # Desktop surface
    prims.append(Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.height - ht), c))

    # Legs — chunky square
    leg_full = params.height - params.top_thickness
    leg_half = leg_full / 2
    lx = hw - hlw - 0.01
    ly = hd - hlw - 0.01

    prims.extend(
        [
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, -ly, leg_half), c),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, -ly, leg_half), c),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, ly, leg_half), c),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, ly, leg_half), c),
        ]
    )

    # Modesty panel: at rear edge (+Y), from floor to underside of desktop
    if params.has_panel:
        panel_half_h = leg_full / 2
        panel_y = hd - params.panel_thickness / 2
        prims.append(
            Prim(
                GeomType.BOX,
                (hw - params.leg_width, params.panel_thickness / 2, panel_half_h),
                (0, panel_y, panel_half_h),
                params.panel_color,
            )
        )

    # Drawer: small box inset under the desktop on the front side
    if params.has_drawer and len(prims) < 8:
        drawer_h = leg_full * 0.22
        drawer_hh = drawer_h / 2
        drawer_z = params.height - params.top_thickness - drawer_hh
        drawer_hw = hw * 0.45
        drawer_hd = hd * 0.70
        prims.append(
            Prim(
                GeomType.BOX,
                (drawer_hw, drawer_hd, drawer_hh),
                (0, -hd + drawer_hd + 0.01, drawer_z),
                params.panel_color,
            )
        )

    # Monitor shelf: narrow raised platform on the back half of the desk
    if params.has_shelf and len(prims) < 8:
        shelf_w = hw * 0.80
        shelf_d = hd * 0.35
        shelf_ht = params.top_thickness / 2
        shelf_z = params.height + params.shelf_height - shelf_ht
        prims.append(
            Prim(
                GeomType.BOX,
                (shelf_w, shelf_d, shelf_ht),
                (0, hd - shelf_d, shelf_z),
                c,
            )
        )

    return tuple(prims[:8])


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "writing desk": Params(),
    "computer desk": Params(
        width=1.40,
        depth=0.70,
        height=0.74,
        has_shelf=True,
        shelf_height=0.12,
        color=WOOD_DARK,
        panel_color=WOOD_DARK,
    ),
    "standing desk": Params(
        width=1.40,
        depth=0.70,
        height=1.05,
        top_thickness=0.04,
        leg_width=0.06,
        has_panel=False,
        color=WOOD_BIRCH,
    ),
    "compact w/ drawer": Params(
        width=0.85,
        depth=0.55,
        height=0.74,
        has_drawer=True,
        color=WOOD_LIGHT,
        panel_color=WOOD_MEDIUM,
    ),
    "industrial": Params(
        width=1.50,
        depth=0.65,
        height=0.76,
        top_thickness=0.045,
        leg_width=0.04,
        has_panel=False,
        color=WOOD_DARK,
        panel_color=METAL_GRAY,
    ),
    "modern white": Params(
        width=1.30,
        depth=0.60,
        height=0.74,
        top_thickness=0.03,
        leg_width=0.04,
        has_drawer=True,
        color=PLASTIC_WHITE,
        panel_color=PLASTIC_BLACK,
    ),
    "executive": Params(
        width=1.60,
        depth=0.75,
        height=0.76,
        top_thickness=0.05,
        leg_width=0.06,
        has_panel=True,
        has_drawer=True,
        color=WOOD_DARK,
        panel_color=WOOD_DARK,
    ),
    "student desk": Params(
        width=1.00,
        depth=0.55,
        height=0.74,
        top_thickness=0.03,
        leg_width=0.04,
        has_panel=True,
        has_shelf=True,
        shelf_height=0.10,
        color=WOOD_BIRCH,
        panel_color=WOOD_LIGHT,
    ),
}
