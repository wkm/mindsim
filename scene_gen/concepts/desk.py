"""Desk — a work surface with legs and optional modesty panel.

Parameters:
    width:           Desktop width (X) in meters
    depth:           Desktop depth (Y) in meters
    height:          Desktop height from floor
    top_thickness:   Desktop slab thickness
    leg_width:       Leg cross-section width
    has_panel:       Include modesty panel (back panel between legs)
    panel_thickness: Modesty panel thickness
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
    Prim,
)


@dataclass(frozen=True)
class Params:
    width: float = 1.20
    depth: float = 0.60
    height: float = 0.74
    top_thickness: float = 0.03
    leg_width: float = 0.04
    has_panel: bool = True
    panel_thickness: float = 0.015
    color: tuple[float, float, float, float] = WOOD_MEDIUM
    panel_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a desk (top + 4 legs + optional panel = up to 6 prims)."""
    hw = params.width / 2
    hd = params.depth / 2
    ht = params.top_thickness / 2
    hlw = params.leg_width / 2
    c = params.color

    prims: list[Prim] = []

    # Desktop surface
    prims.append(Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.height - ht), c))

    # Legs
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

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "writing desk": Params(),
    "computer desk": Params(
        width=1.40,
        depth=0.70,
        height=0.74,
        color=WOOD_DARK,
        panel_color=WOOD_DARK,
    ),
    "standing desk": Params(
        width=1.40,
        depth=0.70,
        height=1.05,
        top_thickness=0.035,
        leg_width=0.05,
        has_panel=False,
        color=WOOD_BIRCH,
    ),
    "compact": Params(
        width=0.80,
        depth=0.50,
        height=0.74,
        color=WOOD_LIGHT,
        panel_color=WOOD_MEDIUM,
    ),
    "industrial": Params(
        width=1.50,
        depth=0.65,
        height=0.76,
        top_thickness=0.04,
        leg_width=0.035,
        has_panel=False,
        color=WOOD_DARK,
        panel_color=METAL_GRAY,
    ),
    "modern": Params(
        width=1.30,
        depth=0.60,
        height=0.74,
        top_thickness=0.025,
        leg_width=0.03,
        color=PLASTIC_WHITE,
        panel_color=PLASTIC_BLACK,
    ),
}
