"""Nightstand — small bedside table with optional drawer.

Parameters:
    width:           Width (X) in meters
    depth:           Depth (Y) in meters
    height:          Height (Z) in meters
    top_thickness:   Top slab thickness in meters
    leg_width:       Leg cross-section width
    has_drawer:      Include a visible drawer front
    body_color:      RGBA for top and legs
    drawer_color:    RGBA for drawer front
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
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
    width: float = 0.45
    depth: float = 0.40
    height: float = 0.55
    top_thickness: float = 0.025
    leg_width: float = 0.035
    has_drawer: bool = True
    body_color: tuple[float, float, float, float] = WOOD_MEDIUM
    drawer_color: tuple[float, float, float, float] = WOOD_LIGHT


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a nightstand (top + 4 legs + optional drawer = 5-6 prims)."""
    hw = params.width / 2
    hd = params.depth / 2
    ht = params.top_thickness / 2
    hlw = params.leg_width / 2
    bc = params.body_color

    prims: list[Prim] = []

    # Top
    prims.append(Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.height - ht), bc))

    # Legs
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
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "nightstand": Params(),
    "modern": Params(
        width=0.50,
        depth=0.40,
        height=0.50,
        leg_width=0.025,
        body_color=WOOD_DARK,
        drawer_color=PLASTIC_WHITE,
    ),
    "tall": Params(height=0.65, body_color=WOOD_DARK),
    "minimal": Params(
        has_drawer=False,
        leg_width=0.025,
        body_color=WOOD_BIRCH,
    ),
    "compact": Params(
        width=0.38,
        depth=0.35,
        height=0.48,
        body_color=WOOD_LIGHT,
        drawer_color=WOOD_LIGHT,
    ),
}
