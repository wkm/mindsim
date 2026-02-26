"""TV stand — a wide, low media console with optional screen on top.

Sims 1 style: chunky media furniture with bold proportions. The TV screen
(when present) is a thin dark box sitting on top — instantly recognizable
as a living room setup.

Variations: media console, entertainment center (tall with shelves),
corner unit, modern floating, retro, with big screen, without screen.

Parameters:
    width:           Total width (X) in meters
    depth:           Total depth (Y) in meters
    height:          Console height (Z) in meters
    top_thickness:   Top slab thickness
    leg_height:      Height of legs (0 = sits on floor)
    leg_width:       Leg cross-section
    has_back_panel:  Include a thin back panel
    has_shelf:       Include a middle shelf between top and bottom
    has_screen:      Include a thin screen panel on top
    screen_width:    Screen width (can be wider than console)
    screen_height:   Screen height above console top
    body_color:      RGBA for the console body
    screen_color:    RGBA for the screen (dark/black)
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_DARK,
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

# Sims 1 colors
SIMS_RED_BROWN = (0.55, 0.25, 0.18, 1.0)
SIMS_OLIVE = (0.45, 0.50, 0.30, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 1.40
    depth: float = 0.40
    height: float = 0.50
    top_thickness: float = 0.03
    leg_height: float = 0.08
    leg_width: float = 0.04
    has_back_panel: bool = True
    has_shelf: bool = False
    has_screen: bool = True
    screen_width: float = 0.90
    screen_height: float = 0.55
    body_color: tuple[float, float, float, float] = WOOD_DARK
    screen_color: tuple[float, float, float, float] = PLASTIC_BLACK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a TV stand (up to 8 prims).

    Budget breakdown:
    - top + bottom shelf + 2 legs + back + mid shelf + screen = 7
    - top + bottom shelf + 2 legs + back + screen = 6
    - top + 2 legs + back + screen = 5 (no bottom shelf when leg_height=0)
    """
    hw = params.width / 2
    hd = params.depth / 2
    ht = params.top_thickness / 2
    bc = params.body_color

    prims: list[Prim] = []

    # Top surface
    top_z = params.height - ht
    prims.append(Prim(GeomType.BOX, (hw, hd, ht), (0, 0, top_z), bc))

    # Bottom shelf (at leg height) and legs
    if params.leg_height > 0:
        shelf_z = params.leg_height + ht
        prims.append(
            Prim(GeomType.BOX, (hw - 0.02, hd - 0.02, ht), (0, 0, shelf_z), bc)
        )

        # 2 short legs (front corners — chunky Sims style)
        hlw = params.leg_width / 2
        lh = params.leg_height / 2
        lx = hw - hlw - 0.01
        ly = hd - hlw - 0.01
        prims.extend(
            [
                Prim(GeomType.BOX, (hlw, hlw, lh), (-lx, -ly, lh), bc),
                Prim(GeomType.BOX, (hlw, hlw, lh), (lx, ly, lh), bc),
            ]
        )

    # Back panel
    if params.has_back_panel:
        back_t = 0.012
        back_h = (params.height - params.leg_height - params.top_thickness) / 2
        back_z = params.leg_height + params.top_thickness + back_h
        prims.append(
            Prim(
                GeomType.BOX,
                (hw - 0.01, back_t / 2, back_h),
                (0, hd - back_t / 2, back_z),
                bc,
            )
        )

    # Middle shelf (between top and bottom)
    if params.has_shelf:
        mid_z = (params.height + params.leg_height) / 2
        prims.append(Prim(GeomType.BOX, (hw - 0.02, hd - 0.02, ht), (0, 0, mid_z), bc))

    # Screen on top (thin flat panel — the TV)
    if params.has_screen:
        screen_hw = params.screen_width / 2
        screen_hh = params.screen_height / 2
        screen_t = 0.015
        screen_z = params.height + screen_hh
        prims.append(
            Prim(
                GeomType.BOX,
                (screen_hw, screen_t / 2, screen_hh),
                (0, hd * 0.5, screen_z),
                params.screen_color,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations (Sims 1 inspired — chunky, bold, recognizable)
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "media console": Params(),
    "wide console": Params(
        width=1.80,
        screen_width=1.20,
        screen_height=0.65,
        body_color=WOOD_MEDIUM,
    ),
    "entertainment center": Params(
        width=1.80,
        height=0.65,
        depth=0.50,
        has_shelf=True,
        screen_width=1.30,
        screen_height=0.70,
        body_color=WOOD_DARK,
        screen_color=PLASTIC_BLACK,
    ),
    "compact": Params(
        width=1.00,
        height=0.45,
        screen_width=0.70,
        screen_height=0.45,
        body_color=WOOD_BIRCH,
    ),
    "no screen": Params(
        has_screen=False,
        body_color=WOOD_LIGHT,
    ),
    "modern floating": Params(
        width=1.50,
        height=0.35,
        leg_height=0.15,
        leg_width=0.025,
        has_screen=True,
        screen_width=1.10,
        screen_height=0.60,
        body_color=METAL_DARK,
        screen_color=PLASTIC_BLACK,
    ),
    "retro": Params(
        width=1.20,
        height=0.55,
        depth=0.45,
        leg_height=0.10,
        leg_width=0.05,
        has_screen=True,
        screen_width=0.65,
        screen_height=0.50,
        body_color=SIMS_RED_BROWN,
        screen_color=PLASTIC_BLACK,
    ),
    "corner unit": Params(
        width=1.00,
        height=0.50,
        depth=0.50,
        has_shelf=True,
        has_screen=True,
        screen_width=0.75,
        screen_height=0.50,
        body_color=SIMS_OLIVE,
        screen_color=PLASTIC_BLACK,
    ),
}
