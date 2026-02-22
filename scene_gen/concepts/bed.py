"""Bed — a mattress on legs with a headboard.

Parameters:
    width:              Bed width (X) in meters
    length:             Bed length (Y) in meters
    height:             Mattress top height from floor
    mattress_thickness: Mattress thickness in meters
    headboard_height:   Headboard height above mattress
    headboard_thickness: Headboard thickness in meters
    has_footboard:      Include a footboard
    footboard_height:   Footboard height above mattress
    leg_width:          Leg cross-section width
    frame_color:        RGBA for frame, headboard, legs
    mattress_color:     RGBA for mattress
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    FABRIC_BEIGE,
    FABRIC_CREAM,
    FABRIC_GRAY,
    FABRIC_WHITE,
    WOOD_BIRCH,
    WOOD_DARK,
    WOOD_MEDIUM,
    GeomType,
    Prim,
)


@dataclass(frozen=True)
class Params:
    width: float = 1.40
    length: float = 2.00
    height: float = 0.50
    mattress_thickness: float = 0.20
    headboard_height: float = 0.50
    headboard_thickness: float = 0.04
    has_footboard: bool = False
    footboard_height: float = 0.15
    leg_width: float = 0.05
    frame_color: tuple[float, float, float, float] = WOOD_MEDIUM
    mattress_color: tuple[float, float, float, float] = FABRIC_WHITE


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a bed (mattress + headboard + 4 legs + optional footboard, up to 7 prims)."""
    hw = params.width / 2
    hl = params.length / 2
    hlw = params.leg_width / 2
    mt = params.mattress_thickness / 2
    fc = params.frame_color
    mc = params.mattress_color

    prims: list[Prim] = []

    # Mattress: thick box at the specified height
    mattress_z = params.height - mt
    prims.append(Prim(GeomType.BOX, (hw, hl, mt), (0, 0, mattress_z), mc))

    # Headboard: at +Y end, rising above mattress
    hb_half_h = params.headboard_height / 2
    hb_z = params.height + hb_half_h
    hb_y = hl - params.headboard_thickness / 2
    prims.append(
        Prim(
            GeomType.BOX,
            (hw, params.headboard_thickness / 2, hb_half_h),
            (0, hb_y, hb_z),
            fc,
        )
    )

    # Legs: floor to underside of mattress
    leg_top = params.height - params.mattress_thickness
    leg_half = leg_top / 2
    lx = hw - hlw
    ly = hl - hlw

    prims.extend(
        [
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, -ly, leg_half), fc),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, -ly, leg_half), fc),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, ly, leg_half), fc),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, ly, leg_half), fc),
        ]
    )

    # Footboard: at -Y end
    if params.has_footboard:
        fb_half_h = params.footboard_height / 2
        fb_z = params.height + fb_half_h
        fb_y = -(hl - params.headboard_thickness / 2)
        prims.append(
            Prim(
                GeomType.BOX,
                (hw, params.headboard_thickness / 2, fb_half_h),
                (0, fb_y, fb_z),
                fc,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "double": Params(),
    "single": Params(width=0.90, length=1.90),
    "queen": Params(width=1.50, length=2.00),
    "king": Params(
        width=1.80,
        length=2.00,
        headboard_height=0.60,
        frame_color=WOOD_DARK,
    ),
    "platform": Params(
        height=0.35,
        mattress_thickness=0.15,
        headboard_height=0.30,
        leg_width=0.04,
        frame_color=WOOD_BIRCH,
        mattress_color=FABRIC_CREAM,
    ),
    "daybed": Params(
        width=0.90,
        length=1.90,
        height=0.45,
        headboard_height=0.25,
        has_footboard=True,
        footboard_height=0.25,
        frame_color=WOOD_DARK,
        mattress_color=FABRIC_BEIGE,
    ),
    "tall frame": Params(
        height=0.60,
        headboard_height=0.65,
        has_footboard=True,
        footboard_height=0.20,
        frame_color=WOOD_DARK,
        mattress_color=FABRIC_GRAY,
    ),
}
