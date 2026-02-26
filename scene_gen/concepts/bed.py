"""Bed — a mattress on legs with headboard, optional pillows and footboard.

Sims 1 style: chunky mattress, visible pillows, bold headboard, toylike proportions.
Variations: single, double, queen, bunk-bed feel, platform, daybed.

Parameters:
    width:              Bed width (X) in meters
    length:             Bed length (Y) in meters
    height:             Mattress top height from floor
    mattress_thickness: Mattress thickness in meters
    headboard_height:   Headboard height above mattress
    headboard_thickness: Headboard thickness in meters
    has_footboard:      Include a footboard
    footboard_height:   Footboard height above mattress
    has_pillow:         Include visible pillow boxes at head end
    num_pillows:        Number of pillows (1 or 2)
    pillow_color:       RGBA for pillows
    leg_width:          Leg cross-section width
    frame_color:        RGBA for frame, headboard, legs
    mattress_color:     RGBA for mattress
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    FABRIC_BEIGE,
    FABRIC_BLUE,
    FABRIC_CREAM,
    FABRIC_PINK,
    FABRIC_PURPLE,
    FABRIC_WHITE,
    FABRIC_YELLOW,
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
    width: float = 1.40
    length: float = 2.00
    height: float = 0.52
    mattress_thickness: float = 0.22
    headboard_height: float = 0.52
    headboard_thickness: float = 0.05
    has_footboard: bool = False
    footboard_height: float = 0.15
    has_pillow: bool = True
    num_pillows: int = 2
    pillow_color: tuple[float, float, float, float] = FABRIC_WHITE
    leg_width: float = 0.06
    frame_color: tuple[float, float, float, float] = WOOD_MEDIUM
    mattress_color: tuple[float, float, float, float] = FABRIC_WHITE


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a bed (up to 8 prims: mattress + headboard + pillows + 4 legs + footboard)."""
    hw = params.width / 2
    hl = params.length / 2
    hlw = params.leg_width / 2
    mt = params.mattress_thickness / 2
    fc = params.frame_color
    mc = params.mattress_color

    prims: list[Prim] = []

    # Mattress: chunky thick slab
    mattress_z = params.height - mt
    prims.append(Prim(GeomType.BOX, (hw, hl, mt), (0, 0, mattress_z), mc))

    # Headboard: at +Y end, rising above mattress -- bold and chunky
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

    # Pillows: small chunky boxes at the head end of the mattress
    if params.has_pillow:
        pillow_h = 0.06  # half-height
        pillow_d = 0.10  # half-depth (Y)
        pillow_z = params.height + pillow_h
        pillow_y = hl - params.headboard_thickness - pillow_d - 0.02

        if params.num_pillows >= 2 and params.width > 0.80:
            # Two pillows side by side
            pillow_w = hw * 0.42
            gap = 0.03
            prims.extend(
                [
                    Prim(
                        GeomType.BOX,
                        (pillow_w, pillow_d, pillow_h),
                        (-(pillow_w + gap), pillow_y, pillow_z),
                        params.pillow_color,
                    ),
                    Prim(
                        GeomType.BOX,
                        (pillow_w, pillow_d, pillow_h),
                        (pillow_w + gap, pillow_y, pillow_z),
                        params.pillow_color,
                    ),
                ]
            )
        else:
            # Single centered pillow
            pillow_w = hw * 0.65
            prims.append(
                Prim(
                    GeomType.BOX,
                    (pillow_w, pillow_d, pillow_h),
                    (0, pillow_y, pillow_z),
                    params.pillow_color,
                )
            )

    # Legs: chunky stubby supports
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

    # Footboard: at -Y end (shorter than headboard)
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
    "single": Params(
        width=0.92,
        length=1.92,
        height=0.50,
        mattress_thickness=0.20,
        headboard_height=0.44,
        num_pillows=1,
        frame_color=WOOD_LIGHT,
        mattress_color=FABRIC_CREAM,
        pillow_color=FABRIC_WHITE,
    ),
    "double": Params(
        width=1.40,
        length=2.00,
        height=0.52,
        mattress_thickness=0.22,
        headboard_height=0.52,
        frame_color=WOOD_MEDIUM,
        mattress_color=FABRIC_WHITE,
        pillow_color=FABRIC_WHITE,
    ),
    "queen": Params(
        width=1.54,
        length=2.04,
        height=0.54,
        mattress_thickness=0.24,
        headboard_height=0.58,
        headboard_thickness=0.06,
        frame_color=WOOD_DARK,
        mattress_color=FABRIC_CREAM,
        pillow_color=FABRIC_BEIGE,
    ),
    "king": Params(
        width=1.80,
        length=2.04,
        height=0.56,
        mattress_thickness=0.26,
        headboard_height=0.64,
        headboard_thickness=0.06,
        has_footboard=True,
        footboard_height=0.18,
        has_pillow=False,
        frame_color=WOOD_DARK,
        mattress_color=FABRIC_WHITE,
    ),
    "bunk bed": Params(
        width=0.92,
        length=1.92,
        height=0.90,
        mattress_thickness=0.16,
        headboard_height=0.30,
        headboard_thickness=0.05,
        has_footboard=True,
        footboard_height=0.30,
        has_pillow=True,
        num_pillows=1,
        leg_width=0.06,
        frame_color=WOOD_BIRCH,
        mattress_color=FABRIC_BLUE,
        pillow_color=FABRIC_WHITE,
    ),
    "platform": Params(
        width=1.50,
        length=2.00,
        height=0.36,
        mattress_thickness=0.16,
        headboard_height=0.32,
        headboard_thickness=0.05,
        leg_width=0.04,
        frame_color=WOOD_BIRCH,
        mattress_color=FABRIC_CREAM,
        pillow_color=FABRIC_BEIGE,
    ),
    "daybed": Params(
        width=0.92,
        length=1.92,
        height=0.46,
        mattress_thickness=0.18,
        headboard_height=0.28,
        has_footboard=True,
        footboard_height=0.28,
        has_pillow=True,
        num_pillows=1,
        frame_color=WOOD_DARK,
        mattress_color=FABRIC_BEIGE,
        pillow_color=FABRIC_PURPLE,
    ),
    "kids bed": Params(
        width=0.85,
        length=1.70,
        height=0.42,
        mattress_thickness=0.18,
        headboard_height=0.40,
        headboard_thickness=0.05,
        has_footboard=True,
        footboard_height=0.20,
        has_pillow=True,
        num_pillows=1,
        frame_color=WOOD_LIGHT,
        mattress_color=FABRIC_PINK,
        pillow_color=FABRIC_YELLOW,
    ),
}
