"""Chair — a seat with legs, backrest, and optional armrests.

Sims 1 style: chunky, slightly oversized, colorful, recognizable silhouettes.
Distinct types: dining chair, office chair, armchair, stool.

Parameters:
    seat_width:     Seat width (X) in meters
    seat_depth:     Seat depth (Y) in meters
    seat_height:    Seat height from floor in meters
    seat_thickness: Seat slab thickness in meters
    back_height:    Backrest height above seat in meters
    back_thickness: Backrest thickness in meters
    leg_width:      Leg cross-section width in meters
    has_back:       Include backrest (False = stool)
    has_arms:       Include armrests
    arm_height:     Armrest height above seat in meters
    arm_width:      Armrest cross-section width in meters
    cushion_inset:  Inset for a visible seat cushion (0 = no cushion)
    seat_color:     RGBA for seat, back, and arms
    leg_color:      RGBA for legs
    cushion_color:  RGBA for seat cushion (if inset > 0)
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    FABRIC_BLUE,
    FABRIC_GRAY,
    FABRIC_PURPLE,
    FABRIC_RED,
    FABRIC_TEAL,
    FABRIC_YELLOW,
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

PLACEMENT = Placement.CENTER


@dataclass(frozen=True)
class Params:
    seat_width: float = 0.45
    seat_depth: float = 0.42
    seat_height: float = 0.46
    seat_thickness: float = 0.05
    back_height: float = 0.42
    back_thickness: float = 0.04
    leg_width: float = 0.035
    has_back: bool = True
    has_arms: bool = False
    arm_height: float = 0.22
    arm_width: float = 0.05
    cushion_inset: float = 0.0
    seat_color: tuple[float, float, float, float] = FABRIC_BLUE
    leg_color: tuple[float, float, float, float] = WOOD_DARK
    cushion_color: tuple[float, float, float, float] = FABRIC_BLUE


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a chair (up to 8 prims: seat + cushion + back + 4 legs + 2 arms)."""
    hw = params.seat_width / 2
    hd = params.seat_depth / 2
    ht = params.seat_thickness / 2
    hlw = params.leg_width / 2

    sc = params.seat_color
    lc = params.leg_color

    prims: list[Prim] = []

    # Seat slab
    prims.append(Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.seat_height - ht), sc))

    # Visible cushion on top of seat (chunky Sims 1 look)
    # Skipped when arms are present to stay within 8-prim budget
    if params.cushion_inset > 0 and not params.has_arms:
        ci = params.cushion_inset
        cush_ht = 0.025
        cush_z = params.seat_height + cush_ht
        prims.append(
            Prim(
                GeomType.BOX,
                (hw - ci, hd - ci, cush_ht),
                (0, 0, cush_z),
                params.cushion_color,
            )
        )

    # Backrest: at rear edge of seat (+Y), rising above seat
    if params.has_back:
        back_half_h = params.back_height / 2
        back_z = params.seat_height + back_half_h
        back_y = hd - params.back_thickness / 2
        prims.append(
            Prim(
                GeomType.BOX,
                (hw, params.back_thickness / 2, back_half_h),
                (0, back_y, back_z),
                sc,
            )
        )

    # Legs: floor to underside of seat
    leg_full = params.seat_height - params.seat_thickness
    leg_half = leg_full / 2
    leg_z = leg_half
    lx = hw - hlw - 0.01
    ly = hd - hlw - 0.01

    prims.extend(
        [
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, -ly, leg_z), lc),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, -ly, leg_z), lc),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, ly, leg_z), lc),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, ly, leg_z), lc),
        ]
    )

    # Armrests: chunky pads on left and right edges
    if params.has_arms:
        arm_hw = params.arm_width / 2
        arm_half_h = params.arm_height / 2
        arm_z = params.seat_height + arm_half_h
        arm_x = hw - arm_hw
        prims.extend(
            [
                Prim(GeomType.BOX, (arm_hw, hd, arm_half_h), (-arm_x, 0, arm_z), sc),
                Prim(GeomType.BOX, (arm_hw, hd, arm_half_h), (arm_x, 0, arm_z), sc),
            ]
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "dining chair": Params(
        seat_width=0.46,
        seat_depth=0.44,
        seat_height=0.47,
        seat_thickness=0.05,
        back_height=0.44,
        back_thickness=0.04,
        leg_width=0.035,
        seat_color=WOOD_MEDIUM,
        leg_color=WOOD_DARK,
    ),
    "office chair": Params(
        seat_width=0.50,
        seat_depth=0.48,
        seat_height=0.50,
        seat_thickness=0.06,
        back_height=0.48,
        back_thickness=0.05,
        leg_width=0.03,
        cushion_inset=0.03,
        seat_color=PLASTIC_BLACK,
        leg_color=METAL_CHROME,
        cushion_color=FABRIC_GRAY,
    ),
    "armchair": Params(
        seat_width=0.58,
        seat_depth=0.52,
        seat_height=0.44,
        seat_thickness=0.07,
        back_height=0.50,
        back_thickness=0.06,
        has_arms=True,
        arm_height=0.24,
        arm_width=0.07,
        cushion_inset=0.04,
        seat_color=FABRIC_RED,
        leg_color=WOOD_DARK,
        cushion_color=FABRIC_RED,
    ),
    "stool": Params(
        seat_width=0.38,
        seat_depth=0.38,
        seat_height=0.46,
        seat_thickness=0.05,
        has_back=False,
        seat_color=WOOD_LIGHT,
        leg_color=WOOD_MEDIUM,
    ),
    "bar stool": Params(
        seat_width=0.36,
        seat_depth=0.36,
        seat_height=0.76,
        seat_thickness=0.05,
        leg_width=0.028,
        has_back=False,
        seat_color=WOOD_BIRCH,
        leg_color=METAL_GRAY,
    ),
    "high back": Params(
        seat_width=0.48,
        seat_depth=0.44,
        seat_height=0.46,
        seat_thickness=0.05,
        back_height=0.58,
        back_thickness=0.04,
        seat_color=FABRIC_PURPLE,
        leg_color=WOOD_DARK,
    ),
    "lounge chair": Params(
        seat_width=0.56,
        seat_depth=0.52,
        seat_height=0.38,
        seat_thickness=0.08,
        back_height=0.36,
        back_thickness=0.06,
        cushion_inset=0.04,
        seat_color=FABRIC_TEAL,
        leg_color=WOOD_MEDIUM,
        cushion_color=FABRIC_TEAL,
    ),
    "kids chair": Params(
        seat_width=0.34,
        seat_depth=0.32,
        seat_height=0.30,
        seat_thickness=0.04,
        back_height=0.28,
        back_thickness=0.04,
        leg_width=0.03,
        seat_color=FABRIC_YELLOW,
        leg_color=WOOD_BIRCH,
    ),
}
