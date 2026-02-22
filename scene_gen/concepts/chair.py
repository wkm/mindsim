"""Chair — a seat with four legs and optional backrest and armrests.

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
    seat_color:     RGBA for seat, back, and arms
    leg_color:      RGBA for legs
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    FABRIC_BLUE,
    FABRIC_GRAY,
    FABRIC_RED,
    WOOD_DARK,
    WOOD_MEDIUM,
    GeomType,
    Prim,
)


@dataclass(frozen=True)
class Params:
    seat_width: float = 0.45
    seat_depth: float = 0.42
    seat_height: float = 0.46
    seat_thickness: float = 0.04
    back_height: float = 0.40
    back_thickness: float = 0.03
    leg_width: float = 0.03
    has_back: bool = True
    has_arms: bool = False
    arm_height: float = 0.22
    arm_width: float = 0.04
    seat_color: tuple[float, float, float, float] = FABRIC_BLUE
    leg_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a chair (up to 8 prims: seat + back + 4 legs + 2 arms)."""
    hw = params.seat_width / 2
    hd = params.seat_depth / 2
    ht = params.seat_thickness / 2
    hlw = params.leg_width / 2

    sc = params.seat_color
    lc = params.leg_color

    prims: list[Prim] = []

    # Seat
    prims.append(Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.seat_height - ht), sc))

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

    # Armrests: on left and right edges, rising from seat
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
    "dining chair": Params(),
    "bar stool": Params(
        seat_width=0.35,
        seat_depth=0.35,
        seat_height=0.75,
        leg_width=0.025,
        has_back=False,
        seat_color=WOOD_MEDIUM,
        leg_color=WOOD_DARK,
    ),
    "lounge": Params(
        seat_width=0.55,
        seat_depth=0.50,
        seat_height=0.38,
        back_height=0.35,
        seat_color=FABRIC_GRAY,
    ),
    "with arms": Params(has_arms=True),
    "arm chair": Params(
        seat_width=0.55,
        seat_depth=0.48,
        seat_height=0.44,
        back_height=0.50,
        has_arms=True,
        arm_height=0.24,
        seat_color=FABRIC_RED,
        leg_color=WOOD_DARK,
    ),
    "stool": Params(
        seat_width=0.36,
        seat_depth=0.36,
        seat_height=0.45,
        has_back=False,
        seat_color=WOOD_MEDIUM,
        leg_color=WOOD_DARK,
    ),
    "high back": Params(
        back_height=0.55,
        seat_color=FABRIC_GRAY,
        leg_color=WOOD_DARK,
    ),
    "compact": Params(
        seat_width=0.38,
        seat_depth=0.36,
        seat_height=0.44,
        back_height=0.32,
        leg_width=0.025,
    ),
}
