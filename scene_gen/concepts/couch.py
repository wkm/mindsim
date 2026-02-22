"""Couch — a wide seat with back and optional arms.

Parameters:
    width:          Overall width (X) in meters
    depth:          Seat depth (Y) in meters
    seat_height:    Seat height from floor in meters
    seat_thickness: Seat cushion thickness
    back_height:    Backrest height above seat
    back_thickness: Backrest thickness
    arm_height:     Armrest height above seat
    arm_width:      Armrest thickness (Y)
    has_arms:       Include armrests
    leg_width:      Leg cross-section width
    seat_color:     RGBA for upholstery
    leg_color:      RGBA for legs
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    FABRIC_BEIGE,
    FABRIC_BLUE,
    FABRIC_BROWN,
    FABRIC_GRAY,
    FABRIC_GREEN,
    FABRIC_RED,
    WOOD_DARK,
    WOOD_MEDIUM,
    GeomType,
    Prim,
)


@dataclass(frozen=True)
class Params:
    width: float = 1.80
    depth: float = 0.80
    seat_height: float = 0.42
    seat_thickness: float = 0.12
    back_height: float = 0.45
    back_thickness: float = 0.12
    arm_height: float = 0.22
    arm_width: float = 0.10
    has_arms: bool = True
    leg_width: float = 0.04
    seat_color: tuple[float, float, float, float] = FABRIC_GRAY
    leg_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a couch (seat + back + 2 arms + 4 legs = up to 8 prims)."""
    hw = params.width / 2
    hd = params.depth / 2
    st = params.seat_thickness / 2
    hlw = params.leg_width / 2
    sc = params.seat_color
    lc = params.leg_color

    prims: list[Prim] = []

    # Seat cushion
    seat_z = params.seat_height - st
    prims.append(Prim(GeomType.BOX, (hw, hd, st), (0, 0, seat_z), sc))

    # Backrest: at rear edge (+Y)
    bk_half_h = params.back_height / 2
    bk_z = params.seat_height + bk_half_h
    bk_y = hd - params.back_thickness / 2
    prims.append(
        Prim(
            GeomType.BOX,
            (hw, params.back_thickness / 2, bk_half_h),
            (0, bk_y, bk_z),
            sc,
        )
    )

    # Legs
    leg_top = params.seat_height - params.seat_thickness
    leg_half = leg_top / 2
    lx = hw - hlw - 0.02
    ly = hd - hlw - 0.02

    prims.extend(
        [
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, -ly, leg_half), lc),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, -ly, leg_half), lc),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, ly, leg_half), lc),
            Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, ly, leg_half), lc),
        ]
    )

    # Armrests
    if params.has_arms:
        arm_half_h = params.arm_height / 2
        arm_z = params.seat_height + arm_half_h
        arm_x = hw - params.arm_width / 2
        arm_hw = params.arm_width / 2
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
    "sofa": Params(),
    "loveseat": Params(width=1.30, seat_color=FABRIC_BLUE),
    "sectional": Params(
        width=2.40,
        depth=0.85,
        back_height=0.50,
        seat_color=FABRIC_BEIGE,
        leg_color=WOOD_MEDIUM,
    ),
    "chaise": Params(
        width=0.80,
        depth=1.50,
        has_arms=False,
        back_height=0.35,
        seat_color=FABRIC_GREEN,
    ),
    "futon": Params(
        width=1.80,
        depth=0.85,
        seat_height=0.30,
        seat_thickness=0.15,
        back_height=0.30,
        has_arms=False,
        leg_width=0.03,
        seat_color=FABRIC_BROWN,
        leg_color=WOOD_DARK,
    ),
    "club": Params(
        width=0.90,
        depth=0.85,
        back_height=0.40,
        arm_height=0.28,
        arm_width=0.14,
        seat_color=FABRIC_RED,
        leg_color=WOOD_DARK,
    ),
}
