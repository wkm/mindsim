"""Couch — a wide seat with back, cushion segments, and chunky arms.

Sims 1 style: oversized, colorful, visible cushion segments, toylike proportions.
Variations: 2-seater, 3-seater, L-shaped feel, loveseat, club chair.

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
    num_cushions:   Number of visible seat cushion segments (0 = single slab)
    cushion_gap:    Gap between cushion segments in meters
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
    FABRIC_GREEN,
    FABRIC_NAVY,
    FABRIC_ORANGE,
    FABRIC_PINK,
    FABRIC_TEAL,
    WOOD_BIRCH,
    WOOD_DARK,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL


@dataclass(frozen=True)
class Params:
    width: float = 1.80
    depth: float = 0.82
    seat_height: float = 0.42
    seat_thickness: float = 0.14
    back_height: float = 0.46
    back_thickness: float = 0.14
    arm_height: float = 0.24
    arm_width: float = 0.12
    has_arms: bool = True
    num_cushions: int = 0
    cushion_gap: float = 0.02
    leg_width: float = 0.05
    seat_color: tuple[float, float, float, float] = FABRIC_BLUE
    leg_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a couch (up to 8 prims: seat/cushions + back + arms + legs).

    Budget management (max 8 prims):
      - seat/cushions: 1-3, back: 1, arms: 0-2, legs: 2-4
      - With segmented cushions, legs are reduced to 2 (front pair) to
        stay within budget.
    """
    hw = params.width / 2
    hd = params.depth / 2
    st = params.seat_thickness / 2
    hlw = params.leg_width / 2
    sc = params.seat_color
    lc = params.leg_color

    prims: list[Prim] = []

    # Interior width (between arms) for cushion layout
    arm_total = params.arm_width if params.has_arms else 0
    interior_w = params.width - 2 * arm_total

    # Seat: either single slab or segmented cushions
    seat_z = params.seat_height - st
    seat_count = max(1, params.num_cushions)
    if params.num_cushions >= 2:
        # Segmented cushions -- chunky individual pads
        total_gap = (params.num_cushions - 1) * params.cushion_gap
        cush_w = (interior_w - total_gap) / params.num_cushions
        cush_hw = cush_w / 2
        start_x = -interior_w / 2 + cush_hw
        for i in range(params.num_cushions):
            cx = start_x + i * (cush_w + params.cushion_gap)
            prims.append(
                Prim(GeomType.BOX, (cush_hw, hd - 0.02, st), (cx, -0.02, seat_z), sc)
            )
    else:
        # Single slab seat
        prims.append(Prim(GeomType.BOX, (hw, hd, st), (0, 0, seat_z), sc))

    # Backrest: chunky pad at rear edge (+Y)
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

    # Legs: chunky stubby legs (Sims 1 style)
    # With segmented cushions, use only 2 front legs to stay within budget
    leg_top = params.seat_height - params.seat_thickness
    leg_half = leg_top / 2
    lx = hw - hlw - 0.02
    ly = hd - hlw - 0.02

    arm_count = 2 if params.has_arms else 0
    remaining = 8 - seat_count - 1 - arm_count  # 1 for back
    num_legs = min(4, remaining)

    if num_legs >= 4:
        prims.extend(
            [
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, -ly, leg_half), lc),
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, -ly, leg_half), lc),
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, ly, leg_half), lc),
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, ly, leg_half), lc),
            ]
        )
    elif num_legs >= 2:
        # Front pair only -- back is covered by the backrest visually
        prims.extend(
            [
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, -ly, leg_half), lc),
                Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, -ly, leg_half), lc),
            ]
        )

    # Armrests: thick chunky pads
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
    "2-seater": Params(
        width=1.50,
        depth=0.82,
        seat_thickness=0.14,
        back_height=0.44,
        num_cushions=2,
        seat_color=FABRIC_BLUE,
        leg_color=WOOD_DARK,
    ),
    "3-seater": Params(
        width=2.20,
        depth=0.85,
        seat_thickness=0.14,
        back_height=0.48,
        num_cushions=3,
        seat_color=FABRIC_TEAL,
        leg_color=WOOD_MEDIUM,
    ),
    "loveseat": Params(
        width=1.30,
        depth=0.80,
        seat_thickness=0.15,
        back_height=0.42,
        arm_width=0.14,
        arm_height=0.26,
        num_cushions=2,
        seat_color=FABRIC_PINK,
        leg_color=WOOD_BIRCH,
    ),
    "club sofa": Params(
        width=0.95,
        depth=0.88,
        seat_thickness=0.16,
        back_height=0.42,
        arm_height=0.30,
        arm_width=0.16,
        seat_color=FABRIC_BROWN,
        leg_color=WOOD_DARK,
    ),
    "futon": Params(
        width=1.80,
        depth=0.88,
        seat_height=0.32,
        seat_thickness=0.16,
        back_height=0.30,
        back_thickness=0.10,
        has_arms=False,
        leg_width=0.04,
        seat_color=FABRIC_NAVY,
        leg_color=WOOD_DARK,
    ),
    "sectional": Params(
        width=2.40,
        depth=0.90,
        seat_thickness=0.14,
        back_height=0.50,
        back_thickness=0.16,
        arm_width=0.10,
        num_cushions=3,
        seat_color=FABRIC_BEIGE,
        leg_color=WOOD_MEDIUM,
    ),
    "retro sofa": Params(
        width=1.70,
        depth=0.78,
        seat_height=0.44,
        seat_thickness=0.12,
        back_height=0.38,
        arm_height=0.20,
        arm_width=0.10,
        num_cushions=2,
        seat_color=FABRIC_ORANGE,
        leg_color=WOOD_BIRCH,
    ),
    "chaise": Params(
        width=0.82,
        depth=1.50,
        seat_thickness=0.14,
        has_arms=False,
        back_height=0.36,
        back_thickness=0.12,
        seat_color=FABRIC_GREEN,
        leg_color=WOOD_DARK,
    ),
}
