"""Bench — a long, low seat for one or more people.

Sims 1 style: chunky proportions, saturated wood tones, toylike design.
Variations include park benches (wood + metal legs), modern concrete
benches (one thick block), and locker room benches (slat top).

Parameters:
    width:          Total width (X) in meters
    depth:          Total depth (Y) in meters
    height:         Seat height (Z) in meters
    has_back:       Include a backrest panel
    back_height:    Height of the backrest from the seat
    leg_style:      "square" (4 thin legs) or "block" (2 thick slabs)
    seat_thickness: Thickness of the seat board
    has_apron:      Add a front apron strip under the seat
    has_stretcher:  Add a lower support bar between legs
    color:          RGBA for the seat and back
    leg_color:      RGBA for the legs
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_DARK,
    METAL_GRAY,
    WOOD_BIRCH,
    WOOD_DARK,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CENTER


@dataclass(frozen=True)
class Params:
    width: float = 1.20
    depth: float = 0.40
    height: float = 0.45
    has_back: bool = False
    back_height: float = 0.40
    leg_style: str = "square"
    seat_thickness: float = 0.04
    has_apron: bool = True
    has_stretcher: bool = True
    color: tuple[float, float, float, float] = WOOD_MEDIUM
    leg_color: tuple[float, float, float, float] = METAL_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a bench (seat + legs + optional backrest)."""
    hw = params.width / 2
    hd = params.depth / 2
    hst = params.seat_thickness / 2

    sc = params.color
    lc = params.leg_color

    prims: list[Prim] = []

    # 1. Seat: centered at (0, 0, height - half_thickness)
    seat = Prim(
        GeomType.BOX,
        (hw, hd, hst),
        (0, 0, params.height - hst),
        sc,
    )
    prims.append(seat)

    # 2. Backrest: two chunky slats at the rear (Sims 1 look)
    if params.has_back:
        slat_thick = max(0.03, params.seat_thickness * 0.75)
        slat_half_h = params.back_height * 0.22
        back_z0 = params.height + slat_half_h
        back_z1 = params.height + params.back_height - slat_half_h
        back_y = hd - slat_thick / 2
        for z in (back_z0, back_z1):
            prims.append(
                Prim(
                    GeomType.BOX,
                    (hw, slat_thick / 2, slat_half_h),
                    (0, back_y, z),
                    sc,
                    euler=(0.10, 0, 0),  # slight backward lean
                )
            )

    # 3. Apron strip under the front edge
    if params.has_apron:
        apron_h = max(0.03, params.seat_thickness * 0.7)
        apron_z = params.height - params.seat_thickness - apron_h / 2
        apron_y = -hd + apron_h / 2
        prims.append(
            Prim(
                GeomType.BOX,
                (hw * 0.95, apron_h / 2, apron_h / 2),
                (0, apron_y, apron_z),
                sc,
            )
        )

    # 4. Legs
    if params.leg_style == "block":
        # Two thick slabs at the ends
        lw = params.width * 0.08
        lh = params.height - params.seat_thickness
        lx = hw - lw / 2 - 0.05
        prims.append(Prim(GeomType.BOX, (lw / 2, hd, lh / 2), (-lx, 0, lh / 2), lc))
        prims.append(Prim(GeomType.BOX, (lw / 2, hd, lh / 2), (lx, 0, lh / 2), lc))
    else:
        # Four square legs
        lw = 0.04
        lh = params.height - params.seat_thickness
        lx = hw - lw / 2 - 0.05
        ly = hd - lw / 2 - 0.05
        prims.append(
            Prim(GeomType.BOX, (lw / 2, lw / 2, lh / 2), (-lx, -ly, lh / 2), lc)
        )
        prims.append(
            Prim(GeomType.BOX, (lw / 2, lw / 2, lh / 2), (lx, -ly, lh / 2), lc)
        )
        prims.append(
            Prim(GeomType.BOX, (lw / 2, lw / 2, lh / 2), (-lx, ly, lh / 2), lc)
        )
        prims.append(Prim(GeomType.BOX, (lw / 2, lw / 2, lh / 2), (lx, ly, lh / 2), lc))

    # 5. Lower stretcher bar between legs
    # Skip when backrest slats are present to stay within 8 prims.
    if params.has_stretcher and not params.has_back:
        stretcher_h = max(0.03, params.seat_thickness * 0.6)
        stretcher_z = stretcher_h / 2 + 0.08
        prims.append(
            Prim(
                GeomType.BOX,
                (hw * 0.85, stretcher_h / 2, stretcher_h / 2),
                (0, 0, stretcher_z),
                lc,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "park bench": Params(
        width=1.50,
        depth=0.50,
        height=0.45,
        has_back=True,
        back_height=0.45,
        leg_style="square",
        color=WOOD_DARK,
        leg_color=METAL_DARK,
    ),
    "modern concrete": Params(
        width=1.20,
        depth=0.45,
        height=0.42,
        leg_style="block",
        seat_thickness=0.15,
        color=METAL_GRAY,
        leg_color=METAL_GRAY,
    ),
    "locker room": Params(
        width=1.00,
        depth=0.30,
        height=0.45,
        leg_style="square",
        color=WOOD_BIRCH,
        leg_color=METAL_GRAY,
    ),
    "waiting bench": Params(
        width=1.80,
        depth=0.50,
        height=0.45,
        has_back=True,
        back_height=0.50,
        leg_style="block",
        color=WOOD_MEDIUM,
        leg_color=METAL_DARK,
    ),
}
