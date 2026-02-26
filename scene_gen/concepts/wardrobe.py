"""Wardrobe — a tall, deep cabinet (armoire).

A serious obstacle for a small robot: tall, wide, and deep.
Placed against walls. Body + 2 doors + optional top trim.

Parameters:
    width:           Total width (X) in meters
    depth:           Total depth (Y) in meters
    height:          Total height (Z) in meters
    panel_thickness: Side/top panel thickness
    door_gap:        Gap between the two doors
    has_top_trim:    Include a decorative top overhang
    body_color:      RGBA for body panels
    door_color:      RGBA for door fronts
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
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
    width: float = 1.00
    depth: float = 0.58
    height: float = 1.90
    panel_thickness: float = 0.025
    door_gap: float = 0.01
    has_top_trim: bool = True
    body_color: tuple[float, float, float, float] = WOOD_DARK
    door_color: tuple[float, float, float, float] = WOOD_MEDIUM


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a wardrobe (top + 2 sides + back + 2 doors + trim = 6-7 prims)."""
    hw = params.width / 2
    hd = params.depth / 2
    t = params.panel_thickness
    ht = t / 2
    bc = params.body_color
    dc = params.door_color

    prims: list[Prim] = []

    # Top panel
    prims.append(Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.height - ht), bc))

    # Left side panel
    side_h = (params.height - t) / 2
    prims.append(Prim(GeomType.BOX, (ht, hd, side_h), (-hw + ht, 0, side_h), bc))

    # Right side panel
    prims.append(Prim(GeomType.BOX, (ht, hd, side_h), (hw - ht, 0, side_h), bc))

    # Back panel
    back_t = 0.012
    prims.append(
        Prim(
            GeomType.BOX,
            (hw - t, back_t / 2, side_h),
            (0, hd - back_t / 2, side_h),
            bc,
        )
    )

    # Two door fronts
    door_front_t = 0.018
    usable_w = params.width - 2 * t - params.door_gap
    door_hw = usable_w / 4  # half-width of each door
    door_hh = side_h - 0.01  # slightly shorter than body
    door_z = side_h

    # Left door
    left_door_x = -params.door_gap / 2 - door_hw
    prims.append(
        Prim(
            GeomType.BOX,
            (door_hw, door_front_t / 2, door_hh),
            (left_door_x, -hd + door_front_t / 2, door_z),
            dc,
        )
    )
    # Right door
    right_door_x = params.door_gap / 2 + door_hw
    prims.append(
        Prim(
            GeomType.BOX,
            (door_hw, door_front_t / 2, door_hh),
            (right_door_x, -hd + door_front_t / 2, door_z),
            dc,
        )
    )

    # Top trim (decorative overhang)
    if params.has_top_trim:
        trim_overhang = 0.015
        prims.append(
            Prim(
                GeomType.BOX,
                (hw + trim_overhang, hd + trim_overhang, 0.02),
                (0, 0, params.height + 0.02),
                bc,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "wardrobe": Params(),
    "double wardrobe": Params(width=1.50, height=2.00),
    "single": Params(width=0.65, height=1.80),
    "armoire": Params(
        width=1.10,
        height=2.00,
        depth=0.62,
        body_color=WOOD_DARK,
        door_color=WOOD_DARK,
    ),
    "compact": Params(
        width=0.80,
        height=1.70,
        depth=0.50,
        has_top_trim=False,
        body_color=WOOD_BIRCH,
        door_color=WOOD_BIRCH,
    ),
    "tall narrow": Params(
        width=0.60,
        height=2.00,
        depth=0.55,
        body_color=WOOD_LIGHT,
        door_color=WOOD_MEDIUM,
    ),
}
