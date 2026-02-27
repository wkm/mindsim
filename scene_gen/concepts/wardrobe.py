"""Wardrobe — a tall, deep cabinet (armoire).

Sims 1 style: chunky, imposing silhouettes with visible door lines,
handles, and trim details.  A serious obstacle for a small robot.

Variations: single door, double door, armoire with crown molding,
modern sliding door, colorful kids wardrobe, rustic pine.

Parameters:
    width:           Total width (X) in meters
    depth:           Total depth (Y) in meters
    height:          Total height (Z) in meters
    panel_thickness: Side/top panel thickness
    n_doors:         Number of door panels (1 or 2)
    door_gap:        Gap between doors / door-to-side gap
    has_crown:       Include decorative crown molding (chunky box on top)
    has_base:        Include base trim (chunky box at bottom)
    body_color:      RGBA for body panels
    door_color:      RGBA for door fronts
    accent_color:    RGBA for crown molding / base trim
    handle_color:    RGBA for door handles
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    METAL_DARK,
    METAL_GRAY,
    WOOD_BIRCH,
    WOOD_DARK,
    WOOD_LIGHT,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Sims 1 accent colors
SIMS_NAVY = (0.15, 0.20, 0.40, 1.0)
SIMS_CREAM = (0.95, 0.90, 0.80, 1.0)
SIMS_SKY = (0.50, 0.70, 0.85, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 1.00
    depth: float = 0.58
    height: float = 1.90
    panel_thickness: float = 0.025
    n_doors: int = 2
    door_gap: float = 0.01
    has_crown: bool = True
    has_base: bool = False
    body_color: tuple[float, float, float, float] = WOOD_DARK
    door_color: tuple[float, float, float, float] = WOOD_MEDIUM
    accent_color: tuple[float, float, float, float] = WOOD_DARK
    handle_color: tuple[float, float, float, float] = METAL_CHROME


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a wardrobe (up to 8 prims).

    Budget breakdown (worst case 8):
        top + 2 sides + back = 4 (always)
        + 1-2 doors           = 5-6
        + 1-2 handles          = 6-8
        + opt crown or base    = 7-8

    Handles and trim are added when budget allows, prioritizing handles
    first (they do the most for recognizability).
    """
    hw = params.width / 2
    hd = params.depth / 2
    t = params.panel_thickness
    ht = t / 2
    bc = params.body_color
    dc = params.door_color

    prims: list[Prim] = []

    # -- Structural panels (4 prims always) --

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

    # -- Door fronts (1-2 prims) --
    door_front_t = 0.018
    door_hh = side_h - 0.01  # slightly shorter than body
    door_z = side_h
    # Y position: flush against front face of body
    door_y = -hd + door_front_t / 2

    # Track door center X positions for handle placement
    door_centers_x: list[float] = []

    if params.n_doors == 1:
        door_hw = (params.width - 2 * t - 2 * params.door_gap) / 2
        prims.append(
            Prim(
                GeomType.BOX,
                (door_hw, door_front_t / 2, door_hh),
                (0, door_y, door_z),
                dc,
            )
        )
        door_centers_x.append(0.0)
    else:
        usable_w = params.width - 2 * t - params.door_gap
        door_hw = usable_w / 4

        left_door_x = -params.door_gap / 2 - door_hw
        prims.append(
            Prim(
                GeomType.BOX,
                (door_hw, door_front_t / 2, door_hh),
                (left_door_x, door_y, door_z),
                dc,
            )
        )
        door_centers_x.append(left_door_x)

        right_door_x = params.door_gap / 2 + door_hw
        prims.append(
            Prim(
                GeomType.BOX,
                (door_hw, door_front_t / 2, door_hh),
                (right_door_x, door_y, door_z),
                dc,
            )
        )
        door_centers_x.append(right_door_x)

    # -- Door handles (chunky Sims 1 knobs, budget permitting) --
    # Small protruding boxes on the front face, near the door gap edge.
    # Placed at ~60% of door height for a natural look.
    handle_hw = 0.012  # chunky little knob
    handle_hh = 0.018
    handle_depth = 0.010  # how far it protrudes
    handle_y = door_y - door_front_t / 2 - handle_depth
    handle_z = params.height * 0.55  # slightly above midpoint

    remaining = 8 - len(prims)
    # Reserve 1 slot for crown or base if requested
    trim_wanted = params.has_crown or params.has_base
    handle_budget = remaining - (1 if trim_wanted else 0)

    for i, cx in enumerate(door_centers_x):
        if i >= handle_budget:
            break
        # Place handle near the gap edge of each door
        if params.n_doors == 1:
            hx = 0.0  # centered on single door
        elif cx < 0:
            hx = cx + door_hw - 0.03  # right edge of left door
        else:
            hx = cx - door_hw + 0.03  # left edge of right door
        prims.append(
            Prim(
                GeomType.BOX,
                (handle_hw, handle_depth, handle_hh),
                (hx, handle_y, handle_z),
                params.handle_color,
            )
        )

    # -- Crown molding — chunky decorative box on top --
    if params.has_crown and len(prims) < 8:
        crown_overhang = 0.02
        crown_h = 0.035
        prims.append(
            Prim(
                GeomType.BOX,
                (hw + crown_overhang, hd + crown_overhang, crown_h),
                (0, 0, params.height + crown_h),
                params.accent_color,
            )
        )

    # -- Base trim — slightly wider plinth at the bottom --
    if params.has_base and len(prims) < 8:
        base_overhang = 0.015
        base_h = 0.03
        prims.append(
            Prim(
                GeomType.BOX,
                (hw + base_overhang, hd + base_overhang, base_h),
                (0, 0, base_h),
                params.accent_color,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations (Sims 1 inspired — chunky, imposing, colorful)
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "wardrobe": Params(),
    "double wardrobe": Params(
        width=1.50,
        height=2.00,
        n_doors=2,
        body_color=WOOD_DARK,
        door_color=WOOD_MEDIUM,
        handle_color=METAL_CHROME,
    ),
    "single door": Params(
        width=0.65,
        height=1.80,
        n_doors=1,
        has_crown=False,
        has_base=True,
        body_color=WOOD_MEDIUM,
        door_color=WOOD_LIGHT,
        accent_color=WOOD_DARK,
        handle_color=METAL_DARK,
    ),
    "armoire": Params(
        width=1.10,
        height=2.00,
        depth=0.62,
        n_doors=2,
        has_crown=True,
        body_color=WOOD_DARK,
        door_color=WOOD_DARK,
        accent_color=WOOD_MEDIUM,
        handle_color=METAL_CHROME,
    ),
    "modern sliding": Params(
        width=1.40,
        height=1.90,
        depth=0.55,
        n_doors=2,
        door_gap=0.005,
        has_crown=False,
        has_base=True,
        body_color=METAL_DARK,
        door_color=SIMS_CREAM,
        accent_color=METAL_DARK,
        handle_color=METAL_GRAY,
    ),
    "kids sky blue": Params(
        width=0.90,
        height=1.60,
        n_doors=2,
        has_crown=True,
        body_color=SIMS_SKY,
        door_color=(0.85, 0.92, 0.98, 1.0),
        accent_color=SIMS_SKY,
        handle_color=SIMS_CREAM,
    ),
    "rustic pine": Params(
        width=1.00,
        height=1.85,
        depth=0.55,
        n_doors=2,
        has_crown=True,
        panel_thickness=0.03,
        body_color=WOOD_BIRCH,
        door_color=WOOD_LIGHT,
        accent_color=WOOD_MEDIUM,
        handle_color=METAL_DARK,
    ),
    "tall narrow": Params(
        width=0.60,
        height=2.00,
        depth=0.55,
        n_doors=1,
        has_crown=False,
        has_base=True,
        body_color=SIMS_NAVY,
        door_color=WOOD_LIGHT,
        accent_color=SIMS_NAVY,
        handle_color=METAL_CHROME,
    ),
}
