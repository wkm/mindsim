"""Dresser — a wide, low cabinet with visible drawer fronts.

Sims 1 style: chunky proportions, saturated colors, visible drawer lines
made from thin contrasting boxes.

Variations: low wide dresser, tall chest of drawers, vanity (with mirror),
mid-century modern, colorful nursery, industrial.

Parameters:
    width:           Total width (X) in meters
    depth:           Total depth (Y) in meters
    height:          Total height (Z) in meters
    n_drawers:       Number of drawer rows (visual only)
    drawer_gap:      Gap between drawers in meters
    panel_thickness: Side panel / top thickness in meters
    has_mirror:      Include a vanity mirror on top (thin tall box)
    body_color:      RGBA for body (sides, top, back)
    drawer_color:    RGBA for drawer fronts
    accent_color:    RGBA for drawer line accents / mirror frame
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
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

PLACEMENT = Placement.WALL

# Sims 1 inspired colors
SIMS_PINK = (0.85, 0.45, 0.55, 1.0)
SIMS_TEAL = (0.30, 0.65, 0.60, 1.0)
SIMS_YELLOW = (0.90, 0.80, 0.35, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 1.20
    depth: float = 0.50
    height: float = 0.80
    n_drawers: int = 3
    drawer_gap: float = 0.015
    panel_thickness: float = 0.025
    has_mirror: bool = False
    body_color: tuple[float, float, float, float] = WOOD_MEDIUM
    drawer_color: tuple[float, float, float, float] = WOOD_LIGHT
    accent_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a dresser (solid body + drawer fronts + handles + opt mirror).

    Uses a single solid body box for efficiency, with contrasting drawer
    fronts on the front face and small protruding handle boxes for detail.

    Prim budget (max 8):
    - Without mirror: 1 body + N drawers + N handles  (N <= 3 → 7 prims)
    - With mirror:    1 body + 1 mirror + N drawers + handles for remaining
    """
    hw = params.width / 2
    hd = params.depth / 2
    hh = params.height / 2
    bc = params.body_color
    dc = params.drawer_color
    ac = params.accent_color

    prims: list[Prim] = []

    # Solid body box — the entire dresser carcass
    prims.append(Prim(GeomType.BOX, (hw, hd, hh), (0, 0, hh), bc))

    # Optional vanity mirror on top
    if params.has_mirror:
        mirror_w = params.width * 0.5
        mirror_h = params.height * 0.5
        mirror_t = 0.02
        prims.append(
            Prim(
                GeomType.BOX,
                (mirror_w / 2, mirror_t / 2, mirror_h / 2),
                (0, hd * 0.3, params.height + mirror_h / 2),
                ac,
            )
        )

    # Drawer fronts — contrasting-color panels on the front face
    remaining = 8 - len(prims)
    # Each drawer needs a front + a handle = 2 prims per drawer
    n = min(params.n_drawers, remaining // 2)

    drawer_inset = params.panel_thickness  # inset from edges
    drawer_hw = hw - drawer_inset
    drawer_front_t = 0.012  # half-extent Y for drawer front

    # Place drawer fronts just proud of the body face
    drawer_y = -(hd + drawer_front_t)

    total_gap = params.drawer_gap * (n + 1)
    drawer_height = (params.height - total_gap) / n if n > 0 else 0
    drawer_hh = drawer_height / 2

    # Handle dimensions — chunky Sims 1 style knobs
    handle_hw = hw * 0.18  # wide, chunky handle bar
    handle_hh = 0.008
    handle_depth = 0.008  # how far handle protrudes
    handle_y = drawer_y - drawer_front_t - handle_depth

    for i in range(n):
        z_bottom = params.drawer_gap * (i + 1) + drawer_height * i
        z_center = z_bottom + drawer_hh

        # Drawer front panel
        prims.append(
            Prim(
                GeomType.BOX,
                (drawer_hw, drawer_front_t, drawer_hh),
                (0, drawer_y, z_center),
                dc,
            )
        )

        # Drawer handle — small protruding box centered on drawer face
        prims.append(
            Prim(
                GeomType.BOX,
                (handle_hw, handle_depth, handle_hh),
                (0, handle_y, z_center),
                ac,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations (Sims 1 inspired — chunky, colorful, recognizable)
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "dresser": Params(),
    "wide dresser": Params(
        width=1.50,
        height=0.75,
        n_drawers=3,
        body_color=WOOD_BIRCH,
        drawer_color=WOOD_LIGHT,
        accent_color=WOOD_MEDIUM,
    ),
    "tall chest": Params(
        width=0.80,
        height=1.20,
        n_drawers=4,
        body_color=WOOD_DARK,
        drawer_color=WOOD_MEDIUM,
        accent_color=WOOD_LIGHT,
    ),
    "vanity": Params(
        width=1.10,
        height=0.75,
        n_drawers=3,
        has_mirror=True,
        body_color=WOOD_LIGHT,
        drawer_color=WOOD_BIRCH,
        accent_color=METAL_CHROME,
    ),
    "mid-century": Params(
        width=1.30,
        height=0.70,
        depth=0.45,
        n_drawers=2,
        drawer_gap=0.02,
        panel_thickness=0.03,
        body_color=SIMS_TEAL,
        drawer_color=WOOD_BIRCH,
        accent_color=METAL_CHROME,
    ),
    "nursery pink": Params(
        width=1.00,
        height=0.80,
        n_drawers=3,
        body_color=SIMS_PINK,
        drawer_color=(0.95, 0.85, 0.88, 1.0),
        accent_color=WOOD_LIGHT,
    ),
    "industrial": Params(
        width=1.20,
        height=0.85,
        n_drawers=3,
        body_color=METAL_GRAY,
        drawer_color=WOOD_MEDIUM,
        accent_color=PLASTIC_BLACK,
    ),
    "low credenza": Params(
        width=1.60,
        height=0.55,
        depth=0.45,
        n_drawers=2,
        body_color=WOOD_DARK,
        drawer_color=WOOD_DARK,
        accent_color=METAL_GRAY,
    ),
}
