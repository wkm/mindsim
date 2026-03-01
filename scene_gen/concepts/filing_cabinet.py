"""Filing cabinet — a narrow tall box with drawer fronts and handles.

Metal or plastic cabinet body with thin drawer front boxes inset on the
front face. Sims 1 style: chunky proportions, visible drawer pulls
(small protruding boxes), saturated body colors.

Parameters:
    width:          Overall width (X) in meters (half-extent internally)
    depth:          Overall depth (Y) in meters
    height:         Overall height (Z) in meters
    n_drawers:      Number of drawer fronts
    drawer_gap:     Vertical gap between drawers in meters
    body_color:     RGBA for the cabinet body
    drawer_color:   RGBA for the drawer fronts
    handle_color:   RGBA for the drawer pull handles
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    METAL_DARK,
    METAL_GRAY,
    PLASTIC_BLACK,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Cabinet body colors — Sims 1 office palette
CABINET_BEIGE = (0.78, 0.72, 0.60, 1.0)
CABINET_BLUE = (0.28, 0.40, 0.58, 1.0)
CABINET_GREEN = (0.30, 0.48, 0.35, 1.0)
CABINET_RED = (0.62, 0.22, 0.18, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.22
    depth: float = 0.25
    height: float = 0.65
    n_drawers: int = 2
    drawer_gap: float = 0.01
    body_color: tuple[float, float, float, float] = METAL_GRAY
    drawer_color: tuple[float, float, float, float] = METAL_DARK
    handle_color: tuple[float, float, float, float] = METAL_CHROME


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a filing cabinet (body + N drawer fronts + N handles).

    Total prims: 1 body + N drawers + min(N, remaining) handles.
    Stays within 8 prim limit (max 3 drawers get handles with 4-drawer).
    """
    hw = params.width / 2
    hd = params.depth / 2
    hh = params.height / 2

    prims: list[Prim] = []

    # Main body
    prims.append(Prim(GeomType.BOX, (hw, hd, hh), (0, 0, hh), params.body_color))

    # Drawer fronts — thin boxes on the front face, evenly spaced
    drawer_inset = 0.005  # how far drawers are inset from edges
    drawer_hw = hw - drawer_inset
    drawer_thickness = 0.012  # half-extent Y for drawer front
    # Place drawer front just in front of the body face
    drawer_y = -(hd + drawer_thickness)

    total_gap = params.drawer_gap * (params.n_drawers + 1)
    drawer_height = (params.height - total_gap) / params.n_drawers
    drawer_hh = drawer_height / 2

    # Handle dimensions — chunky Sims 1 style
    handle_hw = hw * 0.25  # wide, chunky handle
    handle_hh = 0.008
    handle_depth = 0.008  # how far handle protrudes
    handle_y = drawer_y - drawer_thickness - handle_depth

    # Budget: 1 body + n_drawers + n_handles <= 8
    max_handles = min(params.n_drawers, 8 - 1 - params.n_drawers)

    for i in range(params.n_drawers):
        z_bottom = params.drawer_gap * (i + 1) + drawer_height * i
        z_center = z_bottom + drawer_hh
        prims.append(
            Prim(
                GeomType.BOX,
                (drawer_hw, drawer_thickness, drawer_hh),
                (0, drawer_y, z_center),
                params.drawer_color,
            )
        )

        # Drawer handle — small protruding box centered on drawer face
        if i < max_handles:
            prims.append(
                Prim(
                    GeomType.BOX,
                    (handle_hw, handle_depth, handle_hh),
                    (0, handle_y, z_center),
                    params.handle_color,
                )
            )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations — Sims 1 style: chunky, colorful, with handles
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "2-drawer gray": Params(),
    "2-drawer black": Params(
        body_color=PLASTIC_BLACK,
        drawer_color=PLASTIC_BLACK,
        handle_color=METAL_CHROME,
    ),
    "3-drawer": Params(
        n_drawers=3,
        height=0.75,
    ),
    "4-drawer": Params(
        n_drawers=4,
        height=1.0,
        body_color=METAL_GRAY,
        drawer_color=METAL_DARK,
    ),
    "lateral wide": Params(
        width=0.40,
        depth=0.22,
        height=0.50,
        n_drawers=2,
        body_color=METAL_DARK,
        drawer_color=METAL_GRAY,
        handle_color=METAL_CHROME,
    ),
    "beige office": Params(
        n_drawers=3,
        height=0.75,
        body_color=CABINET_BEIGE,
        drawer_color=CABINET_BEIGE,
        handle_color=METAL_DARK,
    ),
    "blue cabinet": Params(
        n_drawers=2,
        height=0.65,
        body_color=CABINET_BLUE,
        drawer_color=CABINET_BLUE,
        handle_color=METAL_CHROME,
    ),
    "compact black": Params(
        width=0.18,
        depth=0.20,
        height=0.45,
        n_drawers=2,
        body_color=PLASTIC_BLACK,
        drawer_color=PLASTIC_BLACK,
        handle_color=METAL_DARK,
    ),
}
