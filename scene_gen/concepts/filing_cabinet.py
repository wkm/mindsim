"""Filing cabinet — a narrow tall box with drawer fronts on the face.

A metal or plastic cabinet body with thin drawer front boxes inset on
the front face. Common office furniture for navigation training.

Parameters:
    width:          Overall width (X) in meters (half-extent internally)
    depth:          Overall depth (Y) in meters
    height:         Overall height (Z) in meters
    n_drawers:      Number of drawer fronts
    drawer_gap:     Vertical gap between drawers in meters
    body_color:     RGBA for the cabinet body
    drawer_color:   RGBA for the drawer fronts
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_DARK,
    METAL_GRAY,
    PLASTIC_BLACK,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL


@dataclass(frozen=True)
class Params:
    width: float = 0.22
    depth: float = 0.25
    height: float = 0.65
    n_drawers: int = 2
    drawer_gap: float = 0.01
    body_color: tuple[float, float, float, float] = METAL_GRAY
    drawer_color: tuple[float, float, float, float] = METAL_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a filing cabinet (body + N drawer fronts)."""
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

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "2-drawer": Params(),
    "3-drawer": Params(
        n_drawers=3,
        height=0.75,
    ),
    "4-drawer": Params(
        n_drawers=4,
        height=1.0,
    ),
    "lateral": Params(
        width=0.40,
        depth=0.22,
        height=0.50,
        n_drawers=2,
        body_color=METAL_DARK,
        drawer_color=METAL_GRAY,
    ),
    "compact": Params(
        width=0.18,
        depth=0.20,
        height=0.45,
        n_drawers=2,
        body_color=PLASTIC_BLACK,
        drawer_color=PLASTIC_BLACK,
    ),
}
