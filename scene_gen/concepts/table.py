"""Table — a flat surface on legs or a pedestal.

Sims 1 style: chunky, toylike proportions with saturated wood tones.

Parameters:
    width:         Table width (X axis) in meters
    depth:         Table depth (Y axis) in meters
    height:        Table height (Z axis) in meters
    top_thickness: Thickness of the table top in meters
    leg_width:     Width of each square leg in meters (ignored if pedestal)
    pedestal:      Use a single thick cylinder instead of 4 legs
    pedestal_r:    Pedestal cylinder radius (only used if pedestal=True)
    color:         RGBA for the whole table
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

PLACEMENT = Placement.CENTER


@dataclass(frozen=True)
class Params:
    width: float = 1.0
    depth: float = 0.6
    height: float = 0.75
    top_thickness: float = 0.04
    leg_width: float = 0.05
    pedestal: bool = False
    pedestal_r: float = 0.10
    color: tuple[float, float, float, float] = WOOD_MEDIUM


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a table (1 top + 4 legs or 1 pedestal = 2-5 prims)."""
    hw = params.width / 2
    hd = params.depth / 2
    ht = params.top_thickness / 2
    c = params.color

    # Table top: centered at (0, 0, height - half_thickness)
    top = Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.height - ht), c)

    if params.pedestal:
        # Single thick cylinder pedestal from floor to underside of top
        leg_full = params.height - params.top_thickness
        leg_half = leg_full / 2
        pedestal = Prim(
            GeomType.CYLINDER,
            (params.pedestal_r, leg_half, 0),
            (0, 0, leg_half),
            c,
        )
        return (top, pedestal)

    # Four chunky square legs
    hlw = params.leg_width / 2
    leg_full = params.height - params.top_thickness
    leg_half = leg_full / 2
    leg_z = leg_half

    # Inset legs slightly from edges
    lx = hw - hlw - 0.01
    ly = hd - hlw - 0.01

    legs = (
        Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, -ly, leg_z), c),
        Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, -ly, leg_z), c),
        Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, ly, leg_z), c),
        Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, ly, leg_z), c),
    )

    return (top, *legs)


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "dining table": Params(
        width=1.10,
        depth=0.70,
        height=0.76,
        top_thickness=0.045,
        leg_width=0.055,
        color=WOOD_MEDIUM,
    ),
    "coffee table": Params(
        width=1.10,
        depth=0.60,
        height=0.42,
        top_thickness=0.05,
        leg_width=0.06,
        color=WOOD_DARK,
    ),
    "end table": Params(
        width=0.45,
        depth=0.45,
        height=0.55,
        top_thickness=0.04,
        leg_width=0.045,
        color=WOOD_LIGHT,
    ),
    "console table": Params(
        width=1.20,
        depth=0.35,
        height=0.78,
        top_thickness=0.035,
        leg_width=0.04,
        color=WOOD_BIRCH,
    ),
    "pedestal dining": Params(
        width=0.90,
        depth=0.90,
        height=0.76,
        top_thickness=0.045,
        pedestal=True,
        pedestal_r=0.12,
        color=WOOD_DARK,
    ),
    "pedestal end table": Params(
        width=0.50,
        depth=0.50,
        height=0.55,
        top_thickness=0.04,
        pedestal=True,
        pedestal_r=0.08,
        color=WOOD_LIGHT,
    ),
    "bar table": Params(
        width=0.60,
        depth=0.60,
        height=1.05,
        top_thickness=0.04,
        pedestal=True,
        pedestal_r=0.09,
        color=WOOD_DARK,
    ),
    "kitchen table": Params(
        width=0.90,
        depth=0.90,
        height=0.76,
        top_thickness=0.05,
        leg_width=0.06,
        color=WOOD_BIRCH,
    ),
}
