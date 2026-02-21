"""Table — a flat surface on four legs.

Parameters:
    width:         Table width (X axis) in meters
    depth:         Table depth (Y axis) in meters
    height:        Table height (Z axis) in meters
    top_thickness: Thickness of the table top in meters
    leg_width:     Width of each square leg in meters
    color:         RGBA for the whole table
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import WOOD_MEDIUM, GeomType, Prim


@dataclass(frozen=True)
class Params:
    width: float = 1.0
    depth: float = 0.6
    height: float = 0.75
    top_thickness: float = 0.03
    leg_width: float = 0.04
    color: tuple[float, float, float, float] = WOOD_MEDIUM


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a table from box primitives (1 top + 4 legs = 5 prims)."""
    hw = params.width / 2  # half-widths for MuJoCo box sizes
    hd = params.depth / 2
    ht = params.top_thickness / 2
    hlw = params.leg_width / 2
    c = params.color

    # Table top: centered at (0, 0, height - half_thickness)
    top = Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.height - ht), c)

    # Legs: run from floor to underside of top
    leg_full = params.height - params.top_thickness
    leg_half = leg_full / 2
    leg_z = leg_half  # center of leg

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
