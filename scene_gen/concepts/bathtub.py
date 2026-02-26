"""Bathtub -- outer shell box + inner basin box.

A bathroom tub placed against a wall. Two boxes create the illusion of
a hollow basin: the outer shell and a slightly inset, lower-colored interior.

Parameters:
    width:        Half-extent X (across the tub)
    depth:        Half-extent Y (length of the tub)
    height:       Half-extent Z (wall height)
    wall_thick:   Wall thickness (difference between outer and inner half-extents)
    shell_color:  RGBA for the outer shell
    basin_color:  RGBA for the inner basin
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    PLASTIC_WHITE,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Basin interior -- slightly blue-white to suggest water/porcelain
BASIN_INTERIOR = (0.85, 0.88, 0.92, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.40
    depth: float = 0.80
    height: float = 0.30
    wall_thick: float = 0.04
    shell_color: tuple[float, float, float, float] = PLASTIC_WHITE
    basin_color: tuple[float, float, float, float] = BASIN_INTERIOR


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a bathtub (outer shell + inner basin = 2 prims)."""
    # Outer shell: box sitting on floor
    shell = Prim(
        GeomType.BOX,
        (params.width, params.depth, params.height),
        (0, 0, params.height),
        params.shell_color,
    )

    # Inner basin: slightly smaller box, top surface sits just below shell top
    inner_w = params.width - params.wall_thick
    inner_d = params.depth - params.wall_thick
    inner_h = params.height - params.wall_thick
    # Basin bottom is raised by wall_thick, so its center z shifts up accordingly
    basin_z = params.wall_thick + inner_h
    basin = Prim(
        GeomType.BOX,
        (inner_w, inner_d, inner_h),
        (0, 0, basin_z),
        params.basin_color,
    )

    return (shell, basin)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "standard bathtub": Params(),
    "freestanding tub": Params(
        width=0.38,
        depth=0.85,
        height=0.35,
        wall_thick=0.05,
    ),
    "corner tub": Params(
        width=0.60,
        depth=0.60,
        height=0.30,
        wall_thick=0.05,
    ),
    "soaking tub": Params(
        width=0.35,
        depth=0.70,
        height=0.40,
        wall_thick=0.04,
    ),
    "clawfoot tub": Params(
        width=0.38,
        depth=0.82,
        height=0.38,
        wall_thick=0.03,
    ),
}
