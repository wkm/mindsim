"""Trash can — a simple cylindrical bin, optionally with a lid.

Provides round obstacles of varying sizes for navigation training.

Parameters:
    radius:     Radius of the can body
    height:     Total height of the can body
    has_lid:    Whether to add a thin lid on top
    body_color: RGBA for the can body
    lid_color:  RGBA for the lid (if present)
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_GRAY,
    PLASTIC_BLACK,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CENTER

# Custom colors for recycling / kitchen
RECYCLING_BLUE = (0.20, 0.35, 0.60, 1.0)
KITCHEN_SILVER = (0.70, 0.70, 0.72, 1.0)


@dataclass(frozen=True)
class Params:
    radius: float = 0.15
    height: float = 0.35
    has_lid: bool = True
    body_color: tuple[float, float, float, float] = METAL_GRAY
    lid_color: tuple[float, float, float, float] = METAL_GRAY


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a trash can (cylinder body + optional lid)."""
    half_h = params.height / 2

    prims: list[Prim] = []

    # Body cylinder
    prims.append(
        Prim(
            GeomType.CYLINDER,
            (params.radius, half_h, 0),
            (0, 0, half_h),
            params.body_color,
        )
    )

    # Optional lid — thin cylinder on top
    if params.has_lid:
        lid_thickness = 0.015
        lid_half_h = lid_thickness / 2
        prims.append(
            Prim(
                GeomType.CYLINDER,
                (params.radius + 0.005, lid_half_h, 0),
                (0, 0, params.height + lid_half_h),
                params.lid_color,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "office bin": Params(
        radius=0.12,
        height=0.30,
        has_lid=False,
        body_color=PLASTIC_BLACK,
    ),
    "kitchen": Params(
        radius=0.16,
        height=0.45,
        body_color=KITCHEN_SILVER,
        lid_color=KITCHEN_SILVER,
    ),
    "outdoor": Params(
        radius=0.25,
        height=0.50,
        body_color=METAL_GRAY,
        lid_color=METAL_GRAY,
    ),
    "slim": Params(
        radius=0.10,
        height=0.42,
        body_color=PLASTIC_BLACK,
        lid_color=PLASTIC_BLACK,
    ),
    "recycling bin": Params(
        radius=0.18,
        height=0.40,
        has_lid=False,
        body_color=RECYCLING_BLUE,
    ),
}
