"""Lamp — a floor lamp with base, pole, and shade.

Parameters:
    base_radius:  Radius of the base disk
    pole_height:  Height of the pole in meters
    pole_radius:  Radius of the pole
    shade_radius: Radius of the lampshade
    shade_height: Height of the lampshade
    base_color:   RGBA for base and pole
    shade_color:  RGBA for the shade
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    LAMP_WARM,
    LAMP_WHITE,
    METAL_CHROME,
    METAL_DARK,
    METAL_GRAY,
    PLASTIC_BLACK,
    WOOD_DARK,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CORNER


@dataclass(frozen=True)
class Params:
    base_radius: float = 0.15
    pole_height: float = 1.50
    pole_radius: float = 0.015
    shade_radius: float = 0.18
    shade_height: float = 0.22
    base_color: tuple[float, float, float, float] = METAL_DARK
    shade_color: tuple[float, float, float, float] = LAMP_WARM


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a floor lamp (base + pole + shade = 3 prims)."""
    # Base: flat cylinder on the floor
    base = Prim(
        GeomType.CYLINDER,
        (params.base_radius, 0.015, 0),
        (0, 0, 0.015),
        params.base_color,
    )

    # Pole: thin cylinder from base to shade
    pole_half = params.pole_height / 2
    pole = Prim(
        GeomType.CYLINDER,
        (params.pole_radius, pole_half, 0),
        (0, 0, 0.03 + pole_half),
        params.base_color,
    )

    # Shade: wider cylinder at top of pole
    shade_half = params.shade_height / 2
    shade_z = 0.03 + params.pole_height - shade_half
    shade = Prim(
        GeomType.CYLINDER,
        (params.shade_radius, shade_half, 0),
        (0, 0, shade_z),
        params.shade_color,
    )

    return (base, pole, shade)


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "floor lamp": Params(),
    "tall": Params(pole_height=1.80, shade_radius=0.20, shade_height=0.25),
    "reading": Params(
        pole_height=1.30,
        shade_radius=0.15,
        shade_height=0.18,
        base_color=METAL_GRAY,
        shade_color=LAMP_WHITE,
    ),
    "minimal": Params(
        base_radius=0.10,
        pole_height=1.40,
        pole_radius=0.01,
        shade_radius=0.12,
        shade_height=0.15,
        base_color=PLASTIC_BLACK,
        shade_color=LAMP_WHITE,
    ),
    "industrial": Params(
        base_radius=0.18,
        pole_height=1.60,
        pole_radius=0.02,
        shade_radius=0.22,
        shade_height=0.20,
        base_color=METAL_DARK,
        shade_color=METAL_GRAY,
    ),
    "accent": Params(
        base_radius=0.12,
        pole_height=1.20,
        pole_radius=0.012,
        shade_radius=0.14,
        shade_height=0.16,
        base_color=WOOD_DARK,
        shade_color=LAMP_WARM,
    ),
    "torchiere": Params(
        base_radius=0.14,
        pole_height=1.70,
        pole_radius=0.012,
        shade_radius=0.16,
        shade_height=0.10,
        base_color=METAL_CHROME,
    ),
}
