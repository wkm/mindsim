"""Stove — a cooking range with burners on top.

Box body with 4 cylindrical burners arranged on the top surface.
Uses 5 prims total (body + 4 burners).

Parameters:
    width:        Half-extent X (meters)
    depth:        Half-extent Y (meters)
    height:       Half-extent Z (meters)
    body_color:   RGBA for the main body
    burner_color: RGBA for the burners
    burner_radius: Radius of each burner cylinder
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    METAL_DARK,
    METAL_GRAY,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Stove-specific colors
APPLIANCE_WHITE = (0.92, 0.91, 0.89, 1.0)
STAINLESS = (0.72, 0.72, 0.74, 1.0)
CAST_IRON = (0.22, 0.22, 0.22, 1.0)
BURNER_BLACK = (0.18, 0.18, 0.18, 1.0)
VINTAGE_CREAM = (0.93, 0.90, 0.82, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.30
    depth: float = 0.30
    height: float = 0.45
    body_color: tuple[float, float, float, float] = APPLIANCE_WHITE
    burner_color: tuple[float, float, float, float] = BURNER_BLACK
    burner_radius: float = 0.06


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a stove (body + 4 burners = 5 prims)."""
    # Body: box sitting on floor
    body = Prim(
        GeomType.BOX,
        (params.width, params.depth, params.height),
        (0, 0, params.height),
        params.body_color,
    )

    # Burners: 4 small cylinders on top surface
    # Arranged in a 2x2 grid, inset from edges
    top_z = params.height * 2 + 0.01  # just above body top
    burner_half_h = 0.01
    inset_x = params.width * 0.5
    inset_y = params.depth * 0.5

    burner_positions = [
        (-inset_x, -inset_y),  # back-left
        (inset_x, -inset_y),  # back-right
        (-inset_x, inset_y),  # front-left
        (inset_x, inset_y),  # front-right
    ]

    burners = tuple(
        Prim(
            GeomType.CYLINDER,
            (params.burner_radius, burner_half_h, 0),
            (bx, by, top_z),
            params.burner_color,
        )
        for bx, by in burner_positions
    )

    return (body, *burners)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "gas range": Params(),
    "electric": Params(
        body_color=APPLIANCE_WHITE,
        burner_color=CAST_IRON,
        burner_radius=0.07,
    ),
    "compact": Params(
        width=0.22,
        depth=0.25,
        height=0.40,
        burner_radius=0.045,
    ),
    "professional": Params(
        width=0.45,
        depth=0.35,
        height=0.45,
        body_color=STAINLESS,
        burner_color=CAST_IRON,
        burner_radius=0.07,
    ),
    "vintage": Params(
        body_color=VINTAGE_CREAM,
        burner_color=METAL_CHROME,
        burner_radius=0.055,
    ),
    "commercial": Params(
        width=0.40,
        depth=0.35,
        height=0.48,
        body_color=METAL_GRAY,
        burner_color=METAL_DARK,
        burner_radius=0.065,
    ),
}
