"""Stove -- a cooking range with burners, oven door, and optional backsplash.

Sims 1 style: chunky body, clearly visible burners with contrasting color,
oven door panel on the front face, bold appliance colors.

Parameters:
    width:         Half-extent X (meters)
    depth:         Half-extent Y (meters)
    height:        Half-extent Z (meters)
    body_color:    RGBA for the main body
    burner_color:  RGBA for the burners
    burner_radius: Radius of each burner cylinder
    door_color:    RGBA for the oven door panel
    has_door:      Whether to show an oven door on the front
    has_backsplash: Whether to add a backsplash panel behind
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
OVEN_DARK = (0.15, 0.15, 0.15, 1.0)
GAS_BLUE = (0.30, 0.45, 0.70, 1.0)
RETRO_TEAL = (0.25, 0.58, 0.55, 1.0)
BURNER_RED = (0.55, 0.15, 0.12, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.30
    depth: float = 0.30
    height: float = 0.45
    body_color: tuple[float, float, float, float] = APPLIANCE_WHITE
    burner_color: tuple[float, float, float, float] = BURNER_BLACK
    burner_radius: float = 0.06
    door_color: tuple[float, float, float, float] = OVEN_DARK
    has_door: bool = True
    has_backsplash: bool = False


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a stove (body + oven door + backsplash + 4 burners, up to 8 prims)."""
    prims: list[Prim] = []

    # Body: box sitting on floor
    body = Prim(
        GeomType.BOX,
        (params.width, params.depth, params.height),
        (0, 0, params.height),
        params.body_color,
    )
    prims.append(body)

    # Oven door: dark panel on the front face (positive Y side)
    if params.has_door:
        door_half_z = params.height * 0.45
        door_z = params.height * 0.55  # lower portion (below burner area)
        door = Prim(
            GeomType.BOX,
            (params.width * 0.80, 0.008, door_half_z),
            (0, params.depth + 0.008, door_z),
            params.door_color,
        )
        prims.append(door)

    # Backsplash: thin tall panel behind the stove
    if params.has_backsplash:
        splash_height = params.height * 0.50
        splash = Prim(
            GeomType.BOX,
            (params.width, 0.01, splash_height / 2),
            (0, -params.depth - 0.01, params.height * 2 + splash_height / 2),
            params.body_color,
        )
        prims.append(splash)

    # Burners: 4 cylinders on top surface in a 2x2 grid
    top_z = params.height * 2 + 0.012
    burner_half_h = 0.012
    inset_x = params.width * 0.50
    inset_y = params.depth * 0.50

    burner_positions = [
        (-inset_x, -inset_y),  # back-left
        (inset_x, -inset_y),  # back-right
        (-inset_x, inset_y),  # front-left
        (inset_x, inset_y),  # front-right
    ]

    for bx, by in burner_positions:
        prims.append(
            Prim(
                GeomType.CYLINDER,
                (params.burner_radius, burner_half_h, 0),
                (bx, by, top_z),
                params.burner_color,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "gas range": Params(
        burner_color=GAS_BLUE,
        burner_radius=0.055,
    ),
    "electric": Params(
        body_color=APPLIANCE_WHITE,
        burner_color=BURNER_RED,
        burner_radius=0.07,
    ),
    "compact": Params(
        width=0.22,
        depth=0.25,
        height=0.40,
        burner_radius=0.045,
        has_door=False,
    ),
    "professional": Params(
        width=0.45,
        depth=0.35,
        height=0.45,
        body_color=STAINLESS,
        burner_color=CAST_IRON,
        burner_radius=0.07,
        door_color=(0.12, 0.12, 0.12, 1.0),
        has_backsplash=True,
    ),
    "vintage cream": Params(
        body_color=VINTAGE_CREAM,
        burner_color=METAL_CHROME,
        burner_radius=0.055,
        door_color=(0.82, 0.78, 0.70, 1.0),
    ),
    "retro teal": Params(
        body_color=RETRO_TEAL,
        burner_color=METAL_CHROME,
        burner_radius=0.055,
        door_color=(0.20, 0.48, 0.45, 1.0),
    ),
    "commercial": Params(
        width=0.40,
        depth=0.35,
        height=0.48,
        body_color=METAL_GRAY,
        burner_color=METAL_DARK,
        burner_radius=0.065,
        door_color=OVEN_DARK,
        has_backsplash=True,
    ),
    "black matte": Params(
        body_color=(0.18, 0.18, 0.18, 1.0),
        burner_color=CAST_IRON,
        burner_radius=0.065,
        door_color=(0.12, 0.12, 0.12, 1.0),
    ),
}
