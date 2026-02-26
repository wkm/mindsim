"""Fridge — a tall rectangular appliance with an optional handle.

A common kitchen landmark. Tall enough to be a significant visual obstacle
for navigation. The handle is a thin box on the front face.

Parameters:
    width:        Half-extent X (meters)
    depth:        Half-extent Y (meters)
    height:       Half-extent Z (meters)
    body_color:   RGBA for the main body
    handle_color: RGBA for the handle
    has_handle:   Whether to include a door handle
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    METAL_DARK,
    METAL_GRAY,
    PLASTIC_WHITE,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Appliance colors
STAINLESS = (0.72, 0.72, 0.74, 1.0)
APPLIANCE_WHITE = (0.92, 0.91, 0.89, 1.0)
RETRO_MINT = (0.65, 0.85, 0.78, 1.0)
RETRO_RED = (0.75, 0.22, 0.20, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.40
    depth: float = 0.35
    height: float = 0.85
    body_color: tuple[float, float, float, float] = APPLIANCE_WHITE
    handle_color: tuple[float, float, float, float] = METAL_GRAY
    has_handle: bool = True


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a fridge (body box + optional handle = 1-2 prims)."""
    # Body: tall box sitting on floor
    body = Prim(
        GeomType.BOX,
        (params.width, params.depth, params.height),
        (0, 0, params.height),
        params.body_color,
    )

    if not params.has_handle:
        return (body,)

    # Handle: thin vertical box on the front face (positive Y side)
    handle_half_z = params.height * 0.25
    handle_z = params.height * 1.2  # upper portion of door
    handle = Prim(
        GeomType.BOX,
        (0.015, 0.02, handle_half_z),
        (params.width * 0.6, params.depth + 0.02, handle_z),
        params.handle_color,
    )

    return (body, handle)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "standard white": Params(),
    "stainless steel": Params(
        body_color=STAINLESS,
        handle_color=METAL_CHROME,
    ),
    "mini fridge": Params(
        width=0.25,
        depth=0.25,
        height=0.45,
        body_color=PLASTIC_WHITE,
        handle_color=METAL_DARK,
    ),
    "double-door": Params(
        width=0.55,
        depth=0.38,
        height=0.90,
        body_color=STAINLESS,
        handle_color=METAL_CHROME,
    ),
    "retro mint": Params(
        body_color=RETRO_MINT,
        handle_color=METAL_CHROME,
    ),
    "retro red": Params(
        body_color=RETRO_RED,
        handle_color=METAL_CHROME,
    ),
}
