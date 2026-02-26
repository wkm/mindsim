"""Painting — a framed canvas mounted on a wall.

Wall-mounted decoration providing visual variety at eye height.
The canvas is a thin flat box with an optional slightly larger frame behind it.

Parameters:
    width:        Half-extent X (meters)
    depth:        Half-extent Y (meters) — very thin
    height:       Half-extent Z (meters)
    center_z:     Center height above floor (meters)
    canvas_color: RGBA for the painted surface
    frame_color:  RGBA for the frame border
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_DARK,
    WOOD_DARK,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Canvas colors (muted tones)
CANVAS_BLUE = (0.30, 0.40, 0.55, 1.0)
CANVAS_GREEN = (0.32, 0.48, 0.35, 1.0)
CANVAS_RED = (0.60, 0.25, 0.22, 1.0)
CANVAS_OCHRE = (0.72, 0.58, 0.32, 1.0)
CANVAS_GRAY = (0.50, 0.50, 0.52, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.175
    depth: float = 0.01
    height: float = 0.125
    center_z: float = 1.4
    canvas_color: tuple[float, float, float, float] = CANVAS_BLUE
    frame_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a wall painting (frame + canvas = 2 prims)."""
    # Frame: slightly larger box behind the canvas
    frame_border = 0.015
    frame = Prim(
        GeomType.BOX,
        (params.width + frame_border, params.depth, params.height + frame_border),
        (0, 0, params.center_z),
        params.frame_color,
    )

    # Canvas: thinner box in front of the frame
    canvas = Prim(
        GeomType.BOX,
        (params.width, params.depth * 0.5, params.height),
        (0, -params.depth, params.center_z),
        params.canvas_color,
    )

    return (frame, canvas)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "landscape painting": Params(),
    "portrait painting": Params(
        width=0.125,
        height=0.20,
        canvas_color=CANVAS_GREEN,
        frame_color=WOOD_MEDIUM,
    ),
    "large canvas": Params(
        width=0.30,
        height=0.22,
        canvas_color=CANVAS_OCHRE,
        frame_color=WOOD_DARK,
    ),
    "small square": Params(
        width=0.10,
        height=0.10,
        center_z=1.35,
        canvas_color=CANVAS_RED,
        frame_color=METAL_DARK,
    ),
    "panoramic": Params(
        width=0.40,
        height=0.10,
        center_z=1.45,
        canvas_color=CANVAS_GRAY,
        frame_color=WOOD_DARK,
    ),
}
