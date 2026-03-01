"""Painting — a framed canvas mounted on a wall.

Wall-mounted decoration providing visual variety at eye height.
Sims 1 style: chunky frames, bright saturated canvases, recognizable
silhouettes from across a room. Variations include portrait, landscape,
modern abstract (multi-color blocks), ornate framed, and poster.

Parameters:
    width:        Half-extent X (meters)
    depth:        Half-extent Y (meters) — very thin
    height:       Half-extent Z (meters)
    center_z:     Center height above floor (meters)
    frame_border: Frame border thickness (meters)
    canvas_color: RGBA for the painted surface
    frame_color:  RGBA for the frame border
    accent_color: Optional RGBA for a second canvas stripe (abstract style)
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    WOOD_DARK,
    WOOD_LIGHT,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Canvas colors — saturated Sims 1 palette
CANVAS_BLUE = (0.22, 0.42, 0.70, 1.0)
CANVAS_GREEN = (0.25, 0.58, 0.30, 1.0)
CANVAS_RED = (0.72, 0.18, 0.15, 1.0)
CANVAS_OCHRE = (0.80, 0.62, 0.22, 1.0)
CANVAS_GRAY = (0.50, 0.50, 0.52, 1.0)
CANVAS_PINK = (0.82, 0.38, 0.52, 1.0)
CANVAS_PURPLE = (0.48, 0.22, 0.60, 1.0)
CANVAS_TEAL = (0.15, 0.58, 0.55, 1.0)
CANVAS_YELLOW = (0.92, 0.82, 0.20, 1.0)
CANVAS_ORANGE = (0.88, 0.48, 0.15, 1.0)

# Frame colors
FRAME_GOLD = (0.78, 0.65, 0.20, 1.0)
FRAME_BLACK = (0.12, 0.12, 0.12, 1.0)
FRAME_WHITE = (0.92, 0.90, 0.88, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.175
    depth: float = 0.01
    height: float = 0.125
    center_z: float = 1.4
    frame_border: float = 0.015
    canvas_color: tuple[float, float, float, float] = CANVAS_BLUE
    frame_color: tuple[float, float, float, float] = WOOD_DARK
    accent_color: tuple[float, float, float, float] | None = None


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a wall painting (frame + canvas, optionally with accent stripe).

    Returns 2 prims normally, 3 with accent stripe.
    """
    prims: list[Prim] = []

    # Frame: slightly larger box behind the canvas
    frame = Prim(
        GeomType.BOX,
        (
            params.width + params.frame_border,
            params.depth,
            params.height + params.frame_border,
        ),
        (0, 0, params.center_z),
        params.frame_color,
    )
    prims.append(frame)

    # Canvas: thinner box in front of the frame
    canvas = Prim(
        GeomType.BOX,
        (params.width, params.depth * 0.5, params.height),
        (0, -params.depth, params.center_z),
        params.canvas_color,
    )
    prims.append(canvas)

    # Optional accent stripe (for abstract / modern style)
    if params.accent_color is not None:
        stripe = Prim(
            GeomType.BOX,
            (params.width * 0.9, params.depth * 0.3, params.height * 0.25),
            (0, -params.depth * 1.5, params.center_z - params.height * 0.2),
            params.accent_color,
        )
        prims.append(stripe)

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations — Sims 1 style: chunky, colorful, recognizable
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "small portrait": Params(
        width=0.10,
        height=0.16,
        center_z=1.40,
        frame_border=0.012,
        canvas_color=CANVAS_GREEN,
        frame_color=FRAME_GOLD,
    ),
    "large landscape": Params(
        width=0.32,
        height=0.20,
        center_z=1.42,
        frame_border=0.018,
        canvas_color=CANVAS_OCHRE,
        frame_color=WOOD_DARK,
    ),
    "modern abstract": Params(
        width=0.22,
        height=0.22,
        center_z=1.38,
        frame_border=0.008,
        canvas_color=CANVAS_TEAL,
        frame_color=FRAME_WHITE,
        accent_color=CANVAS_ORANGE,
    ),
    "ornate framed": Params(
        width=0.18,
        height=0.14,
        center_z=1.45,
        frame_border=0.025,
        canvas_color=CANVAS_RED,
        frame_color=FRAME_GOLD,
    ),
    "poster": Params(
        width=0.20,
        height=0.28,
        center_z=1.35,
        frame_border=0.005,
        canvas_color=CANVAS_PINK,
        frame_color=FRAME_BLACK,
    ),
    "panoramic": Params(
        width=0.38,
        height=0.10,
        center_z=1.45,
        frame_border=0.012,
        canvas_color=CANVAS_BLUE,
        frame_color=WOOD_MEDIUM,
    ),
    "abstract bold": Params(
        width=0.25,
        height=0.25,
        center_z=1.40,
        frame_border=0.010,
        canvas_color=CANVAS_YELLOW,
        frame_color=FRAME_BLACK,
        accent_color=CANVAS_PURPLE,
    ),
    "classic oil": Params(
        width=0.24,
        height=0.18,
        center_z=1.42,
        frame_border=0.022,
        canvas_color=CANVAS_GRAY,
        frame_color=WOOD_LIGHT,
    ),
}
