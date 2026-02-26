"""Mirror — a wall-mounted reflective surface with frame.

Similar to a painting but with a bright, light-colored surface to suggest
reflectivity. Thin flat box at eye height with an optional frame.

Parameters:
    width:         Half-extent X (meters)
    depth:         Half-extent Y (meters) — very thin
    height:        Half-extent Z (meters)
    center_z:      Center height above floor (meters)
    mirror_color:  RGBA for the reflective surface (light gray/white)
    frame_color:   RGBA for the frame
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    METAL_DARK,
    WOOD_DARK,
    WOOD_LIGHT,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.WALL

# Mirror surface colors (high reflectance, light grays)
MIRROR_SILVER = (0.85, 0.88, 0.90, 1.0)
MIRROR_WARM = (0.88, 0.86, 0.82, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.125
    depth: float = 0.01
    height: float = 0.20
    center_z: float = 1.4
    mirror_color: tuple[float, float, float, float] = MIRROR_SILVER
    frame_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a wall mirror (frame + mirror surface = 2 prims)."""
    # Frame: slightly larger box behind the mirror
    frame_border = 0.012
    frame = Prim(
        GeomType.BOX,
        (params.width + frame_border, params.depth, params.height + frame_border),
        (0, 0, params.center_z),
        params.frame_color,
    )

    # Mirror surface: thinner box in front of the frame
    mirror = Prim(
        GeomType.BOX,
        (params.width, params.depth * 0.5, params.height),
        (0, -params.depth, params.center_z),
        params.mirror_color,
    )

    return (frame, mirror)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "wall mirror": Params(),
    "bathroom mirror": Params(
        width=0.20,
        height=0.15,
        center_z=1.45,
        frame_color=METAL_CHROME,
    ),
    "full-length mirror": Params(
        width=0.15,
        height=0.45,
        center_z=1.0,
        frame_color=WOOD_LIGHT,
    ),
    "vanity mirror": Params(
        width=0.15,
        height=0.18,
        center_z=1.35,
        mirror_color=MIRROR_WARM,
        frame_color=WOOD_MEDIUM,
    ),
    "decorative mirror": Params(
        width=0.18,
        height=0.18,
        center_z=1.4,
        frame_color=METAL_DARK,
    ),
}
