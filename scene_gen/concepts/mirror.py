"""Mirror — a reflective surface with visible frame.

Wall-mounted or floor-standing mirrors. Sims 1 style: chunky visible
frames, bright reflective surfaces. Types include rectangular wall mirror,
round mirror (cylinder), full-length floor mirror (tall + stand), and
vanity mirror (small on pedestal).

Parameters:
    width:         Half-extent X (meters) — for rectangular types
    depth:         Half-extent Y (meters) — very thin
    height:        Half-extent Z (meters)
    center_z:      Center height above floor (meters)
    frame_border:  Frame border thickness (meters)
    is_round:      Use cylinder shape instead of box
    round_radius:  Radius for round mirror (meters)
    has_stand:     Include a floor stand (for floor/vanity mirrors)
    stand_height:  Height of stand base from floor (meters)
    mirror_color:  RGBA for the reflective surface
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
MIRROR_BLUE = (0.82, 0.85, 0.92, 1.0)

# Frame colors
FRAME_GOLD = (0.78, 0.65, 0.20, 1.0)
FRAME_BLACK = (0.12, 0.12, 0.12, 1.0)
FRAME_WHITE = (0.92, 0.90, 0.88, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.125
    depth: float = 0.01
    height: float = 0.20
    center_z: float = 1.4
    frame_border: float = 0.012
    is_round: bool = False
    round_radius: float = 0.12
    has_stand: bool = False
    stand_height: float = 0.0
    mirror_color: tuple[float, float, float, float] = MIRROR_SILVER
    frame_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a mirror (frame + surface, optionally with stand).

    Rectangular: 2 prims (frame box + mirror box), +1-2 for stand.
    Round: 2 prims (frame cylinder + mirror cylinder), +1-2 for stand.
    Max 4 prims total.
    """
    prims: list[Prim] = []

    if params.is_round:
        # Round mirror — frame is a thicker cylinder behind a thinner one
        frame_r = params.round_radius + params.frame_border
        frame = Prim(
            GeomType.CYLINDER,
            (frame_r, params.depth, 0),
            (0, 0, params.center_z),
            params.frame_color,
            euler=(1.5708, 0.0, 0.0),  # rotate to face forward (lie on X axis)
        )
        prims.append(frame)

        mirror = Prim(
            GeomType.CYLINDER,
            (params.round_radius, params.depth * 0.5, 0),
            (0, -params.depth, params.center_z),
            params.mirror_color,
            euler=(1.5708, 0.0, 0.0),
        )
        prims.append(mirror)
    else:
        # Rectangular mirror — frame box + mirror box
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

        mirror = Prim(
            GeomType.BOX,
            (params.width, params.depth * 0.5, params.height),
            (0, -params.depth, params.center_z),
            params.mirror_color,
        )
        prims.append(mirror)

    # Optional stand (for floor mirrors and vanity mirrors)
    if params.has_stand:
        stand_w = (
            params.width * 0.6 if not params.is_round else params.round_radius * 0.6
        )
        stand_hh = params.stand_height / 2
        # Vertical post
        prims.append(
            Prim(
                GeomType.BOX,
                (0.015, 0.015, stand_hh),
                (0, 0, stand_hh),
                params.frame_color,
            )
        )
        # Base foot
        prims.append(
            Prim(
                GeomType.BOX,
                (stand_w, 0.05, 0.012),
                (0, 0, 0.012),
                params.frame_color,
            )
        )

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations — Sims 1 style: chunky frames, bright surfaces
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "wall mirror": Params(),
    "bathroom mirror": Params(
        width=0.20,
        height=0.15,
        center_z=1.45,
        frame_border=0.010,
        frame_color=METAL_CHROME,
        mirror_color=MIRROR_BLUE,
    ),
    "round mirror": Params(
        is_round=True,
        round_radius=0.14,
        center_z=1.40,
        frame_border=0.015,
        frame_color=FRAME_GOLD,
    ),
    "small round mirror": Params(
        is_round=True,
        round_radius=0.09,
        center_z=1.42,
        frame_border=0.012,
        frame_color=FRAME_BLACK,
        mirror_color=MIRROR_WARM,
    ),
    "full-length mirror": Params(
        width=0.18,
        height=0.50,
        center_z=0.85,
        frame_border=0.015,
        frame_color=WOOD_LIGHT,
        has_stand=True,
        stand_height=0.35,
    ),
    "vanity mirror": Params(
        width=0.12,
        height=0.14,
        center_z=0.55,
        frame_border=0.010,
        mirror_color=MIRROR_WARM,
        frame_color=WOOD_MEDIUM,
        has_stand=True,
        stand_height=0.35,
    ),
    "ornate wall mirror": Params(
        width=0.16,
        height=0.22,
        center_z=1.40,
        frame_border=0.022,
        frame_color=FRAME_GOLD,
    ),
    "dark framed mirror": Params(
        width=0.14,
        height=0.18,
        center_z=1.38,
        frame_border=0.014,
        frame_color=METAL_DARK,
        mirror_color=MIRROR_SILVER,
    ),
}
