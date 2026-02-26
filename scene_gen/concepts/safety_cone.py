"""Safety Cone — a tapered, high-visibility obstacle.

Sims 1 style: chunky proportions, saturated colors (orange/yellow),
and contrasting reflective bands.

Parameters:
    base_width:    Width/depth of the square base (meters)
    base_height:   Height of the square base plate (meters)
    cone_radius:   Radius at the bottom of the tapered part (meters)
    cone_height:   Total height from floor to tip (meters)
    has_stripe:    Include reflective stripes
    has_cap:       Include a small top cap
    has_base_ring: Include a raised base ring
    color:         RGBA for the cone body
    stripe_color:  RGBA for the reflective stripe
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    PLASTIC_BLACK,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CENTER

# Standard safety colors
SAFETY_ORANGE = (1.0, 0.45, 0.10, 1.0)
SAFETY_YELLOW = (0.95, 0.85, 0.15, 1.0)
REFLECTIVE_WHITE = (0.95, 0.95, 0.98, 1.0)


@dataclass(frozen=True)
class Params:
    base_width: float = 0.25
    base_height: float = 0.02
    cone_radius: float = 0.10
    cone_height: float = 0.45
    has_stripe: bool = True
    has_cap: bool = True
    has_base_ring: bool = True
    color: tuple[float, float, float, float] = SAFETY_ORANGE
    stripe_color: tuple[float, float, float, float] = REFLECTIVE_WHITE


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a safety cone (base plate + tapered cone + stripes + cap)."""
    prims: list[Prim] = []

    # 1. Square base plate
    hb = params.base_width / 2
    hbh = params.base_height / 2
    prims.append(
        Prim(
            GeomType.BOX,
            (hb, hb, hbh),
            (0, 0, hbh),
            params.color,
        )
    )

    # 1b. Raised base ring for extra chunky silhouette
    if params.has_base_ring:
        ring = Prim(
            GeomType.CYLINDER,
            (params.cone_radius * 0.95, params.base_height * 0.4, 0),
            (0, 0, params.base_height + params.base_height * 0.4),
            params.color,
        )
        prims.append(ring)

    # 2. Tapered cone (using GeomType.CYLINDER with 3 size parameters)
    # MuJoCo tapered CYLINDER: (bottom_radius, half_height, top_radius)
    cone_h = params.cone_height - params.base_height
    hch = cone_h / 2
    top_r = params.cone_radius * 0.1  # small top radius for a "pointy" look
    prims.append(
        Prim(
            GeomType.CYLINDER,
            (params.cone_radius, hch, top_r),
            (0, 0, params.base_height + hch),
            params.color,
        )
    )

    # 3. Reflective stripes (thin cylinder bands)
    if params.has_stripe:
        stripe_h = 0.05
        stripe_r = params.cone_radius * 0.55 + 0.005
        for frac in (0.35, 0.65):
            stripe_z = params.base_height + cone_h * frac
            prims.append(
                Prim(
                    GeomType.CYLINDER,
                    (stripe_r, stripe_h / 2, 0),
                    (0, 0, stripe_z),
                    params.stripe_color,
                )
            )

    # 4. Small top cap
    if params.has_cap:
        cap = Prim(
            GeomType.SPHERE,
            (params.cone_radius * 0.12, 0, 0),
            (0, 0, params.base_height + cone_h - params.cone_radius * 0.08),
            params.color,
        )
        prims.append(cap)

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "standard orange": Params(),
    "caution yellow": Params(
        color=SAFETY_YELLOW,
        stripe_color=PLASTIC_BLACK,  # yellow cones often have black stripes
    ),
    "tall pylon": Params(
        base_width=0.35,
        cone_radius=0.14,
        cone_height=0.75,
        color=SAFETY_ORANGE,
    ),
    "small marker": Params(
        base_width=0.15,
        cone_radius=0.06,
        cone_height=0.20,
        has_stripe=False,
    ),
    "traffic barrel": Params(
        base_width=0.45,
        base_height=0.05,
        cone_radius=0.25,
        cone_height=0.90,
        color=SAFETY_ORANGE,
        stripe_color=REFLECTIVE_WHITE,
    ),
}
