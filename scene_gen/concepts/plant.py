"""Plant — a potted plant with cylindrical pot and spherical/ellipsoid foliage.

First non-rectangular concept! Provides organic shapes for visual diversity.

Parameters:
    pot_radius:     Radius of the pot cylinder
    pot_height:     Height of the pot
    foliage_rx:     Foliage X radius (ellipsoid half-extent)
    foliage_ry:     Foliage Y radius
    foliage_rz:     Foliage Z radius (height)
    pot_color:      RGBA for the pot
    foliage_color:  RGBA for the foliage
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CORNER

# Pot colors
TERRACOTTA = (0.72, 0.42, 0.28, 1.0)
CERAMIC_WHITE = (0.90, 0.88, 0.85, 1.0)
CERAMIC_GRAY = (0.55, 0.55, 0.52, 1.0)
POT_BLACK = (0.20, 0.20, 0.20, 1.0)

# Foliage colors
GREEN_DARK = (0.15, 0.38, 0.15, 1.0)
GREEN_MEDIUM = (0.22, 0.50, 0.22, 1.0)
GREEN_LIGHT = (0.35, 0.58, 0.30, 1.0)
GREEN_TROPICAL = (0.18, 0.55, 0.25, 1.0)


@dataclass(frozen=True)
class Params:
    pot_radius: float = 0.15
    pot_height: float = 0.30
    foliage_rx: float = 0.25
    foliage_ry: float = 0.25
    foliage_rz: float = 0.30
    pot_color: tuple[float, float, float, float] = TERRACOTTA
    foliage_color: tuple[float, float, float, float] = GREEN_MEDIUM


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a potted plant (pot cylinder + foliage ellipsoid = 2 prims)."""
    # Pot: cylinder sitting on floor
    pot_half_h = params.pot_height / 2
    pot = Prim(
        GeomType.CYLINDER,
        (params.pot_radius, pot_half_h, 0),
        (0, 0, pot_half_h),
        params.pot_color,
    )

    # Foliage: ellipsoid sitting on top of pot
    foliage_z = params.pot_height + params.foliage_rz * 0.7  # overlap slightly into pot
    foliage = Prim(
        GeomType.ELLIPSOID,
        (params.foliage_rx, params.foliage_ry, params.foliage_rz),
        (0, 0, foliage_z),
        params.foliage_color,
    )

    return (pot, foliage)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "floor plant": Params(),
    "tall plant": Params(
        pot_radius=0.18,
        pot_height=0.35,
        foliage_rx=0.30,
        foliage_ry=0.30,
        foliage_rz=0.45,
        pot_color=CERAMIC_GRAY,
        foliage_color=GREEN_DARK,
    ),
    "small pot": Params(
        pot_radius=0.10,
        pot_height=0.18,
        foliage_rx=0.15,
        foliage_ry=0.15,
        foliage_rz=0.18,
        pot_color=CERAMIC_WHITE,
        foliage_color=GREEN_LIGHT,
    ),
    "bush": Params(
        pot_radius=0.20,
        pot_height=0.25,
        foliage_rx=0.40,
        foliage_ry=0.40,
        foliage_rz=0.30,
        foliage_color=GREEN_DARK,
    ),
    "tree": Params(
        pot_radius=0.20,
        pot_height=0.40,
        foliage_rx=0.35,
        foliage_ry=0.35,
        foliage_rz=0.55,
        pot_color=POT_BLACK,
        foliage_color=GREEN_TROPICAL,
    ),
    "tropical": Params(
        pot_radius=0.16,
        pot_height=0.30,
        foliage_rx=0.35,
        foliage_ry=0.30,
        foliage_rz=0.50,
        pot_color=TERRACOTTA,
        foliage_color=GREEN_TROPICAL,
    ),
}
