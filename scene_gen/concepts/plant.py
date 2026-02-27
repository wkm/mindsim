"""Plant — Sims 1-style potted plants with chunky, recognizable silhouettes.

Each plant type uses multiple prims to create a distinct shape: visible pot
with rim, trunk/stem, and layered foliage clusters instead of a single blob.
Think classic Sims 1 indoor plants — low-poly but you can instantly tell
a ficus from a cactus from a fern.

Parameters:
    plant_type:     Which plant shape to generate (potted/ficus/bush/cactus/fern/palm)
    pot_radius:     Radius of the pot cylinder
    pot_height:     Height of the pot
    scale:          Uniform scale multiplier for the whole plant
    pot_color:      RGBA for the pot body and rim
    foliage_color:  RGBA for foliage / canopy prims
    trunk_color:    RGBA for trunk / stem
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
GREEN_CACTUS = (0.28, 0.52, 0.22, 1.0)

# Trunk color
TRUNK_BROWN = (0.40, 0.28, 0.15, 1.0)


@dataclass(frozen=True)
class Params:
    plant_type: str = "potted"
    pot_radius: float = 0.14
    pot_height: float = 0.25
    scale: float = 1.0
    pot_color: tuple[float, float, float, float] = TERRACOTTA
    foliage_color: tuple[float, float, float, float] = GREEN_MEDIUM
    trunk_color: tuple[float, float, float, float] = TRUNK_BROWN


def _s(val: float, scale: float) -> float:
    """Apply scale factor."""
    return val * scale


def _potted(p: Params) -> tuple[Prim, ...]:
    """Classic potted plant: pot + rim + thin stem + foliage ball (4 prims).

    The pot rim is a wider, thin cylinder on top of the pot body, giving
    that distinctive flowerpot silhouette. A thin stem rises from the pot
    and a round foliage sphere sits on top.
    """
    s = p.scale
    pr = _s(p.pot_radius, s)
    ph = _s(p.pot_height, s)
    ph2 = ph / 2

    # Pot body: tapered (narrower at bottom, classic terracotta shape)
    pot = Prim(GeomType.CONE, (pr * 0.72, ph2, pr), (0, 0, ph2), p.pot_color)

    # Pot rim: wider, thin disc on top of the pot
    rim_r = pr * 1.2
    rim_h = _s(0.025, s)
    rim = Prim(GeomType.CYLINDER, (rim_r, rim_h, 0), (0, 0, ph - rim_h), p.pot_color)

    # Thin stem rising from pot
    stem_r = _s(0.02, s)
    stem_h = _s(0.15, s)
    stem = Prim(
        GeomType.CYLINDER,
        (stem_r, stem_h / 2, 0),
        (0, 0, ph + stem_h / 2),
        p.trunk_color,
    )

    # Foliage ball on top of stem
    foliage_r = _s(0.18, s)
    foliage_z = ph + stem_h + foliage_r * 0.7
    foliage = Prim(
        GeomType.SPHERE, (foliage_r, 0, 0), (0, 0, foliage_z), p.foliage_color
    )

    return (pot, rim, stem, foliage)


def _ficus(p: Params) -> tuple[Prim, ...]:
    """Indoor ficus tree: pot + trunk + 3 stacked foliage clusters (5 prims).

    The classic Sims 1 tree look: a visible trunk cylinder with 2-3
    spherical canopy clusters at different heights, creating a layered
    tree crown silhouette.
    """
    s = p.scale
    pr = _s(p.pot_radius, s)
    ph = _s(p.pot_height, s)
    ph2 = ph / 2

    # Pot: tapered (narrower at bottom)
    pot = Prim(GeomType.CONE, (pr * 0.72, ph2, pr), (0, 0, ph2), p.pot_color)

    # Trunk: from pot top upward
    trunk_r = _s(0.035, s)
    trunk_h = _s(0.45, s)
    trunk = Prim(
        GeomType.CYLINDER,
        (trunk_r, trunk_h / 2, 0),
        (0, 0, ph + trunk_h / 2),
        p.trunk_color,
    )

    # Three foliage spheres at different heights, slightly offset
    top = ph + trunk_h
    r1 = _s(0.20, s)
    r2 = _s(0.24, s)
    r3 = _s(0.16, s)

    canopy_low = Prim(
        GeomType.SPHERE,
        (r1, 0, 0),
        (_s(0.06, s), _s(0.06, s), top - _s(0.05, s)),
        p.foliage_color,
    )
    canopy_mid = Prim(
        GeomType.SPHERE,
        (r2, 0, 0),
        (_s(-0.04, s), _s(-0.03, s), top + _s(0.15, s)),
        p.foliage_color,
    )
    canopy_top = Prim(
        GeomType.SPHERE,
        (r3, 0, 0),
        (_s(0.02, s), _s(0.04, s), top + _s(0.32, s)),
        p.foliage_color,
    )

    return (pot, trunk, canopy_low, canopy_mid, canopy_top)


def _bush(p: Params) -> tuple[Prim, ...]:
    """Leafy bush: pot + rim + 3 overlapping foliage ellipsoids (5 prims).

    Multiple ellipsoids at slightly different positions create a layered,
    bushy silhouette — wider than it is tall, like a classic indoor shrub.
    """
    s = p.scale
    pr = _s(p.pot_radius, s)
    ph = _s(p.pot_height, s)
    ph2 = ph / 2

    # Pot body: tapered (narrower at bottom, classic terracotta shape)
    pot = Prim(GeomType.CONE, (pr * 0.72, ph2, pr), (0, 0, ph2), p.pot_color)

    # Pot rim
    rim_r = pr * 1.15
    rim_h = _s(0.02, s)
    rim = Prim(GeomType.CYLINDER, (rim_r, rim_h, 0), (0, 0, ph - rim_h), p.pot_color)

    # Three overlapping foliage ellipsoids at different offsets
    base_z = ph + _s(0.12, s)
    rx = _s(0.22, s)
    ry = _s(0.20, s)
    rz = _s(0.14, s)

    f1 = Prim(
        GeomType.ELLIPSOID,
        (rx, ry, rz),
        (_s(-0.05, s), _s(0.04, s), base_z),
        p.foliage_color,
    )
    f2 = Prim(
        GeomType.ELLIPSOID,
        (rx * 0.9, ry * 0.95, rz * 1.1),
        (_s(0.06, s), _s(-0.05, s), base_z + _s(0.06, s)),
        p.foliage_color,
    )
    f3 = Prim(
        GeomType.ELLIPSOID,
        (rx * 0.75, ry * 0.8, rz * 0.9),
        (_s(0.0, s), _s(0.02, s), base_z + _s(0.14, s)),
        p.foliage_color,
    )

    return (pot, rim, f1, f2, f3)


def _cactus(p: Params) -> tuple[Prim, ...]:
    """Small cactus: pot + rim + tall body capsule + side arm (4 prims).

    Distinctive cactus silhouette with a tall cylindrical body and a
    stubby arm branching off to one side. The pot has a rim for that
    classic flowerpot look.
    """
    s = p.scale
    pr = _s(p.pot_radius, s)
    ph = _s(p.pot_height, s)
    ph2 = ph / 2

    # Pot body: tapered (narrower at bottom, classic terracotta shape)
    pot = Prim(GeomType.CONE, (pr * 0.72, ph2, pr), (0, 0, ph2), p.pot_color)

    # Pot rim
    rim_r = pr * 1.15
    rim_h = _s(0.02, s)
    rim = Prim(GeomType.CYLINDER, (rim_r, rim_h, 0), (0, 0, ph - rim_h), p.pot_color)

    # Main cactus body: tall capsule
    body_r = _s(0.055, s)
    body_h = _s(0.22, s)
    body = Prim(
        GeomType.CAPSULE,
        (body_r, body_h / 2, 0),
        (0, 0, ph + body_h / 2 + body_r),
        p.foliage_color,
    )

    # Side arm: smaller capsule, offset and tilted
    arm_r = _s(0.035, s)
    arm_h = _s(0.10, s)
    arm_z = ph + body_h * 0.55 + body_r
    arm = Prim(
        GeomType.CAPSULE,
        (arm_r, arm_h / 2, 0),
        (_s(0.08, s), 0, arm_z),
        p.foliage_color,
        euler=(0.0, 0.5, 0.0),  # tilt outward
    )

    return (pot, rim, body, arm)


def _fern(p: Params) -> tuple[Prim, ...]:
    """Fern: pot + rim + 3 wide flat ellipsoids suggesting fronds (5 prims).

    Wide, flat ellipsoids at slightly different angles and heights create
    the drooping, layered look of fern fronds spilling over the pot.
    """
    s = p.scale
    pr = _s(p.pot_radius, s)
    ph = _s(p.pot_height, s)
    ph2 = ph / 2

    # Pot body: tapered (narrower at bottom, classic terracotta shape)
    pot = Prim(GeomType.CONE, (pr * 0.72, ph2, pr), (0, 0, ph2), p.pot_color)

    # Pot rim
    rim_r = pr * 1.2
    rim_h = _s(0.025, s)
    rim = Prim(GeomType.CYLINDER, (rim_r, rim_h, 0), (0, 0, ph - rim_h), p.pot_color)

    # Three wide, flat frond layers at different positions and slight tilts
    base_z = ph + _s(0.06, s)
    frond_rx = _s(0.25, s)
    frond_ry = _s(0.22, s)
    frond_rz = _s(0.05, s)

    f1 = Prim(
        GeomType.ELLIPSOID,
        (frond_rx, frond_ry, frond_rz),
        (_s(0.03, s), _s(-0.02, s), base_z),
        p.foliage_color,
        euler=(0.15, 0.1, 0.0),
    )
    f2 = Prim(
        GeomType.ELLIPSOID,
        (frond_rx * 0.9, frond_ry * 1.05, frond_rz * 1.1),
        (_s(-0.04, s), _s(0.03, s), base_z + _s(0.04, s)),
        p.foliage_color,
        euler=(-0.12, 0.08, 0.8),
    )
    f3 = Prim(
        GeomType.ELLIPSOID,
        (frond_rx * 0.75, frond_ry * 0.8, frond_rz * 1.2),
        (_s(0.01, s), _s(0.0, s), base_z + _s(0.09, s)),
        p.foliage_color,
        euler=(0.1, -0.15, 1.5),
    )

    return (pot, rim, f1, f2, f3)


def _palm(p: Params) -> tuple[Prim, ...]:
    """Tall palm: pot + tall thin trunk + 2 elongated foliage ellipsoids (4 prims).

    A tall, thin trunk with elongated foliage at the top. The foliage
    ellipsoids are stretched horizontally to suggest palm fronds fanning
    out from the crown.
    """
    s = p.scale
    pr = _s(p.pot_radius, s)
    ph = _s(p.pot_height, s)
    ph2 = ph / 2

    # Pot: tapered (narrower at bottom)
    pot = Prim(GeomType.CONE, (pr * 0.72, ph2, pr), (0, 0, ph2), p.pot_color)

    # Tall trunk
    trunk_r = _s(0.03, s)
    trunk_h = _s(0.55, s)
    trunk = Prim(
        GeomType.CYLINDER,
        (trunk_r, trunk_h / 2, 0),
        (0, 0, ph + trunk_h / 2),
        p.trunk_color,
    )

    # Two elongated foliage ellipsoids at the crown, crossing each other
    top = ph + trunk_h
    frond_long = _s(0.28, s)
    frond_short = _s(0.10, s)
    frond_z = _s(0.08, s)

    crown1 = Prim(
        GeomType.ELLIPSOID,
        (frond_long, frond_short, frond_z),
        (0, 0, top + frond_z * 0.5),
        p.foliage_color,
        euler=(0.0, 0.0, 0.4),
    )
    crown2 = Prim(
        GeomType.ELLIPSOID,
        (frond_short, frond_long, frond_z),
        (0, 0, top + frond_z * 0.5),
        p.foliage_color,
        euler=(0.0, 0.0, -0.4),
    )

    return (pot, trunk, crown1, crown2)


# Dispatch table for plant types
_GENERATORS = {
    "potted": _potted,
    "ficus": _ficus,
    "bush": _bush,
    "cactus": _cactus,
    "fern": _fern,
    "palm": _palm,
}


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a plant based on plant_type parameter."""
    gen = _GENERATORS.get(params.plant_type, _potted)
    return gen(params)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "potted plant": Params(
        plant_type="potted",
        pot_radius=0.14,
        pot_height=0.25,
        pot_color=TERRACOTTA,
        foliage_color=GREEN_MEDIUM,
    ),
    "ficus tree": Params(
        plant_type="ficus",
        pot_radius=0.16,
        pot_height=0.30,
        scale=1.1,
        pot_color=CERAMIC_GRAY,
        foliage_color=GREEN_DARK,
    ),
    "bush": Params(
        plant_type="bush",
        pot_radius=0.18,
        pot_height=0.22,
        scale=1.15,
        pot_color=TERRACOTTA,
        foliage_color=GREEN_DARK,
    ),
    "cactus": Params(
        plant_type="cactus",
        pot_radius=0.10,
        pot_height=0.16,
        scale=0.9,
        pot_color=CERAMIC_WHITE,
        foliage_color=GREEN_CACTUS,
    ),
    "fern": Params(
        plant_type="fern",
        pot_radius=0.13,
        pot_height=0.20,
        pot_color=CERAMIC_WHITE,
        foliage_color=GREEN_LIGHT,
    ),
    "tall palm": Params(
        plant_type="palm",
        pot_radius=0.15,
        pot_height=0.28,
        scale=1.2,
        pot_color=POT_BLACK,
        foliage_color=GREEN_TROPICAL,
    ),
}
