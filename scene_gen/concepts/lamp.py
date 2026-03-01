"""Lamp -- various lamp types: floor, table, desk, pendant, wall sconce.

Sims 1 style: recognizable silhouettes, chunky proportions, warm visible shades.
Each lamp type uses different proportions and structure while staying within
the 8-prim budget.

Parameters:
    style:        Lamp style ("floor", "table", "desk", "pendant", "sconce")
    base_radius:  Radius of the base disk
    pole_height:  Height of the pole in meters
    pole_radius:  Radius of the pole
    shade_radius: Radius of the lampshade (bottom)
    shade_height: Height of the lampshade
    base_color:   RGBA for base and pole
    shade_color:  RGBA for the shade
"""

from dataclasses import dataclass
from functools import lru_cache
from math import pi

from scene_gen.primitives import (
    LAMP_WARM,
    LAMP_WHITE,
    METAL_CHROME,
    METAL_DARK,
    METAL_GRAY,
    WOOD_MEDIUM,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CORNER

# Lamp-specific colors
SHADE_CREAM = (0.95, 0.92, 0.82, 0.90)
SHADE_SAGE = (0.70, 0.78, 0.65, 0.90)
SHADE_ROSE = (0.85, 0.60, 0.62, 0.90)
SHADE_TEAL = (0.45, 0.72, 0.70, 0.90)
BULB_WARM = (0.98, 0.92, 0.60, 0.95)


@dataclass(frozen=True)
class Params:
    style: str = "floor"  # floor, table, desk, pendant, sconce
    base_radius: float = 0.15
    pole_height: float = 1.50
    pole_radius: float = 0.015
    shade_radius: float = 0.18
    shade_height: float = 0.22
    base_color: tuple[float, float, float, float] = METAL_DARK
    shade_color: tuple[float, float, float, float] = LAMP_WARM


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a lamp based on style (3-6 prims)."""
    if params.style == "table":
        return _gen_table_lamp(params)
    elif params.style == "desk":
        return _gen_desk_lamp(params)
    elif params.style == "pendant":
        return _gen_pendant(params)
    elif params.style == "sconce":
        return _gen_sconce(params)
    else:
        return _gen_floor_lamp(params)


def _gen_floor_lamp(p: Params) -> tuple[Prim, ...]:
    """Floor lamp: heavy base + tall thin pole + wide shade at top."""
    # Base: flat cylinder on the floor
    base = Prim(
        GeomType.CYLINDER,
        (p.base_radius, 0.018, 0),
        (0, 0, 0.018),
        p.base_color,
    )

    # Pole: thin cylinder from base to shade
    pole_half = p.pole_height / 2
    pole = Prim(
        GeomType.CYLINDER,
        (p.pole_radius, pole_half, 0),
        (0, 0, 0.036 + pole_half),
        p.base_color,
    )

    # Shade: conical frustum (wider at bottom, narrower at top)
    shade_half = p.shade_height / 2
    shade_z = 0.036 + p.pole_height - shade_half
    shade = Prim(
        GeomType.CONE,
        (p.shade_radius, shade_half, p.shade_radius * 0.55),
        (0, 0, shade_z),
        p.shade_color,
    )

    # Bulb glow: small sphere inside shade
    bulb = Prim(
        GeomType.SPHERE,
        (p.shade_radius * 0.30, 0, 0),
        (0, 0, shade_z),
        BULB_WARM,
    )

    return (base, pole, shade, bulb)


def _gen_table_lamp(p: Params) -> tuple[Prim, ...]:
    """Table lamp: short wide base + short pole + broad shade."""
    # Wide flat base
    base = Prim(
        GeomType.CYLINDER,
        (p.base_radius * 0.9, 0.025, 0),
        (0, 0, 0.025),
        p.base_color,
    )

    # Short thick body/pole
    body_h = p.pole_height / 2
    body_half = body_h / 2
    body = Prim(
        GeomType.CYLINDER,
        (p.pole_radius * 2.0, body_half, 0),
        (0, 0, 0.05 + body_half),
        p.base_color,
    )

    # Broad conical shade (wider at bottom, narrower at top)
    shade_half = p.shade_height / 2
    shade_z = 0.05 + body_h + shade_half
    shade = Prim(
        GeomType.CONE,
        (p.shade_radius * 1.2, shade_half, p.shade_radius * 0.65),
        (0, 0, shade_z),
        p.shade_color,
    )

    # Bulb glow
    bulb = Prim(
        GeomType.SPHERE,
        (p.shade_radius * 0.25, 0, 0),
        (0, 0, shade_z),
        BULB_WARM,
    )

    return (base, body, shade, bulb)


def _gen_desk_lamp(p: Params) -> tuple[Prim, ...]:
    """Desk lamp: flat base + angled arm + small cone shade."""
    # Flat square-ish base
    base = Prim(
        GeomType.BOX,
        (p.base_radius * 0.7, p.base_radius * 0.7, 0.02),
        (0, 0, 0.02),
        p.base_color,
    )

    # Vertical lower arm
    arm_lower_h = p.pole_height * 0.45
    arm_lower = Prim(
        GeomType.CYLINDER,
        (p.pole_radius, arm_lower_h / 2, 0),
        (0, 0, 0.04 + arm_lower_h / 2),
        p.base_color,
    )

    # Upper arm angled forward (tilted cylinder)
    arm_upper_h = p.pole_height * 0.40
    arm_top_z = 0.04 + arm_lower_h
    arm_upper = Prim(
        GeomType.CYLINDER,
        (p.pole_radius, arm_upper_h / 2, 0),
        (0, p.base_radius * 0.5, arm_top_z + arm_upper_h * 0.35),
        p.base_color,
        euler=(0.5, 0.0, 0.0),  # tilt forward
    )

    # Conical shade at end of upper arm
    shade_half = p.shade_height * 0.6 / 2
    shade_z = arm_top_z + arm_upper_h * 0.65
    shade = Prim(
        GeomType.CONE,
        (p.shade_radius * 0.65, shade_half, p.shade_radius * 0.35),
        (0, p.base_radius * 0.9, shade_z),
        p.shade_color,
        euler=(0.35, 0.0, 0.0),
    )

    return (base, arm_lower, arm_upper, shade)


def _gen_pendant(p: Params) -> tuple[Prim, ...]:
    """Pendant / hanging lamp: ceiling rod + sphere or dome shade."""
    # Ceiling mount plate
    mount = Prim(
        GeomType.CYLINDER,
        (0.04, 0.01, 0),
        (0, 0, 2.40),
        p.base_color,
    )

    # Drop cord
    cord_half = p.pole_height * 0.25
    cord_z = 2.40 - 0.01 - cord_half
    cord = Prim(
        GeomType.CYLINDER,
        (0.005, cord_half, 0),
        (0, 0, cord_z),
        p.base_color,
    )

    # Globe shade (sphere for pendant look)
    globe_z = cord_z - cord_half - p.shade_radius * 0.7
    globe = Prim(
        GeomType.SPHERE,
        (p.shade_radius * 0.8, 0, 0),
        (0, 0, globe_z),
        p.shade_color,
    )

    # Inner bulb glow
    bulb = Prim(
        GeomType.SPHERE,
        (p.shade_radius * 0.25, 0, 0),
        (0, 0, globe_z),
        BULB_WARM,
    )

    return (mount, cord, globe, bulb)


def _gen_sconce(p: Params) -> tuple[Prim, ...]:
    """Wall sconce: wall plate + arm + small shade, mounted at ~1.6m."""
    mount_z = 1.60

    # Wall mount plate (flat box against wall)
    plate = Prim(
        GeomType.BOX,
        (0.04, 0.015, 0.06),
        (0, 0, mount_z),
        p.base_color,
    )

    # Arm extending outward from wall
    arm = Prim(
        GeomType.CYLINDER,
        (p.pole_radius, 0.06, 0),
        (0, 0.06, mount_z),
        p.base_color,
        euler=(pi / 2, 0.0, 0.0),  # horizontal, pointing outward
    )

    # Small upward-facing conical shade
    shade_half = p.shade_height * 0.40 / 2
    shade = Prim(
        GeomType.CONE,
        (p.shade_radius * 0.50, shade_half, p.shade_radius * 0.30),
        (0, 0.12, mount_z + 0.03),
        p.shade_color,
    )

    # Bulb
    bulb = Prim(
        GeomType.SPHERE,
        (p.shade_radius * 0.15, 0, 0),
        (0, 0.12, mount_z + 0.02),
        BULB_WARM,
    )

    return (plate, arm, shade, bulb)


# ---------------------------------------------------------------------------
# Named variations for the concept catalog
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "floor lamp": Params(),
    "tall floor": Params(
        pole_height=1.80,
        shade_radius=0.20,
        shade_height=0.25,
    ),
    "table lamp": Params(
        style="table",
        base_radius=0.10,
        pole_height=0.50,
        pole_radius=0.02,
        shade_radius=0.14,
        shade_height=0.14,
        base_color=WOOD_MEDIUM,
        shade_color=SHADE_CREAM,
    ),
    "desk lamp": Params(
        style="desk",
        base_radius=0.10,
        pole_height=0.55,
        pole_radius=0.012,
        shade_radius=0.10,
        shade_height=0.10,
        base_color=METAL_GRAY,
        shade_color=METAL_GRAY,
    ),
    "pendant globe": Params(
        style="pendant",
        pole_height=1.20,
        shade_radius=0.18,
        shade_color=LAMP_WHITE,
        base_color=METAL_DARK,
    ),
    "pendant warm": Params(
        style="pendant",
        pole_height=0.80,
        shade_radius=0.14,
        shade_color=SHADE_ROSE,
        base_color=METAL_CHROME,
    ),
    "wall sconce": Params(
        style="sconce",
        pole_radius=0.01,
        shade_radius=0.10,
        shade_height=0.10,
        base_color=METAL_CHROME,
        shade_color=LAMP_WARM,
    ),
    "industrial floor": Params(
        base_radius=0.18,
        pole_height=1.60,
        pole_radius=0.02,
        shade_radius=0.22,
        shade_height=0.20,
        base_color=METAL_DARK,
        shade_color=METAL_GRAY,
    ),
}
