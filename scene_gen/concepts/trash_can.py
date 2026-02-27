"""Trash can -- chunky Sims 1-style bins and waste baskets.

Variations include kitchen pedal bins, office waste baskets, recycling bins,
and large outdoor bins. Some have visible lids, pedals, or handles.

Parameters:
    radius:      Radius of the can body
    height:      Total height of the can body
    has_lid:     Whether to add a lid on top
    lid_style:   Lid shape: "flat" (cylinder) or "dome" (sphere cap)
    has_pedal:   Whether to add a foot pedal at the base
    has_handle:  Whether to add a handle on the lid
    body_color:  RGBA for the can body
    lid_color:   RGBA for the lid
    accent_color: RGBA for pedal/handle accents
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import (
    METAL_CHROME,
    METAL_DARK,
    METAL_GRAY,
    PLASTIC_BLACK,
    PLASTIC_WHITE,
    GeomType,
    Placement,
    Prim,
)

PLACEMENT = Placement.CENTER

# Custom colors
RECYCLING_BLUE = (0.20, 0.40, 0.65, 1.0)
KITCHEN_SILVER = (0.72, 0.72, 0.74, 1.0)
OUTDOOR_GREEN = (0.22, 0.42, 0.25, 1.0)
WARM_RED = (0.70, 0.22, 0.18, 1.0)
OFFICE_MESH = (0.35, 0.35, 0.35, 1.0)


@dataclass(frozen=True)
class Params:
    radius: float = 0.15
    height: float = 0.35
    has_lid: bool = True
    lid_style: str = "flat"
    has_pedal: bool = False
    has_handle: bool = False
    body_color: tuple[float, float, float, float] = METAL_GRAY
    lid_color: tuple[float, float, float, float] = METAL_GRAY
    accent_color: tuple[float, float, float, float] = METAL_CHROME


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a trash can (up to 5 prims: body + base ring + lid + pedal + handle)."""
    half_h = params.height / 2
    prims: list[Prim] = []

    # 1. Body: slight outward taper (narrower at bottom)
    body = Prim(
        GeomType.CONE,
        (params.radius * 0.88, half_h, params.radius),
        (0, 0, half_h),
        params.body_color,
    )
    prims.append(body)

    # 2. Base ring: slightly wider cylinder at the bottom for stability look
    base_ring = Prim(
        GeomType.CYLINDER,
        (params.radius + 0.008, 0.012, 0),
        (0, 0, 0.012),
        params.body_color,
    )
    prims.append(base_ring)

    # 3. Lid
    if params.has_lid:
        if params.lid_style == "dome":
            # Dome lid: flattened sphere on top
            lid = Prim(
                GeomType.ELLIPSOID,
                (params.radius * 0.85, params.radius * 0.85, params.radius * 0.3),
                (0, 0, params.height + params.radius * 0.25),
                params.lid_color,
            )
        else:
            # Flat lid: thin cylinder on top
            lid_half_h = 0.015
            lid = Prim(
                GeomType.CYLINDER,
                (params.radius + 0.005, lid_half_h, 0),
                (0, 0, params.height + lid_half_h),
                params.lid_color,
            )
        prims.append(lid)

    # 4. Pedal: small box at the base, sticking out front
    if params.has_pedal:
        pedal = Prim(
            GeomType.BOX,
            (0.03, 0.025, 0.008),
            (0, params.radius + 0.025, 0.008),
            params.accent_color,
        )
        prims.append(pedal)

    # 5. Handle: small cylinder on top of lid
    if params.has_handle and params.has_lid:
        handle_z = params.height + 0.04
        if params.lid_style == "dome":
            handle_z = params.height + params.radius * 0.45
        handle = Prim(
            GeomType.CYLINDER,
            (0.008, 0.015, 0),
            (0, 0, handle_z),
            params.accent_color,
        )
        prims.append(handle)

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "kitchen pedal bin": Params(
        radius=0.14,
        height=0.42,
        has_lid=True,
        lid_style="dome",
        has_pedal=True,
        has_handle=True,
        body_color=KITCHEN_SILVER,
        lid_color=KITCHEN_SILVER,
        accent_color=METAL_CHROME,
    ),
    "office waste basket": Params(
        radius=0.13,
        height=0.30,
        has_lid=False,
        has_pedal=False,
        body_color=OFFICE_MESH,
    ),
    "recycling bin": Params(
        radius=0.18,
        height=0.40,
        has_lid=False,
        has_pedal=False,
        body_color=RECYCLING_BLUE,
    ),
    "outdoor bin": Params(
        radius=0.25,
        height=0.55,
        has_lid=True,
        lid_style="flat",
        has_handle=True,
        body_color=OUTDOOR_GREEN,
        lid_color=OUTDOOR_GREEN,
        accent_color=METAL_DARK,
    ),
    "slim bathroom": Params(
        radius=0.10,
        height=0.35,
        has_lid=True,
        lid_style="dome",
        has_pedal=True,
        body_color=PLASTIC_WHITE,
        lid_color=PLASTIC_WHITE,
        accent_color=METAL_CHROME,
    ),
    "red retro bin": Params(
        radius=0.15,
        height=0.40,
        has_lid=True,
        lid_style="dome",
        has_pedal=True,
        has_handle=True,
        body_color=WARM_RED,
        lid_color=WARM_RED,
        accent_color=METAL_CHROME,
    ),
    "black tall bin": Params(
        radius=0.12,
        height=0.48,
        has_lid=True,
        lid_style="flat",
        has_pedal=True,
        body_color=PLASTIC_BLACK,
        lid_color=PLASTIC_BLACK,
        accent_color=METAL_DARK,
    ),
    "small desktop": Params(
        radius=0.08,
        height=0.18,
        has_lid=False,
        has_pedal=False,
        body_color=PLASTIC_BLACK,
    ),
}
