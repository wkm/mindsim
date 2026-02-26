"""Fridge -- a tall rectangular appliance with handle, freezer line, and character.

Sims 1 style: chunky proportions, bold color schemes, visible details like
door handles and freezer divider lines. Recognizable silhouette from any angle.

Parameters:
    width:         Half-extent X (meters)
    depth:         Half-extent Y (meters)
    height:        Half-extent Z (meters)
    body_color:    RGBA for the main body
    handle_color:  RGBA for the handle
    has_handle:    Whether to include a door handle
    has_freezer:   Whether to show a freezer divider line
    freezer_color: RGBA for the freezer divider strip
    freezer_frac:  Vertical fraction where the freezer line sits (from top)
    has_kickplate: Add a darker base kickplate
    has_panel:     Add an inset door panel
    has_top_vent:  Add a top vent strip
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

PLACEMENT = Placement.WALL

# Appliance colors
STAINLESS = (0.72, 0.72, 0.74, 1.0)
APPLIANCE_WHITE = (0.92, 0.91, 0.89, 1.0)
RETRO_MINT = (0.65, 0.85, 0.78, 1.0)
RETRO_RED = (0.75, 0.22, 0.20, 1.0)
RETRO_YELLOW = (0.92, 0.82, 0.35, 1.0)
RETRO_BLUE = (0.35, 0.55, 0.78, 1.0)
APPLIANCE_BLACK = (0.18, 0.18, 0.18, 1.0)
FREEZER_STRIP = (0.60, 0.60, 0.62, 1.0)


@dataclass(frozen=True)
class Params:
    width: float = 0.40
    depth: float = 0.35
    height: float = 0.85
    body_color: tuple[float, float, float, float] = APPLIANCE_WHITE
    handle_color: tuple[float, float, float, float] = METAL_GRAY
    has_handle: bool = True
    has_freezer: bool = True
    freezer_color: tuple[float, float, float, float] = FREEZER_STRIP
    freezer_frac: float = 0.30  # freezer is top 30%
    has_kickplate: bool = True
    has_panel: bool = True
    has_top_vent: bool = True


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a fridge (body + details + handles)."""
    prims: list[Prim] = []

    # Body: tall box sitting on floor
    body = Prim(
        GeomType.BOX,
        (params.width, params.depth, params.height),
        (0, 0, params.height),
        params.body_color,
    )
    prims.append(body)

    # Inset door panel to add depth
    if params.has_panel:
        panel = Prim(
            GeomType.BOX,
            (params.width * 0.92, params.depth * 0.90, params.height * 0.92),
            (0, params.depth * 0.08, params.height * 0.98),
            params.body_color,
        )
        prims.append(panel)

    # Freezer divider: thin strip across the front face
    if params.has_freezer:
        divider_z = params.height * 2 * (1.0 - params.freezer_frac)
        divider = Prim(
            GeomType.BOX,
            (params.width * 0.92, 0.005, 0.008),
            (0, params.depth + 0.005, divider_z),
            params.freezer_color,
        )
        prims.append(divider)

    # Base kickplate
    if params.has_kickplate:
        kick = Prim(
            GeomType.BOX,
            (params.width * 0.95, 0.02, 0.03),
            (0, params.depth + 0.02, 0.06),
            METAL_DARK,
        )
        prims.append(kick)

    # Top vent strip
    if params.has_top_vent:
        vent = Prim(
            GeomType.BOX,
            (params.width * 0.85, 0.008, 0.01),
            (0, params.depth + 0.01, params.height * 1.92),
            METAL_DARK,
        )
        prims.append(vent)

    # Handle: thin vertical bar on the front face
    if params.has_handle:
        # Lower door handle (main compartment)
        handle_half_z = params.height * 0.20
        handle_z = params.height * 0.65  # mid-height of lower section
        if params.has_freezer:
            # Position below the divider line
            divider_z_val = params.height * 2 * (1.0 - params.freezer_frac)
            handle_z = divider_z_val * 0.50
        handle = Prim(
            GeomType.BOX,
            (0.015, 0.025, handle_half_z),
            (params.width * 0.65, params.depth + 0.025, handle_z),
            params.handle_color,
        )
        prims.append(handle)

        # Upper handle (freezer section) -- smaller
        if params.has_freezer:
            freezer_mid_z = params.height * 2 * (1.0 - params.freezer_frac * 0.5)
            freezer_handle = Prim(
                GeomType.BOX,
                (0.015, 0.025, params.height * 0.08),
                (params.width * 0.65, params.depth + 0.025, freezer_mid_z),
                params.handle_color,
            )
            prims.append(freezer_handle)

    return tuple(prims)


# ---------------------------------------------------------------------------
# Named variations
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, Params] = {
    "standard white": Params(),
    "stainless steel": Params(
        body_color=STAINLESS,
        handle_color=METAL_CHROME,
        freezer_color=(0.65, 0.65, 0.67, 1.0),
    ),
    "mini fridge": Params(
        width=0.25,
        depth=0.25,
        height=0.45,
        body_color=PLASTIC_WHITE,
        handle_color=METAL_DARK,
        has_freezer=False,
    ),
    "double-door": Params(
        width=0.55,
        depth=0.38,
        height=0.90,
        body_color=STAINLESS,
        handle_color=METAL_CHROME,
        freezer_frac=0.25,
    ),
    "retro mint": Params(
        body_color=RETRO_MINT,
        handle_color=METAL_CHROME,
        freezer_color=(0.55, 0.75, 0.68, 1.0),
    ),
    "retro red": Params(
        body_color=RETRO_RED,
        handle_color=METAL_CHROME,
        freezer_color=(0.60, 0.18, 0.16, 1.0),
    ),
    "retro yellow": Params(
        body_color=RETRO_YELLOW,
        handle_color=METAL_CHROME,
        freezer_color=(0.82, 0.72, 0.30, 1.0),
    ),
    "black matte": Params(
        body_color=APPLIANCE_BLACK,
        handle_color=PLASTIC_BLACK,
        freezer_color=(0.25, 0.25, 0.25, 1.0),
    ),
}
