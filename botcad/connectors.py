"""Connector catalog with real-world dimensions and CAD geometry.

Physical connector housings used in robot wiring. Each connector type
gets accurate body dimensions and a simplified-but-recognizable solid
("recognizable from 3 feet away").

All dimensions in meters (SI), matching the rest of botcad.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from botcad.component import Vec3


class ConnectorType(StrEnum):
    MOLEX_5264_3PIN = "5264_3pin"
    CSI_15PIN = "csi_15pin"
    XT30 = "xt30"
    JST_XH_3PIN = "jst_xh_3pin"
    GPIO_2X20 = "gpio_2x20"


@dataclass(frozen=True)
class ConnectorSpec:
    """Physical connector housing specification."""

    connector_type: ConnectorType
    label: str
    body_dimensions: Vec3  # (x, y, z) housing size in meters
    wire_exit_direction: Vec3  # unit vec, cable leaves this way
    wire_exit_offset: Vec3  # from connector origin to cable start
    cable_bend_radius: float  # meters
    mating_direction: Vec3  # push direction to mate


# ── Connector Catalog ────────────────────────────────────────────────

_CATALOG: dict[str, ConnectorSpec] = {}


def _register(spec: ConnectorSpec) -> ConnectorSpec:
    _CATALOG[spec.connector_type.value] = spec
    return spec


# Molex 5264-2.54 3-pin (servo UART bus)
MOLEX_5264_3PIN = _register(
    ConnectorSpec(
        connector_type=ConnectorType.MOLEX_5264_3PIN,
        label="Molex 5264 3-pin",
        body_dimensions=(0.0075, 0.0045, 0.006),  # 7.5 x 4.5 x 6.0mm
        wire_exit_direction=(0.0, 0.0, -1.0),
        wire_exit_offset=(0.0, 0.0, -0.003),
        cable_bend_radius=0.005,
        mating_direction=(0.0, 0.0, 1.0),
    )
)

# CSI 15-pin FFC/FPC (camera ribbon cable)
CSI_15PIN = _register(
    ConnectorSpec(
        connector_type=ConnectorType.CSI_15PIN,
        label="CSI 15-pin FFC",
        body_dimensions=(0.022, 0.003, 0.0055),  # 22 x 3 x 5.5mm
        wire_exit_direction=(0.0, -1.0, 0.0),
        wire_exit_offset=(0.0, -0.0015, 0.0),
        cable_bend_radius=0.003,
        mating_direction=(0.0, 0.0, -1.0),
    )
)

# XT30 (battery power)
XT30 = _register(
    ConnectorSpec(
        connector_type=ConnectorType.XT30,
        label="XT30",
        body_dimensions=(0.0121, 0.0097, 0.0065),  # 12.1 x 9.7 x 6.5mm
        wire_exit_direction=(0.0, 0.0, -1.0),
        wire_exit_offset=(0.0, 0.0, -0.00325),
        cable_bend_radius=0.008,
        mating_direction=(1.0, 0.0, 0.0),
    )
)

# JST-XH 3-pin (battery balance lead)
JST_XH_3PIN = _register(
    ConnectorSpec(
        connector_type=ConnectorType.JST_XH_3PIN,
        label="JST-XH 3-pin",
        body_dimensions=(0.0075, 0.0061, 0.0083),  # 7.5 x 6.1 x 8.3mm
        wire_exit_direction=(0.0, 0.0, -1.0),
        wire_exit_offset=(0.0, 0.0, -0.00415),
        cable_bend_radius=0.005,
        mating_direction=(0.0, 0.0, 1.0),
    )
)

# GPIO 2x20 header (Raspberry Pi)
GPIO_2X20 = _register(
    ConnectorSpec(
        connector_type=ConnectorType.GPIO_2X20,
        label="GPIO 2x20",
        body_dimensions=(0.051, 0.005, 0.0085),  # 51 x 5 x 8.5mm
        wire_exit_direction=(0.0, 0.0, 1.0),
        wire_exit_offset=(0.0, 0.0, 0.00425),
        cable_bend_radius=0.010,
        mating_direction=(0.0, 0.0, -1.0),
    )
)


def connector_spec(connector_type: str) -> ConnectorSpec:
    """Look up a connector by type string.

    Raises KeyError if not found.
    """
    return _CATALOG[connector_type]


# ── CAD geometry builders ────────────────────────────────────────────


@lru_cache(maxsize=16)
def connector_solid(spec: ConnectorSpec):
    """Build a simplified connector housing solid.

    Centered at origin, correct proportions with retention features
    as bumps. No individual pins — "recognizable from 3 feet away."

    Returns a build123d Solid.
    """
    from build123d import Align, Box, Location

    from botcad.cad_utils import as_solid as _as_solid

    C = (Align.CENTER, Align.CENTER, Align.CENTER)
    bx, by, bz = spec.body_dimensions

    # Main housing body
    body = Box(bx, by, bz, align=C)

    if spec.connector_type == ConnectorType.MOLEX_5264_3PIN:
        # Retention bump on one side
        bump = Box(bx * 0.6, by * 0.15, bz * 0.3, align=C)
        bump = bump.locate(Location((0, by / 2, bz * 0.1)))
        body = body.fuse(bump)

    elif spec.connector_type == ConnectorType.CSI_15PIN:
        # ZIF lever on top
        lever = Box(bx * 0.9, by * 0.3, bz * 0.15, align=C)
        lever = lever.locate(Location((0, 0, bz / 2)))
        body = body.fuse(lever)

    elif spec.connector_type == ConnectorType.XT30:
        # Chamfered mating face (slight taper)
        chamfer_block = Box(bx * 0.1, by * 0.1, bz, align=C)
        chamfer_block = chamfer_block.locate(Location((bx / 2, by / 2, 0)))
        body = body - chamfer_block
        chamfer_block2 = Box(bx * 0.1, by * 0.1, bz, align=C)
        chamfer_block2 = chamfer_block2.locate(Location((bx / 2, -by / 2, 0)))
        body = body - chamfer_block2

    elif spec.connector_type == ConnectorType.JST_XH_3PIN:
        # Retention bump
        bump = Box(bx * 0.5, by * 0.12, bz * 0.25, align=C)
        bump = bump.locate(Location((0, by / 2, bz * 0.15)))
        body = body.fuse(bump)

    elif spec.connector_type == ConnectorType.GPIO_2X20:
        # Pin header rows (two ridges on bottom)
        row_w = bx * 0.95
        row_d = 0.001
        row_h = bz * 0.3
        for y_off in [-0.00127, 0.00127]:  # 2.54mm pitch / 2
            row = Box(
                row_w, row_d, row_h, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            row = row.locate(Location((0, y_off, -bz / 2)))
            body = body.fuse(row)

    return _as_solid(body)
