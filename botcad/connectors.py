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
from botcad.units import Meters, Position, Size3D


class ConnectorType(StrEnum):
    MOLEX_5264_3PIN = "5264_3pin"
    CSI_15PIN = "csi_15pin"
    XT30 = "xt30"
    JST_XH_3PIN = "jst_xh_3pin"
    GPIO_2X20 = "gpio_2x20"
    USB_C = "usb_c"


@dataclass(frozen=True)
class ConnectorSpec:
    """Physical connector housing specification."""

    connector_type: ConnectorType
    label: str
    body_dimensions: Size3D  # (x, y, z) housing size in meters
    wire_exit_direction: Vec3  # unit vec, cable leaves this way
    wire_exit_offset: Position  # from connector origin to cable start
    cable_bend_radius: Meters  # meters
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


# USB Type-C receptacle
USB_C_SPEC = _register(
    ConnectorSpec(
        connector_type=ConnectorType.USB_C,
        label="USB Type-C",
        body_dimensions=(0.0084, 0.0026, 0.0065),  # 8.4 x 2.6 x 6.5mm
        wire_exit_direction=(-1.0, 0.0, 0.0),
        wire_exit_offset=(-0.0042, 0.0, 0.0),
        cable_bend_radius=0.008,
        mating_direction=(1.0, 0.0, 0.0),
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
    """Build a detailed connector plug/jack solid.

    Centered at origin with recognizable features: pin blades on the
    mating face, polarization keys, locking tabs, wire entry channels.
    Correct proportions from datasheets — "recognizable from 3 feet away."

    Returns a build123d Solid.
    """
    from build123d import Align, Box, Cylinder, Location

    from botcad.cad_utils import as_solid as _as_solid

    C = (Align.CENTER, Align.CENTER, Align.CENTER)
    bx, by, bz = spec.body_dimensions

    # Main housing body
    body = Box(bx, by, bz, align=C)

    if spec.connector_type == ConnectorType.MOLEX_5264_3PIN:
        # Polarization rib on one side (asymmetric key)
        rib = Box(bx * 0.15, by * 0.12, bz * 0.7, align=C)
        rib = rib.moved(Location((-bx * 0.35, by / 2, -bz * 0.05)))
        body = body.fuse(rib)

        # Locking tab on opposite side
        tab = Box(bx * 0.4, by * 0.08, bz * 0.15, align=C)
        tab = tab.moved(Location((0, -by / 2, bz * 0.25)))
        body = body.fuse(tab)

        # 3 pin blades protruding from mating face (+Z)
        pin_w = 0.0006  # 0.6mm wide blade
        pin_d = 0.0002  # 0.2mm thick
        pin_h = 0.002  # 2mm protrusion
        pitch = 0.00254  # 2.54mm pitch
        for i in range(3):
            px = -pitch + i * pitch
            pin = Box(
                pin_w, pin_d, pin_h, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            pin = pin.moved(Location((px, 0, bz / 2)))
            body = body.fuse(pin)

        # Wire entry channels on back face (-Z) — 3 circular dips
        for i in range(3):
            px = -pitch + i * pitch
            chan = Cylinder(0.0005, by * 0.3, align=C)
            chan = chan.moved(Location((px, 0, -bz / 2)))
            body = body - chan

    elif spec.connector_type == ConnectorType.CSI_15PIN:
        # ZIF lever on top (hinged flap)
        lever_h = bz * 0.2
        lever = Box(bx * 0.92, by * 0.6, lever_h, align=C)
        lever = lever.moved(Location((0, 0, bz / 2 + lever_h / 2 - 0.0002)))
        body = body.fuse(lever)

        # Ribbon cable slot on bottom face — wide thin opening
        slot = Box(bx * 0.85, by * 0.4, bz * 0.3, align=C)
        slot = slot.moved(Location((0, -by * 0.1, -bz * 0.35)))
        body = body - slot

        # Contact ridges inside (visible through slot)
        for i in range(7):
            rx = -bx * 0.38 + i * bx * 0.76 / 6
            ridge = Box(0.0003, by * 0.2, bz * 0.15, align=C)
            ridge = ridge.moved(Location((rx, -by * 0.1, -bz * 0.15)))
            body = body.fuse(ridge)

    elif spec.connector_type == ConnectorType.XT30:
        # Keying: one corner chamfered (polarization)
        chamfer_block = Box(bx * 0.15, by * 0.15, bz + 0.001, align=C)
        chamfer_block = chamfer_block.moved(Location((bx / 2, by / 2, 0)))
        body = body - chamfer_block

        # Two pin sockets on mating face (+X) — round and square
        # XT30 has one round pin and one flat pin for polarity
        round_pin = Cylinder(
            0.001, 0.003, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
        round_pin = round_pin.moved(Location((bx / 2, -by * 0.2, 0)))
        body = body.fuse(round_pin)

        flat_pin = Box(
            0.002, 0.001, 0.003, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
        flat_pin = flat_pin.moved(Location((bx / 2, by * 0.2, 0)))
        body = body.fuse(flat_pin)

        # Grip ridges on sides
        for i in range(3):
            gz = -bz * 0.3 + i * bz * 0.3
            ridge = Box(bx * 0.8, by * 0.06, 0.0005, align=C)
            ridge = ridge.moved(Location((0, by / 2, gz)))
            body = body.fuse(ridge)
            ridge2 = Box(bx * 0.8, by * 0.06, 0.0005, align=C)
            ridge2 = ridge2.moved(Location((0, -by / 2, gz)))
            body = body.fuse(ridge2)

    elif spec.connector_type == ConnectorType.JST_XH_3PIN:
        # Locking tab (sprung clip on one side)
        tab_w = bx * 0.35
        tab_h = bz * 0.12
        tab_d = by * 0.08
        tab = Box(tab_w, tab_d, tab_h, align=C)
        tab = tab.moved(Location((0, by / 2 + tab_d / 2, bz * 0.2)))
        body = body.fuse(tab)

        # Polarization groove on housing
        groove = Box(bx * 0.12, by + 0.001, bz * 0.15, align=C)
        groove = groove.moved(Location((-bx * 0.35, 0, bz * 0.3)))
        body = body - groove

        # 3 pin blades on mating face (+Z)
        pitch = 0.0025  # 2.5mm XH pitch
        pin_w = 0.0006
        pin_d = 0.0002
        pin_h = 0.002
        for i in range(3):
            px = -pitch + i * pitch
            pin = Box(
                pin_w, pin_d, pin_h, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            pin = pin.moved(Location((px, 0, bz / 2)))
            body = body.fuse(pin)

        # Wire entry on back face (-Z) — strain relief bumps
        for i in range(3):
            px = -pitch + i * pitch
            bump = Cylinder(
                0.0006, 0.001, align=(Align.CENTER, Align.CENTER, Align.MAX)
            )
            bump = bump.moved(Location((px, 0, -bz / 2)))
            body = body.fuse(bump)

    elif spec.connector_type == ConnectorType.USB_C:
        # Rounded-rectangle tongue on mating face (+X)
        # USB-C: bx=8.4mm (long), by=2.6mm (short), bz=6.5mm
        tongue_d = 0.002  # 2mm protrusion along X (mating axis)
        tongue = Box(
            tongue_d,
            by * 0.6,
            bz * 0.5,
            align=C,
        )
        tongue = tongue.moved(Location((bx / 2 + tongue_d / 2, 0, 0)))
        body = body.fuse(tongue)

        # Receptacle cavity on mating face
        cavity = Box(0.003, by * 0.4, bz * 0.4, align=C)
        cavity = cavity.moved(Location((bx / 2 - 0.001, 0, 0)))
        body = body - cavity

    elif spec.connector_type == ConnectorType.GPIO_2X20:
        # 2x20 socket connector (female) — shroud with pin cavities
        # Keying notch on one side
        notch = Box(0.003, by + 0.001, bz * 0.4, align=C)
        notch = notch.moved(Location((bx / 2 - 0.001, 0, bz * 0.15)))
        body = body - notch

        # Two rows of socket holes on mating face (-Z) — batch into one cut
        pitch = 0.00254
        holes = None
        for row_y in [-0.00127, 0.00127]:
            for i in range(20):
                px = -pitch * 9.5 + i * pitch
                hole = Cylinder(
                    0.0004,
                    bz * 0.5,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                )
                hole = hole.moved(Location((px, row_y, -bz / 2)))
                holes = hole if holes is None else holes.fuse(hole)
        body = body - holes

        # Strain relief cable exit on top
        cable_exit = Box(bx * 0.8, by * 0.6, 0.002, align=C)
        cable_exit = cable_exit.moved(Location((0, 0, bz / 2 + 0.001)))
        body = body.fuse(cable_exit)

    return _as_solid(body)


@lru_cache(maxsize=16)
def receptacle_solid(spec: ConnectorSpec):
    """Build a simplified receptacle (PCB header / panel socket) solid.

    The mating counterpart to connector_solid(). Slightly larger housing
    with an open cavity facing the mating direction where the plug inserts.

    Centered at origin. Returns a build123d Solid.
    """
    from build123d import Align, Box, Cylinder, Location

    from botcad.cad_utils import as_solid as _as_solid

    C = (Align.CENTER, Align.CENTER, Align.CENTER)
    bx, by, bz = spec.body_dimensions

    # Receptacle is slightly larger than the plug
    wall = 0.0008  # 0.8mm housing wall
    rx, ry, rz = bx + 2 * wall, by + 2 * wall, bz * 0.7

    body = Box(rx, ry, rz, align=C)

    # Cavity facing the mating direction
    mx, my, mz = spec.mating_direction
    cavity_depth = rz * 0.6
    cavity = Box(bx + 0.0002, by + 0.0002, cavity_depth, align=C)

    # Position cavity at the mating face
    if abs(mz) > 0.5:
        cz = (rz / 2 - cavity_depth / 2) * (1 if mz > 0 else -1)
        cavity = cavity.moved(Location((0, 0, cz)))
    elif abs(mx) > 0.5:
        cavity = Box(cavity_depth, by + 0.0002, bz * 0.5, align=C)
        cx = (rx / 2 - cavity_depth / 2) * (1 if mx > 0 else -1)
        cavity = cavity.moved(Location((cx, 0, 0)))
    elif abs(my) > 0.5:
        cavity = Box(bx + 0.0002, cavity_depth, bz * 0.5, align=C)
        cy = (ry / 2 - cavity_depth / 2) * (1 if my > 0 else -1)
        cavity = cavity.moved(Location((0, cy, 0)))

    body = body - cavity

    if spec.connector_type == ConnectorType.GPIO_2X20:
        # Shroud walls around pin header
        shroud_h = rz * 0.4
        shroud = Box(rx, ry, shroud_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        shroud = shroud.moved(Location((0, 0, rz / 2)))
        inner = Box(
            bx - 0.001,
            by - 0.001,
            shroud_h + 0.001,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        inner = inner.moved(Location((0, 0, rz / 2)))
        shroud = shroud - inner
        body = body.fuse(shroud)

    elif spec.connector_type == ConnectorType.XT30:
        # Panel mount flange
        flange_w = rx + 0.004
        flange_h = ry + 0.004
        flange_t = 0.001
        flange = Box(flange_w, flange_h, flange_t, align=C)
        # Position at the back face (opposite mating direction)
        fx = (
            (-rx / 2 - flange_t / 2)
            if mx > 0
            else (rx / 2 + flange_t / 2)
            if mx < 0
            else 0
        )
        flange = flange.moved(Location((fx, 0, 0)))
        body = body.fuse(flange)

    # Solder pins on bottom
    pin_h = 0.003
    n_pins = max(2, int(bx / 0.00254))  # ~2.54mm pitch
    if spec.connector_type != ConnectorType.GPIO_2X20:
        for i in range(min(n_pins, 6)):
            px = -bx / 2 + bx * (i + 0.5) / min(n_pins, 6)
            pin = Cylinder(0.0003, pin_h, align=(Align.CENTER, Align.CENTER, Align.MAX))
            pin = pin.moved(Location((px, 0, -rz / 2)))
            body = body.fuse(pin)

    return _as_solid(body)
