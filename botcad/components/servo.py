"""Servo motor components with real-world specs.

Geometry extracted from the official Feetech STS3215 STEP model.
Reference: https://github.com/sk-t/so-arm100/blob/main/STEP/SO100/STS3215_03a.step
Local: references/components/STS3215_03a.step
Datasheet: https://core-electronics.com.au/files/attachments/STS3215%20Serial%20Bus%20Servo%20User%20Manual.pdf

STS3250 datasheet: https://ecksteinimg.de/Datasheet/Feetech/FT01021.pdf
SCS0009 product page: https://www.feetechrc.com/6v-23kg-serial-bus-steering-gear_65522.html

Servo local frame convention:
    X = long axis (length direction)
    Y = width axis
    Z = shaft axis (shaft protrudes in +Z)
    Origin = geometric center of the main body (not including ears/horn)
"""

from __future__ import annotations

import math

from botcad.colors import COLOR_STRUCTURE_DARK
from botcad.component import BusType, MountingEar, MountPoint, ServoSpec, WirePort

# ── Feetech STS3215 (C018, 12V variant) ─────────────────────────────
#
# Datasheet: 45.2 x 24.7 x 35mm (with ears), 55g, 30 kg-cm @ 12V
# 0.222s/60° no-load, UART half-duplex (TTL), 25T spline output shaft
#
# STEP model body spans (mm):
#   X: -22.7 to +22.7 (45.4mm, long axis)
#   Y: -12.4 to +12.4 (24.8mm, width)
#   Z: -15.9 to +15.9 (31.8mm, body only)
#
# Mounting flanges extend 3.2mm below body bottom (35.0 - 31.8 = 3.2mm)
#   Flange bottom at Z = -19.1mm, screw holes at Z ≈ -17.5mm
#   6x M3 clearance holes (ø3.2mm) at X: +17.0, 0.0, -17.0mm
#
# Output shaft center at X=+12.5, Y=0, Z=+15.9 (body top face)
# Rear (blind) shaft at same XY, support bearing in -Z
# Connector (5264-2.54 3-pin) on bottom face, toward -X (back) end

# Overall envelope (with ears and shaft boss, per refined CAD)
_STS3215_DIMS = (0.0454, 0.0248, 0.0379)

# Main body only (no ears, no horn — the solid rectangular block)
_STS3215_BODY_DIMS = (0.0454, 0.0248, 0.0326)

_STS3215_MASS = 0.055  # kg
_STS3215_STALL_TORQUE = 2.942  # N-m (30 kg-cm @ 12V)
_STS3215_NO_LOAD_SPEED_60DEG_S = 0.222  # s/60° @ 12V
_STS3215_NO_LOAD_SPEED = (math.pi / 3) / _STS3215_NO_LOAD_SPEED_60DEG_S  # rad/s
_STS3215_GEAR_RATIO = 345.0  # 1:345 total reduction
_STS3215_TYPICAL_CURRENT = 0.180  # A (no-load current at 12V)
_STS3215_VOLTAGE = 12.0

# Output shaft offset from body center (meters)
# Shaft is at X=+12.5mm along the long axis, at the +Z face (16.3mm from center)
_STS3215_SHAFT_OFFSET = (0.0125, 0.0, 0.0163)

# Horn mount XY coordinates (4x M2.5 pattern around shaft center)
# Spacing: 9.9mm x 9.9mm; shared between front horn and rear horn
_HORN_XY = (
    (+0.00755, +0.00495),
    (+0.01745, +0.00495),
    (+0.00755, -0.00495),
    (+0.01745, -0.00495),
)
_HORN_FRONT_Z = +0.0180  # output face (1.7mm above body top)
_HORN_REAR_Z = -0.0176  # blind/rear face (1.3mm below body bottom)
_HORN_HOLE_DIA = 0.0025  # M2.5


def STS3215(continuous: bool = False) -> ServoSpec:
    """Feetech STS3215 serial bus servo (C018, 12V 30kg-cm).

    Dual-axis design with 25T spline output shaft and blind support shaft.
    Mounting ears on the body sides with M3 clearance holes for bracket
    attachment.  5264-2.54 3-pin daisy-chain connector on the bottom face.

    Args:
        continuous: If True, servo is in continuous rotation mode (for wheels).
    """
    range_rad = (
        -math.pi,
        math.pi,
    )  # STS3215 runs 360° (position mode reports 0–4096 over one rotation)

    return ServoSpec(
        name="STS3215",
        dimensions=_STS3215_DIMS,
        mass=_STS3215_MASS,
        wire_ports=(
            # 5264-2.54 3-pin connector on bottom face, toward back (-X) end
            # Also has a short 5264/2.54 pigtail lead (~15cm)
            WirePort(
                "uart_bus",
                pos=(-0.0080, 0.0, -0.0163),
                bus_type=BusType.UART_HALF_DUPLEX,
                connector_type="5264_3pin",
            ),
        ),
        mounting_points=tuple(
            MountPoint(
                f"horn_{i + 1}", pos=(x, y, _HORN_FRONT_Z), diameter=_HORN_HOLE_DIA
            )
            for i, (x, y) in enumerate(_HORN_XY)
        ),
        color=COLOR_STRUCTURE_DARK.rgba,
        stall_torque=_STS3215_STALL_TORQUE,
        no_load_speed=_STS3215_NO_LOAD_SPEED,
        voltage=_STS3215_VOLTAGE,
        typical_current=_STS3215_TYPICAL_CURRENT,
        bus_type=BusType.UART_HALF_DUPLEX,
        shaft_offset=_STS3215_SHAFT_OFFSET,
        shaft_axis=(0.0, 0.0, 1.0),
        range_rad=range_rad,
        gear_ratio=_STS3215_GEAR_RATIO,
        continuous=continuous,
        # Extended geometry
        body_dimensions=_STS3215_BODY_DIMS,
        shaft_boss_radius=0.0045,  # 4.5mm radius
        shaft_boss_height=0.0032,  # 3.2mm protrusion above body top
        mounting_ears=(
            # 6x M3 clearance holes (ø3.2mm) in mounting flanges below body
            # Flanges extend 2.1mm below body bottom (Z=-16.3mm)
            # Screw hole centers at Z=-17.35mm (midpoint of flange)
            # Y=±10.25mm (inside body width), symmetric X positions
            MountingEar(
                "ear_1", pos=(+0.0170, -0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_2", pos=(0.0000, -0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_3", pos=(+0.0170, +0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_4", pos=(0.0000, +0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_5", pos=(-0.0170, -0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_6", pos=(-0.0170, +0.01025, -0.01735), hole_diameter=0.0032
            ),
        ),
        horn_mounting_points=tuple(
            MountPoint(
                f"out_{i + 1}",
                pos=(x, y, _HORN_FRONT_Z),
                diameter=_HORN_HOLE_DIA,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M2.5",
            )
            for i, (x, y) in enumerate(_HORN_XY)
        ),
        rear_horn_mounting_points=tuple(
            MountPoint(
                f"rear_{i + 1}",
                pos=(x, y, _HORN_REAR_Z),
                diameter=_HORN_HOLE_DIA,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            )
            for i, (x, y) in enumerate(_HORN_XY)
        ),
        # 5264-2.54 3-pin connector on bottom face, toward back (-X) end
        connector_pos=(-0.0080, 0.0, -0.0163),
    )


# ── Feetech STS3250 (HLS3950M / ST-3250-C001, 12V variant) ──────────
#
# Identical form factor to the STS3215 — same outer dimensions, same
# mounting pattern, same horn/accessory.  Internal differences: coreless
# DC motor, steel gears, higher torque (50 kg-cm vs 30 kg-cm) and
# heavier (74.5g vs 55g).
#
# Datasheet: HLS3950M-C001 Edition A/0 (2024-04-25)
# Dimensions: 45.22 x 24.72 x 35mm (with ears) — matches STS3215 envelope
# Weight: 74.5 ± 1g
# Stall torque: 50 kg-cm @ 12V (4.905 N-m)
# No-load speed: 0.133s/60° @ 12V (75 RPM)
# Gear ratio: 1:345, steel gears, coreless motor, ball bearings
# Encoder: 12-bit / 4096 counts
# Protocol: UART half-duplex TTL, same as STS3215

_STS3250_MASS = 0.0745  # kg
_STS3250_STALL_TORQUE = 4.905  # N-m (50 kg-cm @ 12V)
_STS3250_NO_LOAD_SPEED_60DEG_S = 0.133  # s/60° @ 12V
_STS3250_NO_LOAD_SPEED = (math.pi / 3) / _STS3250_NO_LOAD_SPEED_60DEG_S  # rad/s
_STS3250_TYPICAL_CURRENT = 0.330  # A (no-load current at 12V)


def STS3250(continuous: bool = False) -> ServoSpec:
    """Feetech STS3250 serial bus servo (12V 50kg-cm).

    Same form factor as the STS3215 but with a coreless motor and steel
    gears — higher torque (50 vs 30 kg-cm) and heavier (74.5g vs 55g).
    Dual-axis 25T spline output shaft, M3 mounting ears, 5264-2.54 3-pin
    daisy-chain connector.

    Args:
        continuous: If True, servo is in continuous rotation mode (for wheels).
    """
    range_rad = (-math.pi, math.pi)  # 360° (12-bit encoder, 4096 counts)

    # Geometry is identical to STS3215 — reuse all dimensional constants.
    return ServoSpec(
        name="STS3250",
        dimensions=_STS3215_DIMS,
        mass=_STS3250_MASS,
        wire_ports=(
            WirePort(
                "uart_bus",
                pos=(-0.0080, 0.0, -0.0163),
                bus_type=BusType.UART_HALF_DUPLEX,
                connector_type="5264_3pin",
            ),
        ),
        mounting_points=tuple(
            MountPoint(
                f"horn_{i + 1}", pos=(x, y, _HORN_FRONT_Z), diameter=_HORN_HOLE_DIA
            )
            for i, (x, y) in enumerate(_HORN_XY)
        ),
        color=COLOR_STRUCTURE_DARK.rgba,
        stall_torque=_STS3250_STALL_TORQUE,
        no_load_speed=_STS3250_NO_LOAD_SPEED,
        voltage=_STS3215_VOLTAGE,  # same 12V
        typical_current=_STS3250_TYPICAL_CURRENT,
        bus_type=BusType.UART_HALF_DUPLEX,
        shaft_offset=_STS3215_SHAFT_OFFSET,
        shaft_axis=(0.0, 0.0, 1.0),
        range_rad=range_rad,
        gear_ratio=_STS3215_GEAR_RATIO,  # same 1:345
        continuous=continuous,
        # Extended geometry — identical to STS3215
        body_dimensions=_STS3215_BODY_DIMS,
        shaft_boss_radius=0.0045,
        shaft_boss_height=0.0032,
        mounting_ears=(
            MountingEar(
                "ear_1", pos=(+0.0170, -0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_2", pos=(0.0000, -0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_3", pos=(+0.0170, +0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_4", pos=(0.0000, +0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_5", pos=(-0.0170, -0.01025, -0.01735), hole_diameter=0.0032
            ),
            MountingEar(
                "ear_6", pos=(-0.0170, +0.01025, -0.01735), hole_diameter=0.0032
            ),
        ),
        horn_mounting_points=tuple(
            MountPoint(
                f"out_{i + 1}",
                pos=(x, y, _HORN_FRONT_Z),
                diameter=_HORN_HOLE_DIA,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M2.5",
            )
            for i, (x, y) in enumerate(_HORN_XY)
        ),
        rear_horn_mounting_points=tuple(
            MountPoint(
                f"rear_{i + 1}",
                pos=(x, y, _HORN_REAR_Z),
                diameter=_HORN_HOLE_DIA,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2.5",
            )
            for i, (x, y) in enumerate(_HORN_XY)
        ),
        connector_pos=(-0.0080, 0.0, -0.0163),
    )


# ── Feetech SCS0009 (SC-0090-C001, 6V micro servo) ──────────────────
#
# Micro servo form factor (~SG90 size). Single output shaft (no blind
# rear shaft). PC plastic case, copper+steel gears, cored motor.
#
# Datasheet dims: 23.2 x 12.1 x 25.25mm (overall envelope with ears)
# Weight: 13.2 ± 1g
# Stall torque: 2.3 kg-cm @ 6V (0.226 N-m)
# No-load speed: 0.10s/60° @ 6V
# Angular range: 300° (0-1024, 10-bit)
# Shaft: 20T spline, OD 3.95mm
# Protocol: UART half-duplex TTL (same bus as STS series)
#
# Micro servo body (standard SG90 form factor):
#   Body: 23.2 x 12.1 x 22.5mm
#   Mounting ears: side tabs protruding ±4.65mm in X beyond body, 2.5mm
#     thick in Z, positioned ~7.75mm below body top.  Tabs span 32.5mm
#     total in X.  2x M2.0 screw holes per tab, ~28mm apart in X.
#   Shaft: 20T spline, OD 3.95mm, offset +5.8mm from body center in X,
#     at +Z face.

# Overall envelope (with ears and shaft boss)
_SCS0009_DIMS = (0.0325, 0.0121, 0.02525)  # 32.5mm with ear tabs

# Main body only (no ears/horn)
_SCS0009_BODY_DIMS = (0.0232, 0.0121, 0.0225)

_SCS0009_MASS = 0.0132  # kg
_SCS0009_STALL_TORQUE = 0.226  # N-m (2.3 kg-cm @ 6V)
_SCS0009_NO_LOAD_SPEED_60DEG_S = 0.10  # s/60° @ 6V
_SCS0009_NO_LOAD_SPEED = (math.pi / 3) / _SCS0009_NO_LOAD_SPEED_60DEG_S  # rad/s
_SCS0009_GEAR_RATIO = 1.0  # not published by Feetech
_SCS0009_TYPICAL_CURRENT = 0.150  # A (no-load current at 6V)
_SCS0009_VOLTAGE = 6.0

# Output shaft center offset from body center (meters)
# Shaft is offset toward +X end: ~5.8mm from body center, at top face
_SCS0009_SHAFT_OFFSET = (0.0058, 0.0, 0.01125)

# Horn mount XY coordinates (4x M2 pattern around shaft center)
# Micro servo horn: ~8mm bolt circle, 4 holes at 90° spacing
_SCS0009_HORN_XY = (
    (+0.0058 + 0.004, +0.004),
    (+0.0058 + 0.004, -0.004),
    (+0.0058 - 0.004, +0.004),
    (+0.0058 - 0.004, -0.004),
)
_SCS0009_HORN_FRONT_Z = +0.01325  # above body top
_SCS0009_HORN_HOLE_DIA = 0.0020  # M2


def SCS0009(continuous: bool = False) -> ServoSpec:
    """Feetech SCS0009 micro serial bus servo (6V 2.3kg-cm).

    Standard micro servo form factor (SG90-size).  Single output shaft
    with 20T spline.  PC plastic case.  Same UART half-duplex protocol
    as the STS series — can share the same bus.

    Note: 300° angular range (10-bit, 0-1024 steps).

    Args:
        continuous: If True, servo is in continuous rotation mode.
    """
    range_rad = (
        -5 * math.pi / 6,
        5 * math.pi / 6,
    )  # 300° total (±150° from center)

    return ServoSpec(
        name="SCS0009",
        dimensions=_SCS0009_DIMS,
        mass=_SCS0009_MASS,
        wire_ports=(
            WirePort(
                "uart_bus",
                pos=(-0.0058, 0.0, -0.01125),
                bus_type=BusType.UART_HALF_DUPLEX,
                connector_type="5264_3pin",
            ),
        ),
        mounting_points=tuple(
            MountPoint(
                f"horn_{i + 1}",
                pos=(x, y, _SCS0009_HORN_FRONT_Z),
                diameter=_SCS0009_HORN_HOLE_DIA,
            )
            for i, (x, y) in enumerate(_SCS0009_HORN_XY)
        ),
        color=COLOR_STRUCTURE_DARK.rgba,
        stall_torque=_SCS0009_STALL_TORQUE,
        no_load_speed=_SCS0009_NO_LOAD_SPEED,
        voltage=_SCS0009_VOLTAGE,
        typical_current=_SCS0009_TYPICAL_CURRENT,
        bus_type=BusType.UART_HALF_DUPLEX,
        shaft_offset=_SCS0009_SHAFT_OFFSET,
        shaft_axis=(0.0, 0.0, 1.0),
        range_rad=range_rad,
        gear_ratio=_SCS0009_GEAR_RATIO,
        continuous=continuous,
        # Extended geometry
        body_dimensions=_SCS0009_BODY_DIMS,
        shaft_boss_radius=0.00198,  # 1.98mm (half of 3.95mm spline OD)
        shaft_boss_height=0.00200,  # ~2mm protrusion above body
        mounting_ears=(
            # 4x M2.0 holes in side-mounted ear tabs (SG90-style).
            # Ears protrude in ±X beyond body at Z ≈ +3.5mm (7.75mm
            # below body top).  Holes at ±14mm in X (28mm apart),
            # through the tab thickness in Z.
            MountingEar(
                "ear_1",
                pos=(+0.0140, 0.0, 0.0035),
                hole_diameter=0.0020,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            MountingEar(
                "ear_2",
                pos=(-0.0140, 0.0, 0.0035),
                hole_diameter=0.0020,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
        ),
        horn_mounting_points=tuple(
            MountPoint(
                f"out_{i + 1}",
                pos=(x, y, _SCS0009_HORN_FRONT_Z),
                diameter=_SCS0009_HORN_HOLE_DIA,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M2",
            )
            for i, (x, y) in enumerate(_SCS0009_HORN_XY)
        ),
        # Single-axis design — no rear horn/blind shaft
        rear_horn_mounting_points=(),
        connector_pos=(-0.0058, 0.0, -0.01125),
    )
