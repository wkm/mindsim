"""Servo motor components with real-world specs.

Geometry extracted from the official Feetech STS3215 STEP model
(SO-ARM100 repo: STEP/SO100/STS3215_03a.step) and the translated
datasheet (core-electronics.com.au).

Servo local frame convention:
    X = long axis (length direction, 45.2mm)
    Y = width axis (24.7mm)
    Z = shaft axis (shaft protrudes in +Z)
    Origin = geometric center of the main body (not including ears/horn)
"""

from __future__ import annotations

from botcad.component import MountingEar, MountPoint, ServoSpec, WirePort

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
#   6x M3 clearance holes (ø4.2mm from STEP) at X: +4.2, -16.5, -20.3mm
#
# Output shaft center at X=+12.5, Y=0, Z=+15.9 (body top face)
# Rear (blind) shaft at same XY, support bearing in -Z
# Connector (PA2.0 4-pin) on bottom face, toward -X (back) end

# Overall envelope (with ears, per datasheet)
_STS3215_DIMS = (0.0452, 0.0247, 0.035)

# Main body only (no ears, no horn — the solid rectangular block)
_STS3215_BODY_DIMS = (0.0454, 0.0248, 0.0318)

_STS3215_MASS = 0.055  # kg
_STS3215_STALL_TORQUE = 2.942  # N-m (30 kg-cm @ 12V)
_STS3215_NO_LOAD_SPEED = 4.71  # rad/s (0.222s/60° → 270°/s)
_STS3215_VOLTAGE = 12.0

# Output shaft offset from body center (meters)
# Shaft is at X=+12.5mm along the long axis, at the +Z face
_STS3215_SHAFT_OFFSET = (0.0125, 0.0, 0.0159)


def STS3215(continuous: bool = False) -> ServoSpec:
    """Feetech STS3215 serial bus servo (C018, 12V 30kg-cm).

    Dual-axis design with 25T spline output shaft and blind support shaft.
    Mounting ears on the body sides with M3 clearance holes for bracket
    attachment.  PA2.0 4-pin daisy-chain connector on the bottom face.

    Args:
        continuous: If True, servo is in continuous rotation mode (for wheels).
    """
    range_rad = (-3.14159, 3.14159) if continuous else (-2.618, 2.618)  # ±150° std

    return ServoSpec(
        name="STS3215",
        dimensions=_STS3215_DIMS,
        mass=_STS3215_MASS,
        wire_ports=(
            # PA2.0 4-pin connector on bottom face, toward back (-X) end
            # Also has 5264/2.54 3P terminal nearby; 15cm cable
            WirePort(
                "uart_bus",
                pos=(-0.0080, 0.0, -0.0159),
                bus_type="uart_half_duplex",
            ),
        ),
        mounting_points=(
            # Horn mounting holes — 4x M2.5 (ø2.5mm) on the output face
            # Pattern centered on shaft at X=+12.5mm, Y=0, Z=+18.7mm
            # Spacing: 9.9mm x 9.9mm; positions in body-center coords
            MountPoint("horn_1", pos=(+0.00755, +0.00495, +0.0187), diameter=0.0025),
            MountPoint("horn_2", pos=(+0.01745, +0.00495, +0.0187), diameter=0.0025),
            MountPoint("horn_3", pos=(+0.00755, -0.00495, +0.0187), diameter=0.0025),
            MountPoint("horn_4", pos=(+0.01745, -0.00495, +0.0187), diameter=0.0025),
        ),
        color=(0.15, 0.15, 0.15, 1.0),
        stall_torque=_STS3215_STALL_TORQUE,
        no_load_speed=_STS3215_NO_LOAD_SPEED,
        voltage=_STS3215_VOLTAGE,
        bus_type="uart_half_duplex",
        shaft_offset=_STS3215_SHAFT_OFFSET,
        shaft_axis=(0.0, 0.0, 1.0),
        range_rad=range_rad,
        gear_ratio=1.0,
        continuous=continuous,
        # Extended geometry
        body_dimensions=_STS3215_BODY_DIMS,
        mounting_ears=(
            # 6x M3 clearance holes (ø4.2mm) in mounting flanges below body
            # Flanges extend 3.2mm below body bottom (Z=-15.9mm)
            # Screw hole centers at Z=-17.5mm (midpoint of flange)
            # Y=±10.25mm (inside body width), 3 X positions per side
            MountingEar(
                "ear_1", pos=(+0.0042, -0.01025, -0.0175), hole_diameter=0.0042
            ),
            MountingEar(
                "ear_2", pos=(-0.0165, -0.01025, -0.0175), hole_diameter=0.0042
            ),
            MountingEar(
                "ear_3", pos=(+0.0042, +0.01025, -0.0175), hole_diameter=0.0042
            ),
            MountingEar(
                "ear_4", pos=(-0.0165, +0.01025, -0.0175), hole_diameter=0.0042
            ),
            # Additional pair at far back end (X=-20.3mm)
            MountingEar(
                "ear_5", pos=(-0.0203, -0.01025, -0.0175), hole_diameter=0.0042
            ),
            MountingEar(
                "ear_6", pos=(-0.0203, +0.01025, -0.0175), hole_diameter=0.0042
            ),
        ),
        horn_mounting_points=(
            # Same as mounting_points above (4x M2.5 on output side)
            # Positions in body-center coords, centered on shaft at X=+12.5mm
            MountPoint("out_1", pos=(+0.00755, +0.00495, +0.0187), diameter=0.0025),
            MountPoint("out_2", pos=(+0.01745, +0.00495, +0.0187), diameter=0.0025),
            MountPoint("out_3", pos=(+0.00755, -0.00495, +0.0187), diameter=0.0025),
            MountPoint("out_4", pos=(+0.01745, -0.00495, +0.0187), diameter=0.0025),
        ),
        rear_horn_mounting_points=(
            # 4x M2.5 on blind/rear side, same XY pattern as front horn
            # At Z=-19.0mm (below body bottom, where rear support bracket attaches)
            MountPoint("rear_1", pos=(+0.00755, +0.00495, -0.0190), diameter=0.0025),
            MountPoint("rear_2", pos=(+0.01745, +0.00495, -0.0190), diameter=0.0025),
            MountPoint("rear_3", pos=(+0.00755, -0.00495, -0.0190), diameter=0.0025),
            MountPoint("rear_4", pos=(+0.01745, -0.00495, -0.0190), diameter=0.0025),
        ),
        # PA2.0 connector on bottom face, toward back (-X) end
        connector_pos=(-0.0080, 0.0, -0.0159),
    )
