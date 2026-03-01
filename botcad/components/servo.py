"""Servo motor components with real-world specs."""

from __future__ import annotations

from botcad.component import MountPoint, ServoSpec, WirePort

# Feetech STS3215 — 30 kg-cm serial bus servo
# Datasheet: 45.2 x 24.7 x 35mm, 55g, 30 kg-cm @ 12V, 0.222s/60° no-load
# UART half-duplex (TTL), 25T spline output shaft
_STS3215_DIMS = (0.0452, 0.0247, 0.035)  # meters
_STS3215_MASS = 0.055  # kg
_STS3215_STALL_TORQUE = 2.942  # N-m (30 kg-cm)
_STS3215_NO_LOAD_SPEED = 4.71  # rad/s (0.222s/60° → 270°/s → 4.71 rad/s)
_STS3215_VOLTAGE = 12.0


def STS3215(continuous: bool = False) -> ServoSpec:
    """Feetech STS3215 serial bus servo.

    Args:
        continuous: If True, servo is in continuous rotation mode (for wheels).
    """
    range_rad = (-3.14159, 3.14159) if continuous else (-2.618, 2.618)  # ±150° std

    return ServoSpec(
        name="STS3215",
        dimensions=_STS3215_DIMS,
        mass=_STS3215_MASS,
        wire_ports=(
            WirePort("uart_in", pos=(-0.0226, 0.0, 0.0), bus_type="uart_half_duplex"),
            WirePort("uart_out", pos=(0.0226, 0.0, 0.0), bus_type="uart_half_duplex"),
        ),
        mounting_points=(
            # Four corner mounting holes (M2.5, 4mm from edges)
            MountPoint("m1", pos=(-0.019, -0.008, -0.0175), diameter=0.0025),
            MountPoint("m2", pos=(0.019, -0.008, -0.0175), diameter=0.0025),
            MountPoint("m3", pos=(-0.019, 0.008, -0.0175), diameter=0.0025),
            MountPoint("m4", pos=(0.019, 0.008, -0.0175), diameter=0.0025),
        ),
        color=(0.15, 0.15, 0.15, 1.0),
        stall_torque=_STS3215_STALL_TORQUE,
        no_load_speed=_STS3215_NO_LOAD_SPEED,
        voltage=_STS3215_VOLTAGE,
        bus_type="uart_half_duplex",
        # Shaft center is at top-center, 12mm from back face
        shaft_offset=(0.0, 0.012, 0.0175),
        shaft_axis=(0.0, 0.0, 1.0),
        range_rad=range_rad,
        gear_ratio=1.0,
        continuous=continuous,
    )
