"""Brushless DC motor components with real-world specs.

Motor local frame convention:
    Z = shaft axis (shaft protrudes in +Z)
    X, Y = radial axes
    Origin = center of motor can base (mount face)
"""

from __future__ import annotations

from botcad.component import BusType, MotorSpec, WirePort
from botcad.materials import MAT_ABS_DARK
from botcad.units import Amps, Meters, Volts, grams, mm

# ── Emax MT2213 (935KV, common RC plane outrunner) ──────────────────
#
# Lightweight outrunner brushless motor for small fixed-wing planes.
# Reference: Emax MT2213 datasheet
# Dimensions: 27.9mm diameter x 26mm can length (without shaft)
# Weight: 56g (with cables)
# KV: 935 RPM/V
# Max continuous current: 15A
# Recommended battery: 2S-3S LiPo
# Recommended prop: 8x4.5 to 10x4.5
# Shaft diameter: 3mm
# Mounting pattern: M3 x 16mm / M3 x 19mm

_MT2213_CAN_DIA = mm(27.9)
_MT2213_CAN_LENGTH = mm(26.0)
_MT2213_SHAFT_DIA = mm(3.0)
_MT2213_SHAFT_LENGTH = mm(12.0)
_MT2213_TOTAL_LENGTH = Meters(_MT2213_CAN_LENGTH + _MT2213_SHAFT_LENGTH)
_MT2213_MASS = grams(56)


def MT2213() -> MotorSpec:
    """Emax MT2213 935KV brushless outrunner motor.

    Common choice for small fixed-wing RC planes (800-1200mm wingspan).
    Paired with 8x4.5 to 10x4.5 propellers on 3S LiPo.
    Max thrust ~800g with 9x4.5 prop on 3S.
    """
    return MotorSpec(
        name="MT2213",
        dimensions=(
            _MT2213_CAN_DIA,
            _MT2213_CAN_DIA,
            _MT2213_TOTAL_LENGTH,
        ),
        mass=_MT2213_MASS,
        wire_ports=(
            WirePort(
                "power_a",
                pos=(mm(5.0), Meters(0.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="bullet_3.5mm",
                permanent=True,
            ),
            WirePort(
                "power_b",
                pos=(mm(-5.0), Meters(0.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="bullet_3.5mm",
                permanent=True,
            ),
            WirePort(
                "power_c",
                pos=(Meters(0.0), mm(5.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="bullet_3.5mm",
                permanent=True,
            ),
        ),
        default_material=MAT_ABS_DARK,
        voltage=Volts(11.1),  # 3S nominal
        typical_current=Amps(10.0),
        kv=935.0,
        max_thrust_n=7.85,  # ~800g thrust with 9x4.5 prop on 3S
        shaft_diameter=_MT2213_SHAFT_DIA,
        can_diameter=_MT2213_CAN_DIA,
        can_length=_MT2213_CAN_LENGTH,
    )
