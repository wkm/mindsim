"""Dimension types for physical quantities.

All physical quantities use NewType wrappers — never bare float.
Zero runtime cost; caught by type checkers (pyright/mypy).

Convention: values are always in SI base units.
Factory functions convert from common datasheet units.
"""

from __future__ import annotations

import math
from typing import NewType

# ── Scalar types ────────────────────────────────────────────────────────

Meters = NewType("Meters", float)  # length
Kg = NewType("Kg", float)  # mass
Degrees = NewType("Degrees", float)  # angle (human-facing: FOV, display)
Radians = NewType("Radians", float)  # angle (math/physics: joint range)
Volts = NewType("Volts", float)  # voltage
Amps = NewType("Amps", float)  # current
NewtonM = NewType("NewtonM", float)  # torque
RadPerSec = NewType("RadPerSec", float)  # angular velocity
Pascals = NewType("Pascals", float)  # pressure / stress / modulus
KgPerM3 = NewType("KgPerM3", float)  # density

# ── Compound types ──────────────────────────────────────────────────────

Position = tuple[Meters, Meters, Meters]  # spatial coordinates (point in space)
Size3D = tuple[Meters, Meters, Meters]  # bounding box extents (w, h, d)

# ── Factory functions ───────────────────────────────────────────────────


def mm(val: float) -> Meters:
    """Millimeters → Meters."""
    return Meters(val / 1000.0)


def mm3(x: float, y: float, z: float) -> tuple[Meters, Meters, Meters]:
    """Three mm values → (Meters, Meters, Meters)."""
    return (mm(x), mm(y), mm(z))


def grams(val: float) -> Kg:
    """Grams → Kg."""
    return Kg(val / 1000.0)


def mpa(val: float) -> Pascals:
    """Megapascals → Pascals."""
    return Pascals(val * 1e6)


def gpa(val: float) -> Pascals:
    """Gigapascals → Pascals."""
    return Pascals(val * 1e9)


def deg_to_rad(val: float) -> Radians:
    """Degrees → Radians."""
    return Radians(val * math.pi / 180.0)
