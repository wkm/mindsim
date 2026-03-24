"""Material and print process definitions.

Data module — declares physical properties that affect mass computation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrintProcess:
    """FDM print parameters that determine effective mass."""

    wall_layers: int = 2
    nozzle_width: float = 0.0004  # 0.4mm
    infill: float = 0.20


@dataclass(frozen=True)
class Material:
    """Physical material with density and optional print process."""

    name: str
    density: float  # kg/m^3
    youngs_modulus: float = 2.3e9  # Pa (default PLA)
    poisson_ratio: float = 0.35
    yield_strength: float = 40e6  # Pa
    process: PrintProcess | None = None


# Standard instances
PLA = Material("PLA", 1200.0, 2.3e9, 0.35, 40e6, PrintProcess())
TPU = Material("TPU", 1120.0, 0.1e9, 0.45, 15e6, PrintProcess(infill=0.15))
ALUMINUM = Material("aluminum", 2700.0, 70e9, 0.33, 270e6, None)
