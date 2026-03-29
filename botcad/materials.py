"""Material definitions — unified visual + physical properties.

Every material has both visual properties (color, metallic, roughness, opacity)
for rendering and physical properties (density, print process) for mass/sim.

The material catalog provides real-world materials for purchased components
(PCB substrate, IC packages, rubber, metal) and fabricated parts (PLA, TPU).
"""

from __future__ import annotations

from dataclasses import dataclass

from botcad.colors import (
    BP_BLUE3,
    BP_DARK_GRAY1,
    BP_DARK_GRAY3,
    BP_FOREST3,
    BP_GOLD3,
    BP_GRAY4,
    BP_GRAY5,
    BP_GREEN3,
    BP_LIGHT_GRAY1,
)
from botcad.units import KgPerM3, Meters, Pascals, gpa, mm, mpa

RGBA = tuple[float, float, float, float]


@dataclass(frozen=True)
class PrintProcess:
    """FDM print parameters that determine effective mass."""

    wall_layers: int = 2
    nozzle_width: Meters = mm(0.4)
    infill: float = 0.20


@dataclass(frozen=True)
class Material:
    """Unified material: visual + physical properties.

    Visual properties drive rendering (viewer MeshPhysicalMaterial, MuJoCo rgba).
    Physical properties drive mass computation and sim fidelity.
    """

    name: str
    # Visual
    color: RGBA = (0.541, 0.608, 0.659, 1.0)  # default gray
    metallic: float = 0.0
    roughness: float = 0.7
    opacity: float = 1.0
    # Physical
    density: KgPerM3 | None = None  # kg/m^3; None for purchased materials
    youngs_modulus: Pascals = gpa(2.3)  # Pa (default PLA)
    poisson_ratio: float = 0.35
    yield_strength: Pascals = mpa(40)  # Pa
    process: PrintProcess | None = None  # FDM params; None for purchased

    @property
    def effective_youngs_modulus(self) -> float:
        """Young's modulus scaled by infill (Gibson-Ashby bending-dominated).

        E_eff = E_solid * infill^2. Conservative for FDM grid/rectilinear.
        """
        if self.process is None:
            return self.youngs_modulus
        return self.youngs_modulus * self.process.infill**2

    @property
    def effective_yield_strength(self) -> float:
        """Yield strength scaled by infill (Gibson-Ashby bending-dominated).

        sigma_eff = sigma_solid * infill^1.5. Conservative for FDM.
        """
        if self.process is None:
            return self.yield_strength
        return self.yield_strength * self.process.infill**1.5


# ── Fabricated materials (with density + print process) ──

PLA = Material(
    "PLA",
    color=(*BP_LIGHT_GRAY1, 1.0),
    roughness=0.8,
    density=KgPerM3(1200.0),
    youngs_modulus=gpa(2.3),
    poisson_ratio=0.35,
    yield_strength=mpa(40),
    process=PrintProcess(),
)
TPU = Material(
    "TPU",
    color=(*BP_DARK_GRAY3, 1.0),
    roughness=0.9,
    density=KgPerM3(1120.0),
    youngs_modulus=gpa(0.1),
    poisson_ratio=0.45,
    yield_strength=mpa(15),
    process=PrintProcess(infill=0.15),
)
ALUMINUM = Material(
    "aluminum",
    color=(*BP_GRAY4, 1.0),
    metallic=0.9,
    roughness=0.3,
    density=KgPerM3(2700.0),
    youngs_modulus=gpa(70),
    poisson_ratio=0.33,
    yield_strength=mpa(270),
)

# ── Material catalog: purchased component materials ──

# PCB substrate — green solder mask
MAT_FR4_GREEN = Material(
    "fr4_green",
    color=(*BP_GREEN3, 1.0),
    roughness=0.85,
)

# IC package epoxy — dark matte
MAT_IC_PACKAGE = Material(
    "ic_package",
    color=(*BP_DARK_GRAY1, 1.0),
    roughness=0.9,
)

# Nickel plating — connector pins, shield cans
MAT_NICKEL = Material(
    "nickel",
    color=(*BP_GRAY4, 1.0),
    metallic=0.85,
    roughness=0.25,
)

# ABS dark — servo housing, camera body
MAT_ABS_DARK = Material(
    "abs_dark",
    color=(*BP_DARK_GRAY1, 1.0),
    roughness=0.6,
)

# Rubber — tire surface
MAT_RUBBER = Material(
    "rubber",
    color=(*BP_DARK_GRAY3, 1.0),
    roughness=0.95,
)

# Polycarbonate clear — camera lens
MAT_POLYCARBONATE_CLEAR = Material(
    "polycarbonate_clear",
    color=(0.2, 0.2, 0.25, 0.7),
    roughness=0.1,
    opacity=0.7,
)

# PLA light — 3D printed structural parts (same as PLA)
MAT_PLA_LIGHT = PLA

# Steel — fastener bodies
MAT_STEEL = Material(
    "steel",
    color=(*BP_GRAY4, 1.0),
    metallic=0.9,
    roughness=0.35,
    density=KgPerM3(7800.0),
)

# Aluminum — servo horns, brackets (same as ALUMINUM)
MAT_ALUMINUM = ALUMINUM

# Battery — blue LiPo wrap
MAT_LIPO_WRAP = Material(
    "lipo_wrap",
    color=(*BP_BLUE3, 1.0),
    roughness=0.7,
)

# Electronics controller PCB — darker green
MAT_PCB_DARK_GREEN = Material(
    "pcb_dark_green",
    color=(*BP_FOREST3, 1.0),
    roughness=0.85,
)

# Bearing steel
MAT_BEARING_STEEL = Material(
    "bearing_steel",
    color=(*BP_GRAY4, 1.0),
    metallic=1.0,
    roughness=0.3,
)

# Brass — decorative fasteners
MAT_BRASS = Material(
    "brass",
    color=(*BP_GOLD3, 1.0),
    metallic=0.9,
    roughness=0.3,
)

# Tube default — gray structural tube
MAT_TUBE_DEFAULT = Material(
    "tube_default",
    color=(*BP_GRAY5, 1.0),
    roughness=0.7,
)
