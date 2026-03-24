"""Material definitions — unified visual + physical properties.

Every material has both visual properties (color, metallic, roughness, opacity)
for rendering and physical properties (density, print process) for mass/sim.

The material catalog provides real-world materials for purchased components
(PCB substrate, IC packages, rubber, metal) and fabricated parts (PLA, TPU).
"""

from __future__ import annotations

from dataclasses import dataclass

RGBA = tuple[float, float, float, float]


@dataclass(frozen=True)
class PrintProcess:
    """FDM print parameters that determine effective mass."""

    wall_layers: int = 2
    nozzle_width: float = 0.0004  # 0.4mm
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
    density: float | None = None  # kg/m^3; None for purchased materials
    process: PrintProcess | None = None  # FDM params; None for purchased


# ── Fabricated materials (with density + print process) ──

PLA = Material(
    "PLA",
    color=(0.808, 0.851, 0.878, 1.0),  # light gray
    roughness=0.8,
    density=1200.0,
    process=PrintProcess(),
)
TPU = Material(
    "TPU",
    color=(0.161, 0.216, 0.278, 1.0),  # dark gray
    roughness=0.9,
    density=1120.0,
    process=PrintProcess(infill=0.15),
)
ALUMINUM = Material(
    "aluminum",
    color=(0.655, 0.761, 0.831, 1.0),  # silver-gray
    metallic=0.9,
    roughness=0.3,
    density=2700.0,
)

# ── Material catalog: purchased component materials ──

# PCB substrate — green solder mask
MAT_FR4_GREEN = Material(
    "fr4_green",
    color=(0.059, 0.600, 0.373, 1.0),  # BP_GREEN3
    roughness=0.85,
)

# IC package epoxy — dark matte
MAT_IC_PACKAGE = Material(
    "ic_package",
    color=(0.094, 0.133, 0.157, 1.0),  # BP_DARK_GRAY1
    roughness=0.9,
)

# Nickel plating — connector pins, shield cans
MAT_NICKEL = Material(
    "nickel",
    color=(0.655, 0.761, 0.831, 1.0),  # BP_GRAY4
    metallic=0.85,
    roughness=0.25,
)

# ABS dark — servo housing, camera body
MAT_ABS_DARK = Material(
    "abs_dark",
    color=(0.094, 0.133, 0.157, 1.0),  # BP_DARK_GRAY1
    roughness=0.6,
)

# Rubber — tire surface
MAT_RUBBER = Material(
    "rubber",
    color=(0.161, 0.216, 0.278, 1.0),  # BP_DARK_GRAY3
    roughness=0.95,
)

# Polycarbonate clear — camera lens
MAT_POLYCARBONATE_CLEAR = Material(
    "polycarbonate_clear",
    color=(0.2, 0.2, 0.25, 0.7),
    roughness=0.1,
    opacity=0.7,
)

# PLA light — 3D printed structural parts
MAT_PLA_LIGHT = Material(
    "pla_light",
    color=(0.808, 0.851, 0.878, 1.0),  # BP_LIGHT_GRAY1
    roughness=0.8,
    density=1200.0,
    process=PrintProcess(),
)

# Steel — fastener bodies
MAT_STEEL = Material(
    "steel",
    color=(0.655, 0.761, 0.831, 1.0),  # BP_GRAY4
    metallic=0.9,
    roughness=0.35,
    density=7800.0,
)

# Aluminum — servo horns, brackets
MAT_ALUMINUM = Material(
    "aluminum_part",
    color=(0.808, 0.851, 0.878, 1.0),  # light silver
    metallic=0.85,
    roughness=0.3,
    density=2700.0,
)

# Battery — blue LiPo wrap
MAT_LIPO_WRAP = Material(
    "lipo_wrap",
    color=(0.075, 0.486, 0.741, 1.0),  # BP_BLUE3
    roughness=0.7,
)

# Electronics controller PCB — darker green
MAT_PCB_DARK_GREEN = Material(
    "pcb_dark_green",
    color=(0.137, 0.549, 0.173, 1.0),  # BP_FOREST3
    roughness=0.85,
)

# Bearing steel
MAT_BEARING_STEEL = Material(
    "bearing_steel",
    color=(0.655, 0.761, 0.831, 1.0),  # BP_GRAY4
    metallic=1.0,
    roughness=0.3,
)

# Brass — decorative fasteners
MAT_BRASS = Material(
    "brass",
    color=(0.851, 0.620, 0.043, 1.0),  # BP_GOLD3
    metallic=0.9,
    roughness=0.3,
)
