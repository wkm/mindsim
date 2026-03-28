"""Parametric bot CAD system — design to sim to fabrication.

Define a robot as a kinematic tree of bodies, joints, and real components.
The system generates everything needed to simulate and build the physical bot:
STEP assemblies, per-body STL meshes, MuJoCo XML, BOMs, and assembly guides.

The CAD geometry IS the sim geometry — MuJoCo references the same STL files
you'd send to a slicer. No separate "sim mesh" that can diverge from reality.
"""

from botcad.component import (
    BusType,
    Component,
    MountingEar,
    MountPoint,
    ServoSpec,
    WirePort,
)
from botcad.skeleton import (
    Assembly,
    BaseType,
    Body,
    BodyKind,
    BodyShape,
    Bot,
    BracketStyle,
    Joint,
    Module,
)

__all__ = [
    "Assembly",
    "BaseType",
    "Body",
    "BodyKind",
    "BodyShape",
    "Bot",
    "BracketStyle",
    "BusType",
    "Component",
    "Joint",
    "Module",
    "MountPoint",
    "MountingEar",
    "ServoSpec",
    "WirePort",
]
