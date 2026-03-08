"""Parametric bot CAD system — design to sim to fabrication.

Define a robot as a kinematic tree of bodies, joints, and real components.
The system generates everything needed to simulate and build the physical bot:
STEP assemblies, per-body STL meshes, MuJoCo XML, BOMs, and assembly guides.

The CAD geometry IS the sim geometry — MuJoCo references the same STL files
you'd send to a slicer. No separate "sim mesh" that can diverge from reality.
"""

from botcad.component import Component, MountingEar, MountPoint, ServoSpec, WirePort
from botcad.skeleton import Body, Bot, Joint

__all__ = [
    "Bot",
    "Body",
    "Joint",
    "Component",
    "ServoSpec",
    "WirePort",
    "MountPoint",
    "MountingEar",
]
