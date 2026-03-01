"""
Parametric bot CAD system.

Component-driven robot design: select real components (servos, compute boards,
batteries, cameras) and the system generates MuJoCo XML, CAD assemblies, BOMs,
and assembly guides automatically.
"""

from botcad.component import Component, MountPoint, ServoSpec, WirePort
from botcad.skeleton import Body, Bot, Joint

__all__ = [
    "Bot",
    "Body",
    "Joint",
    "Component",
    "ServoSpec",
    "WirePort",
    "MountPoint",
]
