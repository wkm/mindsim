"""Base component dataclasses for parametric bot design.

Every physical part (servo, battery, compute board, camera) is a Component
with real-world dimensions, mass, wire ports, and mounting points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

Vec3 = tuple[float, float, float]
RGBA = tuple[float, float, float, float]


class BusType(StrEnum):
    """Wire bus protocol type."""

    UART_HALF_DUPLEX = "uart_half_duplex"
    PWM = "pwm"
    CSI = "csi"
    POWER = "power"
    USB = "usb"
    GPIO = "gpio"
    BALANCE = "balance"


@dataclass(frozen=True)
class WirePort:
    """Where a wire attaches to a component."""

    label: str  # e.g. "uart_in", "uart_out", "power", "csi"
    pos: Vec3  # position relative to component origin (meters)
    bus_type: BusType


@dataclass(frozen=True)
class MountPoint:
    """A screw hole or snap-fit point on a component."""

    label: str
    pos: Vec3  # position relative to component origin (meters)
    diameter: float  # hole diameter (meters)
    axis: Vec3 = (0.0, 0.0, 1.0)  # fastener insertion direction
    fastener_type: str = ""  # "M2", "M2.5", "M3", "press_fit", etc.


def MountingEar(
    label: str,
    pos: Vec3,
    hole_diameter: float,
    axis: Vec3 = (0.0, 0.0, -1.0),
    fastener_type: str = "M3",
) -> MountPoint:
    """Factory for servo mounting ear points (returns MountPoint)."""
    return MountPoint(
        label=label,
        pos=pos,
        diameter=hole_diameter,
        axis=axis,
        fastener_type=fastener_type,
    )


@dataclass(frozen=True)
class Component:
    """Base class for all physical components."""

    name: str
    dimensions: Vec3  # bounding box (x, y, z) in meters
    mass: float  # kg
    wire_ports: tuple[WirePort, ...] = ()
    mounting_points: tuple[MountPoint, ...] = ()
    color: RGBA = (0.541, 0.608, 0.659, 1.0)  # BP_GRAY3 — default
    voltage: float = 0.0  # operating voltage (V), 0 = unpowered
    typical_current: float = 0.0  # typical draw (A), 0 = unpowered


@dataclass(frozen=True)
class BatterySpec(Component):
    """A battery pack with chemistries and cell counts."""

    chemistry: str = "LiPo"
    cells_s: int = 1
    cells_p: int = 1


@dataclass(frozen=True)
class BearingSpec(Component):
    """A ball bearing with dimensions and mounting type."""

    od: float = 0.0  # outer diameter (meters)
    id: float = 0.0  # inner diameter (meters)
    width: float = 0.0  # thickness (meters)


@dataclass(frozen=True)
class CameraSpec(Component):
    """A camera module with optical parameters."""

    fov_deg: float = 72.0  # diagonal field of view
    resolution: tuple[int, int] = (640, 480)  # width x height pixels


@dataclass(frozen=True)
class ServoSpec(Component):
    """A servo motor with mechanical and electrical specs.

    Servo local frame convention (matching the STS3215 STEP model):
        X = long axis (length, 45.2mm for STS3215)
        Y = width axis (24.7mm)
        Z = shaft axis (height, shaft protrudes in +Z)

    The servo is NOT symmetric: the output shaft is offset along X from
    the body center.  ``shaft_offset`` encodes this displacement so that
    ``servo_placement()`` can position the body correctly at a joint.
    """

    stall_torque: float = 0.0  # N-m
    no_load_speed: float = 0.0  # rad/s
    voltage: float = 0.0  # V
    bus_type: BusType = BusType.PWM
    shaft_offset: Vec3 = (0.0, 0.0, 0.0)  # output shaft center relative to body origin
    shaft_axis: Vec3 = (0.0, 0.0, 1.0)  # rotation axis (in servo local frame)
    range_rad: tuple[float, float] = (-3.14159, 3.14159)  # angular range
    gear_ratio: float = 1.0
    continuous: bool = False  # continuous rotation mode (wheels)

    # Extended geometry (optional, for detailed CAD / visualization)
    body_dimensions: Vec3 = (0.0, 0.0, 0.0)  # main body only (no ears/horn)
    shaft_boss_radius: float = 0.0  # bearing housing radius (meters)
    shaft_boss_height: float = 0.0  # protrusion above body top face (meters)
    mounting_ears: tuple[MountPoint, ...] = ()  # bracket attachment tabs
    horn_mounting_points: tuple[MountPoint, ...] = ()  # screw holes on output horn
    rear_horn_mounting_points: tuple[MountPoint, ...] = ()  # screw holes on blind side
    connector_pos: Vec3 | None = None  # wire connector center position

    @property
    def effective_body_dims(self) -> Vec3:
        """Body dimensions (without ears/horn), falling back to overall dims."""
        bd = self.body_dimensions
        if bd[0] > 0.0:
            return bd
        return self.dimensions

    @property
    def kp(self) -> float:
        """Position gain derived from stall torque.

        kp = stall_torque / max_acceptable_error. A real servo with a
        position encoder holds position within a few degrees. We use
        ~3° (0.05 rad) as the max steady-state error under full load.
        """
        max_error_rad = 0.05
        return self.stall_torque / max_error_rad

    @property
    def damping(self) -> float:
        """Joint damping derived from servo specs.

        damping ≈ stall_torque / no_load_speed (viscous friction model).
        """
        if self.no_load_speed <= 0:
            return 0.1
        return self.stall_torque / self.no_load_speed
