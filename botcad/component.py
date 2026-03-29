"""Base component dataclasses for parametric bot design.

Every physical part (servo, battery, compute board, camera) is a Component
with real-world dimensions, mass, wire ports, and mounting points.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from botcad.materials import Material
from botcad.units import (
    Amps,
    Degrees,
    Kg,
    Meters,
    NewtonM,
    Position,
    Radians,
    RadPerSec,
    Size3D,
    Volts,
)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z)


@dataclass(frozen=True, slots=True)
class Pose:
    """Position + orientation in 3D space."""

    pos: Position = (Meters(0.0), Meters(0.0), Meters(0.0))
    quat: Quat = (1.0, 0.0, 0.0, 0.0)  # identity quaternion (w, x, y, z)


POSE_IDENTITY = Pose()


class ComponentKind(StrEnum):
    """Semantic category of a physical component."""

    SERVO = "servo"
    CAMERA = "camera"
    BATTERY = "battery"
    COMPUTE = "compute"
    WHEEL = "wheel"
    BEARING = "bearing"
    MOTOR = "motor"
    PROPELLER = "propeller"
    GENERIC = "generic"


class MountOrientation(StrEnum):
    """How a component sits relative to its mounting surface."""

    FLAT = "flat"  # lies flat on mounting surface
    FACE_NORMAL = "face_normal"  # functional axis aligns with mount face normal


@dataclass(frozen=True)
class ComponentMeta:
    """Metadata for a component kind — script emitter, layers, category."""

    category: str  # BOM category: "electronics", "actuator", "structure"
    layers: tuple[str, ...]  # viewer STL layers
    mount_orientation: MountOrientation
    script_emitter: Callable | None = None  # ShapeScript emitter, set after import
    multi_material_emitter: Callable | None = None  # returns MultiMaterialResult


_COMPONENT_REGISTRY: dict[ComponentKind, ComponentMeta] | None = None


def get_component_meta(kind: ComponentKind) -> ComponentMeta:
    """Look up metadata for a component kind. Lazy-loads the registry."""
    global _COMPONENT_REGISTRY
    if _COMPONENT_REGISTRY is None:
        _COMPONENT_REGISTRY = _build_registry()
    return _COMPONENT_REGISTRY[kind]


def _build_registry() -> dict[ComponentKind, ComponentMeta]:
    from botcad.shapescript.emit_components import (
        battery_script,
        bearing_script,
        camera_multi_material,
        camera_script,
        compute_multi_material,
        compute_script,
        generic_multi_material,
        generic_pcb_script,
        wheel_component_script,
    )
    from botcad.shapescript.emit_servo import servo_script

    return {
        ComponentKind.SERVO: ComponentMeta(
            category="actuator",
            layers=(
                "servo",
                "bracket",
                "cradle",
                "coupler",
                "bracket_envelope",
                "cradle_envelope",
                "bracket_insertion_channel",
                "cradle_insertion_channel",
                "horn",
                "fasteners",
            ),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=servo_script,
        ),
        ComponentKind.CAMERA: ComponentMeta(
            category="electronics",
            layers=("body", "fasteners"),
            mount_orientation=MountOrientation.FACE_NORMAL,
            script_emitter=camera_script,
            multi_material_emitter=camera_multi_material,
        ),
        ComponentKind.BATTERY: ComponentMeta(
            category="electronics",
            layers=("body", "fasteners"),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=battery_script,
        ),
        ComponentKind.COMPUTE: ComponentMeta(
            category="electronics",
            layers=("body", "fasteners"),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=compute_script,
            multi_material_emitter=compute_multi_material,
        ),
        ComponentKind.WHEEL: ComponentMeta(
            category="structure",
            layers=("body",),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=wheel_component_script,
        ),
        ComponentKind.BEARING: ComponentMeta(
            category="structure",
            layers=("body",),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=bearing_script,
        ),
        ComponentKind.MOTOR: ComponentMeta(
            category="actuator",
            layers=("body",),
            mount_orientation=MountOrientation.FACE_NORMAL,
            script_emitter=generic_pcb_script,
        ),
        ComponentKind.PROPELLER: ComponentMeta(
            category="structure",
            layers=("body",),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=generic_pcb_script,
        ),
        ComponentKind.GENERIC: ComponentMeta(
            category="component",
            layers=("body",),
            mount_orientation=MountOrientation.FLAT,
            script_emitter=generic_pcb_script,
            multi_material_emitter=generic_multi_material,
        ),
    }


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
    pos: Position  # position relative to component origin (meters)
    bus_type: BusType
    connector_type: str = ""  # e.g. "5264_3pin", "xt30", "csi_15pin"
    permanent: bool = False  # True if wire is soldered/molded (not a removable plug)


@dataclass(frozen=True)
class MountPoint:
    """A screw hole or snap-fit point on a component."""

    label: str
    pos: Position  # position relative to component origin (meters)
    diameter: Meters  # hole diameter (meters)
    axis: Vec3 = (0.0, 0.0, 1.0)  # fastener insertion direction (dimensionless)
    fastener_type: str = ""  # "M2", "M2.5", "M3", "press_fit", etc.
    head_type: str = ""  # HeadType value, empty = socket_head_cap


def MountingEar(
    label: str,
    pos: Position,
    hole_diameter: Meters,
    axis: Vec3 = (0.0, 0.0, 1.0),
    fastener_type: str = "M3",
) -> MountPoint:
    """Factory for servo mounting ear points (returns MountPoint).

    axis = insertion direction (where the shank goes into material).
    Default +Z means screws insert upward through the ear into the bracket.
    """
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
    dimensions: Size3D  # bounding box (x, y, z) in meters
    mass: Kg  # kg
    kind: ComponentKind = ComponentKind.GENERIC
    wire_ports: tuple[WirePort, ...] = ()
    mounting_points: tuple[MountPoint, ...] = ()
    default_material: Material | None = None
    voltage: Volts = Volts(0.0)  # operating voltage (V), 0 = unpowered
    typical_current: Amps = Amps(0.0)  # typical draw (A), 0 = unpowered


@dataclass(frozen=True)
class BatterySpec(Component):
    """A battery pack with chemistries and cell counts."""

    kind: ComponentKind = ComponentKind.BATTERY
    chemistry: str = "LiPo"
    cells_s: int = 1
    cells_p: int = 1


@dataclass(frozen=True)
class BearingSpec(Component):
    """A ball bearing with dimensions and mounting type."""

    kind: ComponentKind = ComponentKind.BEARING
    od: Meters = Meters(0.0)  # outer diameter (meters)
    id: Meters = Meters(0.0)  # inner diameter (meters)
    width: Meters = Meters(0.0)  # thickness (meters)


@dataclass(frozen=True)
class MotorSpec(Component):
    """A brushless DC motor for propulsion."""

    kind: ComponentKind = ComponentKind.MOTOR
    kv: float = 0.0  # motor velocity constant (RPM per volt)
    max_thrust_n: float = 0.0  # max thrust in Newtons
    shaft_diameter: Meters = Meters(0.0)  # output shaft diameter
    can_diameter: Meters = Meters(0.0)  # motor can outer diameter
    can_length: Meters = Meters(0.0)  # motor can length (without shaft)


@dataclass(frozen=True)
class PropellerSpec(Component):
    """A propeller for aerodynamic thrust."""

    kind: ComponentKind = ComponentKind.PROPELLER
    diameter: Meters = Meters(0.0)  # prop diameter
    pitch: Meters = Meters(0.0)  # prop pitch
    blades: int = 2


@dataclass(frozen=True)
class CameraSpec(Component):
    """A camera module with optical parameters."""

    kind: ComponentKind = ComponentKind.CAMERA
    fov: Degrees = Degrees(72.0)  # diagonal field of view
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

    kind: ComponentKind = ComponentKind.SERVO
    stall_torque: NewtonM = NewtonM(0.0)  # N-m
    no_load_speed: RadPerSec = RadPerSec(0.0)  # rad/s
    voltage: Volts = Volts(0.0)  # V
    bus_type: BusType = BusType.PWM
    shaft_offset: Position = (
        Meters(0.0),
        Meters(0.0),
        Meters(0.0),
    )  # output shaft center relative to body origin
    shaft_axis: Vec3 = (
        0.0,
        0.0,
        1.0,
    )  # rotation axis (dimensionless, in servo local frame)
    range_rad: tuple[Radians, Radians] = (
        Radians(-3.14159),
        Radians(3.14159),
    )  # angular range
    gear_ratio: float = 1.0
    continuous: bool = False  # continuous rotation mode (wheels)

    # Extended geometry (optional, for detailed CAD / visualization)
    body_dimensions: Size3D = (
        Meters(0.0),
        Meters(0.0),
        Meters(0.0),
    )  # main body only (no ears/horn)
    shaft_boss_radius: Meters = Meters(0.0)  # bearing housing radius (meters)
    shaft_boss_height: Meters = Meters(0.0)  # protrusion above body top face (meters)
    mounting_ears: tuple[MountPoint, ...] = ()  # bracket attachment tabs
    horn_mounting_points: tuple[MountPoint, ...] = ()  # screw holes on output horn
    rear_horn_mounting_points: tuple[MountPoint, ...] = ()  # screw holes on blind side
    connector_pos: Position | None = None  # wire connector center position

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
        ~3 deg (0.05 rad) as the max steady-state error under full load.
        """
        max_error_rad = 0.05
        return self.stall_torque / max_error_rad

    @property
    def damping(self) -> float:
        """Joint damping derived from servo specs.

        damping = stall_torque / no_load_speed (viscous friction model).
        """
        if self.no_load_speed <= 0:
            return 0.1
        return self.stall_torque / self.no_load_speed
