"""Kinematic tree DSL for parametric robot design.

Defines Bot, Body, Assembly, and Joint classes that form the user-facing API
for building robots from real components. The design taxonomy is:

    Assembly > Body > Feature

    Bot (root assembly)
    └── Assembly (functional grouping, can nest)
        ├── Assembly (sub-assembly)
        │     ├── Body (fabricated — printed)
        │     └── Body (purchased — servo, battery)
        └── Body (fabricated or purchased)

    Body (root)
    ├── mounted Components
    ├── Joint (with ServoSpec)
    │   └── Body (child)
    │       ├── ...
    └── Joint
        └── Body

Every Joint requires a ServoSpec — the servo's physical dimensions, torque,
and speed determine the joint's geometry, control limits, and damping.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from botcad.component import (
    POSE_IDENTITY,
    Component,
    ComponentKind,
    MountOrientation,
    Pose,
    ServoSpec,
    Vec3,
    get_component_meta,
)
from botcad.geometry import (
    EULER_IDENTITY,
    EULER_RX_180,
    EULER_RX_NEG90,
    EULER_RX_POS90,
    EULER_RY_NEG90,
    EULER_RY_POS90,
    MOUNT_NO_ROTATION,
    MountRotation,
    PackingResult,
)
from botcad.materials import PLA, Material
from botcad.units import Kg, Meters, Position, Size3D

_ZERO_M = Meters(0.0)
_DEFAULT_PADDING = Meters(0.005)
_ZERO_KG = Kg(0.0)
_ZERO_POS: Position = (Meters(0.0), Meters(0.0), Meters(0.0))

FacePosition = Literal["center", "bottom", "top", "front", "back", "left", "right"]

# Face rotation table: position → (euler_deg, dims_reorder, point_transform).
#   euler_deg:  rotation for build123d Location
#   dims_reorder: indices to reorder (cx, cy, cz), or None if unchanged
#   point_transform: (x, y, z) → rotated (x', y', z')
_FaceRotEntry = tuple[
    tuple[float, float, float],
    tuple[int, int, int] | None,
    Callable[[float, float, float], Vec3] | None,
]
_FACE_ROTATION: dict[str, _FaceRotEntry] = {
    "front": (EULER_RX_NEG90, (0, 2, 1), lambda x, y, z: (x, z, -y)),
    "back": (EULER_RX_POS90, (0, 2, 1), lambda x, y, z: (x, -z, y)),
    "left": (EULER_RY_NEG90, (2, 1, 0), lambda x, y, z: (-z, y, x)),
    "right": (EULER_RY_POS90, (2, 1, 0), lambda x, y, z: (z, y, -x)),
    "bottom": (EULER_RX_180, None, lambda x, y, z: (x, -y, -z)),
}


@dataclass(frozen=True)
class ClearanceConstraint:
    """Expected clearance between two bodies in the assembly."""

    body_a: str  # body name
    body_b: str  # body name
    min_distance: Meters = _ZERO_M  # meters — minimum acceptable gap
    label: str = ""  # human-readable description


class BodyShape(StrEnum):
    """Geometric shape of a rigid body."""

    BOX = "box"
    CYLINDER = "cylinder"
    TUBE = "tube"
    SPHERE = "sphere"
    JAW = "jaw"


class BodyKind(StrEnum):
    """Whether a body is fabricated (3D printed) or purchased off-the-shelf."""

    FABRICATED = "fabricated"
    PURCHASED = "purchased"


class BracketStyle(StrEnum):
    """Servo bracket mounting style."""

    POCKET = "pocket"
    COUPLER = "coupler"


class BaseType(StrEnum):
    """How the robot root is attached to the world."""

    FREE = "free"
    FIXED = "fixed"


@dataclass  # plint: disable=frozen-dataclass
class Assembly:
    """A group of bodies and sub-assemblies — a functional unit.

    Assemblies nest: a gripper is an assembly inside an arm assembly
    inside a robot assembly. The Bot itself is the root assembly.
    """

    name: str
    _bot: Bot = field(repr=False)
    parent: Assembly | None = None
    _sub_assemblies: dict[str, Assembly] = field(default_factory=dict)

    def assembly(self, name: str) -> Assembly:
        """Create a named sub-assembly."""
        if name in self._sub_assemblies:
            return self._sub_assemblies[name]
        sub = Assembly(name=name, _bot=self._bot, parent=self)
        self._sub_assemblies[name] = sub
        return sub

    def body(self, name: str, shape: BodyShape = BodyShape.BOX, **kwargs) -> Body:
        """Create a body in this assembly. First body created becomes bot root."""
        b = Body(name=name, shape=shape, assembly=self, **kwargs)
        if self._bot.root is None:
            self._bot.root = b
        return b

    @property
    def path(self) -> str:
        """Full path from root: 'robot/arm/gripper'."""
        parts = []
        node: Assembly | None = self
        while node is not None:
            parts.append(node.name)
            node = node.parent
        return "/".join(reversed(parts))


# Backward compatibility
Module = Assembly


@dataclass  # plint: disable=frozen-dataclass
class Mount:
    """A component placed inside a body."""

    component: Component
    label: str
    position: FacePosition | Vec3  # heuristic position or explicit (x, y, z)
    insertion_axis: Vec3 | None = (
        None  # explicit override, or None = derive from position
    )
    rotation: MountRotation = MOUNT_NO_ROTATION
    resolved_pos: Vec3 = (0.0, 0.0, 0.0)  # filled by packing solver
    resolved_insertion_axis: Vec3 = (0.0, 0.0, 1.0)  # filled by packing solver
    solved_bbox: Vec3 | None = None  # actual bounding box from ShapeScript execution

    @property
    def rotate_z(self) -> bool:
        """Backward-compat: True when mount rotation is 90 deg yaw."""
        return self.rotation.yaw == 90.0

    @property
    def face_outward(self) -> bool:
        """Whether this component should face outward from its mount face.

        Cameras always face outward — their lens axis (+Z in component frame)
        is rotated to align with the mount face normal.
        """

        return (
            get_component_meta(self.component.kind).mount_orientation
            == MountOrientation.FACE_NORMAL
        )

    @property
    def _face_rotation_entry(self) -> _FaceRotEntry | None:
        """Look up the face rotation for this mount, or None if no rotation."""
        if not self.face_outward or not isinstance(self.position, str):
            return None
        return _FACE_ROTATION.get(self.position)

    @property
    def face_euler_deg(self) -> tuple[float, float, float]:
        """Euler angles (X, Y, Z) in degrees to rotate component +Z to face
        the mount normal.  Identity when face_outward is False or position
        is "top"/"center" (already +Z).
        """
        entry = self._face_rotation_entry
        return entry[0] if entry else EULER_IDENTITY

    @property
    def placed_dimensions(self) -> Vec3:
        """Component dimensions in the body frame (actual bbox if computed, else declared).

        Prefers solved_bbox (derived from ShapeScript geometry) over the
        component's declared dimensions.  X/Y are swapped when rotate_z is set.
        Face rotation swaps axes as needed (e.g. front-mounted camera swaps Y/Z).
        """
        d = (
            self.solved_bbox
            if self.solved_bbox is not None
            else self.component.dimensions
        )
        if self.rotate_z:
            d = (d[1], d[0], d[2])
        entry = self._face_rotation_entry
        if entry:
            reorder = entry[1]
            if reorder is not None:
                d = (d[reorder[0]], d[reorder[1]], d[reorder[2]])
        return d

    def rotate_point(self, p: Vec3) -> Vec3:
        """Rotate a component-local point into the body frame.

        Applies rotate_z first (if set), then face rotation (if face_outward).
        """
        if self.rotate_z:
            p = (-p[1], p[0], p[2])
        entry = self._face_rotation_entry
        if entry:
            xform = entry[2]
            if xform is not None:
                p = xform(p[0], p[1], p[2])
        return p

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass  # plint: disable=frozen-dataclass
class Attachment:
    """A rigid (non-jointed) connection from parent to child body.

    Used for structural parts that don't move: wings, stabilizers,
    motor mounts, etc. In MuJoCo this emits a child <body> without
    a <joint> element (welded to parent).
    """

    name: str
    pos: Vec3  # position in parent body frame
    child: Body | None = None

    def body(
        self,
        name: str,
        shape: BodyShape = BodyShape.BOX,
        *,
        radius: Meters = _ZERO_M,
        width: Meters = _ZERO_M,
        height: Meters = _ZERO_M,
        length: Meters = _ZERO_M,
        outer_r: Meters = _ZERO_M,
        padding: Meters = _DEFAULT_PADDING,
        dimensions: Size3D | None = None,
        assembly: Assembly | None = None,
        module: Assembly | None = None,
    ) -> Body:
        """Create and attach a child body at this attachment point."""
        effective_assembly = assembly or module
        b = Body(
            name=name,
            shape=shape,
            radius=radius,
            width=width,
            height=height,
            length=length,
            outer_r=outer_r,
            padding=padding,
            explicit_dimensions=dimensions,
            assembly=effective_assembly,
        )
        self.child = b
        return b

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass  # plint: disable=frozen-dataclass
class Joint:
    """A revolute joint connecting a parent body to a child body via a servo.

    Identity-based hashing allows functools.lru_cache on geometry functions
    that take Joint arguments.  Within a bot build, each Joint is a singleton.
    """

    name: str
    servo: ServoSpec
    axis: Vec3  # rotation axis in parent frame
    pos: Vec3  # joint position relative to parent body origin
    range_rad: tuple[float, float] | None = None  # override servo default
    grip: bool = False  # force-limited gripper actuator
    bracket_style: BracketStyle = BracketStyle.POCKET
    child: Body | None = None

    # Cached servo placement (computed once in packing solver, read by emitters).
    # servo_placement() positions the servo body at this joint; the center and
    # quaternion are in the parent body frame.
    solved_servo_center: Vec3 = (0.0, 0.0, 0.0)
    solved_servo_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def effective_range(self) -> tuple[float, float]:
        if self.range_rad is not None:
            return self.range_rad
        return self.servo.range_rad

    def wheel_outboard_offset(self) -> float:
        """How far a wheel child body sits outboard from the joint shaft.

        Physical stack: shaft_boss → horn disc → wheel hub.
        Offset = shaft_boss_height + horn_thickness + half_wheel_width.
        Returns 0 if no child.
        """
        if self.child is None:
            return 0.0
        boss_h = self.servo.shaft_boss_height or 0.0
        half_w = (self.child.width or self.child.dimensions[2]) / 2

        # Horn disc thickness (the mechanical interface between servo and wheel)
        from botcad.bracket import horn_disc_params

        horn_h = 0.0
        params = horn_disc_params(self.servo)
        if params is not None:
            horn_h = params.thickness

        return boss_h + horn_h + half_w

    def body(
        self,
        name: str,
        shape: BodyShape = BodyShape.BOX,
        *,
        radius: Meters = _ZERO_M,
        width: Meters = _ZERO_M,
        height: Meters = _ZERO_M,
        length: Meters = _ZERO_M,
        outer_r: Meters = _ZERO_M,
        jaw_length: Meters = _ZERO_M,
        jaw_width: Meters = _ZERO_M,
        jaw_thickness: Meters = _ZERO_M,
        padding: Meters = _DEFAULT_PADDING,
        dimensions: Size3D | None = None,
        custom_solid: object | None = None,
        assembly: Assembly | None = None,
        module: Assembly | None = None,
    ) -> Body:
        """Create and attach a child body to this joint."""
        # Accept both 'assembly' and 'module' (backward compat)
        effective_assembly = assembly or module
        b = Body(
            name=name,
            shape=shape,
            radius=radius,
            width=width,
            height=height,
            length=length,
            outer_r=outer_r,
            jaw_length=jaw_length,
            jaw_width=jaw_width,
            jaw_thickness=jaw_thickness,
            padding=padding,
            explicit_dimensions=dimensions,
            custom_solid=custom_solid,
            assembly=effective_assembly,
        )
        self.child = b
        return b

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass  # plint: disable=frozen-dataclass
class Body:
    """A rigid body in the kinematic tree.

    Bodies contain mounted components and child joints. Their size is either
    set explicitly or computed by the packing solver to fit all contents.

    Identity-based hashing allows functools.lru_cache on geometry functions
    that take Body arguments.  Within a bot build, each Body is a singleton.
    """

    name: str
    shape: BodyShape = BodyShape.BOX
    kind: BodyKind = BodyKind.FABRICATED
    radius: Meters = _ZERO_M
    width: Meters = _ZERO_M
    height: Meters = _ZERO_M
    length: Meters = _ZERO_M
    outer_r: Meters = _ZERO_M
    jaw_length: Meters = _ZERO_M
    jaw_width: Meters = _ZERO_M
    jaw_thickness: Meters = _ZERO_M
    padding: Meters = _DEFAULT_PADDING  # clearance around components (meters)
    explicit_dimensions: Size3D | None = None
    custom_solid: object | None = None
    assembly: Assembly | None = None

    mounts: list[Mount] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)

    # Computed by packing solver
    solved_dimensions: Size3D | None = None
    solved_mass: Kg = _ZERO_KG
    solved_com: Position = _ZERO_POS
    solved_inertia: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    # Body frame orientation — rotation from canonical (Z-up) to actual body
    # frame.  Computed during solve() for bodies whose geometry is reoriented
    # (e.g. cylinders aligned to a joint axis).  Identity means no rotation.
    frame_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # True if this body has a wheel component mounted on it.
    # Computed once during _collect_tree(), read by all emitters.
    is_wheel_body: bool = False

    # World-frame placement (computed during solve/build_cad)
    world_pose: Pose = POSE_IDENTITY

    # ShapeScript program that generates this body's geometry (set during build_cad)
    shapescript: object | None = None

    # For purchased parts: which structural body this part is associated with
    parent_body_name: str | None = None

    # Mesh filename for this body (e.g., "base.stl", "servo_STS3215.stl")
    mesh_file: str | None = None

    # For purchased bodies: reference to the component that generated them.
    # Used by build_cad() to assign the correct ShapeScript.
    _component: Component | None = field(default=None, repr=False)

    # Material — set during solve() or at construction time.
    # `material` provides both physical props (density, print process) and
    # visual props (color, metallic, roughness) for rendering.
    material: Material = PLA

    @property
    def world_pos(self) -> Vec3:
        """Compat: read position from world_pose."""
        return self.world_pose.pos

    @property
    def world_quat(self) -> tuple[float, float, float, float]:
        """Compat: read orientation from world_pose."""
        return self.world_pose.quat

    @property
    def component(self) -> Component | None:
        """The component that generated this purchased body (servo, horn, mounted part)."""
        return self._component

    def to_body_frame(self, p: Vec3) -> Vec3:
        """Transform a canonical-frame point/vector into body-frame coordinates."""
        w, x, y, z = self.frame_quat
        if abs(w - 1.0) < 1e-9 and abs(x) + abs(y) + abs(z) < 1e-9:
            return p  # identity — skip math
        from botcad.geometry import rotate_vec

        return rotate_vec(self.frame_quat, p)

    @property
    def dimensions(self) -> Vec3:
        """Effective dimensions (explicit > solved > shape-derived default)."""
        if self.explicit_dimensions is not None:
            return self.explicit_dimensions
        if self.solved_dimensions is not None:
            return self.solved_dimensions
        if self.custom_solid is not None:
            # Fall back to bounding box of custom solid if no dimensions explicitly given
            ab = self.custom_solid.bounding_box()  # type: ignore
            return (ab.size.X, ab.size.Y, ab.size.Z)
        return self._shape_default_dimensions()

    def _shape_default_dimensions(self) -> Vec3:
        if self.shape is BodyShape.CYLINDER:
            r = self.radius or 0.02
            h = self.width or self.height or 0.01
            return (r * 2, r * 2, h)
        if self.shape is BodyShape.TUBE:
            r = self.outer_r or 0.02
            ln = self.length or 0.1
            return (r * 2, r * 2, ln)
        if self.shape is BodyShape.SPHERE:
            r = self.radius or 0.02
            return (r * 2, r * 2, r * 2)
        if self.shape is BodyShape.JAW:
            jl = self.jaw_length or 0.04
            jw = self.jaw_width or 0.03
            jt = self.jaw_thickness or 0.005
            return (jw, jt, jl)  # X=width, Y=thickness, Z=length
        # box: use explicit or small default
        return (
            self.width or 0.05,
            self.length or 0.05,
            self.height or 0.05,
        )

    def mount(
        self,
        component: Component,
        position: FacePosition | Vec3 = "center",
        label: str = "",
        insertion_axis: Vec3 | None = None,
        rotation: MountRotation = MOUNT_NO_ROTATION,
    ) -> Mount:
        """Mount a component inside this body.

        rotation: design-time rotation of the component on its mount surface.
        """
        if not label:
            label = component.name.lower()
        m = Mount(
            component=component,
            label=label,
            position=position,
            insertion_axis=insertion_axis,
            rotation=rotation,
        )
        self.mounts.append(m)
        return m

    def joint(
        self,
        name: str,
        servo: ServoSpec,
        axis: str | Vec3 = "z",
        pos: Vec3 = (0.0, 0.0, 0.0),
        range: tuple[float, float] | None = None,
        grip: bool = False,
        bracket_style: BracketStyle = BracketStyle.POCKET,
    ) -> Joint:
        """Add a joint (with servo) connecting to a new child body."""
        axis_vec = _parse_axis(axis)
        j = Joint(
            name=name,
            servo=servo,
            axis=axis_vec,
            pos=pos,
            range_rad=range,
            grip=grip,
            bracket_style=bracket_style,
        )
        self.joints.append(j)
        return j

    def attach(
        self,
        name: str,
        pos: Vec3 = (0.0, 0.0, 0.0),
    ) -> Attachment:
        """Attach a rigid (non-jointed) child body at the given position."""
        a = Attachment(name=name, pos=pos)
        self.attachments.append(a)
        return a

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass  # plint: disable=frozen-dataclass
class Bot:
    """Top-level robot definition.

    Usage:
        bot = Bot("wheeler_arm")
        base = bot.body("base", padding=0.01)
        base.mount(STS3215(), ...)
        # ... define kinematic tree ...
        bot.solve()           # run packing + routing
        bot.emit()            # generate all outputs (calls all write_*())
        bot.write_mujoco()    # or generate only what you need
        bot.write_step()
        bot.write_docs()
        bot.write_renders()
        bot.write_viewer_manifest()
    """

    name: str
    root: Body | None = None
    base_type: BaseType = BaseType.FREE

    # Populated by solve()
    all_bodies: list[Body] = field(default_factory=list)
    all_joints: list[Joint] = field(default_factory=list)
    wire_routes: list = field(default_factory=list)
    packing_result: PackingResult | None = None

    _assemblies: dict[str, Assembly] = field(default_factory=dict)
    _clearance_constraints: list[ClearanceConstraint] = field(default_factory=list)
    _cad_model: object = field(default=None, init=False, repr=False)
    clearance_results: list = field(default_factory=list)  # populated by build_cad()

    def assembly(self, name: str) -> Assembly:
        """Create or retrieve a named assembly."""
        if name not in self._assemblies:
            self._assemblies[name] = Assembly(name=name, _bot=self)
        return self._assemblies[name]

    def module(self, name: str) -> Assembly:
        """Backward-compatible alias for assembly()."""
        return self.assembly(name)

    def clearance(
        self,
        body_a: str,
        body_b: str,
        min_distance: Meters = _ZERO_M,
        label: str = "",
    ) -> None:
        """Declare an expected clearance between two bodies."""
        self._clearance_constraints.append(
            ClearanceConstraint(
                body_a=body_a,
                body_b=body_b,
                min_distance=min_distance,
                label=label,
            )
        )

    def _generate_implicit_constraints(self) -> None:
        """Auto-generate clearance constraints from assembly structure.

        Default rule: NO two bodies should intersect (min_distance=0).
        For each joint, all bodies in the joint neighborhood are checked
        against each other: parent body, child body, servo, horn.
        Mounted components are checked against their parent body.
        """
        existing = {(c.body_a, c.body_b) for c in self._clearance_constraints}
        existing |= {(c.body_b, c.body_a) for c in self._clearance_constraints}
        body_names = {b.name for b in self.all_bodies}

        def _add(a: str, b: str, min_dist: float, label: str) -> None:
            if a == b:
                return
            if (a, b) not in existing and (b, a) not in existing:
                self._clearance_constraints.append(
                    ClearanceConstraint(a, b, min_dist, label)
                )
                existing.add((a, b))

        for body in self.all_bodies:
            if body.kind != BodyKind.FABRICATED:
                continue
            for joint in body.joints:
                # Collect all bodies at this joint
                joint_bodies = []
                joint_bodies.append((body.name, "parent"))
                if joint.child:
                    joint_bodies.append((joint.child.name, "child"))
                servo_name = f"servo_{joint.name}"
                if servo_name in body_names:
                    joint_bodies.append((servo_name, "servo"))
                horn_name = f"horn_{joint.name}"
                if horn_name in body_names:
                    joint_bodies.append((horn_name, "horn"))

                # Check pairs — skip parent-servo (servo is intentionally
                # inside the bracket pocket which is part of the parent body).
                for ia in range(len(joint_bodies)):
                    for ib in range(ia + 1, len(joint_bodies)):
                        name_a, role_a = joint_bodies[ia]
                        name_b, role_b = joint_bodies[ib]
                        # Servo sits inside parent body's bracket pocket — expected
                        if {role_a, role_b} == {"parent", "servo"}:
                            continue
                        _add(
                            name_a,
                            name_b,
                            0.0,
                            f"{joint.name} {role_a}-{role_b} clearance",
                        )

            # Attachment (rigid) children — check parent-child clearance
            for attachment in body.attachments:
                if attachment.child is not None:
                    _add(
                        body.name,
                        attachment.child.name,
                        0.0,
                        f"{attachment.name} attachment clearance",
                    )

            # Mounted components must not intersect parent body.
            # Skip wheel components — the wheel component IS the wheel body
            # (same geometry), so they always fully overlap.
            for mount in body.mounts:
                if mount.component.kind == ComponentKind.WHEEL:
                    continue
                comp_name = f"comp_{body.name}_{mount.label}"
                if comp_name in body_names:
                    _add(
                        comp_name,
                        body.name,
                        0.0,
                        f"{mount.label} mount clearance",
                    )

    def body(
        self,
        name: str,
        shape: BodyShape = BodyShape.BOX,
        *,
        padding: float = 0.01,
        dimensions: Vec3 | None = None,
        custom_solid: object | None = None,
    ) -> Body:
        """Create the root body of the robot."""
        b = Body(
            name=name,
            shape=shape,
            padding=padding,
            explicit_dimensions=dimensions,
            custom_solid=custom_solid,
        )
        self.root = b
        return b

    def _collect_tree(self) -> None:
        """Walk the kinematic tree, populate all_bodies/all_joints, resolve assemblies.

        Also computes frame_quat for bodies whose geometry is reoriented
        relative to their canonical (Z-up) frame — currently cylinders that
        are aligned to their parent joint axis.
        """
        from botcad.geometry import rotation_between

        self.all_bodies.clear()
        self.all_joints.clear()

        def _walk(body: Body, parent_assembly: Assembly | None) -> None:
            # Inherit assembly from parent if not set explicitly
            if body.assembly is None:
                body.assembly = parent_assembly
            # Compute is_wheel_body from mounted components
            body.is_wheel_body = any(
                m.component.kind == ComponentKind.WHEEL for m in body.mounts
            )
            self.all_bodies.append(body)
            for joint in body.joints:
                self.all_joints.append(joint)
                if joint.child is not None:
                    child = joint.child
                    # Cylinder children are oriented so their Z axis aligns
                    # with the joint axis.  Record this rotation so mount
                    # point coordinates can be transformed consistently.
                    if child.shape is BodyShape.CYLINDER:
                        child.frame_quat = rotation_between((0.0, 0.0, 1.0), joint.axis)
                    _walk(child, body.assembly)
            for attachment in body.attachments:
                if attachment.child is not None:
                    child = attachment.child
                    if child.shape is BodyShape.CYLINDER:
                        child.frame_quat = rotation_between(
                            (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)
                        )
                    _walk(child, body.assembly)

        if self.root is not None:
            _walk(self.root, None)

    def _assembly_bodies(self, assembly_name: str) -> list[Body]:
        """Return all bodies belonging to the named assembly."""
        return [
            b
            for b in self.all_bodies
            if b.assembly and b.assembly.name == assembly_name
        ]

    def _assembly_joints(self, assembly_name: str) -> list[Joint]:
        """Return joints owned by the named assembly (parent body's assembly)."""
        parent_assembly: dict[str, str | None] = {}
        for body in self.all_bodies:
            for joint in body.joints:
                parent_assembly[joint.name] = (
                    body.assembly.name if body.assembly else None
                )
        return [
            j for j in self.all_joints if parent_assembly.get(j.name) == assembly_name
        ]

    # Backward-compatible aliases
    def _module_bodies(self, module_name: str) -> list[Body]:
        """Return all bodies belonging to the named module. Alias for _assembly_bodies."""
        return self._assembly_bodies(module_name)

    def _module_joints(self, module_name: str) -> list[Joint]:
        """Return joints owned by the named module. Alias for _assembly_joints."""
        return self._assembly_joints(module_name)

    def _compute_component_dimensions(self) -> None:
        """Execute component ShapeScripts to derive true bounding boxes.

        Runs each mounted component's ShapeScript through the OCCT backend
        and stores the actual bounding box on the mount.  This ensures the
        packing solver and pocket cutter use geometry-derived dimensions
        instead of the component's declared (approximate) dimensions.
        """
        import logging

        from botcad.shapescript.backend_occt import OcctBackend

        log = logging.getLogger(__name__)
        backend = OcctBackend()
        cache: dict[str, Vec3] = {}

        for body in self.all_bodies:
            for mount in body.mounts:
                comp = mount.component
                if comp.name in cache:
                    mount.solved_bbox = cache[comp.name]
                    continue

                prog = None
                try:
                    meta = get_component_meta(comp.kind)
                    if meta.script_emitter is not None:
                        prog = meta.script_emitter(comp)
                except Exception:
                    log.debug(
                        "No ShapeScript emitter for component %s, "
                        "using declared dimensions",
                        comp.name,
                    )

                if prog is not None and prog.output_ref is not None:
                    try:
                        result = backend.execute(prog)
                        solid = result.shapes.get(prog.output_ref.id)
                        if solid is not None:
                            bb = solid.bounding_box()
                            bbox: Vec3 = (
                                bb.max.X - bb.min.X,
                                bb.max.Y - bb.min.Y,
                                bb.max.Z - bb.min.Z,
                            )
                            mount.solved_bbox = bbox
                            cache[comp.name] = bbox

                            # Log significant discrepancies
                            declared = comp.dimensions
                            for i, axis in enumerate(["X", "Y", "Z"]):
                                diff = abs(bbox[i] - declared[i])
                                if diff > 0.001:
                                    log.info(
                                        "[dims] %s %s: declared=%.1fmm actual=%.1fmm",
                                        comp.name,
                                        axis,
                                        declared[i] * 1000,
                                        bbox[i] * 1000,
                                    )
                    except Exception:
                        log.debug(
                            "ShapeScript execution failed for %s, "
                            "using declared dimensions",
                            comp.name,
                            exc_info=True,
                        )

    def solve(self) -> None:
        """Run packing solver and wire routing."""
        from botcad.packing import solve_packing

        self._collect_tree()
        self._validate_bracket_rom()
        self._compute_component_dimensions()
        self.packing_result = solve_packing(self)
        self._assign_materials()

        # After packing has positioned servos and components, compute
        # world-frame transforms for structural bodies and create
        # purchased body instances (servos, horns, mounted components).
        self._compute_world_transforms()
        self._create_purchased_bodies()

        from botcad.routing import solve_routing

        self.wire_routes = solve_routing(self)

        # Auto-generate clearance constraints from assembly structure
        self._generate_implicit_constraints()

    def _compute_world_transforms(self) -> None:
        """Walk kinematic tree and set world_pose on each structural body.

        At rest pose (all joints at 0 deg), body frames are axis-aligned with
        their parents, so world position is the cumulative sum of joint.pos
        vectors up the chain.  Same logic as _compute_world_positions() in
        cad.py but stores results on Body objects directly.
        """

        def _walk(body: Body, parent_world_pos: Vec3) -> None:
            # Structural bodies at rest have identity orientation (frame_quat
            # describes local geometry rotation, not world orientation).
            body.world_pose = Pose(parent_world_pos)
            for joint in body.joints:
                if joint.child is not None:
                    child = joint.child
                    if child.is_wheel_body:
                        offset = joint.wheel_outboard_offset()
                        ax, ay, az = joint.axis
                        child_pos: Vec3 = (
                            parent_world_pos[0] + joint.pos[0] + ax * offset,
                            parent_world_pos[1] + joint.pos[1] + ay * offset,
                            parent_world_pos[2] + joint.pos[2] + az * offset,
                        )
                    else:
                        child_pos = (
                            parent_world_pos[0] + joint.pos[0],
                            parent_world_pos[1] + joint.pos[1],
                            parent_world_pos[2] + joint.pos[2],
                        )
                    _walk(child, child_pos)
            for attachment in body.attachments:
                if attachment.child is not None:
                    child_pos = (
                        parent_world_pos[0] + attachment.pos[0],
                        parent_world_pos[1] + attachment.pos[1],
                        parent_world_pos[2] + attachment.pos[2],
                    )
                    _walk(attachment.child, child_pos)

        if self.root is not None:
            _walk(self.root, (0.0, 0.0, 0.0))

    def _create_purchased_bodies(self) -> None:
        """Create Body instances for all purchased parts (servos, horns, mounts).

        Positions are derived from the parent structural body's world_pos
        combined with the solved placement offsets from the packing solver.
        Appended to all_bodies so exporters can iterate a single list.
        """
        from botcad.bracket import horn_disc_params

        # Iterate over a snapshot of structural bodies (avoid mutating
        # all_bodies while iterating).
        structural_bodies = list(self.all_bodies)

        placements = self.packing_result.placements if self.packing_result else {}

        for body in structural_bodies:
            bwp = body.world_pos

            # --- Servo bodies for each joint ---
            for joint in body.joints:
                servo_body = Body(
                    name=f"servo_{joint.name}",
                    kind=BodyKind.PURCHASED,
                    parent_body_name=body.name,
                )
                # Servo center is in the parent body frame; transform to world
                j_place = placements.get(joint)
                sc = j_place.pose.pos if j_place else joint.solved_servo_center
                sq = j_place.pose.quat if j_place else joint.solved_servo_quat
                servo_body.world_pose = Pose(
                    (bwp[0] + sc[0], bwp[1] + sc[1], bwp[2] + sc[2]),
                    sq,
                )
                servo_body.mesh_file = f"servo_{joint.servo.name}.stl"
                servo_body._component = joint.servo
                self.all_bodies.append(servo_body)

                # --- Horn disc body ---
                params = horn_disc_params(joint.servo)
                if params is not None:
                    horn_body = Body(
                        name=f"horn_{joint.name}",
                        kind=BodyKind.PURCHASED,
                        parent_body_name=body.name,
                    )
                    # Horn sits on the shaft boss tip, offset outboard from
                    # the joint point by shaft_boss_height + half_horn_thickness.
                    jx = bwp[0] + joint.pos[0]
                    jy = bwp[1] + joint.pos[1]
                    jz = bwp[2] + joint.pos[2]
                    ax, ay, az = joint.axis
                    boss_h = joint.servo.shaft_boss_height or 0.0
                    half_t = params.thickness / 2
                    outboard = boss_h + half_t
                    # Orientation: Z-up rotated to joint axis
                    from botcad.geometry import rotation_between

                    horn_body.world_pose = Pose(
                        (jx + ax * outboard, jy + ay * outboard, jz + az * outboard),
                        rotation_between((0.0, 0.0, 1.0), joint.axis),
                    )
                    horn_body.mesh_file = f"horn_{joint.name}.stl"
                    horn_body._component = joint.servo  # horn is derived from servo
                    self.all_bodies.append(horn_body)

            # --- Mounted components (battery, camera, Pi, etc.) ---
            for mount in body.mounts:
                comp_body = Body(
                    name=f"comp_{body.name}_{mount.label}",
                    kind=BodyKind.PURCHASED,
                    parent_body_name=body.name,
                )
                m_place = placements.get(mount)
                rp = m_place.pose.pos if m_place else mount.resolved_pos
                comp_body.world_pose = Pose(
                    (bwp[0] + rp[0], bwp[1] + rp[1], bwp[2] + rp[2]),
                    m_place.pose.quat if m_place else body.world_quat,
                )
                comp_body.mesh_file = f"comp_{body.name}_{mount.label}.stl"
                comp_body._component = mount.component
                self.all_bodies.append(comp_body)

    def _validate_bracket_rom(self) -> None:
        """Warn if any joint's ROM exceeds its bracket style's safe range."""
        import math
        import warnings

        for joint in self.all_joints:
            if joint.bracket_style is not BracketStyle.COUPLER:
                continue

            from botcad.bracket import coupler_max_rom_rad

            max_rom = coupler_max_rom_rad(joint.servo)
            lo, hi = joint.effective_range
            if abs(lo) > max_rom or abs(hi) > max_rom:
                requested = max(abs(lo), abs(hi))
                warnings.warn(
                    f"Joint '{joint.name}': coupler bracket safe ROM is "
                    f"±{math.degrees(max_rom):.0f}° but joint requests "
                    f"±{math.degrees(requested):.0f}°. "
                    f"Reduce range_rad or use bracket_style='pocket'.",
                    stacklevel=2,
                )

    def _assign_materials(self) -> None:
        """Set material on every body that doesn't already have one explicitly set.

        Purchased bodies (with mounted components) inherit their component's
        default_material. Fabricated bodies get shape-based defaults.
        """
        from botcad.materials import (
            MAT_ABS_DARK,
            MAT_PLA_LIGHT,
            MAT_TUBE_DEFAULT,
        )

        for body in self.all_bodies:
            # Purchased body: inherit from first component
            if body.mounts and body.mounts[0].component.default_material is not None:
                body.material = body.mounts[0].component.default_material
                continue
            # Fabricated body: shape-based default
            if body.shape is BodyShape.CYLINDER and body.radius and body.radius > 0.03:
                body.material = MAT_ABS_DARK
            elif body.shape is BodyShape.TUBE:
                body.material = MAT_TUBE_DEFAULT
            elif body.shape is BodyShape.JAW:
                body.material = MAT_PLA_LIGHT
            else:
                body.material = MAT_PLA_LIGHT

    def build_cad(self):
        """Build CAD geometry and refine mass/inertia from actual solids."""
        from botcad.emit.cad import build_cad

        self._cad_model = build_cad(self)

    def _resolve_output_dir(self, output_dir: str | None = None):
        """Resolve output directory, creating it if needed."""
        from pathlib import Path

        if output_dir is None:
            output_dir_path = Path("bots") / self.name
        else:
            output_dir_path = Path(output_dir)

        output_dir_path.mkdir(parents=True, exist_ok=True)
        return output_dir_path

    def _ensure_cad(self) -> None:
        """Build CAD geometry if not already done."""
        if self._cad_model is None:
            self.build_cad()

    def write_mujoco(self, output_dir: str | None = None) -> None:
        """Write bot.xml + scene.xml + STL meshes for MuJoCo simulation."""
        output_dir_path = self._resolve_output_dir(output_dir)
        (output_dir_path / "meshes").mkdir(exist_ok=True)

        self._ensure_cad()

        from botcad.emit.cad import emit_cad

        emit_cad(self, output_dir_path, self._cad_model)

        from botcad.emit.mujoco import emit_mujoco

        emit_mujoco(self, output_dir_path)

    def write_step(self, output_dir: str | None = None) -> None:
        """Write STEP assembly files for manufacturing/CAD."""
        output_dir_path = self._resolve_output_dir(output_dir)
        (output_dir_path / "meshes").mkdir(exist_ok=True)

        self._ensure_cad()

        from botcad.emit.cad import emit_cad

        parts = emit_cad(self, output_dir_path, self._cad_model)

        # Per-assembly STEP files
        if self._assemblies:
            from botcad.emit.cad import emit_cad_for_assembly

            for asm_name in self._assemblies:
                emit_cad_for_assembly(self, asm_name, output_dir_path, parts)

    def write_docs(self, output_dir: str | None = None) -> None:
        """Write BOM, assembly guide, and technical drawings."""
        output_dir_path = self._resolve_output_dir(output_dir)

        from botcad.emit.bom import emit_bom

        emit_bom(self, output_dir_path)

        from botcad.emit.readme import emit_assembly_guide

        emit_assembly_guide(self, output_dir_path)

        from botcad.emit.drawings import emit_drawings

        emit_drawings(self, output_dir_path)

        # Per-assembly docs
        if self._assemblies:
            from botcad.emit.bom import emit_bom_for_module
            from botcad.emit.readme import emit_assembly_guide_for_module

            for asm_name in self._assemblies:
                emit_bom_for_module(self, asm_name, output_dir_path)
                emit_assembly_guide_for_module(self, asm_name, output_dir_path)

    def write_renders(self, output_dir: str | None = None) -> None:
        """Write overview and sweep render PNGs."""
        output_dir_path = self._resolve_output_dir(output_dir)

        from botcad.emit.renders import emit_renders

        emit_renders(self, output_dir_path)

    def write_viewer_manifest(self, output_dir: str | None = None) -> None:
        """Write viewer_manifest.json — metadata for the web viewer."""
        output_dir_path = self._resolve_output_dir(output_dir)

        from botcad.emit.viewer import emit_viewer_manifest

        emit_viewer_manifest(self, output_dir_path)

    def analyze_stresses(self):
        """Analyze the safety factor of all fabricated joints in the bot."""
        from botcad.fea import analyze_joint_stresses

        return analyze_joint_stresses(self)

    def print_fea_report(self):
        """Print a human-readable FEA report for the bot."""
        from botcad.fea import print_fea_report

        print_fea_report(self)

    def emit(self, output_dir: str | None = None) -> None:
        """Generate all output files.

        Convenience method that calls all write_*() methods. Prefer calling
        individual methods when you only need specific outputs.
        """
        self.write_mujoco(output_dir)
        self.write_step(output_dir)
        self.write_docs(output_dir)
        self.write_renders(output_dir)
        self.write_viewer_manifest(output_dir)


def _parse_axis(axis: str | Vec3) -> Vec3:
    """Convert axis shorthand to a 3-tuple."""
    if isinstance(axis, str):
        mapping = {
            "x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
            "-z": (0.0, 0.0, -1.0),
        }
        return mapping.get(axis.lower(), (0.0, 0.0, 1.0))
    return axis
