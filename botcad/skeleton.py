"""Kinematic tree DSL for parametric robot design.

Defines Bot, Body, and Joint classes that form the user-facing API for
building robots from real components. The kinematic tree is:

    Bot
    └── Body (root)
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

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from botcad.component import Component, ServoSpec, Vec3

Position = Literal["center", "bottom", "top", "front", "back", "left", "right"]


class BodyShape(StrEnum):
    """Geometric shape of a rigid body."""

    BOX = "box"
    CYLINDER = "cylinder"
    TUBE = "tube"
    SPHERE = "sphere"
    JAW = "jaw"


class BracketStyle(StrEnum):
    """Servo bracket mounting style."""

    POCKET = "pocket"
    COUPLER = "coupler"


class BaseType(StrEnum):
    """How the robot root is attached to the world."""

    FREE = "free"
    FIXED = "fixed"


@dataclass
class Module:
    """A named fabrication unit — a group of bodies printed/assembled together."""

    name: str
    _bot: Bot = field(repr=False)

    def body(self, name: str, shape: BodyShape = BodyShape.BOX, **kwargs) -> Body:
        """Create a body in this module. First body created becomes bot root."""
        b = Body(name=name, shape=shape, module=self, **kwargs)
        if self._bot.root is None:
            self._bot.root = b
        return b


@dataclass
class Mount:
    """A component placed inside a body."""

    component: Component
    label: str
    position: Position | Vec3  # heuristic position or explicit (x, y, z)
    insertion_axis: Vec3 | None = (
        None  # explicit override, or None = derive from position
    )
    rotate_z: bool = False  # if True, swap X/Y dimensions (90° around Z)
    resolved_pos: Vec3 = (0.0, 0.0, 0.0)  # filled by packing solver
    resolved_insertion_axis: Vec3 = (0.0, 0.0, 1.0)  # filled by packing solver

    @property
    def placed_dimensions(self) -> Vec3:
        """Component dimensions in the body frame (X/Y swapped if rotate_z)."""
        d = self.component.dimensions
        if self.rotate_z:
            return (d[1], d[0], d[2])
        return d

    def rotate_point(self, p: Vec3) -> Vec3:
        """Rotate a component-local point into the body frame."""
        if self.rotate_z:
            return (-p[1], p[0], p[2])
        return p


@dataclass
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

        Offset = shaft_boss_height + half_wheel_width.  Returns 0 if no child.
        """
        if self.child is None:
            return 0.0
        boss_h = self.servo.shaft_boss_height or 0.0
        half_w = (self.child.width or self.child.dimensions[2]) / 2
        return boss_h + half_w

    def body(
        self,
        name: str,
        shape: BodyShape = BodyShape.BOX,
        *,
        radius: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
        length: float = 0.0,
        outer_r: float = 0.0,
        jaw_length: float = 0.0,
        jaw_width: float = 0.0,
        jaw_thickness: float = 0.0,
        padding: float = 0.005,
        dimensions: Vec3 | None = None,
        custom_solid: object | None = None,
        module: Module | None = None,
    ) -> Body:
        """Create and attach a child body to this joint."""
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
            module=module,
        )
        self.child = b
        return b

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass
class Body:
    """A rigid body in the kinematic tree.

    Bodies contain mounted components and child joints. Their size is either
    set explicitly or computed by the packing solver to fit all contents.

    Identity-based hashing allows functools.lru_cache on geometry functions
    that take Body arguments.  Within a bot build, each Body is a singleton.
    """

    name: str
    shape: BodyShape = BodyShape.BOX
    radius: float = 0.0
    width: float = 0.0
    height: float = 0.0
    length: float = 0.0
    outer_r: float = 0.0
    jaw_length: float = 0.0
    jaw_width: float = 0.0
    jaw_thickness: float = 0.0
    padding: float = 0.005  # clearance around components (meters)
    explicit_dimensions: Vec3 | None = None
    custom_solid: object | None = None
    module: Module | None = None

    mounts: list[Mount] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)

    # Computed by packing solver
    solved_dimensions: Vec3 | None = None
    solved_mass: float = 0.0
    solved_com: Vec3 = (0.0, 0.0, 0.0)
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
        position: Position | Vec3 = "center",
        label: str = "",
        insertion_axis: Vec3 | None = None,
        rotate_z: bool = False,
    ) -> Mount:
        """Mount a component inside this body.

        rotate_z: if True, component is rotated 90° around Z (swaps X/Y dims).
        Use this when the component's long axis should run along Y (e.g. Pi
        board running front-to-back in a wheeler chassis).
        """
        if not label:
            label = component.name.lower()
        m = Mount(
            component=component,
            label=label,
            position=position,
            insertion_axis=insertion_axis,
            rotate_z=rotate_z,
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

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass
class Bot:
    """Top-level robot definition.

    Usage:
        bot = Bot("wheeler_arm")
        base = bot.body("base", padding=0.01)
        base.mount(STS3215(), ...)
        # ... define kinematic tree ...
        bot.solve()  # run packing + routing
        bot.emit()   # generate all outputs
    """

    name: str
    root: Body | None = None
    base_type: BaseType = BaseType.FREE

    # Populated by solve()
    all_bodies: list[Body] = field(default_factory=list)
    all_joints: list[Joint] = field(default_factory=list)
    wire_routes: list = field(default_factory=list)

    _modules: dict[str, Module] = field(default_factory=dict)
    _cad_model: object = field(default=None, init=False, repr=False)

    def module(self, name: str) -> Module:
        """Create or retrieve a named module (fabrication unit)."""
        if name not in self._modules:
            self._modules[name] = Module(name=name, _bot=self)
        return self._modules[name]

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
        """Walk the kinematic tree, populate all_bodies/all_joints, resolve modules.

        Also computes frame_quat for bodies whose geometry is reoriented
        relative to their canonical (Z-up) frame — currently cylinders that
        are aligned to their parent joint axis.
        """
        from botcad.geometry import rotation_between

        self.all_bodies.clear()
        self.all_joints.clear()

        def _walk(body: Body, parent_module: Module | None) -> None:
            # Inherit module from parent if not set explicitly
            if body.module is None:
                body.module = parent_module
            # Compute is_wheel_body from mounted components
            body.is_wheel_body = any(m.component.is_wheel for m in body.mounts)
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
                    _walk(child, body.module)

        if self.root is not None:
            _walk(self.root, None)

    def _module_bodies(self, module_name: str) -> list[Body]:
        """Return all bodies belonging to the named module."""
        return [b for b in self.all_bodies if b.module and b.module.name == module_name]

    def _module_joints(self, module_name: str) -> list[Joint]:
        """Return joints owned by the named module (parent body's module)."""
        parent_module: dict[str, str | None] = {}
        for body in self.all_bodies:
            for joint in body.joints:
                parent_module[joint.name] = body.module.name if body.module else None
        return [j for j in self.all_joints if parent_module.get(j.name) == module_name]

    def solve(self) -> None:
        """Run packing solver and wire routing."""
        from botcad.packing import solve_packing

        self._collect_tree()
        self._validate_bracket_rom()
        solve_packing(self)

        from botcad.routing import solve_routing

        self.wire_routes = solve_routing(self)

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

    def build_cad(self):
        """Build CAD geometry and refine mass/inertia from actual solids."""
        from botcad.emit.cad import build_cad

        self._cad_model = build_cad(self)

    def emit(self, output_dir: str | None = None) -> None:
        """Generate all output files."""
        from pathlib import Path

        if output_dir is None:
            output_dir_path = Path("bots") / self.name
        else:
            output_dir_path = Path(output_dir)

        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "meshes").mkdir(exist_ok=True)

        # Build geometry + refine mass (if not already done)
        if self._cad_model is None:
            self.build_cad()

        from botcad.emit.cad import emit_cad

        parts = emit_cad(self, output_dir_path, self._cad_model)

        from botcad.emit.mujoco import emit_mujoco

        emit_mujoco(self, output_dir_path)

        from botcad.emit.bom import emit_bom

        emit_bom(self, output_dir_path)

        from botcad.emit.readme import emit_assembly_guide

        emit_assembly_guide(self, output_dir_path)

        from botcad.emit.renders import emit_renders

        emit_renders(self, output_dir_path)

        from botcad.emit.component_renders import emit_component_renders

        emit_component_renders(self, output_dir_path)

        from botcad.emit.assembly_renders import emit_assembly_renders

        emit_assembly_renders(self, output_dir_path)

        from botcad.emit.drawings import emit_drawings

        emit_drawings(self, output_dir_path)

        # Per-module outputs (STEP, BOM, assembly guide)
        if self._modules:
            from botcad.emit.bom import emit_bom_for_module
            from botcad.emit.cad import emit_cad_for_module
            from botcad.emit.readme import emit_assembly_guide_for_module

            for mod_name in self._modules:
                emit_cad_for_module(self, mod_name, output_dir_path, parts)
                emit_bom_for_module(self, mod_name, output_dir_path)
                emit_assembly_guide_for_module(self, mod_name, output_dir_path)

        from botcad.emit.viewer import emit_viewer_manifest

        emit_viewer_manifest(self, output_dir_path)


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
