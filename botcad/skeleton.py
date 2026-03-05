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

import math
from dataclasses import dataclass, field
from typing import Literal

from botcad.component import Component, ServoSpec, Vec3

Position = Literal["center", "bottom", "top", "front", "back", "left", "right"]


@dataclass
class Mount:
    """A component placed inside a body."""

    component: Component
    label: str
    position: Position | Vec3  # heuristic position or explicit (x, y, z)
    resolved_pos: Vec3 = (0.0, 0.0, 0.0)  # filled by packing solver


@dataclass
class Joint:
    """A revolute joint connecting a parent body to a child body via a servo."""

    name: str
    servo: ServoSpec
    axis: Vec3  # rotation axis in parent frame
    pos: Vec3  # joint position relative to parent body origin
    range_rad: tuple[float, float] | None = None  # override servo default
    child: Body | None = None

    @property
    def effective_range(self) -> tuple[float, float]:
        if self.range_rad is not None:
            return self.range_rad
        return self.servo.range_rad

    def body(
        self,
        name: str,
        shape: str = "box",
        *,
        radius: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
        length: float = 0.0,
        outer_r: float = 0.0,
        padding: float = 0.005,
        dimensions: Vec3 | None = None,
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
            padding=padding,
            explicit_dimensions=dimensions,
        )
        self.child = b
        return b


@dataclass
class Body:
    """A rigid body in the kinematic tree.

    Bodies contain mounted components and child joints. Their size is either
    set explicitly or computed by the packing solver to fit all contents.
    """

    name: str
    shape: str = "box"  # box, cylinder, tube, sphere
    radius: float = 0.0
    width: float = 0.0
    height: float = 0.0
    length: float = 0.0
    outer_r: float = 0.0
    padding: float = 0.005  # clearance around components (meters)
    explicit_dimensions: Vec3 | None = None

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

    @property
    def dimensions(self) -> Vec3:
        """Effective dimensions (explicit > solved > shape-derived default)."""
        if self.explicit_dimensions is not None:
            return self.explicit_dimensions
        if self.solved_dimensions is not None:
            return self.solved_dimensions
        return self._shape_default_dimensions()

    def _shape_default_dimensions(self) -> Vec3:
        if self.shape == "cylinder":
            r = self.radius or 0.02
            h = self.width or self.height or 0.01
            return (r * 2, r * 2, h)
        if self.shape == "tube":
            r = self.outer_r or 0.02
            ln = self.length or 0.1
            return (r * 2, r * 2, ln)
        if self.shape == "sphere":
            r = self.radius or 0.02
            return (r * 2, r * 2, r * 2)
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
    ) -> Mount:
        """Mount a component inside this body."""
        if not label:
            label = component.name.lower()
        m = Mount(component=component, label=label, position=position)
        self.mounts.append(m)
        return m

    def joint(
        self,
        name: str,
        servo: ServoSpec,
        axis: str | Vec3 = "z",
        pos: Vec3 = (0.0, 0.0, 0.0),
        range: tuple[float, float] | None = None,
    ) -> Joint:
        """Add a joint (with servo) connecting to a new child body."""
        axis_vec = _parse_axis(axis)
        j = Joint(name=name, servo=servo, axis=axis_vec, pos=pos, range_rad=range)
        self.joints.append(j)
        return j


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
    base_type: Literal["free", "fixed"] = (
        "free"  # free = freejoint, fixed = bolted down
    )

    # Populated by solve()
    all_bodies: list[Body] = field(default_factory=list)
    all_joints: list[Joint] = field(default_factory=list)
    wire_routes: list = field(default_factory=list)

    def body(
        self,
        name: str,
        shape: str = "box",
        *,
        padding: float = 0.01,
        dimensions: Vec3 | None = None,
    ) -> Body:
        """Create the root body of the robot."""
        b = Body(
            name=name, shape=shape, padding=padding, explicit_dimensions=dimensions
        )
        self.root = b
        return b

    def _collect_tree(self) -> None:
        """Walk the kinematic tree and populate all_bodies / all_joints."""
        self.all_bodies.clear()
        self.all_joints.clear()

        def _walk(body: Body) -> None:
            self.all_bodies.append(body)
            for joint in body.joints:
                self.all_joints.append(joint)
                if joint.child is not None:
                    _walk(joint.child)

        if self.root is not None:
            _walk(self.root)

    def solve(self) -> None:
        """Run packing solver and wire routing."""
        from botcad.packing import solve_packing

        self._collect_tree()
        solve_packing(self)

        from botcad.routing import solve_routing

        self.wire_routes = solve_routing(self)

    def emit(self, output_dir: str | None = None) -> None:
        """Generate all output files."""
        from pathlib import Path

        if output_dir is None:
            output_dir_path = Path("bots") / self.name
        else:
            output_dir_path = Path(output_dir)

        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "meshes").mkdir(exist_ok=True)

        self._collect_tree()

        from botcad.emit.cad import emit_cad

        emit_cad(self, output_dir_path)

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


def _magnitude(v: Vec3) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
