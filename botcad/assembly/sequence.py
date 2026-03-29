"""Assembly sequence model.

Represents the step-by-step process of building a robot:
what gets installed when, what tools are needed, and what
geometry exists at each point in the build.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from botcad.assembly.refs import ComponentRef, FastenerRef, WireRef
from botcad.assembly.tools import ToolKind
from botcad.ids import BodyId, JointId

if TYPE_CHECKING:
    from botcad.units import Radians


class AssemblyAction(Enum):
    INSERT = "insert"
    FASTEN = "fasten"
    ROUTE_WIRE = "route_wire"
    CONNECT = "connect"
    ARTICULATE = "articulate"


@dataclass(frozen=True)
class AssemblyOp:
    """A single step in the assembly sequence."""

    step: int
    action: AssemblyAction
    target: ComponentRef | FastenerRef | WireRef | JointId
    body: BodyId
    tool: ToolKind | None
    approach_axis: tuple[float, float, float] | None
    angle: Radians | None  # only for ARTICULATE
    prerequisites: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class AssemblyState:
    """Snapshot of what's physically present at a point in the assembly."""

    installed_components: frozenset[ComponentRef]
    installed_fasteners: frozenset[FastenerRef]
    routed_wires: frozenset[WireRef]
    joint_angles: dict[JointId, float]  # can't freeze dicts, but this is fine


@dataclass(frozen=True)
class AssemblySequence:
    """Ordered list of assembly operations with state replay."""

    ops: tuple[AssemblyOp, ...]

    def state_at(self, step: int) -> AssemblyState:
        """Compute what's installed after completing step N.

        Replays ops 0..step, accumulating installed refs and joint angles.
        Joint angles start at 0.0; ARTICULATE ops set absolute values.
        step=-1 means before any ops (empty state).
        """
        components: set[ComponentRef] = set()
        fasteners: set[FastenerRef] = set()
        wires: set[WireRef] = set()
        angles: dict[JointId, float] = {}

        for op in self.ops:
            if op.step > step:
                break
            if op.action == AssemblyAction.INSERT and isinstance(
                op.target, ComponentRef
            ):
                components.add(op.target)
            elif op.action == AssemblyAction.FASTEN and isinstance(
                op.target, FastenerRef
            ):
                fasteners.add(op.target)
            elif op.action in (
                AssemblyAction.ROUTE_WIRE,
                AssemblyAction.CONNECT,
            ) and isinstance(op.target, WireRef):
                wires.add(op.target)
            elif op.action == AssemblyAction.ARTICULATE and isinstance(
                op.target, JointId
            ):
                angles[op.target] = op.angle

        return AssemblyState(
            installed_components=frozenset(components),
            installed_fasteners=frozenset(fasteners),
            routed_wires=frozenset(wires),
            joint_angles=angles,
        )
