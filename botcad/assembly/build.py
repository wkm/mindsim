"""Generate an assembly sequence from a bot skeleton.

Walks the kinematic tree breadth-first, emitting typed AssemblyOps
in assembly order: components first, then fasteners, then wires.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from botcad.assembly.refs import ComponentRef, FastenerRef, WireRef
from botcad.assembly.sequence import AssemblyAction, AssemblyOp, AssemblySequence
from botcad.assembly.tools import ToolKind
from botcad.component import MountPoint
from botcad.fasteners import HeadType, resolve_fastener
from botcad.ids import BodyId
from botcad.routing import WireRoute, solve_routing

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot


def build_assembly_sequence(bot: Bot) -> AssemblySequence:
    """Walk bot skeleton and emit assembly ops in build order."""
    if bot.root is None:
        return AssemblySequence(ops=())

    ops: list[AssemblyOp] = []
    step = 0

    # BFS walk: parent bodies before children
    queue: deque[Body] = deque([bot.root])
    visited: set[BodyId] = set()

    while queue:
        body = queue.popleft()
        if body.name in visited:
            continue
        visited.add(body.name)

        # Track insert step indices so fasteners can reference them as prereqs
        insert_steps: dict[str, int] = {}  # mount_label -> step index

        # --- INSERT ops for mounted components ---
        for mount in body.mounts:
            approach = _mount_approach_axis(mount)
            ref = ComponentRef(body=body.name, mount_label=mount.label)
            op = AssemblyOp(
                step=step,
                action=AssemblyAction.INSERT,
                target=ref,
                body=body.name,
                tool=ToolKind.FINGERS,
                approach_axis=approach,
                angle=None,
                prerequisites=(),
                description=f"Insert {mount.component.name} into {body.name} ({mount.label})",
            )
            insert_steps[mount.label] = step
            ops.append(op)
            step += 1

        # --- INSERT ops for servos on joints of this body ---
        servo_insert_steps: dict[str, int] = {}  # joint_name -> step index
        for joint in body.joints:
            ref = ComponentRef(
                body=body.name,
                mount_label=f"servo_{joint.name}",
            )
            op = AssemblyOp(
                step=step,
                action=AssemblyAction.INSERT,
                target=ref,
                body=body.name,
                tool=ToolKind.FINGERS,
                approach_axis=(0.0, 0.0, -1.0),
                angle=None,
                prerequisites=(),
                description=f"Insert servo {joint.servo.name} for joint {joint.name}",
            )
            servo_insert_steps[str(joint.name)] = step
            ops.append(op)
            step += 1

        # --- FASTEN ops for component mounting points ---
        fastener_index = 0
        for mount in body.mounts:
            for mp in mount.component.mounting_points:
                tool = _tool_for_mount_point(mp)
                approach = _negate_vec3(mp.axis)
                prereqs = ()
                if mount.label in insert_steps:
                    prereqs = (insert_steps[mount.label],)
                ref = FastenerRef(body=body.name, index=fastener_index)
                op = AssemblyOp(
                    step=step,
                    action=AssemblyAction.FASTEN,
                    target=ref,
                    body=body.name,
                    tool=tool,
                    approach_axis=approach,
                    angle=None,
                    prerequisites=prereqs,
                    description=f"Fasten {mp.label} on {mount.label} ({body.name})",
                )
                ops.append(op)
                step += 1
                fastener_index += 1

        # --- FASTEN ops for servo mounting ears ---
        for joint in body.joints:
            for mp in joint.servo.mounting_ears:
                tool = _tool_for_mount_point(mp)
                approach = _negate_vec3(mp.axis)
                prereqs = ()
                jname = str(joint.name)
                if jname in servo_insert_steps:
                    prereqs = (servo_insert_steps[jname],)
                ref = FastenerRef(body=body.name, index=fastener_index)
                op = AssemblyOp(
                    step=step,
                    action=AssemblyAction.FASTEN,
                    target=ref,
                    body=body.name,
                    tool=tool,
                    approach_axis=approach,
                    angle=None,
                    prerequisites=prereqs,
                    description=f"Fasten servo ear {mp.label} on joint {joint.name}",
                )
                ops.append(op)
                step += 1
                fastener_index += 1

        # Enqueue child bodies
        for joint in body.joints:
            if joint.child is not None:
                queue.append(joint.child)

    # --- Wire routes ---
    routes = solve_routing(bot)
    wire_step_map: dict[str, int] = {}  # route_label -> ROUTE_WIRE step
    for route in routes:
        if not route.segments:
            continue
        ref = WireRef(label=route.label)
        op = AssemblyOp(
            step=step,
            action=AssemblyAction.ROUTE_WIRE,
            target=ref,
            body=bot.root.name,
            tool=ToolKind.FINGERS,
            approach_axis=None,
            angle=None,
            prerequisites=(),
            description=f"Route {route.label} wire ({route.bus_type})",
        )
        wire_step_map[route.label] = step
        ops.append(op)
        step += 1

    # --- CONNECT ops for wire routes with connectors ---
    for route in routes:
        if not route.segments:
            continue
        # Each route with non-permanent connector endpoints gets a CONNECT op
        if _route_has_connectors(route, bot):
            prereqs = ()
            if route.label in wire_step_map:
                prereqs = (wire_step_map[route.label],)
            ref = WireRef(label=route.label)
            op = AssemblyOp(
                step=step,
                action=AssemblyAction.CONNECT,
                target=ref,
                body=bot.root.name,
                tool=ToolKind.FINGERS,
                approach_axis=None,
                angle=None,
                prerequisites=prereqs,
                description=f"Connect {route.label} wire ({route.bus_type})",
            )
            ops.append(op)
            step += 1

    return AssemblySequence(ops=tuple(ops))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mount_approach_axis(mount) -> tuple[float, float, float]:
    """Determine approach axis for inserting a component."""
    if mount.insertion_axis is not None:
        return mount.insertion_axis
    # Use resolved_insertion_axis if it's been set by solver
    ria = mount.resolved_insertion_axis
    if ria != (0.0, 0.0, 1.0):
        # Non-default = solver set it
        return ria
    # Default: insert from top (-Z approach)
    return (0.0, 0.0, -1.0)


def _tool_for_mount_point(mp: MountPoint) -> ToolKind:
    """Map a MountPoint's fastener spec to the appropriate tool."""
    spec = resolve_fastener(mp)
    if spec.head_type == HeadType.PAN_HEAD_PHILLIPS:
        # Phillips screws
        if spec.thread_diameter <= 0.0025:
            return ToolKind.PHILLIPS_0
        return ToolKind.PHILLIPS_1

    # Socket head cap (hex key)
    d_mm = spec.thread_diameter * 1000
    if d_mm <= 2.0:
        return ToolKind.HEX_KEY_2
    if d_mm <= 2.5:
        return ToolKind.HEX_KEY_2_5
    return ToolKind.HEX_KEY_3


def _negate_vec3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-v[0], -v[1], -v[2])


def _route_has_connectors(route: WireRoute, bot: Bot) -> bool:
    """Check if a wire route has removable connector endpoints."""
    # All wire routes that cross joints or connect to non-permanent ports
    # need a CONNECT step. For simplicity, treat every route as having
    # at least one pluggable connector.
    return True
