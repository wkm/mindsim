"""Wire routing along the kinematic skeleton.

STS3215 servos use a daisy-chain serial bus (UART half-duplex), so instead
of individual wires from each servo to the Pi, we route a single bus along
the kinematic chain.

Wire segments use body-local coordinates: each segment stores its start/end
positions in the frame of the body it belongs to, plus a body_name field.
This maps directly to MuJoCo's body hierarchy — segments become geoms on
their parent body and move with it automatically.

Joint crossings produce two segments. When a wire crosses a joint, it splits
into a parent-body segment (ending at the bracket cable exit) and a child-body
segment (starting near the joint origin). At rest these endpoints are close
together; when the joint moves, the gap between them represents the real
cable flex zone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from botcad.bracket import BracketSpec
from botcad.component import Vec3
from botcad.geometry import rotate_vec, servo_placement

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot, Joint


@dataclass
class WireSegment:
    """A segment of wire between two points in body-local coordinates."""

    start: Vec3  # body-local coordinates
    end: Vec3  # body-local coordinates
    body_name: str  # which body this segment lives on
    joint_name: str | None = None  # if crossing a joint
    slack: float = 0.0  # extra length needed for joint motion (meters)

    @property
    def straight_length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        dz = self.end[2] - self.start[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @property
    def total_length(self) -> float:
        return self.straight_length + self.slack


@dataclass
class WireRoute:
    """A complete wire route from source to destination."""

    label: str  # e.g. "servo_bus", "camera_csi", "power"
    bus_type: str
    segments: list[WireSegment] = field(default_factory=list)

    @property
    def total_length(self) -> float:
        return sum(s.total_length for s in self.segments)

    @property
    def total_slack(self) -> float:
        return sum(s.slack for s in self.segments)


def solve_routing(bot: Bot) -> list[WireRoute]:
    """Compute wire routes for the bot."""
    routes: list[WireRoute] = []

    # 1. Servo daisy-chain bus: Pi UART → servo chain
    servo_route = _route_servo_bus(bot)
    if servo_route.segments:
        routes.append(servo_route)

    # 2. Camera CSI ribbon: camera → Pi
    csi_route = _route_camera_csi(bot)
    if csi_route.segments:
        routes.append(csi_route)

    # 3. Power bus: battery → Pi
    power_route = _route_power(bot)
    if power_route.segments:
        routes.append(power_route)

    return routes


# ── Coordinate helpers ──


def _servo_connector_local(joint: Joint) -> Vec3:
    """Servo connector position in parent body frame.

    Uses servo_placement() to position the servo at the joint, then
    rotates the connector_pos into the parent body frame.
    """
    servo = joint.servo
    if servo.connector_pos is None:
        return joint.pos
    center, quat = servo_placement(
        servo.shaft_offset,
        servo.shaft_axis,
        joint.axis,
        joint.pos,
    )
    return _add_vec3(center, rotate_vec(quat, servo.connector_pos))


def _cable_exit_local(joint: Joint) -> Vec3:
    """Bracket cable exit position in parent body frame.

    The cable exits at the -X face of the bracket (in servo-local frame),
    at the same Y/Z as the connector.
    """
    servo = joint.servo
    if servo.connector_pos is None:
        return joint.pos
    center, quat = servo_placement(
        servo.shaft_offset,
        servo.shaft_axis,
        joint.axis,
        joint.pos,
    )
    body_x = servo.body_dimensions[0] or servo.dimensions[0]
    spec = BracketSpec()
    bracket_half_x = body_x / 2 + spec.tolerance + spec.wall
    _cx, cy, cz = servo.connector_pos
    cable_exit_servo = (-bracket_half_x, cy, cz)
    return _add_vec3(center, rotate_vec(quat, cable_exit_servo))


def _joint_entry_local() -> Vec3:
    """Joint entry point in child body frame.

    The joint is at the child body origin — (0, 0, 0).
    """
    return (0.0, 0.0, 0.0)


def _wireport_local(body: Body, bus_type: str) -> Vec3 | None:
    """Find a wire port of the given bus type on a body.

    Returns the port position in body-local frame, or None if not found.
    """
    for mount in body.mounts:
        for port in mount.component.wire_ports:
            if port.bus_type == bus_type:
                return _add_vec3(mount.resolved_pos, port.pos)
    return None


def _build_parent_map(bot: Bot) -> dict[str, tuple[Joint, str]]:
    """Build body_name → (parent_joint, parent_body_name) mapping.

    For each non-root body, records which joint connects it to its parent
    and what that parent body's name is.
    """
    result: dict[str, tuple[Joint, str]] = {}

    def _walk(body: Body) -> None:
        for joint in body.joints:
            if joint.child is not None:
                result[joint.child.name] = (joint, body.name)
                _walk(joint.child)

    if bot.root is not None:
        _walk(bot.root)
    return result


# ── Body-interior waypoints ──


def _expand_segment(
    bot: Bot,
    start: Vec3,
    end: Vec3,
    body_name: str,
    joint_name: str | None = None,
    slack: float = 0.0,
) -> list[WireSegment]:
    """Expand a logical segment into sub-segments routed through body interior.

    Instead of a direct line from start to end, inserts waypoints that follow
    the body's internal geometry — tube center axis, box wire duct, etc.
    """
    body = next((b for b in bot.all_bodies if b.name == body_name), None)
    if body is None:
        return [
            WireSegment(
                start=start,
                end=end,
                body_name=body_name,
                joint_name=joint_name,
                slack=slack,
            )
        ]

    waypoints = _body_waypoints(body, start, end)
    if not waypoints:
        return [
            WireSegment(
                start=start,
                end=end,
                body_name=body_name,
                joint_name=joint_name,
                slack=slack,
            )
        ]

    points = [start, *waypoints, end]
    segments = []
    for i in range(len(points) - 1):
        d = _dist_vec3(points[i], points[i + 1])
        if d < 0.001:  # skip sub-millimeter segments
            continue
        segments.append(
            WireSegment(
                start=points[i],
                end=points[i + 1],
                body_name=body_name,
                # joint_name and slack on first segment only
                joint_name=joint_name if i == 0 else None,
                slack=slack if i == 0 else 0.0,
            )
        )

    if not segments:
        return [
            WireSegment(
                start=start,
                end=end,
                body_name=body_name,
                joint_name=joint_name,
                slack=slack,
            )
        ]
    return segments


def _body_waypoints(body: Body, start: Vec3, end: Vec3) -> list[Vec3]:
    """Generate intermediate waypoints that route through body interior."""
    if body.shape == "tube":
        return _tube_waypoints(start, end)
    if body.shape == "cylinder":
        return _tube_waypoints(start, end)
    if body.shape == "box":
        return _box_waypoints(start, end)
    return []


def _tube_waypoints(start: Vec3, end: Vec3) -> list[Vec3]:
    """Route along tube center axis (Z).

    Tubes have their long axis along Z. Wires enter/exit at off-center
    positions (servo connectors, bracket exits) but should run along the
    center of the tube in between.

    Path: start → (0, 0, start.z) → (0, 0, end.z) → end
    """
    threshold = 0.003  # 3mm from center before adding a waypoint
    waypoints: list[Vec3] = []

    start_off = math.sqrt(start[0] ** 2 + start[1] ** 2)
    if start_off > threshold:
        waypoints.append((0.0, 0.0, start[2]))

    end_off = math.sqrt(end[0] ** 2 + end[1] ** 2)
    if end_off > threshold:
        wp = (0.0, 0.0, end[2])
        # Don't duplicate if previous waypoint is at basically the same Z
        if not waypoints or abs(waypoints[-1][2] - end[2]) > threshold:
            waypoints.append(wp)

    return waypoints


def _box_waypoints(start: Vec3, end: Vec3) -> list[Vec3]:
    """Route through box interior at duct height (Z=0, center of box).

    Boxes have height along Z. Wire duct runs at Z=0 (center height).
    Wires drop to duct height, run horizontally, then rise to destination.

    Path: start → (start.x, start.y, 0) → (end.x, end.y, 0) → end
    """
    duct_z = 0.0
    threshold = 0.005  # 5mm from duct before adding a waypoint
    waypoints: list[Vec3] = []

    if abs(start[2] - duct_z) > threshold:
        waypoints.append((start[0], start[1], duct_z))

    if abs(end[2] - duct_z) > threshold:
        wp = (end[0], end[1], duct_z)
        if not waypoints or _dist_vec3(waypoints[-1], wp) > 0.003:
            waypoints.append(wp)

    return waypoints


# ── Route algorithms ──


def _route_servo_bus(bot: Bot) -> WireRoute:
    """Route the UART daisy-chain bus through all servos.

    Walks the kinematic tree depth-first, connecting servos in chain order.
    Each servo is on its joint's parent body. When consecutive servos are on
    different bodies, the wire crosses a joint — producing a segment on each
    side of the crossing.
    """
    route = WireRoute(label="servo_bus", bus_type="uart_half_duplex")

    if bot.root is None:
        return route

    # Pi UART port — origin of the bus, in root body frame
    pi_uart_pos = _wireport_local(bot.root, "uart_half_duplex")
    if pi_uart_pos is None:
        pi_uart_pos = (0.0, 0.0, 0.0)

    parent_map = _build_parent_map(bot)

    current_pos = pi_uart_pos
    current_body = bot.root.name

    def _walk(body: Body) -> None:
        nonlocal current_pos, current_body

        for joint in body.joints:
            servo_body = body.name  # servo is mounted on joint's parent body
            connector_pos = _servo_connector_local(joint)

            if servo_body == current_body:
                # Same body — route through interior
                route.segments.extend(
                    _expand_segment(
                        bot,
                        current_pos,
                        connector_pos,
                        current_body,
                    )
                )
            else:
                # Cross-body transition — find the joint connecting them
                crossing_joint, _parent = parent_map[servo_body]

                # Segment on old body: current_pos → cable exit
                route.segments.extend(
                    _expand_segment(
                        bot,
                        current_pos,
                        _cable_exit_local(crossing_joint),
                        current_body,
                    )
                )

                # Segment on new body: joint entry → connector
                route.segments.extend(
                    _expand_segment(
                        bot,
                        _joint_entry_local(),
                        connector_pos,
                        servo_body,
                        joint_name=crossing_joint.name,
                        slack=_joint_slack(crossing_joint.effective_range, 0.01),
                    )
                )

            current_pos = connector_pos
            current_body = servo_body

            # Recurse into child body
            if joint.child is not None:
                _walk(joint.child)

    _walk(bot.root)
    return route


def _route_camera_csi(bot: Bot) -> WireRoute:
    """Route CSI ribbon cable from camera to Pi.

    Walks back through the kinematic tree from the camera body to the root,
    generating body-local segments at each joint crossing.
    """
    route = WireRoute(label="camera_csi", bus_type="csi")

    if bot.root is None:
        return route

    # Find camera body and CSI port (look for OV5647 specifically,
    # not just any CSI port — the Pi also has one)
    camera_body: Body | None = None
    camera_pos: Vec3 | None = None

    def _find_camera(body: Body) -> None:
        nonlocal camera_body, camera_pos
        if camera_body is not None:
            return
        for mount in body.mounts:
            if mount.component.name == "OV5647":
                for port in mount.component.wire_ports:
                    if port.bus_type == "csi":
                        camera_body = body
                        camera_pos = _add_vec3(mount.resolved_pos, port.pos)
                        return
        for joint in body.joints:
            if joint.child is not None:
                _find_camera(joint.child)

    _find_camera(bot.root)

    if camera_body is None or camera_pos is None:
        return route

    # Camera on root body — single direct segment
    pi_csi_pos = _wireport_local(bot.root, "csi")
    if pi_csi_pos is None:
        return route

    if camera_body.name == bot.root.name:
        route.segments.append(
            WireSegment(
                start=camera_pos,
                end=pi_csi_pos,
                body_name=bot.root.name,
            )
        )
        return route

    # Build path from camera body back to root
    parent_map = _build_parent_map(bot)
    path: list[tuple[str, Joint]] = []  # (body_name, joint_to_cross)

    body_name = camera_body.name
    while body_name in parent_map:
        crossing_joint, parent_body_name = parent_map[body_name]
        path.append((body_name, crossing_joint))
        body_name = parent_body_name

    # Walk the path, generating segments on each body
    current_pos = camera_pos
    current_body = camera_body.name

    for _bname, crossing_joint in path:
        # Segment on child body: current_pos → joint origin
        route.segments.extend(
            _expand_segment(
                bot,
                current_pos,
                _joint_entry_local(),
                current_body,
                joint_name=crossing_joint.name,
                slack=_joint_slack(crossing_joint.effective_range, 0.01),
            )
        )

        # Move to parent body at cable exit
        current_pos = _cable_exit_local(crossing_joint)
        current_body = parent_map[current_body][1]

    # Final segment on root body: cable exit → Pi CSI port
    route.segments.extend(_expand_segment(bot, current_pos, pi_csi_pos, bot.root.name))

    return route


def _route_power(bot: Bot) -> WireRoute:
    """Route power from battery to Pi.

    Both components are on the root body, so this is a single segment
    in body-local coordinates.
    """
    route = WireRoute(label="power", bus_type="power")

    if bot.root is None:
        return route

    battery_pos: Vec3 | None = None
    pi_pos: Vec3 | None = None

    for mount in bot.root.mounts:
        if "LiPo" in mount.component.name or "battery" in mount.label:
            for port in mount.component.wire_ports:
                if port.bus_type == "power":
                    battery_pos = _add_vec3(mount.resolved_pos, port.pos)
                    break
        if "Pi" in mount.component.name or "pi" in mount.label:
            for port in mount.component.wire_ports:
                if port.bus_type == "usb":
                    pi_pos = _add_vec3(mount.resolved_pos, port.pos)
                    break

    if battery_pos is not None and pi_pos is not None:
        route.segments.extend(_expand_segment(bot, battery_pos, pi_pos, bot.root.name))

    return route


# ── Utilities ──


def _joint_slack(range_rad: tuple[float, float], distance_from_axis: float) -> float:
    """Compute extra cable length needed for full range of motion.

    slack = total_range * distance_from_joint_axis
    """
    total_range = range_rad[1] - range_rad[0]
    return total_range * distance_from_axis


def _add_vec3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _dist_vec3(a: Vec3, b: Vec3) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
