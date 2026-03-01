"""Wire routing along the kinematic skeleton.

STS3215 servos use a daisy-chain serial bus (UART half-duplex), so instead
of individual wires from each servo to the Pi, we route a single bus along
the kinematic chain.

Outputs per wire: list of 3D waypoints, total length, required slack at
each joint crossing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from botcad.component import Vec3

if TYPE_CHECKING:
    from botcad.skeleton import Bot


@dataclass
class WireSegment:
    """A segment of wire between two points."""

    start: Vec3
    end: Vec3
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

    # 3. Power bus: battery → all servos + Pi
    power_route = _route_power(bot)
    if power_route.segments:
        routes.append(power_route)

    return routes


def _route_servo_bus(bot: Bot) -> WireRoute:
    """Route the UART daisy-chain bus through all servos.

    Traverses the kinematic tree depth-first, connecting servos in chain order.
    Wire paths follow the center of each structural member, with extra slack
    at each joint crossing.
    """
    route = WireRoute(label="servo_bus", bus_type="uart_half_duplex")

    if bot.root is None:
        return route

    # Find the Pi's UART port position (origin of the bus)
    pi_uart_pos = (0.0, 0.0, 0.0)
    for mount in bot.root.mounts:
        if "Pi" in mount.component.name or "pi" in mount.label:
            for port in mount.component.wire_ports:
                if port.bus_type == "uart_half_duplex":
                    pi_uart_pos = _add_vec3(mount.resolved_pos, port.pos)
                    break

    # Walk the kinematic tree, collecting servo positions
    current_pos = pi_uart_pos
    joint_positions = _collect_joint_positions(bot)

    for joint_name, joint_pos, joint_range, _axis in joint_positions:
        segment = WireSegment(
            start=current_pos,
            end=joint_pos,
            joint_name=joint_name,
            slack=_joint_slack(joint_range, 0.01),  # 1cm from axis
        )
        route.segments.append(segment)
        current_pos = joint_pos

    return route


def _route_camera_csi(bot: Bot) -> WireRoute:
    """Route CSI ribbon cable from camera to Pi."""
    route = WireRoute(label="camera_csi", bus_type="csi")

    if bot.root is None:
        return route

    # Find camera and Pi positions
    camera_pos: Vec3 | None = None
    pi_pos: Vec3 | None = None

    def _find_camera(body, world_pos: Vec3) -> None:
        nonlocal camera_pos
        for mount in body.mounts:
            if mount.component.name == "OV5647":
                camera_pos = _add_vec3(world_pos, mount.resolved_pos)
                return
        for joint in body.joints:
            if joint.child is not None:
                _find_camera(joint.child, _add_vec3(world_pos, joint.pos))

    for mount in bot.root.mounts:
        if "Pi" in mount.component.name or "pi" in mount.label:
            pi_pos = mount.resolved_pos

    _find_camera(bot.root, (0.0, 0.0, 0.0))

    if camera_pos is not None and pi_pos is not None:
        route.segments.append(WireSegment(start=camera_pos, end=pi_pos))

    return route


def _route_power(bot: Bot) -> WireRoute:
    """Route power from battery to Pi and servo bus."""
    route = WireRoute(label="power", bus_type="power")

    if bot.root is None:
        return route

    battery_pos: Vec3 | None = None
    pi_pos: Vec3 | None = None

    for mount in bot.root.mounts:
        if "LiPo" in mount.component.name or "battery" in mount.label:
            battery_pos = mount.resolved_pos
        if "Pi" in mount.component.name or "pi" in mount.label:
            pi_pos = mount.resolved_pos

    if battery_pos is not None and pi_pos is not None:
        # Battery → Pi (via buck converter)
        route.segments.append(WireSegment(start=battery_pos, end=pi_pos))

    return route


def _collect_joint_positions(
    bot: Bot,
) -> list[tuple[str, Vec3, tuple[float, float], Vec3]]:
    """Collect all joint positions in kinematic tree traversal order.

    Returns: [(joint_name, world_position, range_rad, axis), ...]
    """
    result = []

    def _walk(body, world_pos: Vec3) -> None:
        for joint in body.joints:
            joint_world_pos = _add_vec3(world_pos, joint.pos)
            result.append(
                (joint.name, joint_world_pos, joint.effective_range, joint.axis)
            )
            if joint.child is not None:
                _walk(joint.child, joint_world_pos)

    if bot.root is not None:
        _walk(bot.root, (0.0, 0.0, 0.0))

    return result


def _joint_slack(range_rad: tuple[float, float], distance_from_axis: float) -> float:
    """Compute extra cable length needed for full range of motion.

    slack = total_range * distance_from_joint_axis
    """
    total_range = range_rad[1] - range_rad[0]
    return total_range * distance_from_axis


def _add_vec3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
