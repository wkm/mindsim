"""Wire channel sizing DFM check.

Detects wire channels (tubes) that are too small for the connector
housings that must pass through them. The classic case: servo UART
connectors (Molex 5264, 7.5 x 4.5 x 6mm) need to pass through a
bracket cable slot or body tube, but the channel is sized only for
the bare wire diameter (3mm dia cylinder).

Compares connector body_dimensions from the connector registry against
the channel cross-section (wire channel radius from emit_wire_channel)
and the bracket cable_slot dimensions.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from botcad.assembly.refs import WireRef
from botcad.assembly.sequence import AssemblyAction
from botcad.bracket import BracketSpec, _cable_slot_dims
from botcad.connectors import ConnectorSpec, connector_spec
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity
from botcad.routing import WireRoute, WireSegment, solve_routing

if TYPE_CHECKING:
    from botcad.assembly.sequence import AssemblySequence
    from botcad.ids import BodyId
    from botcad.skeleton import Body, Bot, Joint

# Channel radius table — must match emit_components._CHANNEL_RADIUS
_CHANNEL_RADIUS: dict[str, float] = {
    "uart_half_duplex": 0.0015,
    "csi": 0.003,
    "power": 0.002,
}
_DEFAULT_CHANNEL_RADIUS = 0.0015


class WireChannelSizing(DFMCheck):
    """Check that wire channels are large enough for connector housings."""

    @property
    def name(self) -> str:
        return "wire_channel_sizing"

    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
        body_solids: dict[BodyId, object],
    ) -> list[DFMFinding]:
        findings: list[DFMFinding] = []

        if bot.root is None:
            return findings

        routes = solve_routing(bot)
        body_map = _build_body_map(bot)

        # Find the ROUTE_WIRE step for each route label
        wire_steps: dict[str, int] = {}
        for op in sequence.ops:
            if op.action == AssemblyAction.ROUTE_WIRE and isinstance(
                op.target, WireRef
            ):
                wire_steps[op.target.label] = op.step

        for route in routes:
            if not route.segments:
                continue

            # Gather connector specs for this route's bus type
            connector_specs = _connectors_for_route(route, bot)
            if not connector_specs:
                continue

            # The largest connector cross-section that must pass through
            max_connector = _max_connector_cross_section(connector_specs)

            # Check channel vs connector at each segment
            channel_radius = _CHANNEL_RADIUS.get(
                str(route.bus_type), _DEFAULT_CHANNEL_RADIUS
            )
            channel_diameter = channel_radius * 2

            step = wire_steps.get(route.label, 0)

            # Check 1: Wire channel tubes are too narrow for connectors
            # The connector must pass through the channel during assembly
            if max_connector.min_cross > channel_diameter:
                body_name = route.segments[0].body_name
                pos = route.segments[0].start
                findings.append(
                    DFMFinding(
                        check_name=self.name,
                        severity=DFMSeverity.ERROR,
                        body=body_name,
                        target=WireRef(label=route.label),
                        assembly_step=step,
                        title=(
                            f"Wire channel too narrow for "
                            f"{max_connector.label} connector"
                        ),
                        description=(
                            f"Channel diameter {channel_diameter * 1000:.1f}mm "
                            f"but {max_connector.label} connector needs "
                            f"{max_connector.min_cross * 1000:.1f}mm "
                            f"(smallest cross-section). "
                            f"Connector body: "
                            f"{max_connector.dims_mm[0]:.1f} x "
                            f"{max_connector.dims_mm[1]:.1f} x "
                            f"{max_connector.dims_mm[2]:.1f}mm."
                        ),
                        pos=pos,
                        direction=None,
                        measured=channel_diameter * 1000,
                        threshold=max_connector.min_cross * 1000,
                        has_overlay=False,
                    )
                )

            # Check 2: Bracket cable slot sizing at joints
            for segment in route.segments:
                if segment.joint_name is None:
                    continue
                body = body_map.get(segment.body_name)
                if body is None:
                    continue
                joint = _find_joint(body, segment.joint_name)
                if joint is None:
                    # Joint may be on the parent body
                    joint = _find_joint_in_bot(bot, segment.joint_name)
                if joint is None:
                    continue

                slot_finding = _check_bracket_slot(
                    joint=joint,
                    connector=max_connector,
                    route=route,
                    segment=segment,
                    step=step,
                    check_name=self.name,
                )
                if slot_finding is not None:
                    findings.append(slot_finding)

        return findings


# ── Data helpers ──


class _ConnectorCrossSection:
    """Precomputed connector dimensions for comparison."""

    __slots__ = ("dims_mm", "label", "max_cross", "min_cross")

    def __init__(self, spec: ConnectorSpec) -> None:
        bx, by, bz = spec.body_dimensions
        self.label = spec.label
        self.dims_mm = (bx * 1000, by * 1000, bz * 1000)
        # Connector must pass through channel in the orientation that
        # minimizes the cross-section. The smallest cross-section is
        # the two smallest dimensions.
        sorted_dims = sorted([bx, by, bz])
        # The minimum circle that fits the smallest rectangular cross-section
        self.min_cross = math.sqrt(sorted_dims[0] ** 2 + sorted_dims[1] ** 2)
        self.max_cross = math.sqrt(sorted_dims[1] ** 2 + sorted_dims[2] ** 2)


def _max_connector_cross_section(
    specs: list[ConnectorSpec],
) -> _ConnectorCrossSection:
    """Return the connector with the largest minimum cross-section."""
    sections = [_ConnectorCrossSection(s) for s in specs]
    return max(sections, key=lambda c: c.min_cross)


def _connectors_for_route(route: WireRoute, bot: Bot) -> list[ConnectorSpec]:
    """Find all connector specs associated with a wire route's bus type.

    Walks the bot looking for WirePorts matching the route's bus_type
    that have a connector_type set, then looks them up in the registry.
    """
    specs: list[ConnectorSpec] = []
    seen: set[str] = set()

    for body in bot.all_bodies:
        # Component wire ports
        for mount in body.mounts:
            for port in mount.component.wire_ports:
                if port.bus_type != route.bus_type:
                    continue
                if not port.connector_type or port.connector_type in seen:
                    continue
                try:
                    specs.append(connector_spec(port.connector_type))
                    seen.add(port.connector_type)
                except KeyError:
                    pass

        # Servo wire ports (on joints of this body)
        for joint in body.joints:
            for port in joint.servo.wire_ports:
                if port.bus_type != route.bus_type:
                    continue
                if not port.connector_type or port.connector_type in seen:
                    continue
                try:
                    specs.append(connector_spec(port.connector_type))
                    seen.add(port.connector_type)
                except KeyError:
                    pass

    return specs


def _check_bracket_slot(
    *,
    joint: Joint,
    connector: _ConnectorCrossSection,
    route: WireRoute,
    segment: WireSegment,
    step: int,
    check_name: str,
) -> DFMFinding | None:
    """Check if the bracket cable slot is large enough for the connector."""
    servo = joint.servo
    spec = BracketSpec()
    slot_w, slot_h = _cable_slot_dims(servo, spec)

    # The connector's smallest two dimensions need to fit in the slot
    bx, by, bz = (d / 1000 for d in connector.dims_mm)  # back to meters
    sorted_dims = sorted([bx, by, bz])
    # The connector passes through oriented to fit: smallest two dims
    # must fit within slot_w x slot_h
    conn_w = sorted_dims[1]  # medium dimension
    conn_h = sorted_dims[0]  # smallest dimension

    fits_normal = conn_w <= slot_w and conn_h <= slot_h
    fits_rotated = conn_h <= slot_w and conn_w <= slot_h

    if fits_normal or fits_rotated:
        return None

    return DFMFinding(
        check_name=check_name,
        severity=DFMSeverity.ERROR,
        body=segment.body_name,
        target=WireRef(label=route.label),
        assembly_step=step,
        title=(
            f"Bracket cable slot too small for {connector.label} on joint {joint.name}"
        ),
        description=(
            f"Cable slot {slot_w * 1000:.1f} x {slot_h * 1000:.1f}mm "
            f"but {connector.label} connector needs at least "
            f"{conn_w * 1000:.1f} x {conn_h * 1000:.1f}mm "
            f"(smallest orientation). "
            f"Full connector: {connector.dims_mm[0]:.1f} x "
            f"{connector.dims_mm[1]:.1f} x {connector.dims_mm[2]:.1f}mm."
        ),
        pos=segment.start,
        direction=None,
        measured=min(slot_w, slot_h) * 1000,
        threshold=conn_h * 1000,
        has_overlay=False,
    )


# ── Tree traversal helpers ──


def _build_body_map(bot: Bot) -> dict[BodyId, Body]:
    """Walk the kinematic tree and collect all bodies by name."""
    from collections import deque

    result: dict[BodyId, Body] = {}
    if bot.root is None:
        return result

    queue: deque[Body] = deque([bot.root])
    while queue:
        body = queue.popleft()
        if body.name in result:
            continue
        result[body.name] = body
        for joint in body.joints:
            if joint.child is not None:
                queue.append(joint.child)
    return result


def _find_joint(body: Body, joint_name: str) -> Joint | None:
    """Find a joint by name on a specific body."""
    for joint in body.joints:
        if str(joint.name) == joint_name:
            return joint
    return None


def _find_joint_in_bot(bot: Bot, joint_name: str) -> Joint | None:
    """Find a joint by name anywhere in the bot."""
    for body in bot.all_bodies:
        for joint in body.joints:
            if str(joint.name) == joint_name:
                return joint
    return None
