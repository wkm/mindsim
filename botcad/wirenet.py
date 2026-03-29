"""Wire netlist: explicit connectivity between component ports.

WireNet captures *what* is connected, independent of *where* wires are
routed in physical space.  ``derive_wirenets()`` extracts the same
connectivity currently hard-coded inside ``solve_routing()`` as first-class
data structures that DFM checks and the viewer can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from botcad.component import BusType
from botcad.ids import ComponentId

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot


class NetTopology(StrEnum):
    """Physical wiring topology of a net."""

    DAISY_CHAIN = "daisy_chain"
    POINT_TO_POINT = "point_to_point"
    STAR = "star"


@dataclass(frozen=True)
class NetPort:
    """One endpoint in a wire net."""

    component_id: ComponentId
    port_label: str


@dataclass(frozen=True)
class WireNet:
    """A logical wire net connecting component ports."""

    label: str
    bus_type: BusType
    topology: NetTopology
    ports: tuple[NetPort, ...]


# ---------------------------------------------------------------------------
# derive_wirenets — extract connectivity from a solved Bot
# ---------------------------------------------------------------------------


def derive_wirenets(bot: Bot) -> tuple[WireNet, ...]:
    """Build explicit wire nets from the bot's component graph.

    Mirrors the connectivity decisions in ``solve_routing()`` but produces
    lightweight ``WireNet`` objects instead of geometric ``WireRoute``s.
    """
    nets: list[WireNet] = []

    if bot.root is None:
        return ()

    # Detect which optional boards are present
    has_controller = _has_mount(bot.root, "controller")
    has_bec = _has_mount(bot.root, "bec")
    has_camera = _find_camera_body(bot) is not None

    # 1. servo_bus — UART daisy-chain through all servos
    servo_net = _derive_servo_bus(bot, has_controller)
    if servo_net is not None:
        nets.append(servo_net)

    # 2. camera_csi — CSI point-to-point
    if has_camera:
        csi_net = _derive_camera_csi()
        if csi_net is not None:
            nets.append(csi_net)

    # 3. power_servo — battery to controller
    if has_controller:
        nets.append(
            WireNet(
                label="power_servo",
                bus_type=BusType.POWER,
                topology=NetTopology.POINT_TO_POINT,
                ports=(
                    NetPort(ComponentId("battery"), "power_out"),
                    NetPort(ComponentId("controller"), "power_in"),
                ),
            )
        )

    # 4. power_pi — battery -> BEC -> Pi
    if has_bec:
        nets.append(
            WireNet(
                label="power_pi",
                bus_type=BusType.POWER,
                topology=NetTopology.DAISY_CHAIN,
                ports=(
                    NetPort(ComponentId("battery"), "power_out"),
                    NetPort(ComponentId("bec"), "power_in"),
                    NetPort(ComponentId("bec"), "power_out"),
                    NetPort(ComponentId("pi"), "usb_power"),
                ),
            )
        )

    # 5. pi_usb_data — USB data link
    if has_controller:
        nets.append(
            WireNet(
                label="pi_usb_data",
                bus_type=BusType.USB,
                topology=NetTopology.POINT_TO_POINT,
                ports=(
                    NetPort(ComponentId("pi"), "usb_data"),
                    NetPort(ComponentId("controller"), "usb_in"),
                ),
            )
        )

    # 6. power fallback — direct battery to Pi (no controller / BEC)
    if not has_controller and not has_bec:
        nets.append(
            WireNet(
                label="power",
                bus_type=BusType.POWER,
                topology=NetTopology.POINT_TO_POINT,
                ports=(
                    NetPort(ComponentId("battery"), "power_out"),
                    NetPort(ComponentId("pi"), "usb_power"),
                ),
            )
        )

    return tuple(nets)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_mount(body: Body, label: str) -> bool:
    """Check if *body* (non-recursive) has a mount with the given label."""
    return any(m.label == label for m in body.mounts)


def _find_camera_body(bot: Bot) -> Body | None:
    """Walk the skeleton looking for a camera component."""
    from botcad.component import ComponentKind

    def _walk(body: Body) -> Body | None:
        for mount in body.mounts:
            if mount.component.kind == ComponentKind.CAMERA:
                return body
        for joint in body.joints:
            if joint.child is not None:
                result = _walk(joint.child)
                if result is not None:
                    return result
        return None

    if bot.root is None:
        return None
    return _walk(bot.root)


def _derive_servo_bus(bot: Bot, has_controller: bool) -> WireNet | None:
    """Build the servo UART daisy-chain net.

    Walks the kinematic tree depth-first, same order as ``_route_servo_bus``.
    The chain starts at the controller (or Pi) uart_out, then alternates
    uart_in / uart_out through each servo.
    """
    assert bot.root is not None

    ports: list[NetPort] = []

    # Bus origin
    if has_controller:
        ports.append(NetPort(ComponentId("controller"), "uart_out"))
    else:
        ports.append(NetPort(ComponentId("pi"), "uart_out"))

    def _walk(body: Body) -> None:
        for joint in body.joints:
            servo_id = ComponentId(str(joint.name))
            ports.append(NetPort(servo_id, "uart_in"))
            ports.append(NetPort(servo_id, "uart_out"))
            if joint.child is not None:
                _walk(joint.child)

    _walk(bot.root)

    if len(ports) < 2:
        return None

    return WireNet(
        label="servo_bus",
        bus_type=BusType.UART_HALF_DUPLEX,
        topology=NetTopology.DAISY_CHAIN,
        ports=tuple(ports),
    )


def _derive_camera_csi() -> WireNet:
    """Camera CSI point-to-point net."""
    return WireNet(
        label="camera_csi",
        bus_type=BusType.CSI,
        topology=NetTopology.POINT_TO_POINT,
        ports=(
            NetPort(ComponentId("camera"), "csi_out"),
            NetPort(ComponentId("pi"), "csi_in"),
        ),
    )
