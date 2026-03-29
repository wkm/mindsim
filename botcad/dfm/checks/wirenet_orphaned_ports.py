"""DFM check: orphaned component ports not connected to any WireNet.

For each component mounted on the bot, verifies that every WirePort on
that component appears in at least one WireNet.  An orphaned port likely
means a cable was forgotten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.assembly.refs import ComponentRef
from botcad.component import WirePort
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity
from botcad.ids import BodyId, ComponentId

if TYPE_CHECKING:
    from botcad.assembly.sequence import AssemblySequence
    from botcad.skeleton import Bot


def _orphan_finding(
    check_name: str,
    body_name: BodyId,
    mount_label: str,
    wp: WirePort,
    *,
    label_prefix: str = "",
) -> DFMFinding:
    return DFMFinding(
        check_name=check_name,
        severity=DFMSeverity.WARNING,
        body=body_name,
        target=ComponentRef(body=body_name, mount_label=mount_label),
        assembly_step=0,
        title=f"Orphaned port {wp.label} on {label_prefix}{mount_label}",
        description=(
            f"Wire port '{wp.label}' (bus={wp.bus_type}) "
            f"on {label_prefix}'{mount_label}' is not connected "
            f"to any WireNet."
        ),
        pos=(0.0, 0.0, 0.0),
        direction=None,
        measured=None,
        threshold=None,
        has_overlay=False,
    )


class WirenetOrphanedPorts(DFMCheck):
    """Warn when a component wire port is not part of any net."""

    @property
    def name(self) -> str:
        return "wirenet_orphaned_ports"

    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
    ) -> list[DFMFinding]:
        findings: list[DFMFinding] = []

        if bot.root is None:
            return findings

        # Collect all (component_id, port_label) pairs present in nets
        connected: set[tuple[str, str]] = set()
        for net in bot.wire_nets:
            for port in net.ports:
                connected.add((str(port.component_id), port.port_label))

        # Walk all mounted components
        for body in bot.all_bodies:
            for mount in body.mounts:
                comp_id = ComponentId(mount.label)
                findings.extend(
                    _orphan_finding(self.name, body.name, mount.label, wp)
                    for wp in mount.component.wire_ports
                    if (str(comp_id), wp.label) not in connected
                )

        # Walk servos on joints — they have uart_in / uart_out ports
        for body in bot.all_bodies:
            for joint in body.joints:
                servo_id = ComponentId(str(joint.name))
                findings.extend(
                    _orphan_finding(
                        self.name,
                        body.name,
                        str(joint.name),
                        wp,
                        label_prefix="servo ",
                    )
                    for wp in joint.servo.wire_ports
                    if (str(servo_id), wp.label) not in connected
                )

        return findings
