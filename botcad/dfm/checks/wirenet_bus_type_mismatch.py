"""DFM check: net bus_type vs. port bus_type mismatch.

For each WireNet, verifies that every port's declared bus_type on the
component matches the net's bus_type.  A mismatch means an incompatible
connector would be wired together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.assembly.refs import WireRef
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity
from botcad.ids import BodyId, ComponentId

if TYPE_CHECKING:
    from botcad.assembly.sequence import AssemblySequence
    from botcad.skeleton import Bot


class WirenetBusTypeMismatch(DFMCheck):
    """Error when a port's bus_type doesn't match its net's bus_type."""

    @property
    def name(self) -> str:
        return "wirenet_bus_type_mismatch"

    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
    ) -> list[DFMFinding]:
        findings: list[DFMFinding] = []

        if bot.root is None:
            return findings

        # Build component_id -> {port_label -> bus_type} from actual components
        port_bus: dict[str, dict[str, str]] = {}

        # Mounted components
        for body in bot.all_bodies:
            for mount in body.mounts:
                comp_id = str(ComponentId(mount.label))
                port_bus.setdefault(comp_id, {})
                for wp in mount.component.wire_ports:
                    port_bus[comp_id][wp.label] = str(wp.bus_type)

        # Servo ports (indexed by joint name)
        for body in bot.all_bodies:
            for joint in body.joints:
                comp_id = str(ComponentId(str(joint.name)))
                port_bus.setdefault(comp_id, {})
                for wp in joint.servo.wire_ports:
                    port_bus[comp_id][wp.label] = str(wp.bus_type)

        # Check each net
        for net in bot.wire_nets:
            net_bus = str(net.bus_type)
            for port in net.ports:
                comp_id = str(port.component_id)
                declared = port_bus.get(comp_id, {}).get(port.port_label)
                if declared is not None and declared != net_bus:
                    findings.append(
                        DFMFinding(
                            check_name=self.name,
                            severity=DFMSeverity.ERROR,
                            body=BodyId(""),
                            target=WireRef(label=net.label),
                            assembly_step=0,
                            title=(f"Bus type mismatch: {comp_id}/{port.port_label}"),
                            description=(
                                f"Port '{port.port_label}' on '{comp_id}' has "
                                f"bus_type={declared}, but net '{net.label}' "
                                f"uses bus_type={net_bus}."
                            ),
                            pos=(0.0, 0.0, 0.0),
                            direction=None,
                            measured=None,
                            threshold=None,
                            has_overlay=False,
                        )
                    )

        return findings
