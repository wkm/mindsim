"""DFM check: port appears in more than one WireNet.

A physical wire port can only carry one signal / power rail.  If the same
(component_id, port_label) pair appears in multiple nets, something is
wrong with the netlist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.assembly.refs import WireRef
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity
from botcad.ids import BodyId

if TYPE_CHECKING:
    from botcad.assembly.sequence import AssemblySequence
    from botcad.skeleton import Bot


class WirenetOverloadedPorts(DFMCheck):
    """Error when a port is wired into multiple nets."""

    @property
    def name(self) -> str:
        return "wirenet_overloaded_ports"

    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
    ) -> list[DFMFinding]:
        findings: list[DFMFinding] = []

        # Map (component_id, port_label) -> list of net labels
        port_nets: dict[tuple[str, str], list[str]] = {}
        for net in bot.wire_nets:
            for port in net.ports:
                key = (str(port.component_id), port.port_label)
                port_nets.setdefault(key, []).append(net.label)

        for (comp_id, port_label), net_labels in port_nets.items():
            if len(net_labels) > 1:
                findings.append(
                    DFMFinding(
                        check_name=self.name,
                        severity=DFMSeverity.ERROR,
                        body=BodyId(""),
                        target=WireRef(label=net_labels[0]),
                        assembly_step=0,
                        title=f"Port {comp_id}/{port_label} in multiple nets",
                        description=(
                            f"Port '{port_label}' on component '{comp_id}' "
                            f"appears in {len(net_labels)} nets: "
                            f"{', '.join(net_labels)}. "
                            f"A physical port can only belong to one net."
                        ),
                        pos=(0.0, 0.0, 0.0),
                        direction=None,
                        measured=None,
                        threshold=None,
                        has_overlay=False,
                    )
                )

        return findings
