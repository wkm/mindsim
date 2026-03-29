"""DFM check runner with automatic subclass discovery."""

from __future__ import annotations

# Import all check modules to trigger subclass registration
import botcad.dfm.checks.component_retention
import botcad.dfm.checks.connector_access
import botcad.dfm.checks.fastener_clearance
import botcad.dfm.checks.wire_bend_radius
import botcad.dfm.checks.wire_channel_sizing
import botcad.dfm.checks.wire_stub_collision
import botcad.dfm.checks.wirenet_bus_type_mismatch
import botcad.dfm.checks.wirenet_orphaned_ports
import botcad.dfm.checks.wirenet_overloaded_ports  # noqa: F401
from botcad.assembly.build import build_assembly_sequence
from botcad.dfm.check import DFMCheck, DFMFinding
from botcad.skeleton import Bot


def discover_checks() -> list[DFMCheck]:
    """Find all DFMCheck subclasses."""
    return [cls() for cls in DFMCheck.__subclasses__()]


def run_dfm(bot: Bot) -> list[DFMFinding]:
    """Run all DFM checks against a bot. Returns all findings."""
    seq = build_assembly_sequence(bot)
    checks = discover_checks()
    findings: list[DFMFinding] = []
    for check in checks:
        findings.extend(check.run(bot, seq))
    return findings
