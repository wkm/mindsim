"""DFM check runner with automatic subclass discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.assembly.build import build_assembly_sequence
from botcad.dfm.check import DFMCheck, DFMFinding

if TYPE_CHECKING:
    from botcad.skeleton import Bot

# Import all check modules to trigger subclass registration
import botcad.dfm.checks.component_retention
import botcad.dfm.checks.connector_access
import botcad.dfm.checks.fastener_clearance
import botcad.dfm.checks.wire_bend_radius
import botcad.dfm.checks.wire_channel_sizing  # noqa: F401


def discover_checks() -> list[DFMCheck]:
    """Find all DFMCheck subclasses."""
    return [cls() for cls in DFMCheck.__subclasses__()]


def run_dfm(bot: Bot) -> list[DFMFinding]:
    """Run all DFM checks against a bot. Returns all findings."""
    seq = build_assembly_sequence(bot)
    checks = discover_checks()
    findings: list[DFMFinding] = []
    for check in checks:
        findings.extend(check.run(bot, seq, {}))
    return findings
