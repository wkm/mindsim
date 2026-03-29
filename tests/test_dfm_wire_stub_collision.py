"""Tests for the wire stub collision DFM check."""

import pytest

from botcad.assembly.build import build_assembly_sequence
from botcad.dfm.check import DFMSeverity
from botcad.dfm.checks.wire_stub_collision import WireStubCollision
from botcad.dfm.runner import discover_checks


def test_check_discovered():
    names = [c.name for c in discover_checks()]
    assert "wire_stub_collision" in names


@pytest.mark.parametrize(
    "bot_module",
    [
        "bots.wheeler_base.design",
        "bots.wheeler_arm.design",
        "bots.so101_arm.design",
    ],
)
def test_wire_stub_on_all_bots(bot_module: str):
    import importlib

    mod = importlib.import_module(bot_module)
    bot = mod.build()
    bot.solve()
    seq = build_assembly_sequence(bot)
    check = WireStubCollision()
    findings = check.run(bot, seq)
    for f in findings:
        assert f.check_name == "wire_stub_collision"
    if findings:
        print(f"\n{bot_module}: {len(findings)} finding(s)")
        for f in findings:
            print(f"  [{f.severity.value}] {f.title}")


def test_deterministic_servo_connector_collision():
    """STS3215 has two UART connectors 5mm apart — should produce a WARNING at body level.

    The connectors are 5264_3pin (4.5mm Y width), spaced 5mm center-to-center
    in Y, giving a 0.5mm gap — well below the 2mm warning threshold.
    """
    from bots.wheeler_base.design import build

    bot = build()
    bot.solve()
    seq = build_assembly_sequence(bot)
    check = WireStubCollision()
    findings = check.run(bot, seq)

    # Find findings that involve servo UART ports specifically
    servo_uart_findings = [
        f for f in findings if "uart_in" in f.title and "uart_out" in f.title
    ]
    assert len(servo_uart_findings) > 0, (
        f"Expected servo UART connector findings, got: {[f.title for f in findings]}"
    )
    # The 0.5mm gap should be a WARNING (< 2mm threshold, but not overlapping)
    for f in servo_uart_findings:
        assert f.severity in (DFMSeverity.WARNING, DFMSeverity.ERROR)
        assert f.measured is not None
