"""Tests for the wire stub collision DFM check."""

import pytest

from botcad.assembly.build import build_assembly_sequence
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
    seq = build_assembly_sequence(bot)
    check = WireStubCollision()
    findings = check.run(bot, seq)
    for f in findings:
        assert f.check_name == "wire_stub_collision"
    if findings:
        print(f"\n{bot_module}: {len(findings)} finding(s)")
        for f in findings:
            print(f"  [{f.severity.value}] {f.title}")
