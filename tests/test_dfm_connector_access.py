"""Tests for connector mating access DFM check."""

from botcad.assembly.build import build_assembly_sequence
from botcad.dfm.checks.connector_access import ConnectorMatingAccess


def _wheeler_base():
    from bots.wheeler_base.design import build

    bot = build()
    bot.solve()
    return bot


def test_check_name():
    check = ConnectorMatingAccess()
    assert check.name == "connector_mating_access"


def test_wheeler_base_connector_access():
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    check = ConnectorMatingAccess()
    findings = check.run(bot, seq, {})
    # Structure should be valid
    for f in findings:
        assert f.check_name == "connector_mating_access"
        assert f.pos is not None
        assert f.assembly_step >= 0


def test_findings_have_direction_and_thresholds():
    """Every connector access finding should report the mating direction."""
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    check = ConnectorMatingAccess()
    findings = check.run(bot, seq, {})
    for f in findings:
        # direction is the mating axis that was checked
        assert f.direction is not None
        assert f.threshold is not None
        assert f.measured is not None
