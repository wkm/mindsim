"""Tests for fastener tool clearance DFM check."""

from botcad.assembly.build import build_assembly_sequence
from botcad.dfm.check import DFMSeverity
from botcad.dfm.checks.fastener_clearance import FastenerToolClearance


def _wheeler_base():
    from bots.wheeler_base.design import build

    bot = build()
    bot.solve()
    return bot


def test_check_has_correct_name():
    check = FastenerToolClearance()
    assert check.name == "fastener_tool_clearance"


def test_wheeler_base_has_fastener_clearance_issues():
    """Wheeler_base servo ears extend beyond body walls — known blocker."""
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    check = FastenerToolClearance()
    findings = check.run(bot, seq)
    # We expect at least some findings
    assert len(findings) > 0
    # At least one should be an error
    errors = [f for f in findings if f.severity == DFMSeverity.ERROR]
    assert len(errors) > 0, (
        f"Expected fastener clearance errors, got: {[f.title for f in findings]}"
    )


def test_findings_have_valid_structure():
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    check = FastenerToolClearance()
    findings = check.run(bot, seq)
    for f in findings:
        assert f.check_name == "fastener_tool_clearance"
        assert f.pos is not None
        assert f.assembly_step >= 0
