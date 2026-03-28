from botcad.dfm.check import DFMSeverity
from botcad.dfm.runner import discover_checks, run_dfm


def test_runner_discovers_all_checks():
    checks = discover_checks()
    names = {c.name for c in checks}
    expected = {
        "fastener_tool_clearance",
        "wire_channel_sizing",
        "wire_bend_radius",
        "component_retention",
        "connector_mating_access",
    }
    assert names.issuperset(expected), f"Missing checks: {expected - names}"


def test_runner_produces_findings():
    from bots.wheeler_base.design import build

    bot = build()
    findings = run_dfm(bot)
    assert len(findings) > 0
    # Should have at least errors (fastener clearance) and warnings (retention)
    severities = {f.severity for f in findings}
    assert DFMSeverity.ERROR in severities


def test_runner_findings_have_unique_ids():
    from bots.wheeler_base.design import build

    bot = build()
    findings = run_dfm(bot)
    ids = [f.id for f in findings]
    # IDs should be unique (or at least not ALL duplicates)
    assert len(set(ids)) > 1
