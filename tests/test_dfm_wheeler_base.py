"""Regression test: run all DFM checks against wheeler_base, assert known findings."""

from botcad.dfm.check import DFMSeverity
from botcad.dfm.runner import run_dfm


def _get_wheeler_findings():
    from bots.wheeler_base.design import build

    bot = build()
    return run_dfm(bot)


def test_wheeler_base_has_errors():
    findings = _get_wheeler_findings()
    errors = [f for f in findings if f.severity == DFMSeverity.ERROR]
    assert len(errors) > 0, "Wheeler_base should have DFM errors"


def test_wheeler_base_fastener_clearance_errors():
    findings = _get_wheeler_findings()
    fastener_errors = [
        f
        for f in findings
        if f.check_name == "fastener_tool_clearance" and f.severity == DFMSeverity.ERROR
    ]
    # Known: 4 servo mounting ear screws on base are inaccessible
    assert len(fastener_errors) >= 4, (
        f"Expected >=4 fastener errors, got {len(fastener_errors)}"
    )


def test_wheeler_base_fastener_clearance_warnings():
    findings = _get_wheeler_findings()
    fastener_warnings = [
        f
        for f in findings
        if f.check_name == "fastener_tool_clearance"
        and f.severity == DFMSeverity.WARNING
    ]
    # Known: 32 fasteners with tight but possible clearance
    assert len(fastener_warnings) >= 20, (
        f"Expected >=20 fastener warnings, got {len(fastener_warnings)}"
    )


def test_wheeler_base_battery_no_retention():
    findings = _get_wheeler_findings()
    retention = [f for f in findings if f.check_name == "component_retention"]
    assert len(retention) > 0, "Components should be flagged for retention issues"


def test_wheeler_base_connector_access():
    findings = _get_wheeler_findings()
    connector = [f for f in findings if f.check_name == "connector_mating_access"]
    assert len(connector) > 0, "Connector mating access should have findings"
    connector_errors = [f for f in connector if f.severity == DFMSeverity.ERROR]
    assert len(connector_errors) >= 1, (
        f"Expected >=1 connector access error, got {len(connector_errors)}"
    )


def test_wheeler_base_finding_summary():
    """Print a summary of all findings for human review."""
    findings = _get_wheeler_findings()
    by_check: dict[str, list] = {}
    for f in findings:
        by_check.setdefault(f.check_name, []).append(f)

    for check_name, check_findings in sorted(by_check.items()):
        errors = sum(1 for f in check_findings if f.severity == DFMSeverity.ERROR)
        warnings = sum(1 for f in check_findings if f.severity == DFMSeverity.WARNING)
        infos = sum(1 for f in check_findings if f.severity == DFMSeverity.INFO)
        print(f"  {check_name}: {errors}E {warnings}W {infos}I")
