import pytest

from botcad.assembly.refs import FastenerRef
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity
from botcad.ids import BodyId


def test_finding_frozen():
    f = DFMFinding(
        check_name="fastener_tool_clearance",
        severity=DFMSeverity.ERROR,
        body=BodyId("base"),
        target=FastenerRef(BodyId("base"), 0),
        assembly_step=3,
        title="M3 screw inaccessible",
        description="No tool clearance",
        pos=(0.065, 0.0, -0.017),
        direction=(0.0, 0.0, 1.0),
        measured=0.0,
        threshold=0.025,
        has_overlay=False,
    )
    with pytest.raises(AttributeError):
        f.severity = DFMSeverity.WARNING


def test_finding_id_deterministic():
    f = DFMFinding(
        check_name="fastener_tool_clearance",
        severity=DFMSeverity.ERROR,
        body=BodyId("base"),
        target=FastenerRef(BodyId("base"), 0),
        assembly_step=3,
        title="M3 screw inaccessible",
        description="No tool clearance",
        pos=(0.065, 0.0, -0.017),
        direction=None,
        measured=None,
        threshold=None,
        has_overlay=False,
    )
    # ID should be stable across runs
    assert "fastener_tool_clearance" in f.id
    assert "base" in f.id
    # Same inputs = same ID
    f2 = DFMFinding(
        check_name="fastener_tool_clearance",
        severity=DFMSeverity.ERROR,
        body=BodyId("base"),
        target=FastenerRef(BodyId("base"), 0),
        assembly_step=3,
        title="M3 screw inaccessible",
        description="No tool clearance",
        pos=(0.065, 0.0, -0.017),
        direction=None,
        measured=None,
        threshold=None,
        has_overlay=False,
    )
    assert f.id == f2.id


def test_severity_ordering():
    assert DFMSeverity.ERROR.value == "error"
    assert DFMSeverity.WARNING.value == "warning"
    assert DFMSeverity.INFO.value == "info"


def test_dfm_check_is_abstract():
    """DFMCheck can't be instantiated directly."""
    with pytest.raises(TypeError):
        DFMCheck()
