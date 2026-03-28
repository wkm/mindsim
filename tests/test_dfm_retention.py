"""Tests for the component retention DFM check."""

from botcad.assembly.refs import ComponentRef, FastenerRef
from botcad.assembly.sequence import (
    AssemblyAction,
    AssemblyOp,
    AssemblySequence,
)
from botcad.assembly.tools import ToolKind
from botcad.component import Component, ComponentKind, MountPoint
from botcad.dfm.check import DFMSeverity
from botcad.dfm.checks.component_retention import ComponentRetention
from botcad.ids import BodyId
from botcad.skeleton import BodyShape, Bot


def test_check_name():
    check = ComponentRetention()
    assert check.name == "component_retention"


def _make_bot_with_mounts(
    *,
    component_has_mounting_points: bool = False,
    include_fasten_ops: bool = False,
) -> tuple[Bot, AssemblySequence]:
    """Helper: single-body bot with one mounted component."""
    bot = Bot("test_bot")
    asm = bot.assembly("main")
    base = asm.body("base", shape=BodyShape.BOX)

    mounting_points = ()
    if component_has_mounting_points:
        mounting_points = (
            MountPoint(
                label="m2_screw_1",
                pos=(0.01, 0.01, 0.0),
                diameter=0.002,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M2",
            ),
        )

    comp = Component(
        name="TestWidget",
        dimensions=(0.03, 0.02, 0.01),
        mass=0.05,
        kind=ComponentKind.GENERIC,
        mounting_points=mounting_points,
    )

    base.mount(comp, position="bottom", label="widget")

    ops: list[AssemblyOp] = []
    step = 0

    # INSERT the component
    ops.append(
        AssemblyOp(
            step=step,
            action=AssemblyAction.INSERT,
            target=ComponentRef(body=BodyId("base"), mount_label="widget"),
            body=BodyId("base"),
            tool=ToolKind.FINGERS,
            approach_axis=(0.0, 0.0, -1.0),
            angle=None,
            prerequisites=(),
            description="Insert TestWidget into base (widget)",
        )
    )
    step += 1

    if include_fasten_ops and component_has_mounting_points:
        ops.append(
            AssemblyOp(
                step=step,
                action=AssemblyAction.FASTEN,
                target=FastenerRef(body=BodyId("base"), index=0),
                body=BodyId("base"),
                tool=ToolKind.HEX_KEY_2,
                approach_axis=(0.0, 0.0, -1.0),
                angle=None,
                prerequisites=(0,),
                description="Fasten m2_screw_1 on widget (base)",
            )
        )
        step += 1

    seq = AssemblySequence(ops=tuple(ops))
    return bot, seq


def test_no_mounting_points_no_fasteners_warns():
    """Component with no mounting points and no fasteners -> WARNING."""
    bot, seq = _make_bot_with_mounts(
        component_has_mounting_points=False,
        include_fasten_ops=False,
    )
    check = ComponentRetention()
    findings = check.run(bot, seq, {})

    assert len(findings) == 1
    assert findings[0].severity == DFMSeverity.WARNING
    assert "widget" in findings[0].title.lower()
    assert (
        "no mounting points" in findings[0].description.lower()
        or "no retention" in findings[0].title.lower()
    )


def test_has_mounting_points_and_fasteners_ok():
    """Component with mounting points AND fasteners -> no findings."""
    bot, seq = _make_bot_with_mounts(
        component_has_mounting_points=True,
        include_fasten_ops=True,
    )
    check = ComponentRetention()
    findings = check.run(bot, seq, {})

    assert len(findings) == 0


def test_has_mounting_points_but_no_fasteners_warns():
    """Component with mounting points but no FASTEN ops -> WARNING."""
    bot, seq = _make_bot_with_mounts(
        component_has_mounting_points=True,
        include_fasten_ops=False,
    )
    check = ComponentRetention()
    findings = check.run(bot, seq, {})

    assert len(findings) == 1
    assert findings[0].severity == DFMSeverity.WARNING
    assert "unfastened" in findings[0].title.lower()


def test_finding_target_is_component_ref():
    """Finding target should be a ComponentRef."""
    bot, seq = _make_bot_with_mounts(
        component_has_mounting_points=False,
        include_fasten_ops=False,
    )
    check = ComponentRetention()
    findings = check.run(bot, seq, {})

    assert len(findings) == 1
    target = findings[0].target
    assert isinstance(target, ComponentRef)
    assert target.body == BodyId("base")
    assert target.mount_label == "widget"


def test_wheeler_base_battery_no_retention():
    from bots.wheeler_base.design import build

    bot = build()

    from botcad.assembly.build import build_assembly_sequence

    seq = build_assembly_sequence(bot)
    check = ComponentRetention()
    findings = check.run(bot, seq, {})
    # Battery should be flagged
    battery_findings = [
        f
        for f in findings
        if "battery" in f.description.lower() or "battery" in f.title.lower()
    ]
    assert len(battery_findings) > 0
    # Should be at least a warning
    assert any(
        f.severity in (DFMSeverity.WARNING, DFMSeverity.ERROR) for f in battery_findings
    )


def test_empty_bot_no_findings():
    """Bot with no bodies -> no findings."""
    bot = Bot("empty")
    seq = AssemblySequence(ops=())
    check = ComponentRetention()
    findings = check.run(bot, seq, {})
    assert findings == []
