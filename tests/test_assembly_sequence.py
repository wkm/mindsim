from botcad.assembly.refs import ComponentRef, FastenerRef
from botcad.assembly.sequence import (
    AssemblyAction,
    AssemblyOp,
    AssemblySequence,
)
from botcad.assembly.tools import ToolKind
from botcad.ids import BodyId, JointId
from botcad.units import Radians


def test_assembly_op_frozen():
    op = AssemblyOp(
        step=0,
        action=AssemblyAction.INSERT,
        target=ComponentRef(body=BodyId("base"), mount_label="battery"),
        body=BodyId("base"),
        tool=ToolKind.FINGERS,
        approach_axis=(0, 0, -1),
        angle=None,
        prerequisites=(),
        description="Insert battery into base pocket",
    )
    assert op.action == AssemblyAction.INSERT
    import pytest

    with pytest.raises(AttributeError):
        op.step = 1


def test_state_at_accumulates():
    ops = (
        AssemblyOp(
            step=0,
            action=AssemblyAction.INSERT,
            target=ComponentRef(BodyId("base"), "battery"),
            body=BodyId("base"),
            tool=ToolKind.FINGERS,
            approach_axis=(0, 0, -1),
            angle=None,
            prerequisites=(),
            description="Insert battery",
        ),
        AssemblyOp(
            step=1,
            action=AssemblyAction.FASTEN,
            target=FastenerRef(BodyId("base"), 0),
            body=BodyId("base"),
            tool=ToolKind.HEX_KEY_2_5,
            approach_axis=(0, 0, 1),
            angle=None,
            prerequisites=(0,),
            description="Fasten bracket",
        ),
    )
    seq = AssemblySequence(ops=ops)
    state = seq.state_at(0)
    assert ComponentRef(BodyId("base"), "battery") in state.installed_components
    assert len(state.installed_fasteners) == 0

    state = seq.state_at(1)
    assert FastenerRef(BodyId("base"), 0) in state.installed_fasteners


def test_state_at_before_any_ops():
    ops = (
        AssemblyOp(
            step=0,
            action=AssemblyAction.INSERT,
            target=ComponentRef(BodyId("base"), "battery"),
            body=BodyId("base"),
            tool=ToolKind.FINGERS,
            approach_axis=(0, 0, -1),
            angle=None,
            prerequisites=(),
            description="Insert battery",
        ),
    )
    seq = AssemblySequence(ops=ops)
    # state_at(-1) = nothing installed yet
    state = seq.state_at(-1)
    assert len(state.installed_components) == 0
    assert len(state.installed_fasteners) == 0
    assert len(state.routed_wires) == 0


def test_articulate_sets_joint_angle():
    ops = (
        AssemblyOp(
            step=0,
            action=AssemblyAction.ARTICULATE,
            target=JointId("shoulder_yaw"),
            body=BodyId("turntable"),
            tool=None,
            approach_axis=None,
            angle=Radians(1.5708),
            prerequisites=(),
            description="Rotate shoulder",
        ),
    )
    seq = AssemblySequence(ops=ops)
    state = seq.state_at(0)
    assert state.joint_angles[JointId("shoulder_yaw")] == Radians(1.5708)


def test_assembly_sequence_step_count():
    ops = tuple(
        AssemblyOp(
            step=i,
            action=AssemblyAction.INSERT,
            target=ComponentRef(BodyId("base"), f"comp_{i}"),
            body=BodyId("base"),
            tool=ToolKind.FINGERS,
            approach_axis=(0, 0, -1),
            angle=None,
            prerequisites=(),
            description=f"Insert comp {i}",
        )
        for i in range(5)
    )
    seq = AssemblySequence(ops=ops)
    assert len(seq.ops) == 5
    state = seq.state_at(4)
    assert len(state.installed_components) == 5
