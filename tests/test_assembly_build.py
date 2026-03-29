"""Tests for assembly sequence generation from bot skeleton."""

from botcad.assembly.build import build_assembly_sequence
from botcad.assembly.sequence import AssemblyAction


def _wheeler_base():
    from bots.wheeler_base.design import build

    return build()


def test_wheeler_base_sequence_structure():
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    assert len(seq.ops) > 0
    # Steps are sequential
    steps = [op.step for op in seq.ops]
    assert steps == list(range(len(seq.ops)))
    # Prerequisites reference valid earlier steps
    for op in seq.ops:
        for prereq in op.prerequisites:
            assert prereq < op.step


def test_wheeler_base_has_fasten_ops():
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    fasten_ops = [op for op in seq.ops if op.action == AssemblyAction.FASTEN]
    assert len(fasten_ops) > 0
    for op in fasten_ops:
        assert op.tool is not None


def test_wheeler_base_has_insert_ops():
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    insert_ops = [op for op in seq.ops if op.action == AssemblyAction.INSERT]
    # At least battery, Pi, camera, waveshare, BEC
    assert len(insert_ops) >= 5


def test_wheeler_base_has_route_and_connect_ops():
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    route_ops = [op for op in seq.ops if op.action == AssemblyAction.ROUTE_WIRE]
    connect_ops = [op for op in seq.ops if op.action == AssemblyAction.CONNECT]
    assert len(route_ops) > 0, "Expected ROUTE_WIRE ops for servo wiring"
    assert len(connect_ops) > 0, "Expected CONNECT ops for servo connectors"
