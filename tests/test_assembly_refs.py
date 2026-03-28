from botcad.assembly.refs import ComponentRef, FastenerRef, WireRef
from botcad.ids import BodyId


def test_component_ref_frozen():
    ref = ComponentRef(body=BodyId("base"), mount_label="battery")
    assert ref.body == BodyId("base")
    assert ref.mount_label == "battery"
    import pytest

    with pytest.raises(AttributeError):
        ref.body = BodyId("other")


def test_component_ref_hashable():
    a = ComponentRef(body=BodyId("base"), mount_label="battery")
    b = ComponentRef(body=BodyId("base"), mount_label="battery")
    assert a == b
    assert len({a, b}) == 1


def test_fastener_ref():
    ref = FastenerRef(body=BodyId("base"), index=0)
    assert ref.body.name == "base"
    assert ref.index == 0


def test_wire_ref():
    ref = WireRef(label="left_wheel_uart")
    assert ref.label == "left_wheel_uart"


def test_refs_not_equal_across_types():
    """Different ref types with similar data should not be equal."""
    comp = ComponentRef(body=BodyId("base"), mount_label="0")
    fast = FastenerRef(body=BodyId("base"), index=0)
    assert comp != fast
