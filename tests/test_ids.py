from botcad.ids import BodyId, JointId


def test_body_id_frozen():
    bid = BodyId("base")
    assert bid.name == "base"
    import pytest

    with pytest.raises(AttributeError):
        bid.name = "other"


def test_body_id_hashable():
    a = BodyId("base")
    b = BodyId("base")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_body_id_and_joint_id_are_distinct_types():
    bid = BodyId("base")
    jid = JointId("base")
    # String equality holds (both wrap "base"), but types differ
    assert type(bid) is not type(jid)
    assert isinstance(bid, BodyId)
    assert not isinstance(bid, JointId)


def test_body_id_str():
    bid = BodyId("base")
    assert str(bid) == "base"


def test_joint_id_frozen_and_hashable():
    a = JointId("left_wheel")
    b = JointId("left_wheel")
    assert a == b
    assert hash(a) == hash(b)
