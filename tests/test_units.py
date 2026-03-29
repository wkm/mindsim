from botcad.units import Meters, Radians


def test_meters_runtime_transparent():
    """NewType is zero-cost at runtime — Meters(...) returns a plain float."""
    m = Meters(0.003)
    assert isinstance(m, float)
    assert m == 0.003


def test_radians_runtime_transparent():
    r = Radians(1.5708)
    assert isinstance(r, float)


def test_meters_arithmetic():
    a = Meters(0.003)
    b = Meters(0.002)
    assert a + b == 0.005
