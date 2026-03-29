"""Tests for botcad.units — dimension NewTypes and factory functions."""

from __future__ import annotations

import math

import pytest

from botcad.units import (
    Amps,
    Degrees,
    Kg,
    KgPerM3,
    Meters,
    NewtonM,
    Pascals,
    Radians,
    RadPerSec,
    Volts,
    deg_to_rad,
    gpa,
    grams,
    mm,
    mm3,
    mpa,
)


def test_mm_converts_to_meters():
    assert mm(25) == pytest.approx(0.025)


def test_mm3_returns_tuple():
    result = mm3(25, 24, 9)
    assert result == pytest.approx((0.025, 0.024, 0.009))


def test_grams_converts_to_kg():
    assert grams(3) == pytest.approx(0.003)


def test_mpa_converts_to_pascals():
    assert mpa(40) == pytest.approx(40e6)


def test_gpa_converts_to_pascals():
    assert gpa(2.3) == pytest.approx(2.3e9)


def test_deg_to_rad_converts():
    assert deg_to_rad(180) == pytest.approx(math.pi)
    assert deg_to_rad(90) == pytest.approx(math.pi / 2)


def test_types_are_float_at_runtime():
    """NewType erases at runtime — values are plain float."""
    assert isinstance(Meters(1.0), float)
    assert isinstance(Kg(1.0), float)
    assert isinstance(Degrees(45.0), float)
    assert isinstance(Radians(1.0), float)
    assert isinstance(Volts(12.0), float)
    assert isinstance(Amps(0.5), float)
    assert isinstance(NewtonM(3.0), float)
    assert isinstance(RadPerSec(6.28), float)
    assert isinstance(Pascals(1e6), float)
    assert isinstance(KgPerM3(1200.0), float)


def test_arithmetic_works():
    """Typed values participate in normal float arithmetic."""
    a = Meters(0.025)
    b = Meters(0.010)
    assert a + b == pytest.approx(0.035)
    assert a * 2 == pytest.approx(0.050)
