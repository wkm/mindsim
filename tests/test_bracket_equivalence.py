"""Validate that ShapeScript bracket emission matches direct build123d.

This is a migration validation test. It asserts that executing the
ShapeScript IR (now the primary interface) produces volumes within 0.1%
of the legacy direct build123d functions.

Delete this test after the migration is complete and the _b3d functions
are removed from bracket.py.
"""
from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")

from botcad.bracket import (
    BracketSpec,
    _bracket_envelope_b3d,
    _bracket_solid_b3d,
    _coupler_solid_b3d,
    _cradle_envelope_b3d,
    _cradle_solid_b3d,
    _exec_ir,
    _servo_solid_b3d,
    bracket_envelope,
    bracket_solid,
    coupler_solid,
    cradle_envelope,
    cradle_solid,
)
from botcad.shapescript.emit_servo import servo_script


def _servo():
    from botcad.components.servo import STS3215

    return STS3215()


def _vol(solid):
    return abs(solid.volume)


def _bbox(solid):
    """Return bounding box as ((xmin,ymin,zmin), (xmax,ymax,zmax))."""
    bb = solid.bounding_box()
    return (
        (bb.min.X, bb.min.Y, bb.min.Z),
        (bb.max.X, bb.max.Y, bb.max.Z),
    )


def _com(solid):
    """Return center of mass as (x, y, z)."""
    c = solid.center()
    return (c.X, c.Y, c.Z)


class TestBracketEquivalence:
    """ShapeScript emission must match direct build123d within 0.1%."""

    TOLERANCE = 0.001  # 0.1%

    def _assert_equiv(self, direct_solid, ir_solid, label: str):
        """Assert volume, bounding box, and center-of-mass equivalence."""
        dv, iv = _vol(direct_solid), _vol(ir_solid)
        assert abs(dv - iv) / dv < self.TOLERANCE, f"{label} volume mismatch"

        dbb, ibb = _bbox(direct_solid), _bbox(ir_solid)
        for axis in range(3):
            assert abs(dbb[0][axis] - ibb[0][axis]) < 1e-4, f"{label} bbox min[{axis}]"
            assert abs(dbb[1][axis] - ibb[1][axis]) < 1e-4, f"{label} bbox max[{axis}]"

        dc, ic = _com(direct_solid), _com(ir_solid)
        for axis in range(3):
            assert abs(dc[axis] - ic[axis]) < 1e-4, f"{label} COM[{axis}]"

    def test_bracket_envelope(self):
        servo, spec = _servo(), BracketSpec()
        direct = _bracket_envelope_b3d(servo, spec)
        ir = _exec_ir(bracket_envelope(servo, spec))
        self._assert_equiv(direct, ir, "bracket_envelope")

    def test_bracket_solid(self):
        servo, spec = _servo(), BracketSpec()
        direct = _bracket_solid_b3d(servo, spec)
        ir = _exec_ir(bracket_solid(servo, spec))
        self._assert_equiv(direct, ir, "bracket_solid")

    def test_cradle_envelope(self):
        servo, spec = _servo(), BracketSpec()
        direct = _cradle_envelope_b3d(servo, spec)
        ir = _exec_ir(cradle_envelope(servo, spec))
        self._assert_equiv(direct, ir, "cradle_envelope")

    def test_cradle_solid(self):
        servo, spec = _servo(), BracketSpec()
        direct = _cradle_solid_b3d(servo, spec)
        ir = _exec_ir(cradle_solid(servo, spec))
        self._assert_equiv(direct, ir, "cradle_solid")

    def test_coupler_solid(self):
        servo, spec = _servo(), BracketSpec()
        direct = _coupler_solid_b3d(servo, spec)
        ir = _exec_ir(coupler_solid(servo, spec))
        self._assert_equiv(direct, ir, "coupler_solid")

    def test_servo_solid(self):
        servo = _servo()
        direct = _servo_solid_b3d(servo)
        ir = _exec_ir(servo_script(servo))
        self._assert_equiv(direct, ir, "servo_solid")
