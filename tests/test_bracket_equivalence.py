"""Validate that ShapeScript bracket emission matches direct build123d.

This is a migration validation test. It asserts that
backend_occt.execute(bracket_script) produces volumes within 0.1%
of the direct build123d bracket_solid() for all bracket types.

Delete this test after the migration is complete and bracket.py
no longer has a build123d code path.
"""
from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")

from botcad.bracket import (
    BracketSpec,
    bracket_envelope,
    bracket_solid,
    coupler_solid,
    cradle_envelope,
    cradle_solid,
    servo_solid,
)
from botcad.shapescript.backend_occt import OcctBackend
from botcad.shapescript.emit_bracket import (
    bracket_envelope_script,
    bracket_solid_script,
    coupler_solid_script,
    cradle_envelope_script,
    cradle_solid_script,
)
from botcad.shapescript.emit_servo import servo_script


def _servo():
    from botcad.components.servo import STS3215

    return STS3215()


def _exec(prog):
    """Execute a ShapeScript and return the output solid.

    OcctBackend().execute() returns an ExecutionResult with a .shapes dict
    mapping ref IDs to build123d Solids. We extract the output solid.
    """
    result = OcctBackend().execute(prog)
    return result.shapes[prog.output_ref.id]


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
    """ShapeScript emission must match direct build123d within 0.1%.

    Volume alone won't catch positional bugs (e.g. a bracket translated
    to the wrong location). We also compare bounding box extents and
    center-of-mass to catch spatial regressions.
    """

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
        direct = bracket_envelope(servo, spec)
        ir = _exec(bracket_envelope_script(servo, spec))
        self._assert_equiv(direct, ir, "bracket_envelope")

    def test_bracket_solid(self):
        servo, spec = _servo(), BracketSpec()
        direct = bracket_solid(servo, spec)
        ir = _exec(bracket_solid_script(servo, spec))
        self._assert_equiv(direct, ir, "bracket_solid")

    def test_cradle_envelope(self):
        servo, spec = _servo(), BracketSpec()
        direct = cradle_envelope(servo, spec)
        ir = _exec(cradle_envelope_script(servo, spec))
        self._assert_equiv(direct, ir, "cradle_envelope")

    def test_cradle_solid(self):
        servo, spec = _servo(), BracketSpec()
        direct = cradle_solid(servo, spec)
        ir = _exec(cradle_solid_script(servo, spec))
        self._assert_equiv(direct, ir, "cradle_solid")

    def test_coupler_solid(self):
        servo, spec = _servo(), BracketSpec()
        direct = coupler_solid(servo, spec)
        ir = _exec(coupler_solid_script(servo, spec))
        self._assert_equiv(direct, ir, "coupler_solid")

    def test_servo_solid(self):
        servo = _servo()
        direct = servo_solid(servo)
        ir = _exec(servo_script(servo))
        self._assert_equiv(direct, ir, "servo_solid")
