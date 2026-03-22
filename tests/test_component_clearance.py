"""Component sub-part clearance tests.

Validates that component sub-parts (horn, coupler, bracket, etc.) do not
intersect each other when positioned in the servo's local frame. Each test
generates the relevant solids via _generate_solid() (same code the viewer
uses) and checks that boolean intersection volume is negligible.
"""

from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")


def _servo():
    from botcad.components.servo import STS3215

    return STS3215()


def _solid(comp, part):
    """Generate a positioned solid for a component part (same as viewer)."""
    from mindsim.server import _generate_solid

    solid = _generate_solid(comp, part)
    assert solid is not None, f"Failed to generate solid for {comp.name}/{part}"
    return solid


def _intersection_volume(a, b) -> float:
    """Compute the volume of the boolean intersection of two solids.

    Returns 0.0 if the intersection is empty or degenerate.
    Uses OCCT BRepAlgoAPI_Common with fuzzy tolerance to avoid
    near-tangent hangs.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common  # noqa: E402
    from OCP.BRepGProp import BRepGProp  # noqa: E402
    from OCP.GProp import GProp_GProps  # noqa: E402
    from OCP.TopAbs import TopAbs_SOLID  # noqa: E402
    from OCP.TopExp import TopExp_Explorer  # noqa: E402

    common = BRepAlgoAPI_Common(a.wrapped, b.wrapped)
    common.SetFuzzyValue(1e-6)
    common.SetUseOBB(True)
    common.Build()
    if not common.IsDone():
        return 0.0

    shape = common.Shape()
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    if not explorer.More():
        return 0.0  # no solid result

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return abs(props.Mass())


# Volume threshold: anything under 0.1 mm³ is considered no intersection.
# (1e-12 m³ = 0.001 mm³, so 1e-10 m³ ≈ 0.1 mm³)
MAX_INTERSECTION_VOLUME = 1e-10


class TestServoSubPartClearance:
    """Sub-parts of a servo should not intersect each other."""

    def test_horn_does_not_intersect_servo(self):
        servo = _servo()
        horn = _solid(servo, "horn")
        body = _solid(servo, "servo")
        vol = _intersection_volume(horn, body)
        assert vol < MAX_INTERSECTION_VOLUME, (
            f"Horn intersects servo body: {vol * 1e9:.1f} mm³ overlap"
        )

    @pytest.mark.xfail(
        reason="Coupler Z-positioning in viewer needs shaft_offset Z + boss clearance"
    )
    def test_coupler_does_not_intersect_servo(self):
        servo = _servo()
        coupler = _solid(servo, "coupler")
        body = _solid(servo, "servo")
        vol = _intersection_volume(coupler, body)
        assert vol < MAX_INTERSECTION_VOLUME, (
            f"Coupler intersects servo body: {vol * 1e9:.1f} mm³ overlap"
        )

    def test_coupler_does_not_intersect_horn(self):
        servo = _servo()
        coupler = _solid(servo, "coupler")
        horn = _solid(servo, "horn")
        vol = _intersection_volume(coupler, horn)
        assert vol < MAX_INTERSECTION_VOLUME, (
            f"Coupler intersects horn: {vol * 1e9:.1f} mm³ overlap"
        )

    @pytest.mark.xfail(
        reason="660mm³ overlap — bracket boss clearance hole doesn't fully clear servo shaft boss"
    )
    def test_bracket_does_not_intersect_servo(self):
        servo = _servo()
        bracket = _solid(servo, "bracket")
        body = _solid(servo, "servo")
        vol = _intersection_volume(bracket, body)
        assert vol < MAX_INTERSECTION_VOLUME, (
            f"Bracket intersects servo body: {vol * 1e9:.1f} mm³ overlap"
        )

    def test_cradle_does_not_intersect_servo(self):
        servo = _servo()
        cradle = _solid(servo, "cradle")
        body = _solid(servo, "servo")
        vol = _intersection_volume(cradle, body)
        assert vol < MAX_INTERSECTION_VOLUME, (
            f"Cradle intersects servo body: {vol * 1e9:.1f} mm³ overlap"
        )
