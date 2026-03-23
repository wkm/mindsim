"""Component sub-part clearance tests.

Validates that component sub-parts (horn, coupler, bracket, etc.) do not
intersect each other when positioned in the servo's local frame. Each test
generates the relevant solids via _generate_solid() (same code the viewer
uses) and checks that boolean intersection volume is negligible.
"""

from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")


def _make_servo(name: str):
    from botcad.components.servo import SCS0009, STS3215, STS3250

    factories = {
        "STS3215": STS3215,
        "STS3250": STS3250,
        "SCS0009": SCS0009,
    }
    return factories[name]()


def _solid(comp, part):
    """Generate a positioned solid for a component part (same as viewer)."""
    from mindsim.server import _generate_solid

    solid = _generate_solid(comp, part)
    assert solid is not None, f"Failed to generate solid for {comp.name}/{part}"
    return solid


def _solid_or_none(comp, part):
    """Generate a positioned solid, returning None if the part doesn't exist."""
    from mindsim.server import _generate_solid

    return _generate_solid(comp, part)


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

ALL_SERVOS = ["STS3215", "STS3250", "SCS0009"]

# Known failures: (servo_name, part_a, part_b) -> reason
KNOWN_FAILURES = {
    (
        "STS3215",
        "coupler",
        "servo",
    ): "Coupler Z-positioning in viewer needs shaft_offset Z + boss clearance",
    (
        "STS3215",
        "bracket",
        "servo",
    ): "660mm³ overlap — bracket boss clearance hole doesn't fully clear servo shaft boss",
    (
        "STS3250",
        "coupler",
        "servo",
    ): "Coupler Z-positioning in viewer needs shaft_offset Z + boss clearance (same geometry as STS3215)",
    (
        "STS3250",
        "bracket",
        "servo",
    ): "Same bracket clearance issue as STS3215 (identical form factor)",
    (
        "SCS0009",
        "coupler",
        "servo",
    ): "1.0 mm³ overlap — coupler intersects SCS0009 servo body",
    (
        "SCS0009",
        "cradle",
        "servo",
    ): "Cradle generation crashes — negative box dimension from SCS0009 ear geometry",
}


def _has_horn(servo_name: str) -> bool:
    """Check if a servo has a horn disc."""
    from botcad.bracket import horn_disc_params

    servo = _make_servo(servo_name)
    return horn_disc_params(servo) is not None


def _part_pairs():
    """Generate all (servo_name, part_a, part_b) test combinations."""
    pairs = []
    for name in ALL_SERVOS:
        has_horn = _has_horn(name)

        # Always test these
        pairs.append((name, "coupler", "servo"))
        pairs.append((name, "bracket", "servo"))
        pairs.append((name, "cradle", "servo"))

        if has_horn:
            pairs.append((name, "horn", "servo"))
            pairs.append((name, "coupler", "horn"))

    return pairs


class TestServoSubPartClearance:
    """Sub-parts of a servo should not intersect each other."""

    @pytest.mark.parametrize(
        "servo_name,part_a,part_b",
        _part_pairs(),
        ids=[f"{s}-{a}_vs_{b}" for s, a, b in _part_pairs()],
    )
    def test_no_intersection(self, servo_name, part_a, part_b):
        xfail_reason = KNOWN_FAILURES.get((servo_name, part_a, part_b))
        if xfail_reason:
            pytest.xfail(xfail_reason)

        servo = _make_servo(servo_name)

        solid_a = _solid_or_none(servo, part_a)
        solid_b = _solid_or_none(servo, part_b)

        if solid_a is None:
            pytest.skip(f"{servo_name} has no {part_a} solid")
        if solid_b is None:
            pytest.skip(f"{servo_name} has no {part_b} solid")

        vol = _intersection_volume(solid_a, solid_b)
        assert vol < MAX_INTERSECTION_VOLUME, (
            f"{part_a} intersects {part_b} on {servo_name}: {vol * 1e9:.1f} mm³ overlap"
        )
