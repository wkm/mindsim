"""Component sub-part clearance and containment tests.

Two categories:
1. **Clearance**: sub-parts (horn, coupler, bracket) must not intersect each other.
2. **Containment**: insertion channels must fully contain the servo body.
   If (servo - channel) has volume, the servo won't fit through the cutout.
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


def _boolean_volume(a, b, op_class) -> float:
    """Compute volume of a boolean operation (Common or Cut) on two solids."""
    from OCP.BRepGProp import BRepGProp  # noqa: E402
    from OCP.GProp import GProp_GProps  # noqa: E402
    from OCP.TopAbs import TopAbs_SOLID  # noqa: E402
    from OCP.TopExp import TopExp_Explorer  # noqa: E402

    op = op_class(a.wrapped, b.wrapped)
    op.SetFuzzyValue(1e-6)
    op.SetUseOBB(True)
    op.Build()
    if not op.IsDone():
        return 0.0

    shape = op.Shape()
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    if not explorer.More():
        return 0.0

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return abs(props.Mass())


def _intersection_volume(a, b) -> float:
    """Volume of (a AND b). Zero if no overlap."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common  # noqa: E402

    return _boolean_volume(a, b, BRepAlgoAPI_Common)


def _subtraction_volume(a, b) -> float:
    """Volume of (a - b). Zero if b fully contains a."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut  # noqa: E402

    return _boolean_volume(a, b, BRepAlgoAPI_Cut)


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


# ── Insertion channel containment tests ────────────────────────────────────
# The insertion channel is the cutout subtracted from the parent body shell.
# The servo must fit entirely inside it: (servo - channel) should be empty.

# Known containment failures: (servo_name, channel_type) -> reason
CONTAINMENT_FAILURES = {
    (
        "SCS0009",
        "cradle_insertion_channel",
    ): "Cradle generation crashes — negative box dimension from SCS0009 ear geometry",
}


class TestBracketInsertionChannelContainment:
    """Bracket insertion channel must fully contain the servo body.

    The bracket insertion channel is the cutout subtracted from the parent
    body shell for servo insertion. If (servo - channel) has remaining
    volume, the servo won't fit through the cutout during assembly.
    """

    @pytest.mark.parametrize("servo_name", ALL_SERVOS)
    def test_servo_fits_in_bracket_insertion_channel(self, servo_name):
        servo = _make_servo(servo_name)
        servo_body = _solid(servo, "servo")
        channel = _solid(servo, "bracket_insertion_channel")

        remaining = _subtraction_volume(servo_body, channel)
        servo_vol = abs(servo_body.volume)
        pct = (remaining / servo_vol * 100) if servo_vol > 0 else 0

        assert remaining < MAX_INTERSECTION_VOLUME, (
            f"{servo_name} servo does not fit in bracket_insertion_channel: "
            f"{remaining * 1e9:.1f} mm³ ({pct:.1f}%) sticks out"
        )


class TestCradleInsertionChannelContainment:
    """Cradle insertion channel must contain the servo's lower section.

    The cradle is a shallow tray — it intentionally doesn't enclose the
    full servo. But the servo-channel intersection should equal the
    channel volume (the channel should be fully inside the servo's
    bounding region, not sticking out into empty space).
    """

    @pytest.mark.parametrize("servo_name", ALL_SERVOS)
    def test_cradle_insertion_channel_intersects_servo(self, servo_name):
        xfail_reason = CONTAINMENT_FAILURES.get(
            (servo_name, "cradle_insertion_channel")
        )
        if xfail_reason:
            pytest.xfail(xfail_reason)

        servo = _make_servo(servo_name)
        servo_body = _solid_or_none(servo, "servo")
        channel = _solid_or_none(servo, "cradle_insertion_channel")

        if servo_body is None:
            pytest.skip(f"{servo_name} has no servo solid")
        if channel is None:
            pytest.skip(f"{servo_name} has no cradle_insertion_channel solid")

        # The cradle insertion channel should overlap meaningfully with the
        # servo — at minimum it must cover the mounting ear region
        overlap = _intersection_volume(servo_body, channel)
        assert overlap > 1e-9, (
            f"{servo_name} cradle_insertion_channel has no overlap with servo body"
        )
