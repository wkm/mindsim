"""Tests for the clearance constraint system."""

from __future__ import annotations

import pytest

from botcad.skeleton import (
    BodyShape,
    Bot,
    ClearanceConstraint,
)


class TestClearanceConstraint:
    """Unit tests for the ClearanceConstraint dataclass."""

    def test_defaults(self):
        from botcad.ids import BodyId

        c = ClearanceConstraint(body_a=BodyId("a"), body_b=BodyId("b"))
        assert c.min_distance == 0.0
        assert c.label == ""

    def test_frozen(self):
        from botcad.ids import BodyId

        c = ClearanceConstraint(
            body_a=BodyId("a"), body_b=BodyId("b"), min_distance=0.001
        )
        with pytest.raises(AttributeError):
            c.min_distance = 0.002  # type: ignore[misc]


class TestBotClearanceAPI:
    """Tests for Bot.clearance() and implicit constraint generation."""

    def test_explicit_clearance(self):
        bot = Bot("test")
        bot.clearance("body_a", "body_b", min_distance=0.001, label="test gap")
        assert len(bot._clearance_constraints) == 1
        c = bot._clearance_constraints[0]
        assert c.body_a == "body_a"
        assert c.body_b == "body_b"
        assert c.min_distance == 0.001
        assert c.label == "test gap"

    def test_implicit_constraints_child_parent(self):
        """solve() generates child-parent clearance constraints."""
        from botcad.components import STS3215

        bot = Bot("test")
        base = bot.body("base", dimensions=(0.05, 0.05, 0.05))
        j = base.joint("j1", servo=STS3215(), axis="z", pos=(0.0, 0.0, 0.03))
        j.body("arm", dimensions=(0.03, 0.03, 0.06))
        bot.solve()

        # Should have at least a child-parent constraint
        labels = [c.label for c in bot._clearance_constraints]
        assert any("parent-child" in lbl or "child-parent" in lbl for lbl in labels)

    def test_implicit_constraints_wheel_servo(self):
        """solve() generates wheel-servo clearance for wheel bodies."""
        from botcad.components import STS3215, PololuWheel90mm

        bot = Bot("test")
        base = bot.body("base", dimensions=(0.10, 0.05, 0.05))
        j = base.joint(
            "wheel", servo=STS3215(continuous=True), axis="x", pos=(0.05, 0.0, 0.0)
        )
        rim = j.body("rim", shape=BodyShape.CYLINDER, radius=0.045, width=0.010)
        rim.mount(PololuWheel90mm(), label="wheel")
        bot.solve()

        # Should have wheel-servo constraint
        pairs = [(c.body_a, c.body_b) for c in bot._clearance_constraints]
        assert ("rim", "servo_wheel") in pairs

    def test_no_duplicate_implicit_constraints(self):
        """Explicit constraints prevent duplicate implicit ones."""
        from botcad.components import STS3215, PololuWheel90mm

        bot = Bot("test")
        base = bot.body("base", dimensions=(0.10, 0.05, 0.05))
        j = base.joint(
            "wheel", servo=STS3215(continuous=True), axis="x", pos=(0.05, 0.0, 0.0)
        )
        rim = j.body("rim", shape=BodyShape.CYLINDER, radius=0.045, width=0.010)
        rim.mount(PololuWheel90mm(), label="wheel")

        # Add explicit constraint before solve
        bot.clearance("rim", "servo_wheel", min_distance=0.001, label="explicit")
        bot.solve()

        # Should not have a duplicate
        wheel_servo = [
            c
            for c in bot._clearance_constraints
            if {c.body_a, c.body_b} == {"rim", "servo_wheel"}
        ]
        assert len(wheel_servo) == 1
        assert wheel_servo[0].label == "explicit"


class TestClearanceValidation:
    """Integration tests using actual CAD geometry."""

    @pytest.mark.xfail(
        reason="Component pocket centering — protrusions extend beyond centered pockets"
    )
    def test_wheeler_base_no_violations(self):
        """Built wheeler_base should have no clearance violations."""
        from bots.wheeler_base.design import build

        bot = build()
        bot.solve()
        bot.build_cad()

        violations = [r for r in bot.clearance_results if not r.satisfied]
        assert violations == [], "Clearance violations: " + ", ".join(
            f"{v.body_a}<->{v.body_b} {v.distance * 1000:.1f}mm ({v.label})"
            for v in violations
        )

    def test_clearance_results_populated(self):
        """build_cad() populates clearance_results on the bot."""
        from bots.wheeler_base.design import build

        bot = build()
        bot.solve()
        bot.build_cad()

        # Should have results (at least the explicit + implicit ones)
        assert len(bot.clearance_results) > 0
        # Every result should have the expected fields
        for r in bot.clearance_results:
            assert isinstance(r.distance, float)
            assert isinstance(r.satisfied, bool)
            assert r.body_a != ""
            assert r.body_b != ""
