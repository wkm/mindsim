"""Tests for the structural analysis (FEA) system."""

from __future__ import annotations

from botcad.components.servo import STS3215
from botcad.skeleton import BodyShape, Bot


def test_fea_basic():
    """Verify that FEA runs and produces results."""
    bot = Bot("test_bot")
    base = bot.body("base", shape=BodyShape.BOX)

    # Add a joint
    j = base.joint("j1", servo=STS3215(), axis="z")
    j.body("child", shape=BodyShape.BOX)

    bot.solve()
    results = bot.analyze_stresses()

    assert len(results) == 2  # Parent side and Child side
    assert any(r.joint_name == "j1" and r.side == "parent" for r in results)
    assert any(r.joint_name == "j1" and r.side == "child" for r in results)


def test_fea_safety_factor_low_strength():
    """Verify that lower strength material results in lower safety factor."""
    bot = Bot("strong_bot")
    base = bot.body("base")
    j = base.joint("j", servo=STS3215())
    j.body("child")
    bot.solve()
    sf_strong = bot.analyze_stresses()[0].safety_factor

    from botcad.materials import TPU

    bot2 = Bot("weak_bot")
    base2 = bot2.body("base")
    base2.material = TPU
    j2 = base2.joint("j", servo=STS3215())
    j2.body("child")
    child2 = j2.child
    if child2:
        child2.material = TPU
    bot2.solve()
    sf_weak = bot2.analyze_stresses()[0].safety_factor

    # TPU (15 MPa) vs PLA (40 MPa)
    assert sf_weak < sf_strong
