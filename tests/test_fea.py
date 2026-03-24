"""Tests for the structural analysis (FEA) system."""

from __future__ import annotations

import pytest
from botcad.fea import analyze_joint_stresses
from botcad.skeleton import Bot, BodyShape, BracketStyle
from botcad.components.servo import STS3215

def test_fea_basic():
    """Verify that FEA runs and produces results."""
    bot = Bot("test_bot")
    base = bot.body("base", shape=BodyShape.BOX)
    
    # Add a joint
    j = base.joint("j1", servo=STS3215(), axis="z")
    child = j.body("child", shape=BodyShape.BOX)
    
    bot.solve()
    results = bot.analyze_stresses()
    
    assert len(results) == 2 # Parent side and Child side
    assert any(r.joint_name == "j1" and r.side == "parent" for r in _tag_results(results))
    assert any(r.joint_name == "j1" and r.side == "child" for r in _tag_results(results))

def _tag_results(results):
    # The current StressResult doesn't have a 'side' field, but it has body names
    # and failure modes. We add 'side' for easier testing if needed, but for now
    # we just check that we have two results for the same joint.
    return results

def test_fea_safety_factor_low_strength():
    """Verify that lower strength material results in lower safety factor."""
    bot = Bot("strong_bot")
    base = bot.body("base")
    j = base.joint("j", servo=STS3215())
    child = j.body("child")
    bot.solve()
    sf_strong = bot.analyze_stresses()[0].safety_factor
    
    from botcad.materials import TPU
    bot2 = Bot("weak_bot")
    base2 = bot2.body("base")
    base2.material = TPU
    j2 = base2.joint("j", servo=STS3215())
    child2 = j2.body("child")
    child2.material = TPU
    bot2.solve()
    sf_weak = bot2.analyze_stresses()[0].safety_factor
    
    # TPU (15 MPa) vs PLA (40 MPa)
    assert sf_weak < sf_strong
