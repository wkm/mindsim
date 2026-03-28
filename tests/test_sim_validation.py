"""Smoke tests: wheeler_base loads in MuJoCo and sim properties match BOM."""

from pathlib import Path

import mujoco

_ROOT = Path(__file__).parent.parent
SCENE_XML = _ROOT / "bots/wheeler_base/scene.xml"
BOT_XML = _ROOT / "bots/wheeler_base/bot.xml"


def test_loads_without_error():
    """scene.xml parses, meshes resolve, one physics step runs."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)  # one step without crash


def test_mass_matches_bom():
    """Total robot mass within 2% of BOM total (0.468 kg).

    Uses bot.xml (robot only) so world furniture doesn't pollute the sum.
    """
    model = mujoco.MjModel.from_xml_path(str(BOT_XML))
    # body_mass[0] is the immovable MuJoCo world body -- skip it
    total_mass = sum(model.body_mass[1:])
    assert abs(total_mass - 0.468) / 0.468 < 0.02, (
        f"Total mass {total_mass:.4f} kg deviates >2% from BOM 0.468 kg"
    )
