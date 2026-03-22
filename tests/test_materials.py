from __future__ import annotations

import pytest

from botcad.component import Appearance
from botcad.materials import ALUMINUM, PLA, TPU, Material


def test_pla_density():
    assert PLA.density == 1200.0


def test_pla_has_print_process():
    assert PLA.process is not None
    assert PLA.process.wall_layers == 2
    assert PLA.process.nozzle_width == 0.0004
    assert PLA.process.infill == 0.20


def test_tpu_lower_infill():
    assert TPU.process is not None
    assert TPU.process.infill == 0.15


def test_aluminum_no_print_process():
    assert ALUMINUM.process is None
    assert ALUMINUM.density == 2700.0


def test_material_is_frozen():
    import dataclasses

    assert dataclasses.fields(Material)
    try:
        PLA.density = 999  # type: ignore
        assert False, "Should be frozen"
    except dataclasses.FrozenInstanceError:
        pass


def test_print_process_wall_thickness():
    """Wall thickness = wall_layers * nozzle_width."""
    p = PLA.process
    assert p is not None
    assert p.wall_layers * p.nozzle_width == pytest.approx(0.0008)


def test_appearance_defaults():
    a = Appearance(color=(1.0, 0.0, 0.0, 1.0))
    assert a.metallic == 0.0
    assert a.roughness == 0.7
    assert a.opacity == 1.0


def test_appearance_is_frozen():
    import dataclasses

    a = Appearance(color=(1.0, 0.0, 0.0, 1.0))
    try:
        a.color = (0.0, 0.0, 0.0, 1.0)  # type: ignore
        assert False, "Should be frozen"
    except dataclasses.FrozenInstanceError:
        pass


def test_appearance_with_metallic():
    a = Appearance(color=(0.8, 0.8, 0.8, 1.0), metallic=1.0, roughness=0.3)
    assert a.metallic == 1.0
    assert a.roughness == 0.3
