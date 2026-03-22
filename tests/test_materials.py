from __future__ import annotations

import pytest

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
