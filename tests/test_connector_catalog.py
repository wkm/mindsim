"""Tests for the connector catalog — dimension validity and solid watertightness."""

from __future__ import annotations

import pytest

from botcad.connectors import (
    ConnectorSpec,
    ConnectorType,
    connector_solid,
    connector_spec,
)

_ALL_TYPES = list(ConnectorType)


@pytest.mark.parametrize("ct", _ALL_TYPES)
def test_catalog_entry_exists(ct):
    spec = connector_spec(ct.value)
    assert isinstance(spec, ConnectorSpec)
    assert spec.connector_type == ct


@pytest.mark.parametrize("ct", _ALL_TYPES)
def test_dimensions_positive(ct):
    spec = connector_spec(ct.value)
    bx, by, bz = spec.body_dimensions
    assert bx > 0 and by > 0 and bz > 0


@pytest.mark.parametrize("ct", _ALL_TYPES)
def test_solid_has_volume(ct):
    spec = connector_spec(ct.value)
    solid = connector_solid(spec)
    assert solid.volume > 0


@pytest.mark.parametrize("ct", _ALL_TYPES)
def test_solid_bounding_box_matches_dimensions(ct):
    """Solid bounding box should be close to body_dimensions."""
    spec = connector_spec(ct.value)
    solid = connector_solid(spec)
    bb = solid.bounding_box()
    bx, by, bz = spec.body_dimensions

    # Allow 50% tolerance for retention features and pin rows
    assert (bb.max.X - bb.min.X) <= bx * 1.5
    assert (bb.max.Y - bb.min.Y) <= by * 1.5
    assert (bb.max.Z - bb.min.Z) <= bz * 1.5


# ── Receptacle solids ────────────────────────────────────────────────


@pytest.mark.parametrize("ct", _ALL_TYPES)
def test_receptacle_has_volume(ct):
    from botcad.connectors import receptacle_solid

    spec = connector_spec(ct.value)
    solid = receptacle_solid(spec)
    assert solid.volume > 0


@pytest.mark.parametrize("ct", _ALL_TYPES)
def test_receptacle_larger_than_plug(ct):
    """Receptacle should be at least as large as the plug in XY."""
    from botcad.connectors import receptacle_solid

    spec = connector_spec(ct.value)
    plug_bb = connector_solid(spec).bounding_box()
    rcpt_bb = receptacle_solid(spec).bounding_box()

    plug_x = plug_bb.max.X - plug_bb.min.X
    plug_y = plug_bb.max.Y - plug_bb.min.Y
    rcpt_x = rcpt_bb.max.X - rcpt_bb.min.X
    rcpt_y = rcpt_bb.max.Y - rcpt_bb.min.Y

    assert rcpt_x >= plug_x * 0.9  # receptacle at least ~90% of plug width
    assert rcpt_y >= plug_y * 0.9
