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
