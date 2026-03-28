"""Tests for the fastener catalog — catalog consistency and solid validation."""

from __future__ import annotations

import pytest

from botcad.component import MountPoint
from botcad.fasteners import (
    FastenerSpec,
    HeadType,
    fastener_solid,
    fastener_spec,
    resolve_fastener,
)

# ── Catalog consistency ──────────────────────────────────────────────

_ALL_DESIGNATIONS = ["M2", "M2.5", "M3"]
_ALL_HEAD_TYPES = [HeadType.SOCKET_HEAD_CAP, HeadType.PAN_HEAD_PHILLIPS]


@pytest.mark.parametrize("designation", _ALL_DESIGNATIONS)
@pytest.mark.parametrize("head_type", _ALL_HEAD_TYPES)
def test_catalog_entry_exists(designation, head_type):
    spec = fastener_spec(designation, head_type)
    assert isinstance(spec, FastenerSpec)
    assert spec.designation == designation
    assert spec.head_type == head_type


@pytest.mark.parametrize("designation", _ALL_DESIGNATIONS)
@pytest.mark.parametrize("head_type", _ALL_HEAD_TYPES)
def test_head_larger_than_thread(designation, head_type):
    spec = fastener_spec(designation, head_type)
    assert spec.head_diameter > spec.thread_diameter


@pytest.mark.parametrize("designation", _ALL_DESIGNATIONS)
@pytest.mark.parametrize("head_type", _ALL_HEAD_TYPES)
def test_clearance_larger_than_thread(designation, head_type):
    spec = fastener_spec(designation, head_type)
    assert spec.clearance_hole > spec.thread_diameter
    assert spec.close_fit_hole > spec.thread_diameter
    assert spec.clearance_hole >= spec.close_fit_hole


@pytest.mark.parametrize("designation", _ALL_DESIGNATIONS)
def test_head_height_positive(designation):
    spec = fastener_spec(designation)
    assert spec.head_height > 0


# ── resolve_fastener ─────────────────────────────────────────────────


def test_resolve_from_fastener_type():
    mp = MountPoint("test", pos=(0, 0, 0), diameter=0.003, fastener_type="M3")
    spec = resolve_fastener(mp)
    assert spec.designation == "M3"
    assert spec.head_type == HeadType.SOCKET_HEAD_CAP


def test_resolve_from_diameter_fallback():
    mp = MountPoint("test", pos=(0, 0, 0), diameter=0.002)
    spec = resolve_fastener(mp)
    assert spec.designation == "M2"


def test_resolve_with_head_type():
    mp = MountPoint(
        "test",
        pos=(0, 0, 0),
        diameter=0.003,
        fastener_type="M3",
        head_type="pan_head_phillips",
    )
    spec = resolve_fastener(mp)
    assert spec.head_type == HeadType.PAN_HEAD_PHILLIPS


# ── Solid bounding box validation ────────────────────────────────────


@pytest.mark.parametrize("designation", _ALL_DESIGNATIONS)
@pytest.mark.parametrize("head_type", _ALL_HEAD_TYPES)
def test_solid_bounding_box(designation, head_type):
    """Screw solid should have head at Z=0 and shank extending in -Z."""
    spec = fastener_spec(designation, head_type)
    solid = fastener_solid(spec, 0.004)  # 4mm shank
    bb = solid.bounding_box()

    # Head should be near Z=0 (top)
    assert bb.max.Z > -0.001  # top near or above 0
    # Shank extends in -Z: total depth ≈ head_height + shank_length
    expected_depth = spec.head_height + 0.004
    assert -expected_depth * 0.8 > bb.min.Z  # within 20% tolerance

    # XY extent should be roughly head_diameter
    assert bb.max.X > 0  # centered, extends in +X
    assert bb.min.X < 0  # centered, extends in -X
    xy_extent = bb.max.X - bb.min.X
    assert xy_extent <= spec.head_diameter * 1.5  # not wildly oversized
