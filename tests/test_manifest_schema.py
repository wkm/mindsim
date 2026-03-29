"""Tests for the Design Viewer manifest data model.

Validates that build_viewer_manifest() produces a manifest with the expected
schema: body roles, mounts with pos/quat, fasteners with quat (not axis),
and wire stubs with mesh/pos/quat/color.

Uses wheeler_base as the test bot since it has joints, servos, mounts,
fasteners, and wire stubs.
"""

from __future__ import annotations

import importlib
from functools import cache

import pytest

from botcad.emit.viewer import build_viewer_manifest
from botcad.skeleton import Bot


@cache
def _build_and_solve(name: str) -> Bot:
    """Build and solve a bot by name (cached across tests)."""
    mod = importlib.import_module(f"bots.{name}.design")
    bot = mod.build()
    bot.solve()
    return bot


@pytest.fixture(scope="module")
def manifest():
    """Build the wheeler_base manifest once per module."""
    bot = _build_and_solve("wheeler_base")
    return build_viewer_manifest(bot)


# ---------------------------------------------------------------------------
# Body roles
# ---------------------------------------------------------------------------


class TestBodyRoles:
    def test_manifest_has_body_roles(self, manifest):
        """All bodies have a 'role' field -- either 'structure' or 'component'."""
        bodies = manifest["bodies"]
        assert len(bodies) > 0, "manifest should have at least one body"
        for body in bodies:
            assert "role" in body, f"body {body['name']} missing 'role' field"
            assert body["role"] in ("structure", "component"), (
                f"body {body['name']} has unexpected role: {body['role']}"
            )

    def test_at_least_one_structural_body(self, manifest):
        """There should be at least one structural body (the root)."""
        structural = [b for b in manifest["bodies"] if b["role"] == "structure"]
        assert len(structural) >= 1

    def test_component_bodies_have_category(self, manifest):
        """Component bodies should have a category field."""
        components = [b for b in manifest["bodies"] if b["role"] == "component"]
        for body in components:
            assert "category" in body, (
                f"component body {body['name']} missing 'category'"
            )


# ---------------------------------------------------------------------------
# Mounts
# ---------------------------------------------------------------------------


class TestMounts:
    def test_manifest_has_mounts(self, manifest):
        """Manifest has a 'mounts' array for components on structural bodies."""
        assert "mounts" in manifest
        assert isinstance(manifest["mounts"], list)

    def test_manifest_mounts_have_pos_quat(self, manifest):
        """All mounts have pos and quat fields."""
        for mount in manifest["mounts"]:
            assert "pos" in mount, f"mount {mount.get('label', '?')} missing 'pos'"
            assert "quat" in mount, f"mount {mount.get('label', '?')} missing 'quat'"
            assert len(mount["pos"]) == 3, (
                f"mount {mount.get('label', '?')} pos should be 3-vector"
            )
            assert len(mount["quat"]) == 4, (
                f"mount {mount.get('label', '?')} quat should be 4-vector"
            )

    def test_mounts_have_required_fields(self, manifest):
        """Mounts should have body, label, component, mesh fields."""
        for mount in manifest["mounts"]:
            for key in ("body", "label", "component", "mesh"):
                assert key in mount, f"mount {mount.get('label', '?')} missing '{key}'"


# ---------------------------------------------------------------------------
# Fasteners
# ---------------------------------------------------------------------------


class TestFasteners:
    def _fastener_parts(self, manifest):
        return [p for p in manifest.get("parts", []) if p["category"] == "fastener"]

    def test_manifest_fasteners_have_quat(self, manifest):
        """All fastener parts have pos and quat (no axis-only entries)."""
        fasteners = self._fastener_parts(manifest)
        assert len(fasteners) > 0, "wheeler_base should have fasteners"
        for f in fasteners:
            assert "pos" in f, f"fastener {f['id']} missing 'pos'"
            assert "quat" in f, f"fastener {f['id']} missing 'quat'"
            assert len(f["pos"]) == 3, f"fastener {f['id']} pos should be 3-vector"
            assert len(f["quat"]) == 4, f"fastener {f['id']} quat should be 4-vector"

    def test_manifest_no_axis_field_on_fasteners(self, manifest):
        """Fastener entries should NOT have an 'axis' field (quat is the full orientation)."""
        for f in self._fastener_parts(manifest):
            assert "axis" not in f, (
                f"fastener {f['id']} should not have 'axis' (use 'quat' instead)"
            )

    def test_fastener_quats_are_unit(self, manifest):
        """Fastener quaternions should be approximately unit length."""
        import math

        for f in self._fastener_parts(manifest):
            q = f["quat"]
            norm = math.sqrt(sum(x * x for x in q))
            assert abs(norm - 1.0) < 0.01, (
                f"fastener {f['id']} quat not unit: norm={norm}"
            )


# ---------------------------------------------------------------------------
# Wire routes
# ---------------------------------------------------------------------------


class TestWireRoutes:
    def _wire_routes(self, manifest):
        return [p for p in manifest.get("parts", []) if p["category"] == "wire"]

    def test_wire_routes_have_required_fields(self, manifest):
        """Wire route entries have mesh, color, bus_type, route_label."""
        routes = self._wire_routes(manifest)
        if len(routes) == 0:
            pytest.skip("no wire routes in wheeler_base manifest")
        for r in routes:
            assert "mesh" in r, f"wire route {r['id']} missing 'mesh'"
            assert "color" in r, f"wire route {r['id']} missing 'color'"
            assert "bus_type" in r, f"wire route {r['id']} missing 'bus_type'"
            assert "route_label" in r, f"wire route {r['id']} missing 'route_label'"
            assert len(r["color"]) == 4  # RGBA

    def test_wire_routes_no_stub_fields(self, manifest):
        """Wire routes should NOT have stub-era fields (wire_kind)."""
        routes = self._wire_routes(manifest)
        for r in routes:
            assert "wire_kind" not in r, (
                f"wire route {r['id']} has legacy 'wire_kind' field"
            )
