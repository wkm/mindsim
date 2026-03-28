# tests/test_component_manifest.py
"""Tests for the component manifest endpoint."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from mindsim.server import app

    with TestClient(app) as c:
        yield c


def test_component_manifest_structure(client):
    """Manifest for a multi-material component has correct shape."""
    resp = client.get("/api/components/BEC5V/manifest")
    assert resp.status_code == 200
    m = resp.json()

    # Top-level fields
    assert m["bot_name"] == "BEC5V"
    assert isinstance(m["bodies"], list)
    assert isinstance(m["joints"], list)
    assert isinstance(m["mounts"], list)
    assert isinstance(m["parts"], list)
    assert isinstance(m["materials"], dict)

    # Single body, identity pose
    assert len(m["bodies"]) == 1
    body = m["bodies"][0]
    assert body["name"] == "BEC5V"
    assert body["parent"] is None
    assert body["role"] == "structure"
    assert body["mesh"] == "body"
    assert body["pos"] == [0, 0, 0]
    assert body["quat"] == [1, 0, 0, 0]

    # No joints for a standalone component
    assert m["joints"] == []

    # Self-referential mount
    assert len(m["mounts"]) == 1
    mount = m["mounts"][0]
    assert mount["body"] == "BEC5V"
    assert mount["component"] == "BEC5V"
    assert mount["category"] == "component"


def test_component_manifest_multi_material(client):
    """Multi-material components include meshes array and materials dict."""
    resp = client.get("/api/components/BEC5V/manifest")
    m = resp.json()
    mount = m["mounts"][0]

    # Should have meshes array for multi-material rendering
    assert "meshes" in mount
    assert len(mount["meshes"]) > 0
    for mesh_entry in mount["meshes"]:
        assert "file" in mesh_entry
        assert "material" in mesh_entry
        # Material must exist in the materials dict
        assert mesh_entry["material"] in m["materials"]

    # Each material has required fields
    for mat in m["materials"].values():
        assert "color" in mat and len(mat["color"]) == 3
        assert "metallic" in mat
        assert "roughness" in mat
        assert "opacity" in mat


def test_component_manifest_fasteners(client):
    """Components with mounting points include fastener parts."""
    resp = client.get("/api/components/STS3215/manifest")
    m = resp.json()

    fasteners = [p for p in m["parts"] if p["category"] == "fastener"]
    assert len(fasteners) > 0
    for f in fasteners:
        assert "id" in f
        assert "name" in f
        assert f["parent_body"] == "STS3215"
        assert "mesh" in f
        assert "pos" in f
        assert "quat" in f


def test_component_manifest_wires(client):
    """Components with wire ports include wire parts."""
    resp = client.get("/api/components/STS3215/manifest")
    m = resp.json()

    wires = [p for p in m["parts"] if p["category"] == "wire"]
    assert len(wires) > 0
    for w in wires:
        assert "id" in w
        assert "name" in w
        assert w["parent_body"] == "STS3215"
        assert "bus_type" in w
        assert "pos" in w


def test_component_manifest_404(client):
    """Unknown component returns 404."""
    resp = client.get("/api/components/nonexistent/manifest")
    assert resp.status_code == 404


def test_servo_manifest_has_layers(client):
    """Servo components include body layers (bracket, horn, etc.) as additional bodies."""
    resp = client.get("/api/components/STS3215/manifest")
    m = resp.json()

    # Servo manifest should still have one structural body
    assert len(m["bodies"]) == 1
    # The mount should reference the servo component
    assert len(m["mounts"]) >= 1
