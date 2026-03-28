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

    # No joints or assemblies for a standalone component
    assert m["joints"] == []
    assert m["assemblies"] == []

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
        assert "connector_type" in w
        assert "color" in w
        assert "pos" in w

    # Both wire stubs and connector housings should be present
    stub_ids = [w["id"] for w in wires if w["id"].startswith("wire_")]
    connector_ids = [w["id"] for w in wires if w["id"].startswith("connector_")]
    assert len(stub_ids) > 0, "Expected wire stub parts"
    assert len(connector_ids) > 0, "Expected connector housing parts"


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


def test_servo_manifest_design_layers(client):
    """Servo manifest includes design layer mounts (bracket, coupler, etc.)."""
    resp = client.get("/api/components/STS3215/manifest")
    m = resp.json()

    mount_labels = [mt["label"] for mt in m["mounts"]]
    # Should have bracket, coupler, horn etc. as mounts
    assert "bracket" in mount_labels
    assert "coupler" in mount_labels
    assert "horn" in mount_labels

    # Design layers should have correct categories
    bracket_mount = next(mt for mt in m["mounts"] if mt["label"] == "bracket")
    assert bracket_mount["category"] == "design_layer"

    # Insertion channels should be clearance category
    if "bracket_insertion_channel" in mount_labels:
        bic = next(
            mt for mt in m["mounts"] if mt["label"] == "bracket_insertion_channel"
        )
        assert bic["category"] == "clearance"
        assert bic["color"][3] == 0.25  # semi-transparent


def test_non_servo_no_design_layers(client):
    """Non-servo components should not have design layer mounts."""
    resp = client.get("/api/components/BEC5V/manifest")
    m = resp.json()

    design_mounts = [
        mt for mt in m["mounts"] if mt.get("category") in ("design_layer", "clearance")
    ]
    assert len(design_mounts) == 0
