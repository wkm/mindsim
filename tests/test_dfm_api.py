"""Integration tests for DFM-related API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mindsim.server import app

client = TestClient(app)


class TestAssemblySequenceEndpoint:
    """Tests for GET /api/bots/{bot}/assembly-sequence"""

    @pytest.fixture(scope="class")
    def seq_data(self):
        resp = client.get("/api/bots/wheeler_base/assembly-sequence")
        assert resp.status_code == 200, f"assembly-sequence failed: {resp.text}"
        return resp.json()

    def test_response_shape(self, seq_data):
        """Response has ops and tool_library at top level."""
        assert "ops" in seq_data
        assert "tool_library" in seq_data
        assert isinstance(seq_data["ops"], list)
        assert isinstance(seq_data["tool_library"], dict)

    def test_ops_non_empty(self, seq_data):
        """Wheeler_base should have at least one assembly op."""
        assert len(seq_data["ops"]) > 0

    def test_op_fields(self, seq_data):
        """Each op should have the required fields."""
        op = seq_data["ops"][0]
        assert "step" in op
        assert "action" in op
        assert "body" in op
        assert "description" in op
        assert "target" in op
        assert "prerequisites" in op

    def test_op_action_is_string(self, seq_data):
        """Action should be serialized as its enum value string."""
        for op in seq_data["ops"]:
            assert isinstance(op["action"], str)
            assert op["action"] in {
                "insert",
                "fasten",
                "route_wire",
                "connect",
                "articulate",
            }

    def test_tool_library_has_dimensions(self, seq_data):
        """Tool library entries should have physical dimensions, not solid callables."""
        lib = seq_data["tool_library"]
        assert len(lib) > 0
        for spec in lib.values():
            assert "shaft_diameter" in spec
            assert "shaft_length" in spec
            assert "head_diameter" in spec
            assert "grip_clearance" in spec
            # solid callable should NOT be serialized
            assert "solid" not in spec

    def test_404_for_unknown_bot(self):
        resp = client.get("/api/bots/nonexistent_bot_xyz/assembly-sequence")
        assert resp.status_code == 404


def _wait_for_dfm_complete(
    run_id: str, bot: str = "wheeler_base", timeout: float = 30.0
) -> dict:
    """Poll until DFM run reaches 'complete' or 'failed', or timeout."""
    import time

    deadline = time.monotonic() + timeout
    data = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/bots/{bot}/dfm/{run_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        if data["state"] in ("complete", "failed"):
            return data
        time.sleep(0.5)
    return data


class TestDFMRunEndpoint:
    """Tests for POST /api/bots/{bot}/dfm/run"""

    def test_dfm_run_returns_run_id(self):
        resp = client.post("/api/bots/wheeler_base/dfm/run")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert len(data["run_id"]) > 0

    def test_dfm_run_404_for_unknown_bot(self):
        resp = client.post("/api/bots/nonexistent_bot_xyz/dfm/run")
        assert resp.status_code == 404


class TestDFMStatusEndpoint:
    """Tests for GET /api/bots/{bot}/dfm/{run_id}/status"""

    @pytest.fixture(scope="class")
    def completed_run(self):
        run_id = client.post("/api/bots/wheeler_base/dfm/run").json()["run_id"]
        status = _wait_for_dfm_complete(run_id)
        return run_id, status

    def test_dfm_status_completes(self, completed_run):
        _run_id, status = completed_run
        assert status["state"] == "complete"

    def test_dfm_status_tracks_progress(self, completed_run):
        _run_id, status = completed_run
        assert status["checks_total"] > 0
        assert status["checks_complete"] == status["checks_total"]

    def test_dfm_status_checks_have_names(self, completed_run):
        _run_id, status = completed_run
        for check in status["checks"]:
            assert "name" in check
            assert "state" in check
            assert check["state"] in ("complete", "failed")

    def test_dfm_status_404_for_unknown_run(self):
        resp = client.get("/api/bots/wheeler_base/dfm/nonexistent123/status")
        assert resp.status_code == 404


class TestDFMFindingsEndpoint:
    """Tests for GET /api/bots/{bot}/dfm/{run_id}/findings"""

    @pytest.fixture(scope="class")
    def findings_data(self):
        run_id = client.post("/api/bots/wheeler_base/dfm/run").json()["run_id"]
        _wait_for_dfm_complete(run_id)
        resp = client.get(f"/api/bots/wheeler_base/dfm/{run_id}/findings")
        assert resp.status_code == 200
        return resp.json()

    def test_findings_returns_results(self, findings_data):
        assert "findings" in findings_data
        assert len(findings_data["findings"]) > 0

    def test_finding_has_required_fields(self, findings_data):
        f = findings_data["findings"][0]
        assert "id" in f
        assert "severity" in f
        assert "check_name" in f
        assert "body" in f
        assert "target" in f
        assert "assembly_step" in f
        assert "title" in f
        assert "description" in f
        assert "pos" in f
        assert "has_overlay" in f

    def test_finding_severity_is_string(self, findings_data):
        for f in findings_data["findings"]:
            assert f["severity"] in ("error", "warning", "info")

    def test_finding_target_is_typed(self, findings_data):
        for f in findings_data["findings"]:
            assert "type" in f["target"]
            assert f["target"]["type"] in ("component", "fastener", "wire", "joint")

    def test_findings_404_for_unknown_run(self):
        resp = client.get("/api/bots/wheeler_base/dfm/nonexistent123/findings")
        assert resp.status_code == 404
