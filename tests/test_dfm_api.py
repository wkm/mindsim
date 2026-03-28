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
