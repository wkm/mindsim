"""Tests for the tool library."""

from botcad.assembly.tools import TOOL_LIBRARY, ToolKind


def test_tool_library_complete():
    """Every ToolKind has a corresponding ToolSpec."""
    for kind in ToolKind:
        assert kind in TOOL_LIBRARY, f"Missing ToolSpec for {kind}"


def test_hex_key_2_5_dimensions():
    spec = TOOL_LIBRARY[ToolKind.HEX_KEY_2_5]
    assert spec.shaft_diameter == 0.0025  # 2.5mm in meters
    assert spec.shaft_length == 0.050  # 50mm


def test_tool_spec_frozen():
    spec = TOOL_LIBRARY[ToolKind.FINGERS]
    import pytest

    with pytest.raises(AttributeError):
        spec.shaft_diameter = 0.1  # type: ignore[misc]


def test_tool_solid_generates_shape():
    """Tool solid callable returns a build123d shape with nonzero volume."""
    spec = TOOL_LIBRARY[ToolKind.HEX_KEY_2_5]
    shape = spec.solid()
    assert shape is not None
    bb = shape.bounding_box()
    # Should have nonzero extent in all dimensions
    assert bb.max.X - bb.min.X > 0
    assert bb.max.Y - bb.min.Y > 0
    assert bb.max.Z - bb.min.Z > 0
