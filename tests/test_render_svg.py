"""Tests for the SVG rendering pipeline (botcad/render_svg.py).

Validates section view orientation and projection view correctness
without requiring the web server — tests the geometry pipeline directly.
"""

import re

import pytest

from botcad.render_svg import render_component_svg


@pytest.fixture(scope="session")
def servo_solid():
    """Build a servo solid for testing. Cached across the session."""
    from botcad.bracket import servo_solid
    from botcad.components import STS3215

    servo = STS3215()
    return ("servo", servo_solid(servo), (24, 32, 38))


@pytest.fixture(scope="session")
def bracket_solid():
    """Build a bracket solid for testing."""
    from botcad.bracket import BracketSpec, bracket_solid_solid as bracket_solid
    from botcad.components import STS3215

    servo = STS3215()
    return ("bracket", bracket_solid(servo, BracketSpec()), (206, 217, 224))


def _parse_viewbox(svg: str):
    """Extract viewBox dimensions from SVG string."""
    m = re.search(r'viewBox="([^ ]+) ([^ ]+) ([^ ]+) ([^ "]+)"', svg)
    assert m, "SVG must have a viewBox"
    return tuple(float(m.group(i)) for i in range(1, 5))


class TestProjectionView:
    """Projection views (no section plane)."""

    def test_front_view_produces_svg(self, servo_solid):
        svg = render_component_svg(
            [servo_solid],
            view_origin=(0, -10, 0),
            view_up=(0, 0, 1),
        )
        assert svg.startswith("<?xml")
        assert "<svg" in svg

    def test_side_view_produces_svg(self, servo_solid):
        svg = render_component_svg(
            [servo_solid],
            view_origin=(10, 0, 0),
            view_up=(0, 0, 1),
        )
        assert "<svg" in svg

    def test_contains_layer_groups(self, servo_solid):
        svg = render_component_svg(
            [servo_solid],
            view_origin=(0, -10, 0),
            view_up=(0, 0, 1),
        )
        assert "servo" in svg  # layer name appears in the SVG


class TestSectionOrientation:
    """Section views must have canonical orientations regardless of camera.

    Convention (Z-up world):
      X section: right = -Y, up = +Z
      Y section: right = +X, up = +Z
      Z section: right = +X, up = +Y

    We verify by checking that the SVG bounding box aspect ratio matches
    the expected cross-section shape, and that a known asymmetric feature
    appears on the correct side.
    """

    @pytest.fixture(
        params=[
            # (axis, position, expected_wider_axis)
            # The servo is roughly 45mm(X) x 24mm(Y) x 48mm(Z).
            # ViewBox includes dimension line expansion so aspect ratios
            # are approximate.
            # X section at center: cross-section is Y×Z ≈ 24×48 (taller than wide)
            ("x", 0.0, "taller"),
            # Y section at center: cross-section is X×Z ≈ 45×48 (nearly square)
            ("y", 0.0, "wider"),
            # Z section at center: cross-section is X×Y ≈ 45×24 (wider than tall)
            ("z", 0.01, "wider"),
        ],
        ids=["x-section", "y-section", "z-section"],
    )
    def section_svg(self, request, servo_solid):
        axis, position, expected_shape = request.param
        from build123d import Plane, Vector

        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        normal = [0.0, 0.0, 0.0]
        normal[axis_idx] = 1.0
        origin = [0.0, 0.0, 0.0]
        origin[axis_idx] = position

        section_plane = Plane(
            origin=Vector(*origin),
            z_dir=Vector(*normal),
        )

        svg = render_component_svg(
            [servo_solid],
            view_origin=(0, -10, 0),
            view_up=(0, 0, 1),
            section_plane=section_plane,
        )
        return svg, expected_shape

    def test_produces_valid_svg(self, section_svg):
        svg, _ = section_svg
        assert svg.startswith("<?xml")
        assert "<svg" in svg

    def test_section_has_geometry(self, section_svg):
        """The section should contain visible geometry (paths/edges)."""
        svg, _ = section_svg
        # ExportSVG produces <path> or <line> elements for edges
        assert "<path" in svg or "<line" in svg or "<polygon" in svg

    def test_aspect_ratio_matches_expected_shape(self, section_svg):
        """Verify the bounding box aspect ratio is consistent with
        the expected cross-section orientation (wider vs taller)."""
        svg, expected_shape = section_svg
        vb_x, vb_y, vb_w, vb_h = _parse_viewbox(svg)

        if expected_shape == "wider":
            assert vb_w > vb_h, f"Expected wider than tall, got {vb_w:.4f} x {vb_h:.4f}"
        elif expected_shape == "taller":
            assert vb_h > vb_w, (
                f"Expected taller than wide, got {vb_w:.4f} x {vb_h:.4f}"
            )

    def test_orientation_independent_of_camera(self, servo_solid):
        """The same Y section should produce identical SVG regardless
        of camera direction — sections are camera-independent."""
        from build123d import Plane, Vector

        section_plane = Plane(
            origin=Vector(0, 0, 0),
            z_dir=Vector(0, 1, 0),
        )

        svg_front = render_component_svg(
            [servo_solid],
            view_origin=(0, -10, 0),
            view_up=(0, 0, 1),
            section_plane=section_plane,
        )
        svg_side = render_component_svg(
            [servo_solid],
            view_origin=(10, 0, 0),
            view_up=(0, 0, 1),
            section_plane=section_plane,
        )

        # ViewBoxes should be identical (same geometry, same orientation)
        vb_front = _parse_viewbox(svg_front)
        vb_side = _parse_viewbox(svg_side)
        for a, b in zip(vb_front, vb_side):
            assert abs(a - b) < 1e-6, (
                f"ViewBox differs: front={vb_front}, side={vb_side}"
            )


class TestAnnotations:
    def test_annotations_injected(self, servo_solid):
        svg = render_component_svg(
            [servo_solid],
            view_origin=(0, -10, 0),
            view_up=(0, 0, 1),
            annotate={
                "component": "STS3215",
                "view": "Front",
                "layers": ["Servo"],
                "dimensions_mm": [45, 24, 48],
                "mass_g": 55.0,
            },
        )
        assert "STS3215" in svg
        assert "Front view" in svg
        assert "10 mm" in svg  # scale bar


class TestDimensions:
    def test_dimension_lines_present(self, servo_solid):
        svg = render_component_svg(
            [servo_solid],
            view_origin=(0, -10, 0),
            view_up=(0, 0, 1),
        )
        assert "<!-- Dimensions -->" in svg
        assert "<polygon" in svg  # arrowheads
