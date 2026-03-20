"""Single-view SVG renderer for the component browser.

Produces a clean SVG with per-layer <g> groups from build123d solids,
suitable for high-quality 2D export from the web viewer. Supports
both projection views (visible + hidden lines) and section views
(cross-section fills with outlines).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from build123d import (
    Color,
    ExportSVG,
    LineType,
    Location,
    ShapeList,
    Unit,
    section,
)

from botcad.debug_drawing import _project_offset, _viewport_axes

# SVG rendering parameters
LINE_WEIGHT_VISIBLE = 0.5  # mm
LINE_WEIGHT_HIDDEN = 0.2  # mm
LINE_WEIGHT_FILL = 0.01  # mm (effectively invisible outline on fills)
SVG_SCALE = 1000  # 1m → 1000 SVG units


def render_component_svg(
    solids: list[tuple[str, object, tuple[int, int, int]]],
    view_origin: tuple[float, float, float],
    view_up: tuple[float, float, float] = (0, 0, 1),
    section_plane=None,
) -> str:
    """Render a single 2D view of component solids as an SVG string.

    Args:
        solids: List of (layer_id, Solid, rgb) tuples.
        view_origin: Camera position (far from origin, looking at 0,0,0).
        view_up: Camera up vector.
        section_plane: Optional build123d Plane for cross-section view.

    Returns:
        SVG string with per-layer <g> groups.
    """
    svg = ExportSVG(
        unit=Unit.MM,
        scale=SVG_SCALE,
        margin=0,
        line_weight=LINE_WEIGHT_VISIBLE,
    )

    # Register layers
    for layer_id, _solid, (r, g, b) in solids:
        c = Color(r / 255, g / 255, b / 255)
        c_fill = Color(r / 255, g / 255, b / 255, 0.15)
        c_none = Color(r / 255, g / 255, b / 255, 0.0)

        svg.add_layer(layer_id, line_color=c, line_weight=LINE_WEIGHT_VISIBLE)
        svg.add_layer(
            f"{layer_id}_fill",
            fill_color=c_fill,
            line_color=c_none,
            line_weight=LINE_WEIGHT_FILL,
        )
        svg.add_layer(
            f"{layer_id}_hidden",
            line_color=Color(r / 255, g / 255, b / 255, 0.4),
            line_weight=LINE_WEIGHT_HIDDEN,
            line_type=LineType.ISO_DASH,
        )

    if section_plane is not None:
        _add_section_view(svg, solids, section_plane)
    else:
        _add_projection_view(svg, solids, view_origin, view_up)

    # Export to temp file and read back
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=True, mode="w") as tmp:
        svg.write(tmp.name)
        return Path(tmp.name).read_text()


def _add_projection_view(svg, solids, view_origin, view_up):
    """Add a projection view (visible + hidden lines) to the SVG."""
    vp_right, vp_up = _viewport_axes(view_origin, view_up)

    for layer_id, solid, _rgb in solids:
        try:
            visible, hidden = solid.project_to_viewport(
                viewport_origin=view_origin,
                viewport_up=view_up,
            )
        except Exception:
            continue

        # Correct for COM-centered projection output
        com = solid.center()
        dx, dy, dz = float(com.X), float(com.Y), float(com.Z)
        if abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dz) > 1e-9:
            offset_2d = _project_offset((dx, dy, dz), vp_right, vp_up)
            shift = Location((*offset_2d, 0))
            visible = ShapeList([e.moved(shift) for e in visible])
            hidden = ShapeList([e.moved(shift) for e in hidden])

        for edge in visible:
            if edge.length > 1e-6:
                svg.add_shape(edge, layer=layer_id)
        for edge in hidden:
            if edge.length > 1e-6:
                svg.add_shape(edge, layer=f"{layer_id}_hidden")


def _add_section_view(svg, solids, section_plane):
    """Add a section view (cross-section fills with outlines) to the SVG."""
    to_xy = section_plane.location.inverse()

    for layer_id, solid, _rgb in solids:
        try:
            cross = section(solid, section_by=section_plane)
        except Exception:
            continue
        if cross is None or not cross.faces():
            continue

        cross = cross.moved(to_xy)
        for face in cross.faces():
            svg.add_shape(face, layer=f"{layer_id}_fill")
            for edge in face.edges():
                svg.add_shape(edge, layer=layer_id)
