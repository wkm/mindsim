"""Single-view SVG renderer for the component browser.

Produces a clean SVG with per-layer <g> groups from build123d solids,
suitable for high-quality 2D export from the web viewer. Supports
both projection views (visible + hidden lines) and section views
(cross-section fills with outlines), with optional title block
annotation.
"""

from __future__ import annotations

import re
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

# Annotation styling
ANNOT_FONT = "system-ui, -apple-system, sans-serif"
ANNOT_COLOR = "#394B59"  # BP.DARK_GRAY5
ANNOT_LIGHT = "#8A9BA8"  # BP.GRAY3


def render_component_svg(
    solids: list[tuple[str, object, tuple[int, int, int]]],
    view_origin: tuple[float, float, float],
    view_up: tuple[float, float, float] = (0, 0, 1),
    section_plane=None,
    annotate: dict | None = None,
) -> str:
    """Render a single 2D view of component solids as an SVG string.

    Args:
        solids: List of (layer_id, Solid, rgb) tuples.
        view_origin: Camera position (far from origin, looking at 0,0,0).
        view_up: Camera up vector.
        section_plane: Optional build123d Plane for cross-section view.
        annotate: Optional dict with annotation metadata:
            component, view, layers, dimensions_mm, mass_g, section.

    Returns:
        SVG string with per-layer <g> groups and optional annotations.
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
        content = Path(tmp.name).read_text()

    if annotate:
        content = _inject_annotations(content, annotate)

    return content


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


def _inject_annotations(svg_content: str, meta: dict) -> str:
    """Inject a title block and scale bar into the SVG.

    Expands the viewBox downward to make room for annotations below
    the geometry, then adds text elements.
    """
    vb_match = re.search(r'viewBox="([^ ]+) ([^ ]+) ([^ ]+) ([^ "]+)"', svg_content)
    if not vb_match:
        return svg_content

    vb_x = float(vb_match.group(1))
    vb_y = float(vb_match.group(2))
    vb_w = float(vb_match.group(3))
    vb_h = float(vb_match.group(4))

    # Add margin for annotations (in geometry units = meters, scale=1000)
    margin = 0.008  # 8mm in geometry coords
    footer = 0.012  # 12mm for title block

    new_vb_y = vb_y - margin
    new_vb_h = vb_h + margin * 2 + footer
    new_vb_w = vb_w + margin * 2
    new_vb_x = vb_x - margin

    # Update viewBox and SVG dimensions
    new_w_mm = new_vb_w * SVG_SCALE
    new_h_mm = new_vb_h * SVG_SCALE
    svg_content = svg_content.replace(
        vb_match.group(0),
        f'viewBox="{new_vb_x} {new_vb_y} {new_vb_w} {new_vb_h}"',
    )
    # Update width/height attributes
    svg_content = re.sub(
        r'width="[^"]*"', f'width="{new_w_mm:.1f}mm"', svg_content, count=1
    )
    svg_content = re.sub(
        r'height="[^"]*"', f'height="{new_h_mm:.1f}mm"', svg_content, count=1
    )

    # Build annotation elements (positioned in viewBox coords, outside
    # the scale(1,-1) transform group — so Y increases downward as normal)
    annot_y = vb_y + vb_h + margin + 0.002  # just below geometry + margin
    annot_x = vb_x
    font_size_title = 0.003  # ~3mm in geometry coords
    font_size_detail = 0.002  # ~2mm
    line_spacing = 0.003

    lines = []

    # Title: component name
    comp_name = meta.get("component", "")
    view_name = meta.get("view", "")
    lines.append(
        f'<text x="{annot_x}" y="{annot_y}" '
        f'font-family="{ANNOT_FONT}" font-size="{font_size_title}" '
        f'font-weight="600" fill="{ANNOT_COLOR}">'
        f"{comp_name} — {view_name} view</text>"
    )

    # Details line
    details = []
    dims = meta.get("dimensions_mm")
    if dims:
        details.append(f"{' x '.join(f'{d:.1f}' for d in dims)} mm")
    mass = meta.get("mass_g")
    if mass:
        details.append(f"{mass:.1f} g")
    layer_names = meta.get("layers", [])
    if layer_names:
        details.append("Layers: " + ", ".join(layer_names))

    if details:
        lines.append(
            f'<text x="{annot_x}" y="{annot_y + line_spacing}" '
            f'font-family="{ANNOT_FONT}" font-size="{font_size_detail}" '
            f'fill="{ANNOT_LIGHT}">'
            f"{' | '.join(details)}</text>"
        )

    # Section info
    section_text = meta.get("section")
    if section_text:
        lines.append(
            f'<text x="{annot_x}" y="{annot_y + line_spacing * 2}" '
            f'font-family="{ANNOT_FONT}" font-size="{font_size_detail}" '
            f'fill="{ANNOT_LIGHT}">'
            f"{section_text}</text>"
        )

    # Scale bar (10mm)
    bar_len = 0.01  # 10mm in geometry coords
    bar_x = annot_x + new_vb_w - margin - bar_len
    bar_y = annot_y
    lines.append(
        f'<line x1="{bar_x}" y1="{bar_y}" x2="{bar_x + bar_len}" y2="{bar_y}" '
        f'stroke="{ANNOT_COLOR}" stroke-width="0.0003"/>'
    )
    lines.append(
        f'<line x1="{bar_x}" y1="{bar_y - 0.001}" x2="{bar_x}" y2="{bar_y + 0.001}" '
        f'stroke="{ANNOT_COLOR}" stroke-width="0.0002"/>'
    )
    lines.append(
        f'<line x1="{bar_x + bar_len}" y1="{bar_y - 0.001}" '
        f'x2="{bar_x + bar_len}" y2="{bar_y + 0.001}" '
        f'stroke="{ANNOT_COLOR}" stroke-width="0.0002"/>'
    )
    lines.append(
        f'<text x="{bar_x + bar_len / 2}" y="{bar_y + line_spacing}" '
        f'font-family="{ANNOT_FONT}" font-size="{font_size_detail}" '
        f'fill="{ANNOT_LIGHT}" text-anchor="middle">10 mm</text>'
    )

    # Inject annotations before closing </svg>
    annotation_block = "\n  <!-- Annotations -->\n  " + "\n  ".join(lines) + "\n"
    svg_content = svg_content.replace("</svg>", annotation_block + "</svg>")

    return svg_content
