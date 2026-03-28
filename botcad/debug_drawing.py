"""Technical 2D drawings for parametric CAD parts.

Generate cross-section SVGs at specified planes, with multiple parts
overlaid in distinct colors. Sections include hatching, section labels,
and a title block with scale indicator and legend.

Usage:
    from botcad.debug_drawing import DebugDrawingBuilder

    drawing = DebugDrawingBuilder("coupler_debug")
    drawing.add_part("coupler", coupler_solid, color=(200, 80, 50))
    drawing.add_part("servo", servo_solid, color=(60, 60, 60))
    drawing.add_section("front_horn", Plane.XY.offset(front_z))
    drawing.add_section("side_profile", Plane.XZ)
    drawing.save("coupler_debug.svg")
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from pathlib import Path

from build123d import (
    Color,
    Compound,
    Edge,
    ExportSVG,
    Location,
    Plane,
    ShapeList,
    Solid,
    Unit,
    section,
)
from build123d.exporters import LineType

from botcad.colors import DEBUG_PALETTE
from botcad.colors import DIM_COLOR as DIM_COLOR  # re-export
from botcad.component import Vec3

RGB = tuple[int, int, int]

# Default color palette — visually distinct, colorblind-friendly (Blueprint.js)
PALETTE: list[RGB] = DEBUG_PALETTE

# Geometry is in meters; scale=1000 means 1m → 1000 SVG units (mm).
DEFAULT_SCALE = 1000.0

# Line weights (in mm at SVG output scale)
LINE_WEIGHT_THICK = 0.35  # visible outlines
LINE_WEIGHT_THIN = 0.15  # hidden lines
LINE_WEIGHT_FILL = 0.0  # fill regions (no stroke)

# Hatch pattern: 45° parallel lines
HATCH_SPACING_MM = 1.2
HATCH_STROKE_MM = 0.1

# Font stack
FONT_FAMILY = "'Input Sans Narrow', 'DejaVu Sans', Arial, sans-serif"

# Annotation spacing in mm (converted to viewBox meters internally)
HEADER_MM = 12  # space below geometry for title/scale/legend
FOOTER_MM = 9  # space above geometry for section labels
SECTION_GAP_MM = 8  # horizontal gap between sections
MARGIN_MM = 3  # margin around content

# Dimension line styling (mm)
DIM_OFFSET_MM = 3.0  # offset from geometry edge to dimension line
DIM_ARROW_MM = 1.5  # arrowhead length
DIM_TICK_MM = 0.8  # extension line overshoot past dimension line
DIM_LINE_WEIGHT_MM = 0.2  # dimension line stroke width

# Section cutting line styling (mm)
_CUT_OVERHANG_MM = 3.0  # extend past geometry edges
_CUT_LINE_W_MM = 0.25
_CUT_DASH_MM = (2.5, 0.8, 0.5, 0.8)  # long-short chain dash
_CUT_FONT_MM = 2.0
_CUT_LABEL_GAP_MM = 1.5  # gap between line end and letter
_CUT_LABEL_VOFF_MM = 0.6  # vertical centering offset for letter


def _mm(mm: float) -> float:
    """Convert mm to meters (viewBox units)."""
    return mm / 1000.0


# Pre-computed cutting line constants (in viewBox meters)
_CUT_OVERHANG = _mm(_CUT_OVERHANG_MM)
_CUT_LINE_W = _mm(_CUT_LINE_W_MM)
_CUT_DASH = ",".join(f"{_mm(v):.6f}" for v in _CUT_DASH_MM)
_CUT_FONT = _mm(_CUT_FONT_MM)
_CUT_LABEL_GAP = _mm(_CUT_LABEL_GAP_MM)
_CUT_LABEL_VOFF = _mm(_CUT_LABEL_VOFF_MM)


def _dim_line_h(
    x1: float,
    x2: float,
    y: float,
    label: str,
    geo_y: float | None = None,
) -> list[str]:
    """SVG elements for a horizontal dimension line with arrows and label.

    All coordinates in viewBox units (meters). Renders:
    - Extension lines from geo_y to dim line (if geo_y provided)
    - Horizontal line with arrowheads
    - Centered label text (above the line)

    Args:
        geo_y: Y coordinate of geometry edge to draw extension lines from.
    """
    lw = _mm(DIM_LINE_WEIGHT_MM)
    ext_lw = _mm(DIM_LINE_WEIGHT_MM * 0.7)
    arrow = _mm(DIM_ARROW_MM)
    tick = _mm(DIM_TICK_MM)
    gap = _mm(0.8)  # gap between geometry and extension line
    fs = _mm(1.6)

    els: list[str] = []

    # Extension lines from geometry to dimension line
    if geo_y is not None:
        ext_start = geo_y + gap if y > geo_y else geo_y - gap
        ext_end = y + tick
        els.extend(
            f'  <line x1="{x:.6f}" y1="{ext_start:.6f}" '
            f'x2="{x:.6f}" y2="{ext_end:.6f}" '
            f'stroke="{DIM_COLOR}" stroke-width="{ext_lw:.6f}" '
            f'stroke-opacity="0.5"/>'
            for x in [x1, x2]
        )

    # Ticks at dimension line
    els.extend(
        f'  <line x1="{x:.6f}" y1="{y - tick:.6f}" '
        f'x2="{x:.6f}" y2="{y + tick:.6f}" '
        f'stroke="{DIM_COLOR}" stroke-width="{lw:.6f}"/>'
        for x in [x1, x2]
    )
    # Main dimension line
    els.append(
        f'  <line x1="{x1:.6f}" y1="{y:.6f}" '
        f'x2="{x2:.6f}" y2="{y:.6f}" '
        f'stroke="{DIM_COLOR}" stroke-width="{lw:.6f}"/>'
    )
    # Arrowheads (simple triangles)
    ah = arrow * 0.4  # half-height
    els.append(
        f'  <polygon points="{x1:.6f},{y:.6f} '
        f"{x1 + arrow:.6f},{y - ah:.6f} "
        f'{x1 + arrow:.6f},{y + ah:.6f}" '
        f'fill="{DIM_COLOR}"/>'
    )
    els.append(
        f'  <polygon points="{x2:.6f},{y:.6f} '
        f"{x2 - arrow:.6f},{y - ah:.6f} "
        f'{x2 - arrow:.6f},{y + ah:.6f}" '
        f'fill="{DIM_COLOR}"/>'
    )
    # Label (above the line)
    cx = (x1 + x2) / 2
    text_y = y - _mm(1.2)
    els.append(
        f'  <text x="{cx:.6f}" y="{text_y:.6f}" '
        f'text-anchor="middle" '
        f'font-family="{FONT_FAMILY}" '
        f'font-size="{fs:.6f}" '
        f'fill="{DIM_COLOR}">{label}</text>'
    )
    return els


def _dim_line_v(
    y1: float,
    y2: float,
    x: float,
    label: str,
    geo_x: float | None = None,
) -> list[str]:
    """SVG elements for a vertical dimension line with arrows and label.

    y1 < y2 (y1 is top, y2 is bottom in viewBox Y-down coordinates).

    Args:
        geo_x: X coordinate of geometry edge to draw extension lines from.
    """
    lw = _mm(DIM_LINE_WEIGHT_MM)
    ext_lw = _mm(DIM_LINE_WEIGHT_MM * 0.7)
    arrow = _mm(DIM_ARROW_MM)
    tick = _mm(DIM_TICK_MM)
    gap = _mm(0.8)
    fs = _mm(1.6)

    els: list[str] = []

    # Extension lines from geometry to dimension line
    if geo_x is not None:
        ext_start = geo_x + gap
        ext_end = x + tick
        els.extend(
            f'  <line x1="{ext_start:.6f}" y1="{y:.6f}" '
            f'x2="{ext_end:.6f}" y2="{y:.6f}" '
            f'stroke="{DIM_COLOR}" stroke-width="{ext_lw:.6f}" '
            f'stroke-opacity="0.5"/>'
            for y in [y1, y2]
        )

    # Ticks at dimension line
    els.extend(
        f'  <line x1="{x - tick:.6f}" y1="{y:.6f}" '
        f'x2="{x + tick:.6f}" y2="{y:.6f}" '
        f'stroke="{DIM_COLOR}" stroke-width="{lw:.6f}"/>'
        for y in [y1, y2]
    )
    # Main dimension line
    els.append(
        f'  <line x1="{x:.6f}" y1="{y1:.6f}" '
        f'x2="{x:.6f}" y2="{y2:.6f}" '
        f'stroke="{DIM_COLOR}" stroke-width="{lw:.6f}"/>'
    )
    # Arrowheads
    ah = arrow * 0.4
    els.append(
        f'  <polygon points="{x:.6f},{y1:.6f} '
        f"{x - ah:.6f},{y1 + arrow:.6f} "
        f'{x + ah:.6f},{y1 + arrow:.6f}" '
        f'fill="{DIM_COLOR}"/>'
    )
    els.append(
        f'  <polygon points="{x:.6f},{y2:.6f} '
        f"{x - ah:.6f},{y2 - arrow:.6f} "
        f'{x + ah:.6f},{y2 - arrow:.6f}" '
        f'fill="{DIM_COLOR}"/>'
    )
    # Label (rotated 90° to the right of the line)
    cy = (y1 + y2) / 2
    text_x = x + _mm(1.5)
    els.append(
        f'  <text x="{text_x:.6f}" y="{cy:.6f}" '
        f'text-anchor="middle" dominant-baseline="central" '
        f'font-family="{FONT_FAMILY}" '
        f'font-size="{fs:.6f}" '
        f'fill="{DIM_COLOR}" '
        f'transform="rotate(-90 {text_x:.6f} {cy:.6f})">{label}</text>'
    )
    return els


def _fmt_mm(meters: float) -> str:
    """Format a meter value as mm with appropriate precision."""
    mm = abs(meters) * 1000
    if mm >= 10:
        return f"{mm:.1f}"
    return f"{mm:.2f}"


def _viewport_axes(origin: Vec3, up: Vec3) -> tuple[Vec3, Vec3]:
    """Compute orthonormal (right, up) axes for a viewport.

    The viewport looks from *origin* toward the world origin with the given
    *up* hint.  Returns unit vectors for the viewport's horizontal (right)
    and vertical (up) directions in world coordinates.
    """
    ox, oy, oz = origin
    mag = math.sqrt(ox * ox + oy * oy + oz * oz)
    vx, vy, vz = -ox / mag, -oy / mag, -oz / mag

    # Right = view_dir × up
    ux, uy, uz = up
    rx = vy * uz - vz * uy
    ry = vz * ux - vx * uz
    rz = vx * uy - vy * ux
    rmag = math.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rmag, ry / rmag, rz / rmag

    # Re-orthogonalize up = right × view_dir
    ux = ry * vz - rz * vy
    uy = rz * vx - rx * vz
    uz = rx * vy - ry * vx

    return (rx, ry, rz), (ux, uy, uz)


def _project_offset(offset_3d: Vec3, right: Vec3, up: Vec3) -> tuple[float, float]:
    """Project a 3D offset onto pre-computed viewport axes."""
    dx, dy, dz = offset_3d
    rx, ry, rz = right
    ux, uy, uz = up
    return (dx * rx + dy * ry + dz * rz, dx * ux + dy * uy + dz * uz)


@dataclass(frozen=True)
class _Part:
    name: str
    solid: Solid | Compound
    color: RGB


@dataclass(frozen=True)
class _Section:
    name: str
    plane: Plane


@dataclass(frozen=True)
class _Projection:
    """A 2D projection view with visible/hidden line separation."""

    name: str
    origin: Vec3  # camera position
    up: Vec3  # up direction


# A view is either a cross-section or a projection
_View = _Section | _Projection


class DebugDrawingBuilder:
    """Accumulates parts and views (sections + projections), exports SVG.

    All views are laid out side-by-side in one SVG file with:
    - Cross-hatched section fills (ISO 45° pattern per part)
    - Projection views with visible (solid) and hidden (dashed) lines
    - Section/view labels (A-A, B-B, ...)
    - Title text with scale indicator and part legend
    """

    def __init__(
        self,
        title: str,
        scale: float = DEFAULT_SCALE,
        line_weight: float = LINE_WEIGHT_THICK,
    ) -> None:
        self.title = title
        self.parts: list[_Part] = []
        self.sections: list[_Section] = []
        self.projections: list[tuple[int, _Projection]] = []
        self.scale = scale
        self.line_weight = line_weight
        self._view_order: list[_View] = []

    def add_part(
        self,
        name: str,
        solid: Solid | Compound,
        color: RGB | None = None,
    ) -> DebugDrawingBuilder:
        """Add a solid to be sectioned. Color auto-assigned if omitted."""
        if color is None:
            color = PALETTE[len(self.parts) % len(PALETTE)]
        self.parts.append(_Part(name=name, solid=solid, color=color))
        return self

    def add_section(self, name: str, plane: Plane) -> DebugDrawingBuilder:
        """Add a named section plane."""
        sec = _Section(name=name, plane=plane)
        self.sections.append(sec)
        self._view_order.append(sec)
        return self

    def add_projection(
        self,
        name: str,
        origin: Vec3,
        up: Vec3 = (0, 0, 1),
    ) -> DebugDrawingBuilder:
        """Add a 2D projection view with visible/hidden line separation.

        Args:
            name: View label (e.g., "front", "side").
            origin: Camera position — projects toward the origin of the parts.
            up: Camera up direction.
        """
        proj = _Projection(name=name, origin=origin, up=up)
        self._view_order.append(proj)
        return self

    def save(self, path: str | Path) -> Path:
        """Export a single SVG with all sections side by side."""
        path = Path(path)
        self._export_combined(path)
        return path

    def _export_combined(self, out: Path) -> None:
        """Process all views (sections + projections) and compose into SVG."""
        # If no explicit view order, fall back to sections only
        views = self._view_order if self._view_order else list(self.sections)
        if not views:
            return

        # Process each view into column data:
        #   (name, is_section, parts_data, view)
        # For sections: parts_data = [(part, cross_compound)]
        # For projections: parts_data = [(part, visible_edges, hidden_edges)]
        ColumnData = tuple[str, bool, list, _View]
        columns: list[ColumnData] = []

        for view in views:
            if isinstance(view, _Section):
                parts_cross = self._process_section(view)
                if parts_cross:
                    columns.append((view.name, True, parts_cross, view))
            elif isinstance(view, _Projection):
                parts_proj = self._process_projection(view)
                if parts_proj:
                    columns.append((view.name, False, parts_proj, view))

        if not columns:
            return

        gap_m = _mm(SECTION_GAP_MM)

        # Compute bounding box for each column (in meters)
        col_bounds: list[tuple[float, float, float, float]] = []
        for _name, is_section, parts_data, _view in columns:
            xmin = ymin = float("inf")
            xmax = ymax = float("-inf")
            if is_section:
                for _part, cross in parts_data:
                    bb = cross.bounding_box()
                    xmin = min(xmin, bb.min.X)
                    ymin = min(ymin, bb.min.Y)
                    xmax = max(xmax, bb.max.X)
                    ymax = max(ymax, bb.max.Y)
            else:
                for _part, vis, hid in parts_data:
                    all_edges = list(vis) + list(hid)
                    if all_edges:
                        c = Compound(children=all_edges)
                        bb = c.bounding_box()
                        xmin = min(xmin, bb.min.X)
                        ymin = min(ymin, bb.min.Y)
                        xmax = max(xmax, bb.max.X)
                        ymax = max(ymax, bb.max.Y)
            if xmin < float("inf"):
                col_bounds.append((xmin, ymin, xmax, ymax))
            else:
                col_bounds.append((0, 0, 0, 0))

        # Compute X offsets to tile columns side by side
        x_offsets: list[float] = []
        cursor_x = 0.0
        for xmin, _ymin, xmax, _ymax in col_bounds:
            col_w = xmax - xmin
            x_offsets.append(cursor_x - xmin)
            cursor_x += col_w + gap_m

        # Build SVG
        svg = ExportSVG(
            unit=Unit.MM,
            scale=self.scale,
            margin=0,
            line_weight=self.line_weight,
        )

        for part in self.parts:
            r, g, b = part.color
            c = Color(r / 255, g / 255, b / 255)
            c_fill = Color(r / 255, g / 255, b / 255, 0.15)
            c_none = Color(r / 255, g / 255, b / 255, 0.0)

            svg.add_layer(part.name, line_color=c, line_weight=self.line_weight)
            svg.add_layer(
                f"{part.name}_fill",
                fill_color=c_fill,
                line_color=c_none,
                line_weight=LINE_WEIGHT_FILL,
            )
            svg.add_layer(
                f"{part.name}_hidden",
                line_color=Color(r / 255, g / 255, b / 255, 0.4),
                line_weight=LINE_WEIGHT_THIN,
                line_type=LineType.ISO_DASH,
            )

        # Add shapes per column
        for (_col_name, is_section, parts_data, _view), x_off in zip(
            columns, x_offsets, strict=True
        ):
            offset_loc = Location((x_off, 0, 0))
            if is_section:
                for part, cross in parts_data:
                    moved = cross.moved(offset_loc)
                    for face in moved.faces():
                        svg.add_shape(face, layer=f"{part.name}_fill")
                        for edge in face.edges():
                            svg.add_shape(edge, layer=part.name)
            else:
                for part, vis, hid in parts_data:
                    for edge in vis:
                        if edge.length > 1e-6:
                            svg.add_shape(edge.moved(offset_loc), layer=part.name)
                    for edge in hid:
                        if edge.length > 1e-6:
                            svg.add_shape(
                                edge.moved(offset_loc),
                                layer=f"{part.name}_hidden",
                            )

        svg.write(str(out))

        # Post-process: hatching, annotations, expanded viewBox
        self._postprocess_svg(out, columns, col_bounds, x_offsets)
        print(f"  drawing: {out}")

    def _process_section(self, sec: _Section) -> list[tuple[_Part, Compound]]:
        """Section all parts at the given plane."""
        to_xy = sec.plane.location.inverse()
        parts_cross = []
        for part in self.parts:
            try:
                cross = section(part.solid, section_by=sec.plane)
            except Exception:
                continue
            if cross is None or not cross.faces():
                continue
            cross = cross.moved(to_xy)
            parts_cross.append((part, cross))
        return parts_cross

    def _process_projection(
        self, proj: _Projection
    ) -> list[tuple[_Part, ShapeList[Edge], ShapeList[Edge]]]:
        """Project all parts from the given viewpoint.

        build123d's project_to_viewport re-centers each shape's projection
        at its center of mass. We compensate by shifting the 2D edges back
        to the shape's actual world position.
        """
        # Pre-compute viewport axes once for all parts in this projection.
        vp_right, vp_up = _viewport_axes(proj.origin, proj.up)

        parts_proj = []
        for part in self.parts:
            try:
                visible, hidden = part.solid.project_to_viewport(
                    viewport_origin=proj.origin,
                    viewport_up=proj.up,
                )
            except Exception:
                continue

            # project_to_viewport centers output at the shape's COM.
            # Shift edges so each shape appears at its true world position.
            com = part.solid.center()
            dx, dy, dz = float(com.X), float(com.Y), float(com.Z)
            if abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dz) > 1e-9:
                offset_2d = _project_offset((dx, dy, dz), vp_right, vp_up)
                shift = Location((*offset_2d, 0))
                visible = ShapeList([e.moved(shift) for e in visible])
                hidden = ShapeList([e.moved(shift) for e in hidden])

            if visible or hidden:
                parts_proj.append((part, visible, hidden))
        return parts_proj

    def _postprocess_svg(
        self,
        svg_path: Path,
        columns: list[tuple[str, bool, list, _View]],
        col_bounds: list[tuple[float, float, float, float]],
        x_offsets: list[float],
    ) -> None:
        """Inject hatching, section labels, title block into the SVG."""
        content = svg_path.read_text()

        vb_match = re.search(r'viewBox="([^ ]+) ([^ ]+) ([^ ]+) ([^ "]+)"', content)
        if not vb_match:
            return

        # viewBox is in meters (geometry coordinates)
        vb_x = float(vb_match.group(1))
        vb_y = float(vb_match.group(2))
        vb_w = float(vb_match.group(3))
        vb_h = float(vb_match.group(4))

        # Geometry bounds in viewBox coords
        geo_left = vb_x
        geo_top = vb_y  # most negative Y (top in flipped space)
        geo_bottom = vb_y + vb_h

        # Expand viewBox for annotations (in meters)
        margin = _mm(MARGIN_MM)
        header = _mm(HEADER_MM)
        footer = _mm(FOOTER_MM)
        dim_space = _mm(DIM_OFFSET_MM + 5)  # space for dimension lines + label

        # In build123d's SVG, the geometry group has scale(1,-1), so positive
        # Y in geometry = negative Y in viewBox. The viewBox Y range covers
        # the flipped geometry. We add space:
        #   - above geometry (more negative Y) for footer/labels
        #   - below geometry (more positive Y) for header/title + width dims
        #   - right of geometry for height dims
        new_x = geo_left - margin
        new_y = geo_top - footer
        new_w = vb_w + 2 * margin + dim_space
        new_h = vb_h + header + footer + dim_space

        # Update viewBox and SVG dimensions
        content = re.sub(
            r'width="[^"]*mm"',
            f'width="{new_w * self.scale:.2f}mm"',
            content,
        )
        content = re.sub(
            r'height="[^"]*mm"',
            f'height="{new_h * self.scale:.2f}mm"',
            content,
        )
        content = re.sub(
            r'viewBox="[^"]*"',
            f'viewBox="{new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}"',
            content,
        )

        # Apply hatch patterns to fill groups
        defs = self._build_hatch_defs()
        content = self._apply_hatch_fills(content)

        # Inject <defs> before first <g>
        content = content.replace(
            '  <g transform="scale(1,-1)"',
            f'{defs}  <g transform="scale(1,-1)"',
        )

        # Build annotations outside the flipped group
        annotations = self._build_annotations(
            columns,
            col_bounds,
            x_offsets,
            geo_top,
            geo_bottom,
            new_x,
            new_y,
        )
        content = content.replace("</svg>", f"{annotations}</svg>")

        svg_path.write_text(content)

    def _build_hatch_defs(self) -> str:
        """Build SVG <defs> with hatch patterns for each part."""
        # Pattern dimensions are in viewBox units (meters)
        spacing = _mm(HATCH_SPACING_MM)
        stroke = _mm(HATCH_STROKE_MM)
        patterns = []
        for i, part in enumerate(self.parts):
            r, g, b = part.color
            pat_id = f"hatch_{part.name}"
            angle = 45 + i * 30
            patterns.append(
                f'    <pattern id="{pat_id}" width="{spacing}" '
                f'height="{spacing}" '
                f'patternUnits="userSpaceOnUse" '
                f'patternTransform="rotate({angle})">\n'
                f'      <line x1="0" y1="0" x2="0" y2="{spacing}" '
                f'stroke="rgb({r},{g},{b})" stroke-width="{stroke}" '
                f'stroke-opacity="0.6"/>\n'
                f"    </pattern>"
            )
        return f"  <defs>\n{''.join(p + chr(10) for p in patterns)}  </defs>\n"

    def _apply_hatch_fills(self, content: str) -> str:
        """Replace flat color fills in _fill groups with hatch pattern fills."""
        for part in self.parts:
            r, g, b = part.color
            old_fill = (
                f'fill="rgb({r},{g},{b})" fill-opacity="0.15" '
                f'stroke="rgb({r},{g},{b})" stroke-opacity="0.0" '
                f'stroke-width="0.0" id="{part.name}_fill"'
            )
            new_fill = (
                f'fill="url(#hatch_{part.name})" '
                f'stroke="none" '
                f'stroke-width="0" id="{part.name}_fill"'
            )
            content = content.replace(old_fill, new_fill)
        return content

    def _build_annotations(
        self,
        columns: list[tuple[str, bool, list, _View]],
        col_bounds: list[tuple[float, float, float, float]],
        x_offsets: list[float],
        geo_top: float,
        geo_bottom: float,
        new_x: float,
        new_y: float,
    ) -> str:
        """Build SVG annotations in viewBox coordinates (meters).

        Annotations go outside the scale(1,-1) group, so Y increases downward.
        The viewBox Y range: new_y (top) to new_y + new_h (bottom).
        Geometry occupies [geo_top, geo_bottom] in viewBox Y.
        """
        elements = []
        label_letters = list(string.ascii_uppercase)
        section_idx = 0  # separate counter for section A-A labels

        fs_title = _mm(3.5)
        fs_label = _mm(2.5)
        fs_small = _mm(1.8)

        sublabel_y = geo_top - _mm(2)
        label_y = geo_top - _mm(5.5)

        dim_offset = _mm(DIM_OFFSET_MM)
        prev_dims: tuple[str, str] | None = None

        # Collect section info for cutting lines on projections
        section_info: list[tuple[str, float]] = []  # (letter, Z value)

        for _i, (
            (col_name, is_section, _data, view),
            (xmin, ymin, xmax, ymax),
            x_off,
        ) in enumerate(zip(columns, col_bounds, x_offsets, strict=True)):
            cx = (xmin + xmax) / 2 + x_off

            if is_section:
                letter = label_letters[section_idx % len(label_letters)]
                label = f"{letter}-{letter}"
                # Extract the section plane Z offset for cutting lines
                assert isinstance(view, _Section)
                sec_z = float(view.plane.origin.Z)
                section_info.append((letter, sec_z))
                section_idx += 1
            else:
                label = col_name  # projections use their name directly

            elements.append(
                f'  <text x="{cx:.6f}" y="{label_y:.6f}" '
                f'text-anchor="middle" '
                f'font-family="{FONT_FAMILY}" '
                f'font-size="{fs_label:.6f}" font-weight="bold" '
                f'fill="#333">{label}</text>'
            )
            # Sublabel: section name for sections (A-A already has the letter);
            # skip for projections since label already IS the name.
            if is_section:
                elements.append(
                    f'  <text x="{cx:.6f}" y="{sublabel_y:.6f}" '
                    f'text-anchor="middle" '
                    f'font-family="{FONT_FAMILY}" '
                    f'font-size="{fs_small:.6f}" '
                    f'fill="#888">{col_name}</text>'
                )

            # --- Dimension lines ---
            # Geometry Y is up; viewBox Y is down. Mapping: viewBox_y = -geo_y
            vb_top = -ymax  # top of geometry in viewBox
            vb_bottom = -ymin  # bottom of geometry in viewBox
            vb_left = xmin + x_off
            vb_right = xmax + x_off
            col_w = xmax - xmin
            col_h = ymax - ymin

            # Deduplicate: round to 0.1mm and skip if same as previous
            w_label = _fmt_mm(col_w)
            h_label = _fmt_mm(col_h)
            dims_key = (w_label, h_label)
            show_dims = dims_key != prev_dims
            prev_dims = dims_key

            # Width dimension (below geometry)
            if col_w > 1e-6 and show_dims:
                wd_y = vb_bottom + dim_offset
                elements.extend(
                    _dim_line_h(vb_left, vb_right, wd_y, w_label, geo_y=vb_bottom)
                )

            # Height dimension (right of geometry)
            if col_h > 1e-6 and show_dims:
                hd_x = vb_right + dim_offset
                elements.extend(
                    _dim_line_v(vb_top, vb_bottom, hd_x, h_label, geo_x=vb_right)
                )

            # --- Section cutting lines on projection views ---
            if not is_section and section_info:
                for letter, sec_z in section_info:
                    # Section at world Z = sec_z → viewBox Y = -sec_z
                    cut_y = -sec_z
                    # Only draw if within the column's vertical extent
                    if (
                        cut_y < vb_top - _CUT_OVERHANG
                        or cut_y > vb_bottom + _CUT_OVERHANG
                    ):
                        continue

                    line_left = vb_left - _CUT_OVERHANG
                    line_right = vb_right + _CUT_OVERHANG

                    # Chain-dash cutting line
                    elements.append(
                        f'  <line x1="{line_left:.6f}" y1="{cut_y:.6f}" '
                        f'x2="{line_right:.6f}" y2="{cut_y:.6f}" '
                        f'stroke="#555" stroke-width="{_CUT_LINE_W:.6f}" '
                        f'stroke-dasharray="{_CUT_DASH}"/>'
                    )
                    # Letter labels at both ends
                    for lx, anchor in [
                        (line_left - _CUT_LABEL_GAP, "end"),
                        (line_right + _CUT_LABEL_GAP, "start"),
                    ]:
                        elements.append(
                            f'  <text x="{lx:.6f}" y="{cut_y + _CUT_LABEL_VOFF:.6f}" '
                            f'text-anchor="{anchor}" '
                            f'font-family="{FONT_FAMILY}" '
                            f'font-size="{_CUT_FONT:.6f}" font-weight="bold" '
                            f'fill="#555">{letter}</text>'
                        )

        # --- Title block (below geometry in viewBox = positive Y direction) ---
        title_x = new_x + _mm(2)
        title_y = geo_bottom + dim_offset + _mm(5)

        elements.append(
            f'  <text x="{title_x:.6f}" y="{title_y:.6f}" '
            f'font-family="{FONT_FAMILY}" '
            f'font-size="{fs_title:.6f}" font-weight="bold" '
            f'fill="#222">{self.title}</text>'
        )

        # Scale bar (10mm) — on the second line
        bar_y = title_y + _mm(4)
        bar_len = _mm(10)
        tick = _mm(1.0)
        elements.append(
            f'  <line x1="{title_x:.6f}" y1="{bar_y:.6f}" '
            f'x2="{title_x + bar_len:.6f}" y2="{bar_y:.6f}" '
            f'stroke="#333" stroke-width="{_mm(0.35):.6f}"/>'
        )
        elements.extend(
            f'  <line x1="{bx:.6f}" y1="{bar_y - tick:.6f}" '
            f'x2="{bx:.6f}" y2="{bar_y + tick:.6f}" '
            f'stroke="#333" stroke-width="{_mm(0.25):.6f}"/>'
            for bx in [title_x, title_x + bar_len]
        )
        elements.append(
            f'  <text x="{title_x + bar_len + _mm(1.5):.6f}" y="{bar_y + _mm(0.7):.6f}" '
            f'font-family="{FONT_FAMILY}" '
            f'font-size="{fs_small:.6f}" '
            f'fill="#888">10 mm</text>'
        )

        # Part legend — inline to the right
        legend_x = title_x + _mm(30)
        legend_y = title_y - _mm(1)
        swatch_w = _mm(4)
        swatch_h = _mm(2.5)
        for j, part in enumerate(self.parts):
            r, g, b = part.color
            ly = legend_y + j * _mm(3.5)
            elements.append(
                f'  <rect x="{legend_x:.6f}" y="{ly - _mm(1.5):.6f}" '
                f'width="{swatch_w:.6f}" height="{swatch_h:.6f}" '
                f'fill="url(#hatch_{part.name})" '
                f'stroke="rgb({r},{g},{b})" stroke-width="{_mm(0.3):.6f}"/>'
            )
            elements.append(
                f'  <text x="{legend_x + _mm(5.5):.6f}" y="{ly + _mm(0.5):.6f}" '
                f'font-family="{FONT_FAMILY}" '
                f'font-size="{fs_small:.6f}" '
                f'fill="rgb({r},{g},{b})">{part.name}</text>'
            )

        return "\n".join(elements) + "\n"

    def save_and_open(self, path: str | Path) -> Path:
        """Save SVG and open it."""
        import subprocess

        out = self.save(path)
        subprocess.Popen(["open", str(out)])
        return out


# ── Convenience: standard section sets ──


def xy_sections(
    z_values: dict[str, float],
) -> list[_Section]:
    """Create a list of XY section planes at named Z heights."""
    return [
        _Section(name=name, plane=Plane.XY.offset(z)) for name, z in z_values.items()
    ]


def standard_sections() -> list[_Section]:
    """The three principal planes — useful starting point."""
    return [
        _Section("XY", Plane.XY),
        _Section("XZ", Plane.XZ),
        _Section("YZ", Plane.YZ),
    ]
