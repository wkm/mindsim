"""Technical 2D drawings for parametric CAD parts.

Generate cross-section SVGs at specified planes, with multiple parts
overlaid in distinct colors. Sections include hatching, section labels,
and a title block with scale indicator and legend.

Usage:
    from botcad.debug_drawing import DebugDrawing

    drawing = DebugDrawing("coupler_debug")
    drawing.add_part("coupler", coupler_solid, color=(200, 80, 50))
    drawing.add_part("servo", servo_solid, color=(60, 60, 60))
    drawing.add_section("front_horn", Plane.XY.offset(front_z))
    drawing.add_section("side_profile", Plane.XZ)
    drawing.save("coupler_debug.svg")
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from pathlib import Path

from build123d import (
    Color,
    Compound,
    ExportSVG,
    Location,
    Plane,
    Solid,
    Unit,
    section,
)
from build123d.exporters import LineType

RGB = tuple[int, int, int]

# Default color palette — visually distinct, colorblind-friendly
PALETTE: list[RGB] = [
    (220, 60, 40),  # red
    (50, 120, 190),  # blue
    (60, 60, 60),  # dark gray
    (40, 170, 80),  # green
    (200, 140, 40),  # amber
    (140, 80, 170),  # purple
]

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


def _mm(mm: float) -> float:
    """Convert mm to meters (viewBox units)."""
    return mm / 1000.0


@dataclass
class _Part:
    name: str
    solid: Solid | Compound
    color: RGB


@dataclass
class _Section:
    name: str
    plane: Plane


@dataclass
class DebugDrawing:
    """Accumulates parts and section planes, then exports a single SVG.

    All sections are laid out side-by-side in one SVG file with:
    - Cross-hatched section fills (ISO 45° pattern per part)
    - Section labels (A-A, B-B, ...)
    - Title text with scale indicator and part legend
    """

    title: str
    parts: list[_Part] = field(default_factory=list)
    sections: list[_Section] = field(default_factory=list)
    scale: float = DEFAULT_SCALE
    line_weight: float = LINE_WEIGHT_THICK

    def add_part(
        self,
        name: str,
        solid: Solid | Compound,
        color: RGB | None = None,
    ) -> DebugDrawing:
        """Add a solid to be sectioned. Color auto-assigned if omitted."""
        if color is None:
            color = PALETTE[len(self.parts) % len(PALETTE)]
        self.parts.append(_Part(name=name, solid=solid, color=color))
        return self

    def add_section(self, name: str, plane: Plane) -> DebugDrawing:
        """Add a named section plane."""
        self.sections.append(_Section(name=name, plane=plane))
        return self

    def save(self, path: str | Path) -> Path:
        """Export a single SVG with all sections side by side."""
        path = Path(path)
        self._export_combined(path)
        return path

    def _export_combined(self, out: Path) -> None:
        """Section all parts at all planes and compose into one SVG."""
        section_data: list[tuple[str, list[tuple[_Part, object]]]] = []
        gap_m = _mm(SECTION_GAP_MM)

        for sec in self.sections:
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
            if parts_cross:
                section_data.append((sec.name, parts_cross))

        if not section_data:
            return

        # Compute bounding box for each section column (in meters)
        col_bounds: list[tuple[float, float, float, float]] = []
        for _name, parts_cross in section_data:
            xmin = ymin = float("inf")
            xmax = ymax = float("-inf")
            for _part, cross in parts_cross:
                bb = cross.bounding_box()
                xmin = min(xmin, bb.min.X)
                ymin = min(ymin, bb.min.Y)
                xmax = max(xmax, bb.max.X)
                ymax = max(ymax, bb.max.Y)
            col_bounds.append((xmin, ymin, xmax, ymax))

        # Compute X offsets to tile columns side by side (meters)
        x_offsets: list[float] = []
        cursor_x = 0.0
        for xmin, _ymin, xmax, _ymax in col_bounds:
            col_w = xmax - xmin
            x_offsets.append(cursor_x - xmin)
            cursor_x += col_w + gap_m

        # Build SVG with build123d ExportSVG
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

        for (_sec_name, parts_cross), x_off in zip(section_data, x_offsets):
            offset_loc = Location((x_off, 0, 0))
            for part, cross in parts_cross:
                moved = cross.moved(offset_loc)
                for face in moved.faces():
                    svg.add_shape(face, layer=f"{part.name}_fill")
                    for edge in face.edges():
                        svg.add_shape(edge, layer=part.name)

        svg.write(str(out))

        # Post-process: hatching, annotations, expanded viewBox
        self._postprocess_svg(out, section_data, col_bounds, x_offsets)
        print(f"  drawing: {out}")

    def _postprocess_svg(
        self,
        svg_path: Path,
        section_data: list[tuple[str, list]],
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

        # In build123d's SVG, the geometry group has scale(1,-1), so positive
        # Y in geometry = negative Y in viewBox. The viewBox Y range covers
        # the flipped geometry. We add space:
        #   - above geometry (more negative Y) for footer/labels
        #   - below geometry (more positive Y) for header/title
        new_x = geo_left - margin
        new_y = geo_top - footer
        new_w = vb_w + 2 * margin
        new_h = vb_h + header + footer

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
            section_data,
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
        section_data: list[tuple[str, list]],
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

        # Font sizes in viewBox units (meters) — these will render at
        # the right physical size because viewBox maps to the mm dimensions.
        fs_title = _mm(3.5)
        fs_label = _mm(2.5)
        fs_small = _mm(1.8)
        # --- Section labels (above geometry in viewBox = more negative Y) ---
        # geo_top is the most negative Y. Labels go above that with enough room.
        sublabel_y = geo_top - _mm(2)
        label_y = geo_top - _mm(5.5)

        for i, ((sec_name, _parts), (xmin, _ymin, xmax, _ymax), x_off) in enumerate(
            zip(section_data, col_bounds, x_offsets)
        ):
            cx = (xmin + xmax) / 2 + x_off
            letter = label_letters[i % len(label_letters)]
            label = f"{letter}-{letter}"

            elements.append(
                f'  <text x="{cx:.6f}" y="{label_y:.6f}" '
                f'text-anchor="middle" '
                f'font-family="{FONT_FAMILY}" '
                f'font-size="{fs_label:.6f}" font-weight="bold" '
                f'fill="#333">{label}</text>'
            )
            elements.append(
                f'  <text x="{cx:.6f}" y="{sublabel_y:.6f}" '
                f'text-anchor="middle" '
                f'font-family="{FONT_FAMILY}" '
                f'font-size="{fs_small:.6f}" '
                f'fill="#888">{sec_name}</text>'
            )

        # --- Title block (below geometry in viewBox = positive Y direction) ---
        title_x = new_x + _mm(2)
        title_y = geo_bottom + _mm(3.5)

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
        for bx in [title_x, title_x + bar_len]:
            elements.append(
                f'  <line x1="{bx:.6f}" y1="{bar_y - tick:.6f}" '
                f'x2="{bx:.6f}" y2="{bar_y + tick:.6f}" '
                f'stroke="#333" stroke-width="{_mm(0.25):.6f}"/>'
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
