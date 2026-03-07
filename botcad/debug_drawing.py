"""Technical 2D drawings for parametric CAD parts.

Generate cross-section SVGs at specified planes, with multiple parts
overlaid in distinct colors. All sections are combined into a single
SVG with labeled columns. Useful at every design scale:

    Component:    section a single solid to verify dimensions
    Interface:    overlay two mating parts to check clearances
    Subassembly:  section a moving group at a joint angle
    Bot:          section the full kinematic chain at a pose

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

# Geometry is in meters; scale=1000 means 1m = 1000mm in SVG.
# For ~50mm parts this gives ~50mm SVG columns — readable at 1:1.
DEFAULT_SCALE = 1000.0
DEFAULT_LINE_WEIGHT = 0.2  # mm stroke in final SVG


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

    All sections are laid out side-by-side in one SVG file, each column
    labeled with the section name.
    """

    title: str
    parts: list[_Part] = field(default_factory=list)
    sections: list[_Section] = field(default_factory=list)
    scale: float = DEFAULT_SCALE
    line_weight: float = DEFAULT_LINE_WEIGHT

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
        """Export a single SVG with all sections side by side.

        Returns the output path.
        """
        path = Path(path)
        self._export_combined(path)
        return path

    def _export_combined(self, out: Path) -> None:
        """Section all parts at all planes and compose into one SVG."""
        # First pass: section everything, compute bounding boxes
        section_data: list[tuple[str, list[tuple[_Part, object]]]] = []
        gap = 0.005  # 5mm gap between columns (in meters)

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

        # Compute bounding box for each section column
        col_bounds: list[
            tuple[float, float, float, float]
        ] = []  # (xmin, ymin, xmax, ymax)
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

        # Compute X offsets to tile columns side by side
        x_offsets: list[float] = []
        cursor_x = 0.0
        for xmin, _ymin, xmax, _ymax in col_bounds:
            col_w = xmax - xmin
            # Offset so that xmin of this column starts at cursor_x
            x_offsets.append(cursor_x - xmin)
            cursor_x += col_w + gap

        # Build single SVG with all sections
        svg = ExportSVG(
            unit=Unit.MM,
            scale=self.scale,
            margin=3,
            line_weight=self.line_weight,
        )

        # Create layers for each part (shared across all sections)
        for part in self.parts:
            r, g, b = part.color
            svg.add_layer(
                part.name,
                line_color=Color(r / 255, g / 255, b / 255),
                line_weight=self.line_weight,
            )
            svg.add_layer(
                f"{part.name}_fill",
                fill_color=Color(r / 255, g / 255, b / 255, 0.15),
                line_color=Color(r / 255, g / 255, b / 255, 0.0),
                line_weight=0,
            )

        # Add shapes with X offsets
        for (sec_name, parts_cross), x_off in zip(section_data, x_offsets):
            offset_loc = Location((x_off, 0, 0))
            for part, cross in parts_cross:
                moved = cross.moved(offset_loc)
                for face in moved.faces():
                    svg.add_shape(face, layer=f"{part.name}_fill")
                    for edge in face.edges():
                        svg.add_shape(edge, layer=part.name)

        svg.write(str(out))
        print(f"  drawing: {out}")

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
