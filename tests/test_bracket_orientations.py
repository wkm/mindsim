#!/usr/bin/env python3
"""Stress test: 4 servos + brackets at different axis orientations.

Renders each servo+bracket pair individually from the same angle so you can
compare bracket pocket orientation across all 4 axes. Also renders the
combined hub body with all 4 brackets.

Run:  uv run mjpython tests/test_bracket_orientations.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from build123d import Align, Box, Location, export_stl
    from PIL import Image, ImageDraw

    from botcad.bracket import bracket_envelope_solid as bracket_envelope, bracket_solid_solid as bracket_solid, servo_solid
    from botcad.cad_utils import as_solid
    from botcad.colors import COLOR_BRACKET, COLOR_STRUCTURE_DARK, Color
    from botcad.components import STS3215
    from botcad.emit.composite import FONT_LABEL, FONT_TITLE, save_png
    from botcad.emit.render3d import Renderer3D, SceneBuilder
    from botcad.geometry import quat_to_euler, rotate_vec, servo_placement

    servo = STS3215()
    shaft_offset = servo.shaft_offset
    shaft_axis = servo.shaft_axis

    axes = [
        ("axis=+X", (1, 0, 0), (0.06, 0, 0)),
        ("axis=-X", (-1, 0, 0), (-0.06, 0, 0)),
        ("axis=+Y", (0, 1, 0), (0, 0.06, 0)),
        ("axis=-Y", (0, -1, 0), (0, -0.06, 0)),
    ]

    tmpdir = Path(tempfile.mkdtemp(prefix="bracket_stress_"))
    W, H = 600, 600

    # Use a thin body (60mm cube) so the bracket protrusion is clearly visible
    body_size = 0.06
    hub_base = Box(
        body_size * 2,
        body_size * 2,
        body_size,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )

    hub_color = Color(
        COLOR_BRACKET.rgb[0], COLOR_BRACKET.rgb[1], COLOR_BRACKET.rgb[2], 0.35
    )

    # Render each servo+bracket pair individually
    individual_images = []
    for name, axis, pos in axes:
        center, quat = servo_placement(shaft_offset, shaft_axis, axis, pos)
        euler = quat_to_euler(quat)
        z_dir = rotate_vec(quat, (0, 0, 1))

        print(
            f"{name}: euler={tuple(round(e, 1) for e in euler)}, "
            f"shaft→({z_dir[0]:+.2f},{z_dir[1]:+.2f},{z_dir[2]:+.2f})"
        )

        # Build body with just this one bracket
        env = bracket_envelope(servo).locate(Location(center, euler))
        brk = bracket_solid(servo).locate(Location(center, euler))
        shell = (hub_base - env) + brk

        scene = SceneBuilder()

        # Body + bracket
        stl = tmpdir / f"body_{name}.stl"
        export_stl(as_solid(shell), str(stl))
        scene.add_mesh("body", str(stl), hub_color)

        # Servo
        sv = servo_solid(servo).locate(Location(center, euler))
        sv_stl = tmpdir / f"servo_{name}.stl"
        export_stl(as_solid(sv), str(sv_stl))
        scene.add_mesh("servo", str(sv_stl), COLOR_STRUCTURE_DARK)

        xml = scene.to_xml()
        with Renderer3D(xml, width=W, height=H) as r:
            img = r.render_frame(azimuth=135, elevation=-30)

        # Add label
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 10), name, fill="black", font=FONT_TITLE)
        draw.text(
            (10, 40),
            f"shaft→({z_dir[0]:+.1f},{z_dir[1]:+.1f},{z_dir[2]:+.1f})",
            fill="gray",
            font=FONT_LABEL,
        )
        individual_images.append(pil_img)

    # Also render combined hub with all 4 brackets
    shell = hub_base
    scene = SceneBuilder()
    for name, axis, pos in axes:
        center, quat = servo_placement(shaft_offset, shaft_axis, axis, pos)
        euler = quat_to_euler(quat)
        env = bracket_envelope(servo).locate(Location(center, euler))
        brk = bracket_solid(servo).locate(Location(center, euler))
        shell = (shell - env) + brk

    stl = tmpdir / "combined.stl"
    export_stl(as_solid(shell), str(stl))
    scene.add_mesh("body", str(stl), hub_color)

    for i, (name, axis, pos) in enumerate(axes):
        center, quat = servo_placement(shaft_offset, shaft_axis, axis, pos)
        euler = quat_to_euler(quat)
        sv = servo_solid(servo).locate(Location(center, euler))
        sv_stl = tmpdir / f"servo_all_{i}.stl"
        export_stl(as_solid(sv), str(sv_stl))
        scene.add_mesh(f"servo_{i}", str(sv_stl), COLOR_STRUCTURE_DARK)

    xml = scene.to_xml()
    with Renderer3D(xml, width=W, height=H) as r:
        combined_img = Image.fromarray(r.render_frame(azimuth=135, elevation=-30))
    draw = ImageDraw.Draw(combined_img)
    draw.text((10, 10), "All 4 combined", fill="black", font=FONT_TITLE)

    # Front view of combined
    with Renderer3D(xml, width=W, height=H) as r:
        front_img = Image.fromarray(r.render_frame(azimuth=90, elevation=0))
    draw = ImageDraw.Draw(front_img)
    draw.text((10, 10), "Front view (all 4)", fill="black", font=FONT_TITLE)

    # Compose: 3 rows x 2 cols
    # Row 1: +X, -X
    # Row 2: +Y, -Y
    # Row 3: combined 3/4, combined front
    grid = Image.new("RGB", (W * 2, H * 3), "white")
    for i, img in enumerate(individual_images):
        col, row = i % 2, i // 2
        grid.paste(img, (col * W, row * H))
    grid.paste(combined_img, (0, H * 2))
    grid.paste(front_img, (W, H * 2))

    out = Path(__file__).parent.parent / "bots" / "test_bracket_stress.png"
    save_png(grid, out)
    print(f"\nRendered: {out}")

    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
