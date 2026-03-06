"""Validation renders for component inspection.

Generates per-component PNGs with orthographic views showing body envelope,
mounting points, wire ports, axis indicators, and component-specific features.

Uses a unified visual language where the same color always means the same thing:
    Blue   = mounting points (screws, snap-fits)
    Orange = wire ports (connectors, cables)
    RGB    = axis indicators (X=red, Y=green, Z=blue)

ServoSpec components get additional detail: shaft, horn disc, horn holes,
rear holes, and mounting ears.

Called as part of Bot.emit() pipeline, or standalone:
    PYTHONPATH=. uv run python -m botcad.emit.component_renders [bot_dir]
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from botcad.component import Component, ServoSpec
from botcad.emit.renders import white_background

# ── Rendering config ──

WIDTH, HEIGHT = 800, 800  # per-view resolution
PNG_DPI = (150, 150)  # DPI metadata for saved PNGs
VIEWS = {
    "front (+Y)": {"azimuth": 90, "elevation": 0},
    "right (+X)": {"azimuth": 0, "elevation": 0},
    "top (+Z)": {"azimuth": 90, "elevation": -90},
    "bottom (-Z)": {"azimuth": 90, "elevation": 90},
    "back (-X)": {"azimuth": 180, "elevation": 0},
    "three-quarter": {"azimuth": 135, "elevation": -30},
}

# ── Visual language colors ──

COLOR_MOUNTING = (0.2, 0.4, 1.0, 1.0)  # blue — mounting points
COLOR_WIRE_PORT = (1.0, 0.5, 0.1, 1.0)  # orange — wire ports
COLOR_SHAFT = (1.0, 0.2, 0.2, 1.0)  # red — servo shaft
COLOR_HORN = (1.0, 0.9, 0.2, 0.7)  # yellow — horn disc
COLOR_HORN_HOLE = (0.2, 1.0, 0.3, 1.0)  # green — horn mounting holes
COLOR_REAR_HOLE = (0.2, 0.9, 0.9, 1.0)  # cyan — rear horn mounting holes

# Legend entries: (label, RGB for the swatch)
LEGEND_MOUNTING = ("mounting (blue)", (51, 102, 255))
LEGEND_WIRE_PORT = ("wire port (orange)", (255, 128, 26))
LEGEND_SHAFT = ("shaft (red)", (255, 51, 51))
LEGEND_HORN = ("horn (yellow)", (255, 230, 51))
LEGEND_HORN_HOLE = ("horn holes (green)", (51, 255, 77))
LEGEND_REAR_HOLE = ("rear holes (cyan)", (51, 230, 230))


# ── Scene building: layered approach ──


def _add_common_geoms(geoms, legends, component: Component):
    """Body mesh + axis indicators — every component gets these.

    The body is rendered as a mesh from the CAD pipeline (actual STL geometry),
    not a primitive approximation. The mesh asset "comp_mesh" must be defined
    in the scene's <asset> block by the caller.
    """
    c = component.color
    # Lighten very dark components so mesh geometry details (bosses, pockets)
    # are visible in the tear sheet.  The tear sheet is for inspection, not
    # photorealism — readability trumps color accuracy.
    min_lum = 0.35
    r, g, b = c[0], c[1], c[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < min_lum and lum > 0:
        scale = min_lum / lum
        r, g, b = min(r * scale, 1.0), min(g * scale, 1.0), min(b * scale, 1.0)
    body_rgba = f"{r:.2f} {g:.2f} {b:.2f} 0.85"

    # Body: reference the CAD-generated mesh
    geoms.append(f'<geom name="body" type="mesh" mesh="comp_mesh" rgba="{body_rgba}"/>')

    body_rgb = (int(r * 255), int(g * 255), int(b * 255))
    legends.append(("body", body_rgb))

    # Effective dimensions for axis sizing
    d = component.dimensions
    if isinstance(component, ServoSpec):
        d = component.effective_body_dims

    # Axis indicators: thin cylinders for X (red), Y (green), Z (blue)
    axis_len = max(d) * 1.0  # match component's largest dimension
    axis_len = max(axis_len, 0.015)  # minimum visible length
    axis_r = 0.0005
    geoms.append(
        f'<geom name="axis_x" type="cylinder"'
        f' size="{axis_r} {axis_len / 2}"'
        f' pos="{axis_len / 2} 0 0"'
        f' quat="0.7071068 0 0.7071068 0"'
        f' rgba="1 0 0 0.5"/>'
    )
    geoms.append(
        f'<geom name="axis_y" type="cylinder"'
        f' size="{axis_r} {axis_len / 2}"'
        f' pos="0 {axis_len / 2} 0"'
        f' quat="0.7071068 0.7071068 0 0"'
        f' rgba="0 1 0 0.5"/>'
    )
    geoms.append(
        f'<geom name="axis_z" type="cylinder"'
        f' size="{axis_r} {axis_len / 2}"'
        f' pos="0 0 {axis_len / 2}"'
        f' rgba="0 0 1 0.5"/>'
    )


def _add_mounting_geoms(geoms, legends, component: Component):
    """Blue spheres at each mounting_point — skip if none."""
    if not component.mounting_points:
        return
    for i, mp in enumerate(component.mounting_points):
        geoms.append(
            f'<geom name="mount_{i}" type="sphere"'
            f' size="0.002"'
            f' pos="{mp.pos[0]:.6f} {mp.pos[1]:.6f} {mp.pos[2]:.6f}"'
            f' rgba="{COLOR_MOUNTING[0]} {COLOR_MOUNTING[1]} {COLOR_MOUNTING[2]} {COLOR_MOUNTING[3]}"/>'
        )
    legends.append(LEGEND_MOUNTING)


def _add_wire_port_geoms(geoms, legends, component: Component):
    """Orange spheres at each wire_port — skip if none."""
    if not component.wire_ports:
        return
    for i, wp in enumerate(component.wire_ports):
        geoms.append(
            f'<geom name="wire_{i}" type="sphere"'
            f' size="0.003"'
            f' pos="{wp.pos[0]:.6f} {wp.pos[1]:.6f} {wp.pos[2]:.6f}"'
            f' rgba="{COLOR_WIRE_PORT[0]} {COLOR_WIRE_PORT[1]} {COLOR_WIRE_PORT[2]} {COLOR_WIRE_PORT[3]}"/>'
        )
    legends.append(LEGEND_WIRE_PORT)


def _add_servo_geoms(geoms, legends, servo: ServoSpec):
    """Servo-specific: mounting ears, shaft, horn, horn holes, rear holes."""
    so = servo.shaft_offset

    # Mounting ears (blue — same as mounting points, unified language)
    if servo.mounting_ears:
        for i, ear in enumerate(servo.mounting_ears):
            geoms.append(
                f'<geom name="ear_{i}" type="sphere"'
                f' size="0.002"'
                f' pos="{ear.pos[0]:.6f} {ear.pos[1]:.6f} {ear.pos[2]:.6f}"'
                f' rgba="{COLOR_MOUNTING[0]} {COLOR_MOUNTING[1]} {COLOR_MOUNTING[2]} {COLOR_MOUNTING[3]}"/>'
            )
        # mounting ears share the LEGEND_MOUNTING entry added by _add_mounting_geoms
        # but if no mounting_points were present, add the legend now
        if not any(entry == LEGEND_MOUNTING for entry in legends):
            legends.append(LEGEND_MOUNTING)

    # Shaft indicator (red cylinder at shaft position, along +Z)
    shaft_r = 0.003
    shaft_h = 0.008
    shaft_z = so[2] + shaft_h / 2
    geoms.append(
        f'<geom name="shaft" type="cylinder"'
        f' size="{shaft_r:.6f} {shaft_h / 2:.6f}"'
        f' pos="{so[0]:.6f} {so[1]:.6f} {shaft_z:.6f}"'
        f' rgba="{COLOR_SHAFT[0]} {COLOR_SHAFT[1]} {COLOR_SHAFT[2]} {COLOR_SHAFT[3]}"/>'
    )
    legends.append(LEGEND_SHAFT)

    # Horn disc (thin yellow cylinder at shaft, showing horn diameter)
    horn_r = 0.01
    horn_h = 0.001
    horn_z = so[2] + 0.003
    geoms.append(
        f'<geom name="horn" type="cylinder"'
        f' size="{horn_r:.6f} {horn_h / 2:.6f}"'
        f' pos="{so[0]:.6f} {so[1]:.6f} {horn_z:.6f}"'
        f' rgba="{COLOR_HORN[0]} {COLOR_HORN[1]} {COLOR_HORN[2]} {COLOR_HORN[3]}"/>'
    )
    legends.append(LEGEND_HORN)

    # Horn mounting holes (green spheres)
    if servo.horn_mounting_points:
        for i, mp in enumerate(servo.horn_mounting_points):
            geoms.append(
                f'<geom name="horn_mount_{i}" type="sphere"'
                f' size="0.0015"'
                f' pos="{mp.pos[0]:.6f} {mp.pos[1]:.6f} {mp.pos[2]:.6f}"'
                f' rgba="{COLOR_HORN_HOLE[0]} {COLOR_HORN_HOLE[1]} {COLOR_HORN_HOLE[2]} {COLOR_HORN_HOLE[3]}"/>'
            )
        legends.append(LEGEND_HORN_HOLE)

    # Rear horn mounting holes (cyan spheres)
    if servo.rear_horn_mounting_points:
        for i, mp in enumerate(servo.rear_horn_mounting_points):
            geoms.append(
                f'<geom name="rear_mount_{i}" type="sphere"'
                f' size="0.0015"'
                f' pos="{mp.pos[0]:.6f} {mp.pos[1]:.6f} {mp.pos[2]:.6f}"'
                f' rgba="{COLOR_REAR_HOLE[0]} {COLOR_REAR_HOLE[1]} {COLOR_REAR_HOLE[2]} {COLOR_REAR_HOLE[3]}"/>'
            )
        legends.append(LEGEND_REAR_HOLE)


def build_component_scene(component: Component) -> tuple[str, list, Path]:
    """Build MuJoCo XML showing any component with all its features.

    Generates a real STL mesh via the CAD pipeline and loads it as a
    MuJoCo mesh asset. Debug overlays (mounting points, wire ports, etc.)
    are layered on top as primitive geoms.

    Returns (xml_string, legends, temp_dir). Caller must clean up temp_dir.
    """
    from build123d import export_stl

    from botcad.emit.cad import make_component_solid

    # Generate actual CAD solid → temp STL
    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_comp_"))
    solid = make_component_solid(component)
    export_stl(solid, str(temp_dir / "component.stl"))

    geoms: list[str] = []
    legends: list[tuple[str, tuple[int, int, int]]] = []

    _add_common_geoms(geoms, legends, component)
    _add_mounting_geoms(geoms, legends, component)
    _add_wire_port_geoms(geoms, legends, component)

    if isinstance(component, ServoSpec):
        _add_servo_geoms(geoms, legends, component)

    geoms_str = "\n      ".join(geoms)

    xml = f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="component_debug">
  <option gravity="0 0 0"/>
  <compiler meshdir="{temp_dir}"/>
  <visual>
    <rgba haze="1 1 1 1"/>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
  </visual>
  <asset>
    <mesh name="comp_mesh" file="component.stl" scale="1 1 1"/>
  </asset>
  <worldbody>
    <light pos="0.1 0.1 0.2" dir="-0.3 -0.3 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.1 0.1 0.2" dir="0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>
    <body name="component" pos="0 0 0">
      {geoms_str}
    </body>
  </worldbody>
</mujoco>
"""
    return xml, legends, temp_dir


# ── Rendering ──


def _render_view(model, data, renderer, azimuth: float, elevation: float) -> np.ndarray:
    """Render a single view at given camera angles."""
    mujoco.mj_forward(model, data)

    # Tight zoom: 1.4x extent fills most of the viewport
    extent = max(model.stat.extent, 0.01)
    distance = extent * 1.4

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    renderer.update_scene(data, camera=cam)
    img = renderer.render().copy()
    white_background(img)
    return img


def _render_and_collect(xml: str, temp_dir: Path) -> list[tuple[Image.Image, str]]:
    """Load MuJoCo scene, render all views, return (image, label) pairs."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, WIDTH, HEIGHT)

    results = []
    for label, params in VIEWS.items():
        img_array = _render_view(
            model, data, renderer, params["azimuth"], params["elevation"]
        )
        results.append((Image.fromarray(img_array), label))

    renderer.close()
    return results


def _composite_grid(
    title: str,
    legends: list[tuple[str, tuple[int, int, int]]],
    views: list[tuple[Image.Image, str]],
) -> Image.Image:
    """Composite view images into a 3x2 grid with title and legend."""
    n_cols = 3
    n_rows = 2
    margin = 5
    label_h = 25
    grid_w = WIDTH * n_cols + margin * (n_cols + 1)
    grid_h = (HEIGHT + label_h) * n_rows + margin * (n_rows + 1) + 40

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), title, fill=(0, 0, 0))

    # Dynamic legend — only colors that appear in this render
    seen_labels = set()
    unique_legends = []
    for entry in legends:
        if entry[0] not in seen_labels:
            seen_labels.add(entry[0])
            unique_legends.append(entry)

    legend_y = margin + 18
    x_off = margin
    for text, color in unique_legends:
        draw.rectangle([x_off, legend_y, x_off + 10, legend_y + 10], fill=color)
        draw.text((x_off + 13, legend_y - 2), text, fill=(0, 0, 0))
        x_off += len(text) * 7 + 22

    title_total_h = 40
    for idx, (img, label) in enumerate(views):
        col = idx % n_cols
        row = idx // n_cols
        x = margin + col * (WIDTH + margin)
        y = title_total_h + margin + row * (HEIGHT + label_h + margin)
        draw.text((x, y), label, fill=(0, 0, 0))
        img_y = y + label_h
        draw.rectangle(
            [x - 1, img_y - 1, x + WIDTH, img_y + HEIGHT],
            outline=(0, 0, 0),
            width=1,
        )
        canvas.paste(img, (x, img_y))

    return canvas


def render_component_views(component: Component, name: str) -> Image.Image:
    """Render all views of a component and composite into a single image."""
    xml, legends, temp_dir = build_component_scene(component)

    try:
        views = _render_and_collect(xml, temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    d = component.dimensions
    title = f"{name} — {d[0] * 1000:.1f} x {d[1] * 1000:.1f} x {d[2] * 1000:.1f} mm"
    return _composite_grid(title, legends, views)


# ── Bracket rendering ──

COLOR_BRACKET = (0.55, 0.75, 0.85, 0.85)  # light blue — printed bracket
COLOR_SERVO_BODY = (0.15, 0.15, 0.15, 0.6)  # dark translucent — servo body
LEGEND_BRACKET = ("bracket (blue)", (140, 191, 217))
LEGEND_SERVO_BODY = ("servo (dark)", (38, 38, 38))


def build_bracket_scene(servo: ServoSpec) -> tuple[str, list, Path]:
    """Build MuJoCo scene showing bracket + servo + annotations.

    Exports bracket_solid and servo_solid as separate STL meshes, loads
    both into a MuJoCo scene with mounting ear and horn hole annotations.

    Returns (xml_string, legends, temp_dir). Caller must clean up temp_dir.
    """
    from build123d import export_stl

    from botcad.bracket import BracketSpec, bracket_solid, servo_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_bracket_"))

    bracket = bracket_solid(servo, BracketSpec())
    export_stl(bracket, str(temp_dir / "bracket.stl"))

    servo_body = servo_solid(servo)
    export_stl(servo_body, str(temp_dir / "servo.stl"))

    geoms: list[str] = []
    legends: list[tuple[str, tuple[int, int, int]]] = []

    # Bracket mesh (light blue)
    br = COLOR_BRACKET
    geoms.append(
        f'<geom name="bracket" type="mesh" mesh="bracket_mesh"'
        f' rgba="{br[0]} {br[1]} {br[2]} {br[3]}"/>'
    )
    legends.append(LEGEND_BRACKET)

    # Servo body mesh (dark translucent)
    sv = COLOR_SERVO_BODY
    geoms.append(
        f'<geom name="servo" type="mesh" mesh="servo_mesh"'
        f' rgba="{sv[0]} {sv[1]} {sv[2]} {sv[3]}"/>'
    )
    legends.append(LEGEND_SERVO_BODY)

    # Mounting ears (blue spheres)
    if servo.mounting_ears:
        for i, ear in enumerate(servo.mounting_ears):
            geoms.append(
                f'<geom name="ear_{i}" type="sphere" size="0.002"'
                f' pos="{ear.pos[0]:.6f} {ear.pos[1]:.6f} {ear.pos[2]:.6f}"'
                f' rgba="{COLOR_MOUNTING[0]} {COLOR_MOUNTING[1]} {COLOR_MOUNTING[2]} {COLOR_MOUNTING[3]}"/>'
            )
        legends.append(LEGEND_MOUNTING)

    # Front horn holes (green spheres)
    if servo.horn_mounting_points:
        for i, mp in enumerate(servo.horn_mounting_points):
            geoms.append(
                f'<geom name="horn_{i}" type="sphere" size="0.0015"'
                f' pos="{mp.pos[0]:.6f} {mp.pos[1]:.6f} {mp.pos[2]:.6f}"'
                f' rgba="{COLOR_HORN_HOLE[0]} {COLOR_HORN_HOLE[1]} {COLOR_HORN_HOLE[2]} {COLOR_HORN_HOLE[3]}"/>'
            )
        legends.append(LEGEND_HORN_HOLE)

    # Rear horn holes (cyan spheres)
    if servo.rear_horn_mounting_points:
        for i, mp in enumerate(servo.rear_horn_mounting_points):
            geoms.append(
                f'<geom name="rear_{i}" type="sphere" size="0.0015"'
                f' pos="{mp.pos[0]:.6f} {mp.pos[1]:.6f} {mp.pos[2]:.6f}"'
                f' rgba="{COLOR_REAR_HOLE[0]} {COLOR_REAR_HOLE[1]} {COLOR_REAR_HOLE[2]} {COLOR_REAR_HOLE[3]}"/>'
            )
        legends.append(LEGEND_REAR_HOLE)

    # Shaft indicator (red)
    so = servo.shaft_offset
    shaft_h = 0.008
    shaft_z = so[2] + shaft_h / 2
    geoms.append(
        f'<geom name="shaft" type="cylinder" size="0.003 {shaft_h / 2:.6f}"'
        f' pos="{so[0]:.6f} {so[1]:.6f} {shaft_z:.6f}"'
        f' rgba="{COLOR_SHAFT[0]} {COLOR_SHAFT[1]} {COLOR_SHAFT[2]} {COLOR_SHAFT[3]}"/>'
    )
    legends.append(LEGEND_SHAFT)

    # Wire port (orange)
    if servo.connector_pos:
        cp = servo.connector_pos
        geoms.append(
            f'<geom name="wire_0" type="sphere" size="0.003"'
            f' pos="{cp[0]:.6f} {cp[1]:.6f} {cp[2]:.6f}"'
            f' rgba="{COLOR_WIRE_PORT[0]} {COLOR_WIRE_PORT[1]} {COLOR_WIRE_PORT[2]} {COLOR_WIRE_PORT[3]}"/>'
        )
        legends.append(LEGEND_WIRE_PORT)

    geoms_str = "\n      ".join(geoms)

    xml = f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="bracket_debug">
  <option gravity="0 0 0"/>
  <compiler meshdir="{temp_dir}"/>
  <visual>
    <rgba haze="1 1 1 1"/>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
  </visual>
  <asset>
    <mesh name="bracket_mesh" file="bracket.stl" scale="1 1 1"/>
    <mesh name="servo_mesh" file="servo.stl" scale="1 1 1"/>
  </asset>
  <worldbody>
    <light pos="0.1 0.1 0.2" dir="-0.3 -0.3 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.1 0.1 0.2" dir="0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>
    <body name="assembly" pos="0 0 0">
      {geoms_str}
    </body>
  </worldbody>
</mujoco>
"""
    return xml, legends, temp_dir


def render_bracket_views(servo: ServoSpec, name: str) -> Image.Image:
    """Render bracket + servo assembly from all views."""
    xml, legends, temp_dir = build_bracket_scene(servo)

    try:
        views = _render_and_collect(xml, temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    title = f"Pocket Bracket — {name}"
    return _composite_grid(title, legends, views)


# ── Pipeline entry point ──


def _component_category(comp: Component) -> str:
    """Derive category from the component's module (battery, camera, servo, etc.)."""
    import importlib
    import pkgutil

    import botcad.components as pkg

    for info in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"botcad.components.{info.name}")
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if callable(obj) and not isinstance(obj, type):
                try:
                    instance = obj()
                    if isinstance(instance, Component) and instance.name == comp.name:
                        return info.name
                except TypeError:
                    pass
    return "component"


def emit_component_renders(bot, output_dir: Path) -> None:
    """Render every unique component used in the bot.

    Collects components from body mounts and joint servos, deduplicates
    by name, and renders each one. Output goes to botcad/components/ —
    tear sheets live next to the component definitions, not the bot.
    """
    import dataclasses

    t0 = time.perf_counter()

    # Output next to the component source files
    components_dir = Path(__file__).resolve().parent.parent / "components"

    # Collect unique components by name
    seen: dict[str, Component] = {}

    for body in bot.all_bodies:
        for mount in body.mounts:
            comp = mount.component
            if comp.name not in seen:
                seen[comp.name] = comp

    # Collect servos separately (they get position + continuous variants)
    servos: dict[str, ServoSpec] = {}
    for joint in bot.all_joints:
        servo = joint.servo
        if servo.name not in servos:
            servos[servo.name] = servo

    if not seen and not servos:
        print("  component renders: no components found, skipping")
        return

    print("Component renders:")

    # Render non-servo components
    for name, comp in seen.items():
        if isinstance(comp, ServoSpec):
            continue  # servos handled below
        category = _component_category(comp)
        safe_name = name.lower().replace(" ", "_")
        filename = f"test_{category}_{safe_name}.png"

        img = render_component_views(comp, name)
        out_path = components_dir / filename
        img.save(out_path, optimize=True, dpi=PNG_DPI)
        print(f"  {out_path}")

    # Render servo variants (position + continuous)
    for name, servo in servos.items():
        safe_name = name.lower().replace(" ", "_")
        for mode in (False, True):
            mode_name = "continuous" if mode else "position"
            variant = dataclasses.replace(servo, continuous=mode)
            display_name = f"{name} ({mode_name})"
            filename = f"test_servo_{safe_name}_{mode_name}.png"

            img = render_component_views(variant, display_name)
            out_path = components_dir / filename
            img.save(out_path, optimize=True, dpi=PNG_DPI)
            print(f"  {out_path}")

    # Render bracket assemblies for each servo
    for name, servo in servos.items():
        safe_name = name.lower().replace(" ", "_")
        filename = f"test_bracket_{safe_name}.png"

        img = render_bracket_views(servo, name)
        out_path = components_dir / filename
        img.save(out_path, optimize=True, dpi=PNG_DPI)
        print(f"  {out_path}")

    print(f"  component renders done ({time.perf_counter() - t0:.1f}s)")


# ── Standalone ──


if __name__ == "__main__":
    from botcad.components import (
        OV5647,
        STS3215,
        LiPo2S,
        PololuWheel90mm,
        RaspberryPiZero2W,
    )

    out_dir = Path(__file__).resolve().parent.parent / "components"

    # (category, display_name, component)
    components: list[tuple[str, str, Component]] = [
        ("servo", "STS3215 (position)", STS3215(continuous=False)),
        ("servo", "STS3215 (continuous)", STS3215(continuous=True)),
        ("wheel", "Pololu 90x10mm Wheel", PololuWheel90mm()),
        ("camera", "OV5647", OV5647()),
        ("battery", "LiPo2S-1000", LiPo2S(1000)),
        ("compute", "RaspberryPiZero2W", RaspberryPiZero2W()),
    ]

    for category, name, comp in components:
        img = render_component_views(comp, name)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = out_dir / f"test_{category}_{safe_name}.png"
        img.save(out_path, optimize=True, dpi=PNG_DPI)
        print(f"Saved: {out_path}")

    # Bracket tear sheet
    servo = STS3215()
    img = render_bracket_views(servo, "STS3215")
    out_path = out_dir / "test_bracket_sts3215.png"
    img.save(out_path, optimize=True, dpi=PNG_DPI)
    print(f"Saved: {out_path}")
