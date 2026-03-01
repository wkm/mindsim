"""Validation renders for servo component inspection.

Generates per-servo-spec PNGs with orthographic views showing body box,
shaft position, mounting ears, horn mounting holes, and connector.

Called as part of Bot.emit() pipeline, or standalone:
    PYTHONPATH=. uv run python -m botcad.emit.servo_renders [bot_dir]
"""

from __future__ import annotations

import math
import struct
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

# ── Rendering config ──

WIDTH, HEIGHT = 400, 400  # per-view resolution
VIEWS = {
    "front (+Y)": {  # looking at the front face (from +Y toward -Y)
        "azimuth": 90,
        "elevation": 0,
    },
    "right (+X)": {  # looking at the right side (from +X toward -X)
        "azimuth": 0,
        "elevation": 0,
    },
    "top (+Z)": {  # looking down at the shaft face (from +Z toward -Z)
        "azimuth": 90,
        "elevation": -90,
    },
    "bottom (-Z)": {  # looking up at the connector/rear face (from -Z toward +Z)
        "azimuth": 90,
        "elevation": 90,
    },
    "back (-X)": {  # looking at the back face (from -X toward +X)
        "azimuth": 180,
        "elevation": 0,
    },
    "three-quarter": {
        "azimuth": 135,
        "elevation": -30,
    },
}


def _box_stl_bytes(sx: float, sy: float, sz: float) -> bytes:
    """Minimal binary STL for a box centered at origin."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    tris = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (3, 7, 6),
        (3, 6, 2),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (0, 4, 7),
        (0, 7, 3),
    ]
    data = bytearray(b"\x00" * 80)
    data.extend(struct.pack("<I", len(tris)))
    for i0, i1, i2 in tris:
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        m = math.sqrt(nx * nx + ny * ny + nz * nz)
        if m > 0:
            nx, ny, nz = nx / m, ny / m, nz / m
        data.extend(struct.pack("<fff", nx, ny, nz))
        data.extend(struct.pack("<fff", *v0))
        data.extend(struct.pack("<fff", *v1))
        data.extend(struct.pack("<fff", *v2))
        data.extend(struct.pack("<H", 0))
    return bytes(data)


def _cylinder_stl_bytes(radius: float, height: float, segments: int = 16) -> bytes:
    """Minimal binary STL for a Z-axis cylinder centered at origin."""
    hz = height / 2
    verts = [(0, 0, -hz), (0, 0, hz)]
    tris = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        verts.append((radius * math.cos(a), radius * math.sin(a), -hz))
        verts.append((radius * math.cos(a), radius * math.sin(a), hz))
    for i in range(segments):
        bi, ti = 2 + i * 2, 3 + i * 2
        ni = (i + 1) % segments
        bni, tni = 2 + ni * 2, 3 + ni * 2
        tris.extend([(0, bni, bi), (1, ti, tni), (bi, bni, tni), (bi, tni, ti)])
    data = bytearray(b"\x00" * 80)
    data.extend(struct.pack("<I", len(tris)))
    for i0, i1, i2 in tris:
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        m = math.sqrt(nx * nx + ny * ny + nz * nz)
        if m > 0:
            nx, ny, nz = nx / m, ny / m, nz / m
        data.extend(struct.pack("<fff", nx, ny, nz))
        data.extend(struct.pack("<fff", *v0))
        data.extend(struct.pack("<fff", *v1))
        data.extend(struct.pack("<fff", *v2))
        data.extend(struct.pack("<H", 0))
    return bytes(data)


def build_servo_scene(servo) -> str:
    """Build a MuJoCo XML scene showing the servo with all its features."""
    d = servo.dimensions
    bd = servo.body_dimensions if any(x > 0 for x in servo.body_dimensions) else d
    so = servo.shaft_offset

    # Shaft indicator: small cylinder at the shaft position, along +Z
    shaft_r = 0.003  # 3mm radius visual indicator
    shaft_h = 0.008  # 8mm tall
    shaft_z = so[2] + shaft_h / 2  # sits on top of body

    # Build XML
    geoms = []

    # Main body (dark gray, slightly transparent)
    geoms.append(
        f'<geom name="body" type="box"'
        f' size="{bd[0] / 2:.6f} {bd[1] / 2:.6f} {bd[2] / 2:.6f}"'
        f' rgba="0.2 0.2 0.2 0.85"/>'
    )

    # Shaft indicator (red cylinder)
    geoms.append(
        f'<geom name="shaft" type="cylinder"'
        f' size="{shaft_r:.6f} {shaft_h / 2:.6f}"'
        f' pos="{so[0]:.6f} {so[1]:.6f} {shaft_z:.6f}"'
        f' rgba="1.0 0.2 0.2 1.0"/>'
    )

    # Horn circle (thin yellow cylinder at shaft, showing horn diameter)
    horn_r = 0.01  # 10mm radius (ø20mm horn)
    horn_h = 0.001
    horn_z = so[2] + 0.003
    geoms.append(
        f'<geom name="horn" type="cylinder"'
        f' size="{horn_r:.6f} {horn_h / 2:.6f}"'
        f' pos="{so[0]:.6f} {so[1]:.6f} {horn_z:.6f}"'
        f' rgba="1.0 0.9 0.2 0.7"/>'
    )

    # Mounting ears (blue spheres at each ear position)
    for i, ear in enumerate(servo.mounting_ears):
        geoms.append(
            f'<geom name="ear_{i}" type="sphere"'
            f' size="0.002"'
            f' pos="{ear.pos[0]:.6f} {ear.pos[1]:.6f} {ear.pos[2]:.6f}"'
            f' rgba="0.2 0.4 1.0 1.0"/>'
        )

    # Horn mounting holes (green spheres)
    for i, mp in enumerate(servo.horn_mounting_points):
        geoms.append(
            f'<geom name="horn_mount_{i}" type="sphere"'
            f' size="0.0015"'
            f' pos="{mp.pos[0]:.6f} {mp.pos[1]:.6f} {mp.pos[2]:.6f}"'
            f' rgba="0.2 1.0 0.3 1.0"/>'
        )

    # Rear horn mounting holes (cyan spheres)
    for i, mp in enumerate(servo.rear_horn_mounting_points):
        geoms.append(
            f'<geom name="rear_mount_{i}" type="sphere"'
            f' size="0.0015"'
            f' pos="{mp.pos[0]:.6f} {mp.pos[1]:.6f} {mp.pos[2]:.6f}"'
            f' rgba="0.2 0.9 0.9 1.0"/>'
        )

    # Connector (orange sphere)
    if servo.connector_pos:
        cp = servo.connector_pos
        geoms.append(
            f'<geom name="connector" type="sphere"'
            f' size="0.003"'
            f' pos="{cp[0]:.6f} {cp[1]:.6f} {cp[2]:.6f}"'
            f' rgba="1.0 0.5 0.1 1.0"/>'
        )

    # Axis indicators: thin cylinders for X (red), Y (green), Z (blue)
    axis_len = 0.035
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

    geoms_str = "\n      ".join(geoms)

    xml = f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="servo_debug">
  <option gravity="0 0 0"/>
  <worldbody>
    <light pos="0.1 0.1 0.2" dir="-0.3 -0.3 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.1 0.1 0.2" dir="0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>
    <body name="servo" pos="0 0 0">
      {geoms_str}
    </body>
  </worldbody>
</mujoco>
"""
    return xml


def render_view(model, data, renderer, azimuth: float, elevation: float) -> np.ndarray:
    """Render a single view at given camera angles."""
    mujoco.mj_forward(model, data)

    # Compute camera distance to fit the servo
    extent = max(model.stat.extent, 0.05)
    distance = extent * 2.5

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def render_servo_views(servo, name: str) -> Image.Image:
    """Render all views of a servo and composite into a single image."""
    xml = build_servo_scene(servo)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, WIDTH, HEIGHT)

    images = []
    labels = []
    for label, params in VIEWS.items():
        img_array = render_view(
            model, data, renderer, params["azimuth"], params["elevation"]
        )
        images.append(Image.fromarray(img_array))
        labels.append(label)

    renderer.close()

    # Composite: 3x2 grid with labels
    n_cols = 3
    n_rows = 2
    margin = 5
    label_h = 25
    grid_w = WIDTH * n_cols + margin * (n_cols + 1)
    grid_h = (HEIGHT + label_h) * n_rows + margin * (n_rows + 1) + 40  # extra for title

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title
    title = f"{name} — {servo.dimensions[0] * 1000:.1f} x {servo.dimensions[1] * 1000:.1f} x {servo.dimensions[2] * 1000:.1f} mm"
    draw.text((margin, margin), title, fill=(0, 0, 0))

    # Legend
    legend_y = margin + 18
    legend_items = [
        ("body (dark gray)", (51, 51, 51)),
        ("shaft (red)", (255, 51, 51)),
        ("horn (yellow)", (255, 230, 51)),
        ("ears (blue)", (51, 102, 255)),
        ("horn holes (green)", (51, 255, 77)),
        ("rear holes (cyan)", (51, 230, 230)),
        ("connector (orange)", (255, 128, 26)),
    ]
    x_off = margin
    for text, color in legend_items:
        draw.rectangle([x_off, legend_y, x_off + 10, legend_y + 10], fill=color)
        draw.text((x_off + 13, legend_y - 2), text, fill=(0, 0, 0))
        x_off += len(text) * 7 + 22

    title_total_h = 40
    for idx, (img, label) in enumerate(zip(images, labels)):
        col = idx % n_cols
        row = idx // n_cols
        x = margin + col * (WIDTH + margin)
        y = title_total_h + margin + row * (HEIGHT + label_h + margin)
        draw.text((x, y), label, fill=(0, 0, 0))
        canvas.paste(img, (x, y + label_h))

    return canvas


# ── Pipeline entry point ──


def emit_servo_renders(bot, output_dir: Path) -> None:
    """Generate per-servo-spec validation renders.

    Renders each unique ServoSpec used in the bot (deduplicated by name).
    Both position and continuous variants are rendered for each servo.
    """
    import dataclasses

    t0 = time.perf_counter()

    # Collect unique servo specs by name
    seen = {}
    for joint in bot.all_joints:
        servo = joint.servo
        if servo.name not in seen:
            seen[servo.name] = servo

    if not seen:
        print("  servo renders: no servos found, skipping")
        return

    print("Servo renders:")
    for name, servo in seen.items():
        safe_name = name.lower().replace(" ", "_")

        # Render both position and continuous variants
        for mode in (False, True):
            mode_name = "continuous" if mode else "position"
            variant = dataclasses.replace(servo, continuous=mode)
            display_name = f"{name} ({mode_name})"
            filename = f"test_servo_{safe_name}_{mode_name}.png"

            img = render_servo_views(variant, display_name)
            out_path = output_dir / filename
            img.save(out_path)
            print(f"  {out_path}")

    print(f"  servo renders done ({time.perf_counter() - t0:.1f}s)")


# ── Standalone ──


if __name__ == "__main__":
    from botcad.components.servo import STS3215

    bot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bots/wheeler_arm")
    bot_dir.mkdir(parents=True, exist_ok=True)

    servos = [
        ("STS3215 (position)", STS3215(continuous=False)),
        ("STS3215 (continuous)", STS3215(continuous=True)),
    ]

    for name, servo in servos:
        img = render_servo_views(servo, name)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = bot_dir / f"test_servo_{safe_name}.png"
        img.save(out_path)
        print(f"Saved: {out_path}")
