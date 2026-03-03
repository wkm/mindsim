"""Validation renders for assembled bot inspection.

Generates three PNGs in the bot output directory:
  - test_overview.png   — 2x2 grid, full bot from 4 angles
  - test_closeups.png   — per-joint zoomed views of each servo area
  - test_sweep.png      — per-joint filmstrip across range of motion

Called as part of Bot.emit() pipeline, or standalone:
    PYTHONPATH=. uv run python -m botcad.emit.renders [bot_dir]
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

# ── Config ──

VIEW_W, VIEW_H = 1000, 1000  # per-view resolution for overview/closeups
SWEEP_W, SWEEP_H = 600, 600  # per-frame resolution for sweeps
PNG_DPI = (150, 150)  # DPI metadata for saved PNGs
SWEEP_FRAMES = 9

OVERVIEW_VIEWS = {
    "front (+Y)": {"azimuth": 90, "elevation": 0},
    "right (+X)": {"azimuth": 0, "elevation": 0},
    "top (+Z)": {"azimuth": 90, "elevation": -90},
    "three-quarter": {"azimuth": 135, "elevation": -30},
}

CLOSEUP_VIEWS = {
    "front": {"azimuth": 90, "elevation": 0},
    "right": {"azimuth": 0, "elevation": 0},
    "top": {"azimuth": 90, "elevation": -90},
    "three-quarter": {"azimuth": 135, "elevation": -30},
}


# ── Helpers ──


def _configure_spec(spec: mujoco.MjSpec) -> None:
    """Common spec configuration: headlight, offscreen buffer."""
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.global_.offwidth = max(VIEW_W, SWEEP_W) * 2
    spec.visual.global_.offheight = max(VIEW_H, SWEEP_H) * 2


def _load_model(bot_xml: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load bot.xml via MjSpec, add headlight, compile."""
    spec = mujoco.MjSpec.from_file(str(bot_xml))
    _configure_spec(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _render(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    lookat: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
) -> np.ndarray:
    """Render a single frame with the given camera parameters."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, camera=cam)
    img = renderer.render().copy()
    # White background — replace black pixels
    mask = np.all(img == 0, axis=2)
    img[mask] = 255
    return img


def _set_mesh_alpha(model: mujoco.MjModel, alpha: float) -> np.ndarray:
    """Set all mesh geom rgba alpha to `alpha`. Returns original values for restore."""
    original = model.geom_rgba.copy()
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_rgba[gid, 3] = alpha
    return original


def _restore_rgba(model: mujoco.MjModel, original: np.ndarray) -> None:
    """Restore geom rgba from saved values."""
    model.geom_rgba[:] = original


def _sweepable_joints(model: mujoco.MjModel) -> list[int]:
    """Return joint IDs that are hinge joints with a nonzero range (not freejoint, not wheels)."""
    joints = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        lo, hi = model.jnt_range[jid]
        if lo == hi:
            # Continuous/unlimited joint (wheels) — skip
            continue
        joints.append(jid)
    return joints


def _joint_name(model: mujoco.MjModel, jid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)


def _body_name(model: mujoco.MjModel, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)


def _servo_geom_id(model: mujoco.MjModel, joint_name: str) -> int:
    """Find the servo visualization geom for a joint."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{joint_name}_servo")


def _child_body_id(model: mujoco.MjModel, jid: int) -> int:
    """The body that this joint belongs to (child body)."""
    return model.jnt_bodyid[jid]


# ── Step 1: Assembled overview ──


def render_overview(
    model: mujoco.MjModel, data: mujoco.MjData, output_dir: Path
) -> Path:
    """Render 4-view overview of the full bot."""
    t0 = time.perf_counter()
    renderer = mujoco.Renderer(model, VIEW_W, VIEW_H)

    lookat = model.stat.center.copy()
    distance = model.stat.extent * 2.5

    # Make mesh geoms semi-transparent so wires are visible
    saved_rgba = _set_mesh_alpha(model, 0.3)

    images = []
    labels = []
    for label, params in OVERVIEW_VIEWS.items():
        img = _render(
            model,
            data,
            renderer,
            lookat,
            distance,
            params["azimuth"],
            params["elevation"],
        )
        images.append(Image.fromarray(img))
        labels.append(label)

    _restore_rgba(model, saved_rgba)
    renderer.close()

    # Composite 2x2
    margin = 8
    label_h = 22
    title_h = 30
    cols, rows = 2, 2
    cw = VIEW_W * cols + margin * (cols + 1)
    ch = title_h + (VIEW_H + label_h) * rows + margin * (rows + 1)

    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), f"Bot overview — {output_dir.name}", fill=(0, 0, 0))

    for idx, (img, label) in enumerate(zip(images, labels)):
        col = idx % cols
        row = idx // cols
        x = margin + col * (VIEW_W + margin)
        y = title_h + margin + row * (VIEW_H + label_h + margin)
        draw.text((x, y), label, fill=(80, 80, 80))
        img_y = y + label_h
        draw.rectangle(
            [x - 1, img_y - 1, x + VIEW_W, img_y + VIEW_H],
            outline=(0, 0, 0),
            width=1,
        )
        canvas.paste(img, (x, img_y))

    out = output_dir / "test_overview.png"
    canvas.save(out, optimize=True, dpi=PNG_DPI)
    print(f"  overview: {out} ({time.perf_counter() - t0:.1f}s)")
    return out


# ── Step 2: Servo closeups ──


def render_closeups(
    model: mujoco.MjModel, data: mujoco.MjData, output_dir: Path
) -> Path:
    """Render per-joint zoomed views of each servo area."""
    t0 = time.perf_counter()
    renderer = mujoco.Renderer(model, VIEW_W, VIEW_H)
    mujoco.mj_forward(model, data)

    # Make mesh geoms semi-transparent so wires are visible
    saved_rgba = _set_mesh_alpha(model, 0.3)

    sweep_joints = _sweepable_joints(model)
    sections: list[tuple[str, list[Image.Image], list[str]]] = []

    for jid in sweep_joints:
        jname = _joint_name(model, jid)
        gid = _servo_geom_id(model, jname)
        child_bid = _child_body_id(model, jid)

        if gid < 0:
            print(f"  warning: no servo geom for joint '{jname}', skipping closeup")
            continue

        servo_pos = data.geom_xpos[gid].copy()
        child_pos = data.xpos[child_bid].copy()
        midpoint = (servo_pos + child_pos) / 2.0

        # Distance: based on separation or servo size, whichever gives a better view
        separation = np.linalg.norm(servo_pos - child_pos)
        servo_half = model.geom_size[gid]  # box half-sizes
        servo_max_dim = np.max(servo_half) * 2
        distance = max(separation * 2.5, servo_max_dim * 6, 0.08)

        imgs = []
        lbls = []
        for label, params in CLOSEUP_VIEWS.items():
            img = _render(
                model,
                data,
                renderer,
                midpoint,
                distance,
                params["azimuth"],
                params["elevation"],
            )
            imgs.append(Image.fromarray(img))
            lbls.append(label)

        sections.append((jname, imgs, lbls))

    _restore_rgba(model, saved_rgba)
    renderer.close()

    if not sections:
        print("  closeups: no sweepable joints found, skipping")
        return output_dir / "test_closeups.png"

    # Composite: vertical stack of sections, each with header + 2x2 grid
    margin = 8
    label_h = 20
    header_h = 28
    cols, rows = 2, 2
    section_w = VIEW_W * cols + margin * (cols + 1)
    section_h = header_h + (VIEW_H + label_h) * rows + margin * (rows + 1)

    title_h = 30
    canvas_w = section_w
    canvas_h = title_h + section_h * len(sections) + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), f"Servo closeups — {output_dir.name}", fill=(0, 0, 0))

    for sec_idx, (jname, imgs, lbls) in enumerate(sections):
        sec_y = title_h + sec_idx * section_h
        draw.text((margin, sec_y + 4), f"Joint: {jname}", fill=(0, 0, 0))

        for idx, (img, label) in enumerate(zip(imgs, lbls)):
            col = idx % cols
            row = idx // cols
            x = margin + col * (VIEW_W + margin)
            y = sec_y + header_h + margin + row * (VIEW_H + label_h + margin)
            draw.text((x, y), label, fill=(80, 80, 80))
            img_y = y + label_h
            draw.rectangle(
                [x - 1, img_y - 1, x + VIEW_W, img_y + VIEW_H],
                outline=(0, 0, 0),
                width=1,
            )
            canvas.paste(img, (x, img_y))

    out = output_dir / "test_closeups.png"
    canvas.save(out, optimize=True, dpi=PNG_DPI)
    print(f"  closeups: {out} ({time.perf_counter() - t0:.1f}s)")
    return out


# ── Step 3 & 4: Joint sweep filmstrip with collision detection ──


def _build_sweep_model(bot_xml: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load bot.xml a second time and add invisible collision capsules to
    non-wheel bodies for self-collision detection during sweeps."""
    spec = mujoco.MjSpec.from_file(str(bot_xml))
    _configure_spec(spec)

    # Walk spec bodies, add collision geoms to each non-world body
    for body in spec.bodies:
        _add_collision_geoms_recursive(body)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def _add_collision_geoms_recursive(body):
    """Recursively add invisible collision capsules to bodies that have mesh geoms."""
    # Check if this body has any mesh geoms (visual geometry)
    has_mesh = False
    for geom in body.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            has_mesh = True
            break

    if has_mesh:
        # Check if this is a wheel body (has a continuous hinge — range==[0,0])
        is_wheel = False
        for joint in body.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                lo, hi = joint.range
                if lo == 0.0 and hi == 0.0:
                    is_wheel = True
                    break

        if not is_wheel:
            # Add an invisible collision box approximating the body
            g = body.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [0.02, 0.02, 0.02]  # conservative default
            g.contype = 2
            g.conaffinity = 2
            g.rgba = [1, 0, 0, 0]  # invisible
            g.group = 3  # hidden visual group

    # Recurse into child bodies
    for child_body in body.bodies:
        _add_collision_geoms_recursive(child_body)


def render_sweeps(bot_xml: Path, model_base: mujoco.MjModel, output_dir: Path) -> Path:
    """Render per-joint filmstrip across range of motion with collision detection."""
    t0 = time.perf_counter()

    # Build sweep model with collision geoms
    model, data = _build_sweep_model(bot_xml)
    renderer = mujoco.Renderer(model, SWEEP_W, SWEEP_H)

    sweep_joints = _sweepable_joints(model)

    if not sweep_joints:
        renderer.close()
        print("  sweeps: no sweepable joints found, skipping")
        return output_dir / "test_sweep.png"

    strips: list[tuple[str, list[Image.Image], list[str], list[bool]]] = []

    for jid in sweep_joints:
        jname = _joint_name(model, jid)
        child_bid = _child_body_id(model, jid)
        lo, hi = model.jnt_range[jid]
        qposadr = model.jnt_qposadr[jid]
        angles = np.linspace(lo, hi, SWEEP_FRAMES)

        frames = []
        frame_labels = []
        collisions = []

        for angle in angles:
            mujoco.mj_resetData(model, data)
            data.qpos[qposadr] = angle
            mujoco.mj_forward(model, data)

            # Camera follows child body
            child_pos = data.xpos[child_bid].copy()
            distance = model.stat.extent * 1.8

            img = _render(
                model,
                data,
                renderer,
                child_pos,
                distance,
                azimuth=135,
                elevation=-30,
            )
            frames.append(Image.fromarray(img))
            frame_labels.append(f"{math.degrees(angle):+.0f}°")

            # Check for self-collisions among contype=2 geoms
            has_collision = False
            for ci in range(data.ncon):
                contact = data.contact[ci]
                if contact.dist < 0:
                    g1_contype = model.geom_contype[contact.geom1]
                    g2_contype = model.geom_contype[contact.geom2]
                    if g1_contype == 2 and g2_contype == 2:
                        has_collision = True
                        break
            collisions.append(has_collision)

        strips.append((jname, frames, frame_labels, collisions))

        # Report collisions
        collision_angles = [frame_labels[i] for i, c in enumerate(collisions) if c]
        if collision_angles:
            print(f"  sweep {jname}: COLLISION at {', '.join(collision_angles)}")

    renderer.close()

    # Composite: one horizontal filmstrip per joint, stacked vertically
    margin = 8
    header_h = 28
    label_h = 20
    strip_h = header_h + SWEEP_H + label_h + margin
    border_w = 3  # red border width for collision frames

    canvas_w = SWEEP_W * SWEEP_FRAMES + margin * (SWEEP_FRAMES + 1)
    canvas_h = 30 + strip_h * len(strips) + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), f"Joint sweeps — {output_dir.name}", fill=(0, 0, 0))

    for strip_idx, (jname, frames, frame_labels, collisions) in enumerate(strips):
        lo, hi = model.jnt_range[sweep_joints[strip_idx]]
        sy = 30 + strip_idx * strip_h
        draw.text(
            (margin, sy + 4),
            f"{jname}  [{math.degrees(lo):+.0f}° .. {math.degrees(hi):+.0f}°]",
            fill=(0, 0, 0),
        )

        for fi, (frame, label, has_col) in enumerate(
            zip(frames, frame_labels, collisions)
        ):
            x = margin + fi * (SWEEP_W + margin)
            y = sy + header_h

            if has_col:
                # Red border for collision frames
                draw.rectangle(
                    [
                        x - border_w,
                        y - border_w,
                        x + SWEEP_W + border_w,
                        y + SWEEP_H + border_w,
                    ],
                    fill=(220, 40, 40),
                )
            else:
                # Black border for normal frames
                draw.rectangle(
                    [x - 1, y - 1, x + SWEEP_W, y + SWEEP_H],
                    outline=(0, 0, 0),
                    width=1,
                )

            canvas.paste(frame, (x, y))

            # Angle label below frame
            color = (220, 40, 40) if has_col else (80, 80, 80)
            draw.text((x, y + SWEEP_H + 2), label, fill=color)

    out = output_dir / "test_sweep.png"
    canvas.save(out, optimize=True, dpi=PNG_DPI)
    print(f"  sweeps: {out} ({time.perf_counter() - t0:.1f}s)")
    return out


# ── Pipeline entry point ──


def emit_renders(bot, output_dir: Path) -> None:
    """Generate validation renders for a bot.

    Loads bot.xml from output_dir (already generated by the MuJoCo emitter),
    runs the three render passes, and saves PNGs.
    """
    bot_xml = output_dir / "bot.xml"
    if not bot_xml.exists():
        print(f"  renders: skipping — {bot_xml} not found")
        return

    print("Renders:")
    t0 = time.perf_counter()

    model, data = _load_model(bot_xml)

    render_overview(model, data, output_dir)
    render_closeups(model, data, output_dir)
    render_sweeps(bot_xml, model, output_dir)

    print(f"  renders done ({time.perf_counter() - t0:.1f}s total)")


# ── Standalone ──


if __name__ == "__main__":
    bot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bots/wheeler_arm")
    bot_xml = bot_dir / "bot.xml"

    if not bot_xml.exists():
        print(f"Error: {bot_xml} not found")
        sys.exit(1)

    print(f"Validation renders: {bot_xml}")
    t0 = time.perf_counter()

    model, data = _load_model(bot_xml)

    render_overview(model, data, bot_dir)
    render_closeups(model, data, bot_dir)
    render_sweeps(bot_xml, model, bot_dir)

    print(f"Done ({time.perf_counter() - t0:.1f}s total)")
