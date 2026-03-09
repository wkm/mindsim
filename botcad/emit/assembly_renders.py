"""Assembly validation renders - per-joint insertion animations with collision detection.

Generates two PDFs:
  test_assembly.pdf   - Collision test: full filmstrip per joint, red/green borders
  assembly_visual.pdf - Assembly instructions: before/after per joint, IKEA-style

Both use IKEA-style rendering: solid target bodies, wireframe context, orange
insertion arrows. Camera perpendicular to insertion axis.

Called as part of Bot.emit() pipeline, or standalone:
    PYTHONPATH=. uv run python -m botcad.emit.assembly_renders [bot_dir]
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from fpdf import FPDF
from PIL import Image

from botcad.emit.render3d import white_background
from botcad.emit.renders import (
    _configure_spec,
    _joint_name,
    _servo_geom_id,
    _sweepable_joints,
)


class DeterministicFPDF(FPDF):
    """FPDF subclass with deterministic file identifier for git-friendly output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_file_id = None

    def set_custom_file_id(self, file_id: str):
        self._custom_file_id = file_id

    def file_id(self):
        if self._custom_file_id:
            import hashlib

            # PDF ID must be two 16-byte hex strings.
            # We hash the provided string and return it twice.
            h = hashlib.md5(self._custom_file_id.encode()).hexdigest().upper()
            return f"{h}{h}"
        return super().file_id()


# ── Config ──

FRAME_W, FRAME_H = 1200, 1200  # 2x resolution for crisp PDF embedding
DISPLACEMENTS_MM = [50, 35, 20, 10, 5, 2, 0]
BORDER_W_MM = 1.5
COLLISION_COLOR = (220, 40, 40)
CLEAR_COLOR = (40, 180, 40)
ARROW_RGBA = [1.0, 0.4, 0.0, 0.9]  # bright orange
CONTEXT_RGBA = [0.70, 0.72, 0.75, 1.0]  # light steel gray for wireframe
MINIMAP_SIZE = 400

# Geom name prefixes to hide in assembly renders (screws, mounting holes, wires)
_HARDWARE_PREFIXES = ("screw_", "horn_", "rear_", "mount_", "wire_")


@dataclass
class StripData:
    jname: str
    # Collision test (full visibility)
    s_frames: list = field(default_factory=list)
    s_labels: list = field(default_factory=list)
    s_cols: list = field(default_factory=list)
    a_frames: list = field(default_factory=list)
    a_labels: list = field(default_factory=list)
    a_cols: list = field(default_factory=list)
    # Instructions (progressive visibility)
    inst_s_frames: list = field(default_factory=list)
    inst_s_labels: list = field(default_factory=list)
    inst_a_frames: list = field(default_factory=list)
    inst_a_labels: list = field(default_factory=list)
    # Minimap
    minimap: Image.Image | None = None
    # Child body name (for human-readable step text)
    child_body_name: str = ""
    # Custom step text (defaults to servo/joint text if empty)
    step_a_title: str = ""
    step_b_title: str = ""


# ── Camera ──


def _camera_for_axis(joint_axis: np.ndarray) -> tuple[float, float]:
    """Choose camera angle based on joint axis orientation.

    Horizontal axes (X/Y): head-on view — camera looks along +axis so the
    arrow points at the viewer. Shows WHERE things attach clearly.

    Vertical axes (Z): perpendicular side view — camera is horizontal,
    looking from the side so the insertion gap is visible as the part
    slides down. Arrow appears as a clear vertical line.
    """
    ax, ay, az = joint_axis

    if abs(az) > 0.7:
        # Vertical axis (Z) — perpendicular side view
        # Camera horizontal, 3/4 angle, slight downward tilt
        return 135.0, -20.0

    # Horizontal axis (X/Y) — head-on view along the axis
    az_deg = math.degrees(math.atan2(ax, ay))
    return az_deg, -5.0  # slight downward tilt for depth


def _quat_z_to(axis: np.ndarray) -> np.ndarray:
    """Quaternion rotating +Z to the given unit vector."""
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    if np.allclose(a, z):
        return np.array([1, 0, 0, 0], dtype=float)
    if np.allclose(a, -z):
        return np.array([0, 1, 0, 0], dtype=float)
    c = np.cross(z, a)
    d = np.dot(z, a)
    q = np.array([1.0 + d, c[0], c[1], c[2]])
    return q / np.linalg.norm(q)


# ── Cel-shaded rendering (flat fill + wireframe context + edge outlines) ──


def _render_composite(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    lookat: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
    target_bids: set[int],
    visible_bids: set[int] | None = None,
    hide_arrows: bool = False,
    hide_geom_ids: set[int] | None = None,
    unhide_geom_ids: set[int] | None = None,
) -> np.ndarray:
    """Cel-shaded render: lit targets over wireframe context with edge outlines.

    Three-pass pipeline:
    1. Wireframe context (non-target bodies in light gray)
    2. Lit targets with actual geom colors (no segment mode)
    3. Segmentation pass → edge detection → black outlines

    visible_bids: if set, only these body IDs are rendered at all.
                  Bodies not in this set are hidden in all passes.
                  None means all bodies visible (original behavior).
    hide_arrows:  if True, arrow geoms are hidden in all passes.
    hide_geom_ids: if set, these specific geom IDs are hidden in all passes.
    unhide_geom_ids: if set, these geom IDs override hardware prefix hiding.
    """
    saved_rgba = model.geom_rgba.copy()

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    def _is_hidden(gid):
        bid = model.geom_bodyid[gid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if visible_bids is not None and bid not in visible_bids:
            return True
        if hide_arrows and name.startswith("_arrow_"):
            return True
        if hide_geom_ids is not None and gid in hide_geom_ids:
            return True
        # Allow explicit unhiding (overrides hardware prefix hiding)
        if unhide_geom_ids is not None and gid in unhide_geom_ids:
            return False
        # Hide hardware geoms (screws, mounting holes, wires)
        if any(name.startswith(p) for p in _HARDWARE_PREFIXES):
            return True
        return False

    # ── Pass 1: wireframe context (non-target bodies in light gray) ──
    for gid in range(model.ngeom):
        if _is_hidden(gid):
            model.geom_rgba[gid, 3] = 0
        elif model.geom_bodyid[gid] in target_bids:
            model.geom_rgba[gid, 3] = 0  # hide targets in wireframe pass
        elif (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith(
            "_arrow_"
        ):
            model.geom_rgba[gid, 3] = 0  # hide arrows in wireframe pass
        else:
            model.geom_rgba[gid] = CONTEXT_RGBA

    renderer.update_scene(data, camera=cam)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    img_wire = renderer.render().copy()
    white_background(img_wire)

    # ── Pass 2: flat-shaded target bodies + arrow geoms ──
    model.geom_rgba[:] = saved_rgba
    for gid in range(model.ngeom):
        if _is_hidden(gid):
            model.geom_rgba[gid, 3] = 0
        elif model.geom_bodyid[gid] not in target_bids and not (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        ).startswith("_arrow_"):
            model.geom_rgba[gid, 3] = 0  # hide context

    renderer.update_scene(data, camera=cam)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
    img_solid = renderer.render().copy()

    # ── Pass 3: segmentation for edge detection ──
    # Same geom visibility as pass 2 (targets only) — rgba unchanged
    renderer.update_scene(data, camera=cam)
    renderer.enable_segmentation_rendering()
    seg = renderer.render()  # (H, W, 2) int32: (geom_id, obj_type)
    renderer.disable_segmentation_rendering()

    # ── Composite: lit solid targets over wireframe context ──
    ids = seg[:, :, 0]
    solid_mask = ids >= 0  # pixels where target geometry is present
    result = img_wire.copy()
    result[solid_mask] = img_solid[solid_mask]

    # ── Edge outlines from segmentation boundaries ──
    # Find pixels where any neighbor has a different geom ID
    edges = np.zeros(ids.shape, dtype=bool)
    edges[:-1, :] |= ids[:-1, :] != ids[1:, :]
    edges[1:, :] |= ids[:-1, :] != ids[1:, :]
    edges[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    edges[:, 1:] |= ids[:, :-1] != ids[:, 1:]
    # Thicken to ~3px
    thick = edges.copy()
    thick[1:, :] |= edges[:-1, :]
    thick[:-1, :] |= edges[1:, :]
    thick[:, 1:] |= edges[:, :-1]
    thick[:, :-1] |= edges[:, 1:]
    # Paint edges on solid regions (includes silhouette inner edge)
    result[thick & solid_mask] = [40, 40, 40]

    model.geom_rgba[:] = saved_rgba
    return result


# ── Arrow geoms ──


def _add_arrow_geoms(parent_spec_body, joint_name: str, servo_pos, axis):
    """Add 3D arrow geoms to the parent body showing insertion direction.

    Arrow: capsule shaft + two angled capsules forming a chevron arrowhead.
    Positioned beside the servo, pointing along -axis (insertion direction).
    Sized to be clearly visible in renders (5mm radius shaft, 4mm arrowhead).
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    servo_pos = np.array(servo_pos, dtype=float)

    # Perpendicular offset so arrow doesn't overlap the servo body
    if abs(axis[0]) < 0.9:
        perp = np.cross(axis, [1, 0, 0])
    else:
        perp = np.cross(axis, [0, 1, 0])
    perp = perp / np.linalg.norm(perp)
    offset = perp * 0.040  # 40mm to the side

    # Arrow shaft: from 65mm above servo to 15mm above, along +axis
    shaft_start = servo_pos + axis * 0.065 + offset
    shaft_end = servo_pos + axis * 0.015 + offset
    shaft_center = (shaft_start + shaft_end) / 2
    shaft_half_len = np.linalg.norm(shaft_start - shaft_end) / 2

    shaft = parent_spec_body.add_geom()
    shaft.name = f"_arrow_shaft_{joint_name}"
    shaft.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    shaft.size = [0.005, shaft_half_len, 0]
    shaft.pos = shaft_center
    shaft.quat = _quat_z_to(axis)
    shaft.rgba = ARROW_RGBA
    shaft.contype = 0
    shaft.conaffinity = 0
    shaft.group = 0

    # Arrowhead: two angled capsules forming ">"
    tip = servo_pos + axis * 0.010 + offset  # just past shaft end
    head_len = 0.018

    # Second perpendicular for the other branch dimension
    perp2 = np.cross(axis, perp)
    perp2 = perp2 / np.linalg.norm(perp2)

    for i, sign in enumerate([1, -1]):
        branch_end = tip + axis * head_len + perp2 * sign * 0.015
        branch_center = (tip + branch_end) / 2
        branch_half = np.linalg.norm(tip - branch_end) / 2
        branch_dir = branch_end - tip
        branch_dir = branch_dir / np.linalg.norm(branch_dir)

        g = parent_spec_body.add_geom()
        g.name = f"_arrow_head{i}_{joint_name}"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [0.004, branch_half, 0]
        g.pos = branch_center
        g.quat = _quat_z_to(branch_dir)
        g.rgba = ARROW_RGBA
        g.contype = 0
        g.conaffinity = 0
        g.group = 0


# ── Model building ──


def _build_assembly_model(
    bot_xml: Path, joint_name: str
) -> tuple[mujoco.MjModel, mujoco.MjData, dict] | None:
    """Load bot.xml, add slide joint + arrow geoms, compile.

    Returns (model, data, info_dict) or None if joint not found.
    Collision detection is handled by trimesh, not MuJoCo.
    """
    spec = mujoco.MjSpec.from_file(str(bot_xml))
    _configure_spec(spec)
    # Bump offscreen buffer for higher resolution
    spec.visual.global_.offwidth = FRAME_W * 2
    spec.visual.global_.offheight = FRAME_H * 2

    # Find target joint's child body AND its parent in the spec tree
    target_body = None
    target_joint = None
    parent_body = None

    def _find_joint(body, name, parent=None):
        nonlocal target_body, target_joint, parent_body
        for j in body.joints:
            if j.name == name:
                target_body = body
                target_joint = j
                parent_body = parent
                return True
        for child in body.bodies:
            if _find_joint(child, name, body):
                return True
        return False

    for top_body in spec.bodies:
        if _find_joint(top_body, joint_name, None):
            break

    if target_body is None or target_joint is None:
        return None

    # Slide joint for animating child body displacement
    slide_name = f"_assembly_slide_{joint_name}"
    slide = target_body.add_joint()
    slide.name = slide_name
    slide.type = mujoco.mjtJoint.mjJNT_SLIDE
    slide.axis = target_joint.axis.copy()
    slide.range = [0, 0.06]
    slide.limited = False

    # Arrow geoms on parent body showing insertion direction
    servo_geom_name = f"{joint_name}_servo"
    if parent_body is not None:
        servo_local_pos = None
        for geom in parent_body.geoms:
            if geom.name == servo_geom_name:
                servo_local_pos = np.array(geom.pos)
                break
        if servo_local_pos is not None:
            _add_arrow_geoms(
                parent_body, joint_name, servo_local_pos, target_joint.axis
            )

    model = spec.compile()
    data = mujoco.MjData(model)

    child_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body.name)
    parent_bid = model.body_parentid[child_bid]
    servo_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, servo_geom_name)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    joint_axis = model.jnt_axis[jid].copy()

    # Collect horn geom IDs for this joint so they can be unhidden
    # during assembly instruction renders (horn_ is in _HARDWARE_PREFIXES).
    # Split into parent-body (horn mounting points that travel with the servo)
    # and child-body (horn disc that's part of the child).
    horn_gids = set()
    servo_companion_gids = set()  # geoms on parent body that move with the servo
    horn_prefix = f"horn_{joint_name}_"
    horn_disc_name = f"{joint_name}_horn_disc"
    servo_boss_name = f"{joint_name}_servo_boss"
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname.startswith(horn_prefix):
            horn_gids.add(gid)
            if model.geom_bodyid[gid] == parent_bid:
                servo_companion_gids.add(gid)
        elif gname == horn_disc_name:
            horn_gids.add(gid)
        elif gname == servo_boss_name:
            servo_companion_gids.add(gid)

    info = {
        "slide_jnt_name": slide_name,
        "servo_geom_name": servo_geom_name,
        "child_body_name": target_body.name,
        "joint_axis": joint_axis,
        "child_bid": child_bid,
        "parent_bid": parent_bid,
        "servo_gid": servo_gid,
        "joint_name": joint_name,
        "horn_gids": horn_gids,
        "servo_companion_gids": servo_companion_gids,
    }
    return model, data, info


# ── Trimesh collision detection ──


def _check_box_geom_collision(
    meshes_dir: Path, model, data, geom_id: int, parent_bid: int, mesh_cache: dict
) -> bool:
    """Check if a box geom intersects parent body mesh using trimesh.

    Shared by servo insertion and component insertion.
    Handles concave meshes correctly (bracket pockets are respected).
    """
    import trimesh

    parent_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_bid)
    stl_path = meshes_dir / f"{parent_body_name}.stl"
    if not stl_path.exists():
        return False

    if str(stl_path) not in mesh_cache:
        mesh_cache[str(stl_path)] = trimesh.load(stl_path, force="mesh")
    parent_mesh = mesh_cache[str(stl_path)]

    geom_size = model.geom_size[geom_id]  # half-extents
    geom_pos_world = data.geom_xpos[geom_id]
    geom_mat_world = data.geom_xmat[geom_id].reshape(3, 3)

    parent_pos_world = data.xpos[parent_bid]
    parent_mat_world = data.xmat[parent_bid].reshape(3, 3)

    parent_mat_inv = parent_mat_world.T
    geom_pos_local = parent_mat_inv @ (geom_pos_world - parent_pos_world)
    geom_mat_local = parent_mat_inv @ geom_mat_world

    geom_box = trimesh.creation.box(extents=geom_size * 2)
    T = np.eye(4)
    T[:3, :3] = geom_mat_local
    T[:3, 3] = geom_pos_local
    geom_box.apply_transform(T)

    manager = trimesh.collision.CollisionManager()
    manager.add_object("parent", parent_mesh)
    return manager.in_collision_single(geom_box)


def _check_servo_collision(
    meshes_dir: Path, info: dict, model, data, mesh_cache: dict
) -> bool:
    return _check_box_geom_collision(
        meshes_dir, model, data, info["servo_gid"], info["parent_bid"], mesh_cache
    )


def _check_child_collision(
    meshes_dir: Path, info: dict, model, data, mesh_cache: dict
) -> bool:
    """Check if child body mesh intersects parent body mesh using trimesh."""
    import trimesh

    parent_bid = info["parent_bid"]
    child_bid = info["child_bid"]

    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_bid)
    child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, child_bid)

    parent_stl = meshes_dir / f"{parent_name}.stl"
    child_stl = meshes_dir / f"{child_name}.stl"
    if not parent_stl.exists() or not child_stl.exists():
        return False

    if str(parent_stl) not in mesh_cache:
        mesh_cache[str(parent_stl)] = trimesh.load(parent_stl, force="mesh")
    if str(child_stl) not in mesh_cache:
        mesh_cache[str(child_stl)] = trimesh.load(child_stl, force="mesh")

    # Copy meshes before transforming (don't mutate cache)
    parent_mesh = mesh_cache[str(parent_stl)].copy()
    child_mesh = mesh_cache[str(child_stl)].copy()

    # Transform both meshes to world frame using MuJoCo body poses
    for bid, mesh in [(parent_bid, parent_mesh), (child_bid, child_mesh)]:
        T = np.eye(4)
        T[:3, :3] = data.xmat[bid].reshape(3, 3)
        T[:3, 3] = data.xpos[bid]
        mesh.apply_transform(T)

    # Offset child 0.5mm along joint axis to ignore flush-face contacts
    # at the designed interface (e.g., turntable top touching tube bottom).
    # This only eliminates surface-contact false positives; real penetrations
    # (> 0.5mm) are still caught.
    child_mesh.apply_translation(info["joint_axis"] * 0.0005)

    manager = trimesh.collision.CollisionManager()
    manager.add_object("parent", parent_mesh)
    return manager.in_collision_single(child_mesh)


# ── Per-joint animation rendering ──


def _render_servo_insertion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: dict,
    renderer: mujoco.Renderer,
    meshes_dir: Path,
    mesh_cache: dict,
    visible_bids: set[int] | None = None,
    hide_geom_ids: set[int] | None = None,
    unhide_geom_ids: set[int] | None = None,
) -> tuple[list[Image.Image], list[str], list[bool]]:
    """Servo slides along +joint_axis into parent body bracket."""
    servo_gid = info["servo_gid"]
    parent_bid = info["parent_bid"]
    child_bid = info["child_bid"]
    axis = info["joint_axis"]
    azimuth, elevation = _camera_for_axis(axis)

    # Save original positions for servo + companion geoms (horn mounting points)
    companion_gids = info.get("servo_companion_gids", set())
    moving_gids = {servo_gid} | companion_gids
    orig_positions = {gid: model.geom_pos[gid].copy() for gid in moving_gids}

    frames, labels, collisions = [], [], []

    for disp_mm in DISPLACEMENTS_MM:
        disp_m = disp_mm / 1000.0
        for gid in moving_gids:
            model.geom_pos[gid] = orig_positions[gid] + axis * disp_m
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Tight zoom on servo area
        lookat = data.geom_xpos[servo_gid].copy()
        servo_half = model.geom_size[servo_gid]
        distance = max(np.max(servo_half) * 12, 0.10)

        img = _render_composite(
            model,
            data,
            renderer,
            lookat,
            distance,
            azimuth,
            elevation,
            {parent_bid, child_bid},
            visible_bids=visible_bids,
            hide_geom_ids=hide_geom_ids,
            unhide_geom_ids=unhide_geom_ids,
        )
        frames.append(Image.fromarray(img))
        labels.append(f"{disp_mm}mm")
        collisions.append(
            _check_servo_collision(meshes_dir, info, model, data, mesh_cache)
        )

    for gid, pos in orig_positions.items():
        model.geom_pos[gid] = pos
    return frames, labels, collisions


def _render_child_attachment(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: dict,
    renderer: mujoco.Renderer,
    meshes_dir: Path,
    mesh_cache: dict,
    visible_bids: set[int] | None = None,
    hide_geom_ids: set[int] | None = None,
    unhide_geom_ids: set[int] | None = None,
) -> tuple[list[Image.Image], list[str], list[bool]]:
    """Child body slides along +joint_axis onto servo horn via slide joint."""
    slide_jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, info["slide_jnt_name"]
    )
    slide_qadr = model.jnt_qposadr[slide_jid]
    parent_bid = info["parent_bid"]
    child_bid = info["child_bid"]
    servo_gid = info["servo_gid"]
    axis = info["joint_axis"]
    azimuth, elevation = _camera_for_axis(axis)

    frames, labels, collisions = [], [], []

    for disp_mm in DISPLACEMENTS_MM:
        disp_m = disp_mm / 1000.0
        mujoco.mj_resetData(model, data)
        data.qpos[slide_qadr] = disp_m
        mujoco.mj_forward(model, data)

        servo_pos = data.geom_xpos[servo_gid].copy()
        child_pos = data.xpos[child_bid].copy()
        lookat = (servo_pos + child_pos) / 2.0
        servo_half = model.geom_size[servo_gid]
        distance = max(np.max(servo_half) * 14, 0.12)

        img = _render_composite(
            model,
            data,
            renderer,
            lookat,
            distance,
            azimuth,
            elevation,
            {parent_bid, child_bid},
            visible_bids=visible_bids,
            hide_geom_ids=hide_geom_ids,
            unhide_geom_ids=unhide_geom_ids,
        )
        frames.append(Image.fromarray(img))
        labels.append(f"{disp_mm}mm")
        collisions.append(
            _check_child_collision(meshes_dir, info, model, data, mesh_cache)
        )

    return frames, labels, collisions


# ── Minimap ──


def _render_minimap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: dict,
    renderer: mujoco.Renderer,
) -> Image.Image:
    """Full-robot thumbnail with current joint's bodies highlighted solid.

    Fixed 3/4 view, all bodies visible as wireframe, parent+child solid.
    Arrow geoms hidden. Rendered at full res then downscaled.
    """
    parent_bid = info["parent_bid"]
    child_bid = info["child_bid"]

    # Reset to home pose for consistent minimap
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    lookat = model.stat.center.copy()
    distance = model.stat.extent * 2.5

    img = _render_composite(
        model,
        data,
        renderer,
        lookat,
        distance,
        azimuth=135.0,
        elevation=-30.0,
        target_bids={parent_bid, child_bid},
        visible_bids=None,  # all bodies visible
        hide_arrows=True,
    )
    pil = Image.fromarray(img)
    return pil.resize((MINIMAP_SIZE, MINIMAP_SIZE), Image.LANCZOS)


# ── Assembly order ──


def _assembly_order(model: mujoco.MjModel) -> list[int]:
    """Sweepable joints in tree order, then wheel joints."""
    sweepable = _sweepable_joints(model)
    wheels = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        lo, hi = model.jnt_range[jid]
        if lo == 0.0 and hi == 0.0:
            wheels.append(jid)
    return sweepable + wheels


# ── PDF compositing: collision test (full filmstrips) ──

MARGIN = 12


def _composite_collision_pdf(strips, output_dir: Path) -> Path:
    """Full filmstrip per joint with red/green collision borders. Letter portrait."""
    n_frames = len(DISPLACEMENTS_MM)
    page_w = 215.9  # Letter portrait
    usable_w = page_w - 2 * MARGIN
    gap = 3
    frame_w = (usable_w - (n_frames - 1) * gap) / n_frames
    frame_h = frame_w
    row_label_h = 8
    row_spacing = 6

    pdf = DeterministicFPDF(orientation="P", unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=False)

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 0, 0)
    pdf.text(MARGIN, MARGIN + 16, f"Collision Test - {output_dir.name}")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.text(
        MARGIN, MARGIN + 26, f"{len(strips)} joints | {n_frames} steps per animation"
    )

    ly = MARGIN + 38
    pdf.set_fill_color(*CLEAR_COLOR)
    pdf.rect(MARGIN, ly - 3, 8, 4, "F")
    pdf.set_text_color(80, 80, 80)
    pdf.text(MARGIN + 10, ly, "Clear")
    pdf.set_fill_color(*COLLISION_COLOR)
    pdf.rect(MARGIN + 40, ly - 3, 8, 4, "F")
    pdf.text(MARGIN + 50, ly, "Collision")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    toc_y = ly + 14
    for i, strip in enumerate(strips):
        pdf.text(MARGIN + 4, toc_y, f"Step {i + 1}: {strip.jname}")
        toc_y += 6

    # Per-joint pages
    with tempfile.TemporaryDirectory(prefix="botcad_col_") as tmpdir:
        tmp = Path(tmpdir)
        for sec_idx, strip in enumerate(strips):
            jname = strip.jname
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(0, 0, 0)
            pdf.text(MARGIN, MARGIN + 8, f"Step {sec_idx + 1}: {jname}")
            y = MARGIN + 14

            for row_idx, (frames, labels, cols, title) in enumerate(
                [
                    (strip.s_frames, strip.s_labels, strip.s_cols, "Servo insertion"),
                    (strip.a_frames, strip.a_labels, strip.a_cols, "Child attachment"),
                ]
            ):
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(80, 80, 80)
                pdf.text(MARGIN, y + 4, title)
                y += row_label_h

                for fi, (frame, label, has_col) in enumerate(zip(frames, labels, cols)):
                    x = MARGIN + fi * (frame_w + gap)
                    p = tmp / f"c_{jname}_{row_idx}_{fi}.png"
                    frame.save(str(p))

                    pdf.set_fill_color(*(COLLISION_COLOR if has_col else CLEAR_COLOR))
                    pdf.rect(
                        x - BORDER_W_MM,
                        y - BORDER_W_MM,
                        frame_w + 2 * BORDER_W_MM,
                        frame_h + 2 * BORDER_W_MM,
                        "F",
                    )
                    pdf.image(str(p), x, y, frame_w, frame_h)

                    pdf.set_text_color(*(COLLISION_COLOR if has_col else (80, 80, 80)))
                    pdf.set_font("Helvetica", "", 6)
                    pdf.text(x, y + frame_h + 3, label)

                y += frame_h + row_spacing + 4

    out = output_dir / "test_assembly.pdf"
    from datetime import datetime

    pdf.set_creation_date(datetime(2026, 1, 1))
    pdf.set_custom_file_id(f"static-id-{out.name}")
    pdf.output(str(out))
    return out


# ── Overview + component rendering ──


def _render_overview_frame(bot_xml: Path) -> Image.Image:
    """Render completed assembly for the overview page."""
    spec = mujoco.MjSpec.from_file(str(bot_xml))
    _configure_spec(spec)
    spec.visual.global_.offwidth = FRAME_W * 2
    spec.visual.global_.offheight = FRAME_H * 2
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, FRAME_W, FRAME_H)
    all_bids = set(range(1, model.nbody))
    lookat = model.stat.center.copy()
    distance = model.stat.extent * 2.5

    img = _render_composite(
        model,
        data,
        renderer,
        lookat,
        distance,
        azimuth=135.0,
        elevation=-30.0,
        target_bids=all_bids,
    )
    renderer.close()
    return Image.fromarray(img)


def _build_component_model(
    bot_xml: Path,
    comp_geom_name: str,
    insertion_axis_override: tuple[float, float, float] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, dict] | None:
    """Load bot.xml, add arrow geoms for component insertion, compile.

    Returns (model, data, info_dict) or None if comp geom not found.
    """
    spec = mujoco.MjSpec.from_file(str(bot_xml))
    _configure_spec(spec)
    spec.visual.global_.offwidth = FRAME_W * 2
    spec.visual.global_.offheight = FRAME_H * 2

    # Find the comp geom's parent body in the spec tree
    parent_body = None
    comp_geom = None

    def _find_comp(body, name):
        nonlocal parent_body, comp_geom
        for g in body.geoms:
            if g.name == name:
                parent_body = body
                comp_geom = g
                return True
        for child in body.bodies:
            if _find_comp(child, name):
                return True
        return False

    for top_body in spec.bodies:
        if _find_comp(top_body, comp_geom_name):
            break

    if parent_body is None or comp_geom is None:
        return None

    # Insertion axis: use resolved axis from Bot, or default to +Z
    if insertion_axis_override is not None:
        insertion_axis = np.array(insertion_axis_override, dtype=float)
    else:
        insertion_axis = np.array([0.0, 0.0, 1.0])

    # Add arrow geoms showing insertion direction
    _add_arrow_geoms(parent_body, comp_geom_name, comp_geom.pos, insertion_axis)

    model = spec.compile()
    data = mujoco.MjData(model)

    comp_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, comp_geom_name)
    parent_bid = model.geom_bodyid[comp_gid]

    # Find associated mount_* geoms (fastener positions)
    # comp_base_pi → mount_base_pi_m1, mount_base_pi_m2, ...
    mount_prefix = comp_geom_name.replace("comp_", "mount_") + "_m"
    mount_gids = set()
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname.startswith(mount_prefix):
            mount_gids.add(gid)

    # Collect servo geom IDs and sibling comp_ geom IDs on the parent body
    # so callers can hide them during component insertion steps.
    servo_gids = set()
    sibling_comp_gids = set()
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != parent_bid:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname.endswith("_servo") or gname.endswith("_servo_boss"):
            servo_gids.add(gid)
        elif gname.startswith("comp_") and gid != comp_gid:
            sibling_comp_gids.add(gid)

    info = {
        "comp_gid": comp_gid,
        "comp_geom_name": comp_geom_name,
        "parent_bid": parent_bid,
        "insertion_axis": insertion_axis,
        "mount_gids": mount_gids,
        "servo_gids": servo_gids,
        "sibling_comp_gids": sibling_comp_gids,
    }
    return model, data, info


def _render_component_insertion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: dict,
    renderer: mujoco.Renderer,
    meshes_dir: Path,
    mesh_cache: dict,
    visible_bids: set[int] | None = None,
    hide_geom_ids: set[int] | None = None,
) -> tuple[list[Image.Image], list[str], list[bool]]:
    """Component slides along insertion axis into parent body pocket."""
    comp_gid = info["comp_gid"]
    parent_bid = info["parent_bid"]
    axis = info["insertion_axis"]
    azimuth, elevation = _camera_for_axis(axis)

    orig_pos = model.geom_pos[comp_gid].copy()
    frames, labels, collisions = [], [], []

    for disp_mm in DISPLACEMENTS_MM:
        disp_m = disp_mm / 1000.0
        model.geom_pos[comp_gid] = orig_pos + axis * disp_m
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        lookat = data.geom_xpos[comp_gid].copy()
        comp_half = model.geom_size[comp_gid]
        distance = max(np.max(comp_half) * 14, 0.10)

        img = _render_composite(
            model,
            data,
            renderer,
            lookat,
            distance,
            azimuth,
            elevation,
            {parent_bid},
            visible_bids=visible_bids,
            hide_geom_ids=hide_geom_ids,
        )
        frames.append(Image.fromarray(img))
        labels.append(f"{disp_mm}mm")
        collisions.append(
            _check_box_geom_collision(
                meshes_dir, model, data, comp_gid, parent_bid, mesh_cache
            )
        )

    model.geom_pos[comp_gid] = orig_pos
    return frames, labels, collisions


def _render_fastener_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: dict,
    renderer: mujoco.Renderer,
    visible_bids: set[int] | None = None,
    hide_geom_ids: set[int] | None = None,
) -> tuple[list[Image.Image], list[str], list[bool]]:
    """Before/after showing mount screws appearing around the component."""
    comp_gid = info["comp_gid"]
    parent_bid = info["parent_bid"]
    mount_gids = info["mount_gids"]
    axis = info["insertion_axis"]
    azimuth, elevation = _camera_for_axis(axis)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    lookat = data.geom_xpos[comp_gid].copy()
    comp_half = model.geom_size[comp_gid]
    distance = max(np.max(comp_half) * 14, 0.10)

    frames, labels, collisions = [], [], []

    # "Before": component seated, screws hidden (default hardware hiding)
    for disp_mm in DISPLACEMENTS_MM:
        img = _render_composite(
            model,
            data,
            renderer,
            lookat,
            distance,
            azimuth,
            elevation,
            {parent_bid},
            visible_bids=visible_bids,
            hide_arrows=True,
            hide_geom_ids=hide_geom_ids,
        )
        frames.append(Image.fromarray(img))
        labels.append(f"{disp_mm}mm")
        collisions.append(False)

    # Replace last frame with screws visible ("after" state)
    if mount_gids:
        img_fastened = _render_composite(
            model,
            data,
            renderer,
            lookat,
            distance,
            azimuth,
            elevation,
            {parent_bid},
            visible_bids=visible_bids,
            hide_arrows=True,
            hide_geom_ids=hide_geom_ids,
            unhide_geom_ids=mount_gids,
        )
        frames[-1] = Image.fromarray(img_fastened)

    return frames, labels, collisions


def _render_component_strips(
    bot_xml: Path, comp_axes: dict[str, tuple[float, float, float]] | None = None
) -> list[StripData]:
    """Render component insertion strips — one per comp_* geom, same format as joint strips."""
    spec = mujoco.MjSpec.from_file(str(bot_xml))
    _configure_spec(spec)
    base_model = spec.compile()

    # Find comp_* geoms
    comp_geoms: list[tuple[str, str, int]] = []  # (geom_name, label, parent_bid)
    for gid in range(base_model.ngeom):
        name = mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("comp_"):
            bid = base_model.geom_bodyid[gid]
            parts = name.split("_", 2)
            label = parts[2] if len(parts) > 2 else name
            comp_geoms.append((name, label, bid))

    if not comp_geoms:
        return []

    meshes_dir = bot_xml.parent / "meshes"
    mesh_cache: dict = {}
    strips = []

    for comp_name, comp_label, _ in comp_geoms:
        axis_override = comp_axes.get(comp_name) if comp_axes else None
        result = _build_component_model(
            bot_xml, comp_name, insertion_axis_override=axis_override
        )
        if result is None:
            print(f"  assembly: could not build model for '{comp_name}', skipping")
            continue

        model, data, info = result
        parent_bid = info["parent_bid"]
        renderer = mujoco.Renderer(model, FRAME_W, FRAME_H)

        # Collision test renders (all bodies visible)
        s_frames, s_labels, s_cols = _render_component_insertion(
            model,
            data,
            info,
            renderer,
            meshes_dir,
            mesh_cache,
        )
        a_frames, a_labels, a_cols = _render_fastener_step(
            model,
            data,
            info,
            renderer,
        )

        # Instruction renders (only parent body visible — zoomed in)
        # Hide all servo geoms and sibling component geoms — components are
        # assembled before servos, and each component is its own step.
        comp_hide = info["servo_gids"] | info["sibling_comp_gids"]
        inst_s_frames, inst_s_labels, _ = _render_component_insertion(
            model,
            data,
            info,
            renderer,
            meshes_dir,
            mesh_cache,
            visible_bids={parent_bid},
            hide_geom_ids=comp_hide,
        )
        inst_a_frames, inst_a_labels, _ = _render_fastener_step(
            model,
            data,
            info,
            renderer,
            visible_bids={parent_bid},
            hide_geom_ids=comp_hide,
        )

        renderer.close()

        has_fasteners = bool(info["mount_gids"])
        strip = StripData(
            jname=comp_label,
            s_frames=s_frames,
            s_labels=s_labels,
            s_cols=s_cols,
            a_frames=a_frames,
            a_labels=a_labels,
            a_cols=a_cols,
            inst_s_frames=inst_s_frames,
            inst_s_labels=inst_s_labels,
            inst_a_frames=inst_a_frames,
            inst_a_labels=inst_a_labels,
            step_a_title=f"Insert {comp_label} into pocket",
            step_b_title="Fasten with screws" if has_fasteners else "Seat component",
        )
        strips.append(strip)

        hits = [s_labels[i] for i, c in enumerate(s_cols) if c]
        if hits:
            print(f"  assembly {comp_label} insertion: COLLISION at {', '.join(hits)}")

    return strips


# ── PDF compositing: assembly instructions ──


def _draw_before_after(pdf, tmp, prefix, before, after, frame_w, frame_h, gap, y):
    """Draw a before/after frame pair with arrow between them."""
    p_before = tmp / f"{prefix}_before.png"
    p_after = tmp / f"{prefix}_after.png"
    before.save(str(p_before))
    after.save(str(p_after))

    x1 = MARGIN
    x2 = MARGIN + frame_w + gap

    pdf.set_draw_color(200, 200, 200)
    pdf.rect(x1 - 0.5, y - 0.5, frame_w + 1, frame_h + 1, "D")
    pdf.rect(x2 - 0.5, y - 0.5, frame_w + 1, frame_h + 1, "D")

    pdf.image(str(p_before), x1, y, frame_w, frame_h)
    pdf.image(str(p_after), x2, y, frame_w, frame_h)

    # Arrow between frames
    arrow_y = y + frame_h / 2
    arrow_x1 = x1 + frame_w + 1.5
    arrow_x2 = x2 - 1.5
    pdf.set_draw_color(100, 100, 100)
    pdf.set_line_width(0.4)
    pdf.line(arrow_x1, arrow_y, arrow_x2, arrow_y)
    pdf.line(arrow_x2, arrow_y, arrow_x2 - 2, arrow_y - 1.5)
    pdf.line(arrow_x2, arrow_y, arrow_x2 - 2, arrow_y + 1.5)


def _composite_instructions_pdf(
    strips: list[StripData],
    output_dir: Path,
    overview_frame: Image.Image | None = None,
) -> Path:
    """IKEA-style assembly instructions.

    Page 1: Completed assembly overview.
    Then: Per-step pages — each strip gets one page with two rows (a/b).
    Component strips use step_a_title/step_b_title for text.
    Joint strips default to "Insert servo into bracket" / "Attach {child} onto horn".

    Letter portrait, large frames. Progressive visibility for joint steps.
    Minimap in top-right corner for spatial orientation.
    """
    page_w, page_h = 215.9, 279.4  # Letter portrait
    usable_w = page_w - 2 * MARGIN
    gap = 8
    frame_w = (usable_w - gap) / 2  # two frames side by side
    frame_h = frame_w  # 1:1
    # Verify: header(16) + 2*(label(7) + frame + spacing(6)) + margin fits in page_h
    total_h = MARGIN + 16 + 2 * (7 + frame_h + 6) + MARGIN
    if total_h > page_h:
        # Scale down frames to fit
        avail = page_h - MARGIN * 2 - 16 - 2 * (7 + 6)
        frame_h = avail / 2
        frame_w = frame_h

    minimap_mm = 35  # minimap size in mm

    pdf = DeterministicFPDF(orientation="P", unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=False)

    step_num = 0  # global step counter

    with tempfile.TemporaryDirectory(prefix="botcad_inst_") as tmpdir:
        tmp = Path(tmpdir)

        # ── Page 1: Completed assembly overview ──
        if overview_frame is not None:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(0, 0, 0)
            pdf.text(MARGIN, MARGIN + 10, output_dir.name)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(80, 80, 80)
            pdf.text(MARGIN, MARGIN + 18, "Completed assembly")

            # Large centered overview image
            overview_size = min(usable_w, page_h - MARGIN * 2 - 30)
            overview_x = MARGIN + (usable_w - overview_size) / 2
            overview_y = MARGIN + 26

            p_overview = tmp / "overview.png"
            overview_frame.save(str(p_overview))
            pdf.set_draw_color(200, 200, 200)
            pdf.rect(
                overview_x - 0.5,
                overview_y - 0.5,
                overview_size + 1,
                overview_size + 1,
                "D",
            )
            pdf.image(
                str(p_overview), overview_x, overview_y, overview_size, overview_size
            )

        # ── Per-step pages (unified: components + joints) ──
        for strip in strips:
            step_num += 1
            pdf.add_page()

            # Step header
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(0, 0, 0)
            pdf.text(MARGIN, MARGIN + 8, f"Step {step_num}")
            pdf.set_font("Helvetica", "", 12)
            pdf.set_text_color(80, 80, 80)
            pdf.text(MARGIN + 40, MARGIN + 8, strip.jname.replace("_", " "))

            # Minimap (top-right corner)
            if strip.minimap is not None:
                minimap_x = page_w - MARGIN - minimap_mm
                minimap_y = MARGIN
                p_minimap = tmp / f"minimap_{strip.jname}.png"
                strip.minimap.save(str(p_minimap))
                pdf.set_draw_color(200, 200, 200)
                pdf.rect(
                    minimap_x - 0.5,
                    minimap_y - 0.5,
                    minimap_mm + 1,
                    minimap_mm + 1,
                    "D",
                )
                pdf.image(str(p_minimap), minimap_x, minimap_y, minimap_mm, minimap_mm)

            y = MARGIN + 16

            # Resolve step titles — use custom if set, else default joint text
            child_label = strip.child_body_name.replace("_", " ")
            title_a = strip.step_a_title or "Insert servo into bracket"
            title_b = strip.step_b_title or f"Attach {child_label} onto horn"

            for row_idx, (frames, labels, title) in enumerate(
                [
                    (strip.inst_s_frames, strip.inst_s_labels, title_a),
                    (strip.inst_a_frames, strip.inst_a_labels, title_b),
                ]
            ):
                if not frames:
                    continue

                # Row label
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(60, 60, 60)
                sub = chr(ord("a") + row_idx)
                pdf.text(MARGIN, y + 4, f"{step_num}{sub}. {title}")
                y += 7

                _draw_before_after(
                    pdf,
                    tmp,
                    f"i_{strip.jname}_{row_idx}",
                    frames[0],
                    frames[-1],
                    frame_w,
                    frame_h,
                    gap,
                    y,
                )
                y += frame_h + 6

    out = output_dir / "assembly_visual.pdf"
    from datetime import datetime

    pdf.set_creation_date(datetime(2026, 1, 1))
    pdf.set_custom_file_id(f"static-id-{out.name}")
    pdf.output(str(out))
    return out


# ── Pipeline entry point ──


def _render_strips(bot_xml: Path):
    """Render all joints, return StripData list for both PDFs.

    Collision renders use full visibility (all bodies).
    Instruction renders use progressive visibility (only assembled bodies).
    """
    spec = mujoco.MjSpec.from_file(str(bot_xml))
    _configure_spec(spec)
    base_model = spec.compile()

    joint_ids = _assembly_order(base_model)
    if not joint_ids:
        return []

    # Pre-compute joint -> child body name from base model for progressive tracking
    joint_child_names = {}
    for jid in joint_ids:
        jname = _joint_name(base_model, jid)
        # Child body is the body that owns this joint
        child_bid = base_model.jnt_bodyid[jid]
        child_name = mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_BODY, child_bid)
        joint_child_names[jname] = child_name

    # Find root body (first body with parent=world, i.e. body 1)
    root_name = mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_BODY, 1)
    assembled_body_names = {root_name}
    assembled_joint_names: set[str] = set()

    meshes_dir = bot_xml.parent / "meshes"
    mesh_cache: dict = {}
    strips = []
    for jid in joint_ids:
        jname = _joint_name(base_model, jid)
        sgid = _servo_geom_id(base_model, jname)
        if sgid < 0:
            print(f"  assembly: no servo geom for '{jname}', skipping")
            continue

        result = _build_assembly_model(bot_xml, jname)
        if result is None:
            print(f"  assembly: could not build model for '{jname}', skipping")
            continue

        model, data, info = result
        child_name = joint_child_names[jname]
        child_bid = info["child_bid"]
        renderer = mujoco.Renderer(model, FRAME_W, FRAME_H)

        # Translate assembled body names -> body IDs in this per-joint model
        assembled_bids = set()
        for bname in assembled_body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                assembled_bids.add(bid)

        # Collision renders (all bodies visible, original behavior)
        s_frames, s_labels, s_cols = _render_servo_insertion(
            model, data, info, renderer, meshes_dir, mesh_cache
        )
        a_frames, a_labels, a_cols = _render_child_attachment(
            model, data, info, renderer, meshes_dir, mesh_cache
        )

        # Instruction renders (progressive visibility)
        # Hide servo geoms for joints not yet assembled
        unassembled_servo_gids = set()
        for gid in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if gname.endswith("_servo") and gname != f"{jname}_servo":
                joint_for_servo = gname[: -len("_servo")]
                if joint_for_servo not in assembled_joint_names:
                    unassembled_servo_gids.add(gid)
            elif gname.endswith("_servo_boss") and gname != f"{jname}_servo_boss":
                joint_for_boss = gname[: -len("_servo_boss")]
                if joint_for_boss not in assembled_joint_names:
                    unassembled_servo_gids.add(gid)

        # Unhide horn geoms for the current joint (overrides _HARDWARE_PREFIXES)
        horn_gids = info.get("horn_gids", set())

        # Servo insertion: only assembled bodies visible (child not yet attached)
        inst_s_frames, inst_s_labels, _ = _render_servo_insertion(
            model,
            data,
            info,
            renderer,
            meshes_dir,
            mesh_cache,
            visible_bids=assembled_bids,
            hide_geom_ids=unassembled_servo_gids,
            unhide_geom_ids=horn_gids,
        )
        # Child attachment: assembled bodies + child now visible
        inst_a_frames, inst_a_labels, _ = _render_child_attachment(
            model,
            data,
            info,
            renderer,
            meshes_dir,
            mesh_cache,
            visible_bids=assembled_bids | {child_bid},
            hide_geom_ids=unassembled_servo_gids,
            unhide_geom_ids=horn_gids,
        )

        # Minimap (reset to home pose internally)
        minimap = _render_minimap(model, data, info, renderer)

        renderer.close()

        strip = StripData(
            jname=jname,
            s_frames=s_frames,
            s_labels=s_labels,
            s_cols=s_cols,
            a_frames=a_frames,
            a_labels=a_labels,
            a_cols=a_cols,
            inst_s_frames=inst_s_frames,
            inst_s_labels=inst_s_labels,
            inst_a_frames=inst_a_frames,
            inst_a_labels=inst_a_labels,
            minimap=minimap,
            child_body_name=child_name,
        )
        strips.append(strip)

        # Update assembled sets for next joint
        assembled_body_names.add(child_name)
        assembled_joint_names.add(jname)

        for kind, klabels, kcols in [
            ("servo", s_labels, s_cols),
            ("attach", a_labels, a_cols),
        ]:
            hits = [klabels[i] for i, c in enumerate(kcols) if c]
            if hits:
                print(f"  assembly {jname} {kind}: COLLISION at {', '.join(hits)}")

    return strips


def _build_comp_axes(bot) -> dict[str, tuple[float, float, float]]:
    """Build lookup of comp geom name → resolved insertion axis from the Bot."""
    axes: dict[str, tuple[float, float, float]] = {}
    for body in bot.all_bodies:
        for mount in body.mounts:
            key = f"comp_{body.name}_{mount.label}"
            axes[key] = mount.resolved_insertion_axis
    return axes


def emit_assembly_renders(bot, output_dir: Path) -> None:
    """Pipeline entry point called from Bot.emit()."""
    bot_xml = output_dir / "bot.xml"
    if not bot_xml.exists():
        print(f"  assembly renders: skipping - {bot_xml} not found")
        return

    print("Assembly renders:")
    t0 = time.perf_counter()

    comp_axes = _build_comp_axes(bot)
    comp_strips = _render_component_strips(bot_xml, comp_axes=comp_axes)
    joint_strips = _render_strips(bot_xml)
    overview_frame = _render_overview_frame(bot_xml)

    # Components first, then joints — matches physical assembly order
    all_strips = comp_strips + joint_strips

    if joint_strips:
        out1 = _composite_collision_pdf(joint_strips, output_dir)
        print(f"  collision test: {out1}")

    out2 = _composite_instructions_pdf(
        all_strips,
        output_dir,
        overview_frame=overview_frame,
    )
    print(f"  instructions:   {out2}")
    print(f"  assembly done ({time.perf_counter() - t0:.1f}s)")


# ── Standalone ──

if __name__ == "__main__":
    bot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bots/wheeler_arm")
    bot_xml = bot_dir / "bot.xml"

    if not bot_xml.exists():
        print(f"Error: {bot_xml} not found")
        sys.exit(1)

    print(f"Assembly renders: {bot_xml}")
    t0 = time.perf_counter()

    comp_strips = _render_component_strips(bot_xml)
    joint_strips = _render_strips(bot_xml)
    overview_frame = _render_overview_frame(bot_xml)

    all_strips = comp_strips + joint_strips

    if joint_strips:
        print(_composite_collision_pdf(joint_strips, bot_dir))
    print(
        _composite_instructions_pdf(
            all_strips,
            bot_dir,
            overview_frame=overview_frame,
        )
    )

    print(f"Done ({time.perf_counter() - t0:.1f}s)")
