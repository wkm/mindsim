"""Subassembly range-of-motion validation.

Reusable framework for sweeping any subassembly through its range of motion
with collision detection and filmstrip rendering. Use this to validate that
bracket designs, joint assemblies, and other mechanical subassemblies work
correctly before integrating them into full bot definitions.

Usage:
    from botcad.validation import validate_subassembly, SolidPart

    result = validate_subassembly(
        name="coupler_bracket",
        fixed=[
            SolidPart("cradle", cradle_solid(servo)),
            SolidPart("servo", servo_solid(servo)),
        ],
        moving=[
            SolidPart("coupler", coupler_solid(servo)),
        ],
        pivot=(sx, sy, sz),
        axis=(0, 0, 1),
        range_rad=(-2.618, 2.618),
    )
    result.save("test_rom_coupler.png")
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from botcad.emit.renders import white_background

Vec3 = tuple[float, float, float]

# ── Config ──

SWEEP_W, SWEEP_H = 600, 600
SWEEP_FRAMES = 12
PNG_DPI = (150, 150)


# ── Public API ──


@dataclass
class SolidPart:
    """A named solid for subassembly validation.

    Set check_collision=False for parts that are expected to be in contact
    with the moving parts (e.g., the servo body in a coupler assembly —
    the coupler bolts to the horns, so contact is by design).
    """

    name: str
    solid: object  # build123d Solid
    rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 0.85)
    check_collision: bool = True


@dataclass
class SweepResult:
    """Result of a ROM validation sweep."""

    name: str
    frames: list[Image.Image] = field(default_factory=list)
    angles_deg: list[float] = field(default_factory=list)
    collisions: list[bool] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def has_collisions(self) -> bool:
        return any(self.collisions)

    @property
    def collision_angles(self) -> list[float]:
        return [a for a, c in zip(self.angles_deg, self.collisions) if c]

    def image(self) -> Image.Image:
        """Composite filmstrip with collision indicators."""
        return _composite_filmstrip(self)

    def save(self, path: str | Path) -> Path:
        """Save filmstrip to PNG."""
        out = Path(path)
        self.image().save(out, optimize=True, dpi=PNG_DPI)
        return out


def validate_subassembly(
    name: str,
    fixed: list[SolidPart],
    moving: list[SolidPart],
    pivot: Vec3 = (0.0, 0.0, 0.0),
    axis: Vec3 = (0.0, 0.0, 1.0),
    range_rad: tuple[float, float] = (-3.14, 3.14),
    frames: int = SWEEP_FRAMES,
    camera_distance: float | None = None,
    camera_azimuth: float = 135,
    camera_elevation: float = -30,
) -> SweepResult:
    """Sweep a subassembly through its ROM, checking for collisions.

    Args:
        name: Human-readable name for the subassembly.
        fixed: Parts that don't move (bolted to world).
        moving: Parts attached to the hinge joint.
        pivot: Hinge position in the fixed frame.
        axis: Rotation axis (unit vector).
        range_rad: (lo, hi) joint range in radians.
        frames: Number of sweep frames.
        camera_distance: Override auto-computed camera distance.
        camera_azimuth: Camera azimuth angle in degrees.
        camera_elevation: Camera elevation angle in degrees.

    Returns:
        SweepResult with frames, angles, and collision data.
    """
    t0 = time.perf_counter()
    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_rom_"))

    try:
        # Export all solids to STL
        _export_parts(fixed, temp_dir, prefix="fixed")
        _export_parts(moving, temp_dir, prefix="moving")

        # Build MuJoCo XML
        xml = _build_scene_xml(fixed, moving, pivot, axis, range_rad, temp_dir)

        # Load and sweep
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, SWEEP_W, SWEEP_H)

        # Auto camera distance from scene extent
        mujoco.mj_forward(model, data)
        dist = camera_distance or max(model.stat.extent * 2.5, 0.05)

        angles = np.linspace(range_rad[0], range_rad[1], frames)
        qposadr = model.jnt_qposadr[0]  # single hinge joint

        result = SweepResult(name=name)

        # Load trimesh meshes for collision checking (only check_collision parts)
        collision_fixed = [p for p in fixed if p.check_collision]
        fixed_meshes = _load_trimeshes(collision_fixed, temp_dir, "fixed")
        moving_meshes = _load_trimeshes(moving, temp_dir, "moving")

        for angle in angles:
            mujoco.mj_resetData(model, data)
            data.qpos[qposadr] = angle
            mujoco.mj_forward(model, data)

            # Render
            img = _render_frame(
                model, data, renderer, dist, camera_azimuth, camera_elevation
            )
            result.frames.append(Image.fromarray(img))
            result.angles_deg.append(math.degrees(angle))

            # Collision check using trimesh
            has_col = _check_collision(
                model,
                data,
                collision_fixed,
                moving,
                fixed_meshes,
                moving_meshes,
                pivot,
                axis,
            )
            result.collisions.append(has_col)

        renderer.close()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    result.elapsed = time.perf_counter() - t0

    # Report
    if result.has_collisions:
        col_str = ", ".join(f"{a:+.0f}°" for a in result.collision_angles)
        print(f"  ROM {name}: COLLISION at {col_str} ({result.elapsed:.1f}s)")
    else:
        lo_deg = math.degrees(range_rad[0])
        hi_deg = math.degrees(range_rad[1])
        print(
            f"  ROM {name}: clean [{lo_deg:+.0f}°..{hi_deg:+.0f}°]"
            f" ({result.elapsed:.1f}s)"
        )

    return result


# ── Internals ──


def _export_parts(parts: list[SolidPart], temp_dir: Path, prefix: str) -> None:
    """Export solids to STL files. Handles Solid, Compound, and ShapeList."""
    from build123d import Compound, Solid, export_stl

    for part in parts:
        stl_path = temp_dir / f"{prefix}_{part.name}.stl"
        solid = part.solid
        # Wrap non-Solid objects (ShapeList, list) into a Compound for export
        if not isinstance(solid, (Solid, Compound)):
            solid = Compound(children=list(solid))
        export_stl(solid, str(stl_path))


def _build_scene_xml(
    fixed: list[SolidPart],
    moving: list[SolidPart],
    pivot: Vec3,
    axis: Vec3,
    range_rad: tuple[float, float],
    temp_dir: Path,
) -> str:
    """Build MuJoCo XML for the subassembly sweep."""
    assets = []
    for part in fixed:
        assets.append(f'<mesh name="fixed_{part.name}" file="fixed_{part.name}.stl"/>')
    for part in moving:
        assets.append(
            f'<mesh name="moving_{part.name}" file="moving_{part.name}.stl"/>'
        )
    assets_str = "\n    ".join(assets)

    fixed_geoms = []
    for part in fixed:
        r, g, b, a = part.rgba
        fixed_geoms.append(
            f'<geom name="fixed_{part.name}" type="mesh"'
            f' mesh="fixed_{part.name}" rgba="{r} {g} {b} {a}"/>'
        )
    fixed_geoms_str = "\n        ".join(fixed_geoms)

    moving_geoms = []
    for part in moving:
        r, g, b, a = part.rgba
        moving_geoms.append(
            f'<geom name="moving_{part.name}" type="mesh"'
            f' mesh="moving_{part.name}" rgba="{r} {g} {b} {a}"/>'
        )
    moving_geoms_str = "\n        ".join(moving_geoms)

    px, py, pz = pivot
    ax, ay, az = axis
    lo, hi = range_rad

    xml = f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="rom_sweep">
  <option gravity="0 0 0"/>
  <compiler meshdir="{temp_dir}"/>
  <visual>
    <rgba haze="1 1 1 1"/>
    <global offwidth="{SWEEP_W * 2}" offheight="{SWEEP_H * 2}"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.3 0.3 0.3"/>
  </visual>
  <asset>
    {assets_str}
  </asset>
  <worldbody>
    <light pos="0.1 0.1 0.2" dir="-0.3 -0.3 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.1 0.1 0.2" dir="0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>
    <body name="fixed" pos="0 0 0">
      {fixed_geoms_str}
      <body name="moving" pos="{px} {py} {pz}">
        <joint name="sweep" type="hinge"
               axis="{ax} {ay} {az}"
               range="{lo} {hi}" limited="true"/>
        {moving_geoms_str}
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    return xml


def _load_trimeshes(parts: list[SolidPart], temp_dir: Path, prefix: str) -> list:
    """Load trimesh objects for collision detection."""
    import trimesh

    meshes = []
    for part in parts:
        stl_path = temp_dir / f"{prefix}_{part.name}.stl"
        meshes.append(trimesh.load(stl_path, force="mesh"))
    return meshes


def _check_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fixed_parts: list[SolidPart],
    moving_parts: list[SolidPart],
    fixed_meshes: list,
    moving_meshes: list,
    pivot: Vec3,
    axis: Vec3,
) -> bool:
    """Check if any moving part collides with any fixed part using trimesh."""
    import trimesh

    # Get body transforms from MuJoCo
    fixed_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fixed")
    moving_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving")

    fixed_T = np.eye(4)
    fixed_T[:3, :3] = data.xmat[fixed_bid].reshape(3, 3)
    fixed_T[:3, 3] = data.xpos[fixed_bid]

    moving_T = np.eye(4)
    moving_T[:3, :3] = data.xmat[moving_bid].reshape(3, 3)
    moving_T[:3, 3] = data.xpos[moving_bid]

    manager = trimesh.collision.CollisionManager()

    # Add all fixed parts
    for i, mesh in enumerate(fixed_meshes):
        m = mesh.copy()
        m.apply_transform(fixed_T)
        # Small offset along axis to ignore flush-face contacts at pivot
        axis_arr = np.array(axis)
        m.apply_translation(-axis_arr * 0.0005)
        manager.add_object(f"fixed_{i}", m)

    # Check each moving part against all fixed
    for i, mesh in enumerate(moving_meshes):
        m = mesh.copy()
        m.apply_transform(moving_T)
        if manager.in_collision_single(m):
            return True

    return False


def _render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    distance: float,
    azimuth: float,
    elevation: float,
) -> np.ndarray:
    """Render a single sweep frame."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = model.stat.center
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, camera=cam)
    img = renderer.render().copy()
    white_background(img)
    return img


def _composite_filmstrip(result: SweepResult) -> Image.Image:
    """Build a filmstrip image from sweep results."""
    n = len(result.frames)
    if n == 0:
        return Image.new("RGB", (100, 100), (255, 255, 255))

    margin = 8
    header_h = 32
    label_h = 20
    border_w = 3

    # Layout: one row of frames
    canvas_w = SWEEP_W * n + margin * (n + 1)
    canvas_h = header_h + SWEEP_H + label_h + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Header with name and range
    lo = min(result.angles_deg) if result.angles_deg else 0
    hi = max(result.angles_deg) if result.angles_deg else 0
    collision_text = "  COLLISIONS DETECTED" if result.has_collisions else ""
    header = (
        f"ROM sweep: {result.name}  "
        f"[{lo:+.0f}° .. {hi:+.0f}°]"
        f"  ({result.elapsed:.1f}s){collision_text}"
    )
    draw.text((margin, margin), header, fill=(0, 0, 0))

    # ROM bar
    bar_x0 = margin
    bar_w = min(300, canvas_w - 2 * margin)
    bar_y = margin + 16
    bar_h = 10
    draw.rectangle(
        [bar_x0, bar_y, bar_x0 + bar_w, bar_y + bar_h],
        fill=(230, 230, 230),
        outline=(180, 180, 180),
    )
    px_lo = int((lo + 180) / 360 * bar_w)
    px_hi = int((hi + 180) / 360 * bar_w)
    px_lo = max(0, min(bar_w, px_lo))
    px_hi = max(px_lo + 1, min(bar_w, px_hi))
    bar_color = (255, 180, 50) if result.has_collisions else (80, 180, 80)
    draw.rectangle(
        [bar_x0 + px_lo, bar_y, bar_x0 + px_hi, bar_y + bar_h],
        fill=bar_color,
    )
    # Center tick
    center_px = int(180 / 360 * bar_w)
    draw.line(
        [bar_x0 + center_px, bar_y, bar_x0 + center_px, bar_y + bar_h],
        fill=(100, 100, 100),
        width=1,
    )

    # Frames
    for i, (frame, angle, has_col) in enumerate(
        zip(result.frames, result.angles_deg, result.collisions)
    ):
        x = margin + i * (SWEEP_W + margin)
        y = header_h

        if has_col:
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
            draw.rectangle(
                [x - 1, y - 1, x + SWEEP_W, y + SWEEP_H],
                outline=(0, 0, 0),
                width=1,
            )

        canvas.paste(frame, (x, y))
        color = (220, 40, 40) if has_col else (80, 80, 80)
        draw.text((x, y + SWEEP_H + 2), f"{angle:+.0f}°", fill=color)

    return canvas
