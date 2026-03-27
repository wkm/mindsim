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
from PIL import Image

from botcad.component import Vec3
from botcad.emit.composite import filmstrip, save_png
from botcad.emit.render3d import Color, Renderer3D, SceneBuilder

# ── Config ──

SWEEP_W, SWEEP_H = 600, 600
SWEEP_FRAMES = 12

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
    rgba: tuple[float, float, float, float] = (0.541, 0.608, 0.659, 0.85)  # BP_GRAY3
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
        return [a for a, c in zip(self.angles_deg, self.collisions, strict=True) if c]

    def image(self) -> Image.Image:
        """Composite filmstrip with collision indicators."""
        return filmstrip(
            self.name,
            self.frames,
            self.angles_deg,
            self.collisions,
            cell_w=SWEEP_W,
            cell_h=SWEEP_H,
        )

    def save(self, path: str | Path) -> Path:
        """Save filmstrip to PNG."""
        out = Path(path)
        save_png(self.image(), out)
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

        # Build scene via SceneBuilder
        scene = _build_scene(fixed, moving, pivot, axis, range_rad, temp_dir)
        xml = scene.to_xml()

        # Create renderer (gets orthographic, edges, SSAO, CAD lighting)
        r = Renderer3D(xml, SWEEP_W, SWEEP_H)

        # Auto camera distance from scene extent
        dist = camera_distance or max(r._geom_extent * 1.4, 0.05)

        angles = np.linspace(range_rad[0], range_rad[1], frames)
        qposadr = r.model.jnt_qposadr[0]  # single hinge joint

        result = SweepResult(name=name)

        # Load trimesh meshes for collision checking (only check_collision parts)
        collision_fixed = [p for p in fixed if p.check_collision]
        fixed_meshes = _load_trimeshes(collision_fixed, temp_dir, "fixed")
        moving_meshes = _load_trimeshes(moving, temp_dir, "moving")

        for angle in angles:
            mujoco.mj_resetData(r.model, r.data)
            r.data.qpos[qposadr] = angle
            mujoco.mj_forward(r.model, r.data)

            # Render with CAD-style edges
            img = r.render_frame(
                azimuth=camera_azimuth,
                elevation=camera_elevation,
                distance=dist,
            )
            result.frames.append(Image.fromarray(img))
            result.angles_deg.append(math.degrees(angle))

            # Collision check using trimesh
            has_col = _check_collision(
                r.model,
                r.data,
                collision_fixed,
                moving,
                fixed_meshes,
                moving_meshes,
                pivot,
                axis,
            )
            result.collisions.append(has_col)

        r.close()

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


def _build_scene(
    fixed: list[SolidPart],
    moving: list[SolidPart],
    pivot: Vec3,
    axis: Vec3,
    range_rad: tuple[float, float],
    temp_dir: Path,
) -> SceneBuilder:
    """Build MuJoCo scene for the subassembly sweep via SceneBuilder."""
    scene = SceneBuilder(model_name="rom_sweep", width=SWEEP_W, height=SWEEP_H)
    scene.set_mesh_dir(temp_dir)

    # Fixed parts as root-level geoms
    for part in fixed:
        r, g, b, a = part.rgba
        color = Color(r, g, b, a)
        scene.add_mesh(f"fixed_{part.name}", f"fixed_{part.name}.stl", color)

    # Moving parts in a child body with hinge joint
    moving_geoms = []
    for part in moving:
        r, g, b, a = part.rgba
        mesh_name = f"moving_{part.name}_mesh"
        scene._meshes.append((mesh_name, f"moving_{part.name}.stl"))
        moving_geoms.append(
            f'<geom name="moving_{part.name}" type="mesh"'
            f' mesh="{mesh_name}" rgba="{r} {g} {b} {a}"/>'
        )

    scene.add_body("moving", pos=pivot, geoms=moving_geoms)
    scene.add_hinge("sweep", axis=axis, range_rad=range_rad)

    return scene


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
