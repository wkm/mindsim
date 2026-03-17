"""Centralized 3D rendering via MuJoCo.

All 3D imagery in the pipeline flows through this module. It provides:

- Color: unified color definition (MuJoCo RGBA + PIL RGB from one source)
- SceneBuilder: declarative MuJoCo XML construction (replaces hand-rolled f-strings)
- render_frame: single-frame rendering with camera config
- render_views: multi-view rendering with named camera presets

Renders use a CAD-style "shaded with edges" look:
- High ambient, low specular lighting (flat diffuse, no hotspots)
- Segmentation-based edge detection (dark outlines at part boundaries)
- Depth-buffer edge detection (silhouette outlines at depth discontinuities)

Usage:
    scene = SceneBuilder()
    scene.add_mesh("bracket", "/tmp/bracket.stl", COLOR_BRACKET)
    scene.annotate_sphere("ear_0", pos=(0.01, 0, 0.02), size=0.002, color=COLOR_MOUNTING)
    xml = scene.to_xml()

    with Renderer3D(xml, width=800, height=800) as r:
        img = r.render_frame(azimuth=135, elevation=-30)
        views = r.render_views(VIEWS_6)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

# ── Color (re-exported from botcad.colors) ──
from botcad.colors import (  # noqa: E402, F401
    COLOR_BRACKET,
    COLOR_COUPLER,
    COLOR_CRADLE,
    COLOR_ENVELOPE,
    COLOR_HORN,
    COLOR_HORN_HOLE,
    COLOR_MOUNTING,
    COLOR_REAR_HOLE,
    COLOR_SERVO_BODY,
    COLOR_SHAFT,
    COLOR_WIRE_PORT,
    Color,
)

# ── Camera presets ──

# 6-view tear sheet (component renders)
VIEWS_6: dict[str, dict[str, float]] = {
    "front (+Y)": {"azimuth": 90, "elevation": 0},
    "right (+X)": {"azimuth": 0, "elevation": 0},
    "top (+Z)": {"azimuth": 90, "elevation": -90},
    "bottom (-Z)": {"azimuth": 90, "elevation": 90},
    "back (-X)": {"azimuth": 180, "elevation": 0},
    "three-quarter": {"azimuth": 135, "elevation": -30},
}

# 4-view overview (bot renders)
VIEWS_4: dict[str, dict[str, float]] = {
    "front (+Y)": {"azimuth": 90, "elevation": 0},
    "right (+X)": {"azimuth": 0, "elevation": 0},
    "top (+Z)": {"azimuth": 90, "elevation": -90},
    "three-quarter": {"azimuth": 135, "elevation": -30},
}


# ── Scene builder ──


@dataclass
class SceneBuilder:
    """Declarative MuJoCo XML scene construction.

    Accumulate meshes and primitive geoms, then call to_xml() to get
    a complete MuJoCo XML string. Replaces hand-rolled XML f-strings.
    """

    model_name: str = "scene"
    width: int = 800
    height: int = 800
    _meshes: list[tuple[str, str]] = field(default_factory=list)  # (name, filename)
    _geoms: list[str] = field(default_factory=list)
    _legends: list[tuple[str, tuple[int, int, int]]] = field(default_factory=list)
    _mesh_dir: str = ""
    # (name, pos_str, geom_strs, joint_strs)
    _bodies: list[tuple[str, str, list[str], list[str]]] = field(default_factory=list)

    def set_mesh_dir(self, path: str | Path) -> None:
        self._mesh_dir = str(path)

    def add_mesh(
        self,
        name: str,
        filename: str,
        color: Color,
        pos: tuple[float, ...] | None = None,
        quat: str | None = None,
    ) -> None:
        """Add a mesh asset + geom, optionally positioned/oriented."""
        mesh_name = f"{name}_mesh"
        self._meshes.append((mesh_name, filename))
        pos_attr = f' pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"' if pos else ""
        quat_attr = f' quat="{quat}"' if quat else ""
        self._geoms.append(
            f'<geom name="{name}" type="mesh" mesh="{mesh_name}"'
            f"{pos_attr}{quat_attr}"
            f' rgba="{color.rgba_str}"/>'
        )
        self._add_legend(color)

    def annotate_sphere(
        self, name: str, pos: tuple[float, ...], size: float, color: Color
    ) -> None:
        """Add a sphere annotation marker (not physical geometry)."""
        self._geoms.append(
            f'<geom name="{name}" type="sphere" size="{size}"'
            f' pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"'
            f' rgba="{color.rgba_str}"/>'
        )
        self._add_legend(color)

    def annotate_cylinder(
        self,
        name: str,
        pos: tuple[float, ...],
        radius: float,
        half_height: float,
        color: Color,
        quat: str | None = None,
    ) -> None:
        """Add a cylinder annotation marker (not physical geometry)."""
        quat_attr = f' quat="{quat}"' if quat else ""
        self._geoms.append(
            f'<geom name="{name}" type="cylinder"'
            f' size="{radius:.6f} {half_height:.6f}"'
            f' pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"'
            f"{quat_attr}"
            f' rgba="{color.rgba_str}"/>'
        )
        self._add_legend(color)

    def add_raw_geom(self, xml: str) -> None:
        """Add a raw XML geom string (for one-off cases)."""
        self._geoms.append(xml)

    def add_body(
        self,
        name: str,
        pos: tuple[float, ...] = (0, 0, 0),
        geoms: list[str] | None = None,
    ) -> None:
        """Add a child body (for multi-body scenes like ROM validation)."""
        self._bodies.append((name, f"{pos[0]} {pos[1]} {pos[2]}", geoms or [], []))

    def add_hinge(
        self,
        name: str,
        axis: tuple[float, ...],
        range_rad: tuple[float, float],
    ) -> None:
        """Add a hinge joint to the last added body."""
        if not self._bodies:
            raise ValueError("add_hinge requires add_body() before add_hinge()")
        lo, hi = range_rad
        ax, ay, az = axis
        self._bodies[-1][3].append(
            f'<joint name="{name}" type="hinge"'
            f' axis="{ax} {ay} {az}"'
            f' range="{lo} {hi}" limited="true"/>'
        )

    def _add_legend(self, color: Color) -> None:
        if color.label and not any(e[0] == color.label for e in self._legends):
            self._legends.append(color.legend)

    @property
    def legends(self) -> list[tuple[str, tuple[int, int, int]]]:
        return list(self._legends)

    def to_xml(self) -> str:
        assets = "\n    ".join(
            f'<mesh name="{name}" file="{fname}" scale="1 1 1"/>'
            for name, fname in self._meshes
        )
        geoms_str = "\n        ".join(self._geoms)

        # Build body hierarchy
        if self._bodies:
            # Multi-body scene (e.g., ROM validation with fixed + moving)
            body_xml_parts = []
            for bname, bpos, bgeoms, bjoints in self._bodies:
                inner = "\n          ".join([*bgeoms, *bjoints])
                body_xml_parts.append(
                    f'    <body name="{bname}" pos="{bpos}">\n'
                    f"          {inner}\n"
                    f"        </body>"
                )
            bodies_str = "\n".join(body_xml_parts)
            worldbody_content = f"""
    <light pos="0.1 0.1 0.2" dir="-0.3 -0.3 -1" diffuse="0.6 0.6 0.6" specular="0.05 0.05 0.05"/>
    <light pos="-0.1 0.1 0.2" dir="0.3 -0.3 -1" diffuse="0.3 0.3 0.3" specular="0.02 0.02 0.02"/>
    <body name="root" pos="0 0 0">
        {geoms_str}
{bodies_str}
    </body>"""
        else:
            # Single-body scene (most common)
            worldbody_content = f"""
    <light pos="0.1 0.1 0.2" dir="-0.3 -0.3 -1" diffuse="0.6 0.6 0.6" specular="0.05 0.05 0.05"/>
    <light pos="-0.1 0.1 0.2" dir="0.3 -0.3 -1" diffuse="0.3 0.3 0.3" specular="0.02 0.02 0.02"/>
    <body name="root" pos="0 0 0">
        {geoms_str}
    </body>"""

        return f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="{self.model_name}">
  <option gravity="0 0 0"/>
  <compiler meshdir="{self._mesh_dir}"/>
  <visual>
    <rgba haze="1 1 1 1"/>
    <global offwidth="{self.width}" offheight="{self.height}"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <quality offsamples="4"/>
  </visual>
  <asset>
    {assets}
  </asset>
  <worldbody>{worldbody_content}
  </worldbody>
</mujoco>
"""


# ── Post-processing ──


def white_background(img: np.ndarray) -> None:
    """Replace black pixels with white (MuJoCo renders on black)."""
    mask = np.all(img == 0, axis=2)
    img[mask] = 255


def _detect_edges(seg: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Detect edges from segmentation IDs and depth discontinuities.

    Returns a boolean edge mask (True = edge pixel).
    """
    ids = seg[:, :, 0]

    # Segmentation boundaries: neighboring pixels with different geom IDs
    seg_edges = np.zeros(ids.shape, dtype=bool)
    seg_edges[:-1, :] |= ids[:-1, :] != ids[1:, :]
    seg_edges[1:, :] |= ids[:-1, :] != ids[1:, :]
    seg_edges[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    seg_edges[:, 1:] |= ids[:, :-1] != ids[:, 1:]

    # Depth discontinuities: large depth gradient → silhouette edge
    # Normalize depth to [0, 1] range for threshold comparison
    d = depth.astype(np.float32)
    d_range = d.max() - d.min()
    if d_range > 0:
        d_norm = (d - d.min()) / d_range
    else:
        d_norm = np.zeros_like(d)

    # Sobel-like gradient magnitude on depth
    dx = np.zeros_like(d_norm)
    dy = np.zeros_like(d_norm)
    dx[:, 1:] = np.abs(d_norm[:, 1:] - d_norm[:, :-1])
    dy[1:, :] = np.abs(d_norm[1:, :] - d_norm[:-1, :])
    depth_grad = np.maximum(dx, dy)
    depth_edges = depth_grad > 0.015

    # Union of both edge sources
    edges = seg_edges | depth_edges

    # Thicken to ~2px for visibility
    thick = edges.copy()
    thick[1:, :] |= edges[:-1, :]
    thick[:-1, :] |= edges[1:, :]
    thick[:, 1:] |= edges[:, :-1]
    thick[:, :-1] |= edges[:, 1:]

    return thick


def _apply_ssao(img: np.ndarray, depth: np.ndarray, strength: float = 0.35) -> None:
    """Approximate screen-space ambient occlusion from the depth buffer.

    Darkens concavities (crevices, corners, tight joints) to add
    depth perception. Modifies img in-place.
    """
    d = depth.astype(np.float32)
    d_range = d.max() - d.min()
    if d_range < 1e-6:
        return

    d_norm = (d - d.min()) / d_range

    # Compare each pixel's depth to the local average.
    # Where a pixel is deeper than its neighborhood → concavity → darken.
    local_avg = uniform_filter(d_norm, size=9)
    occlusion = np.clip((local_avg - d_norm) * 15.0, 0.0, 1.0)

    # Smooth the occlusion map so it's not noisy
    occlusion = uniform_filter(occlusion, size=5)

    # Apply as darkening factor only on non-background pixels
    bg_mask = np.all(img == 255, axis=2)
    factor = 1.0 - occlusion * strength
    for c in range(3):
        channel = img[:, :, c].astype(np.float32)
        channel *= factor
        img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
    # Restore background
    img[bg_mask] = 255


def _mesh_bounds(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, float]:
    """Compute center and extent from mesh geoms only.

    Falls back to model.stat if no mesh geoms exist (e.g. primitive-only scenes).
    Returns (center, extent).
    """
    positions = []
    sizes = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            pos = data.geom_xpos[gid].copy()
            r = model.geom_rbound[gid]  # bounding sphere radius
            positions.append(pos)
            sizes.append(r)

    if not positions:
        # No meshes — fall back to full scene stats
        return model.stat.center.copy(), max(model.stat.extent, 0.01)

    positions = np.array(positions)
    sizes = np.array(sizes)

    # Bounding box from mesh centers ± their radii
    lo = (positions - sizes[:, None]).min(axis=0)
    hi = (positions + sizes[:, None]).max(axis=0)
    center = (lo + hi) / 2.0
    extent = np.linalg.norm(hi - lo) / 2.0
    return center, max(extent, 0.01)


# ── Renderer ──


class Renderer3D:
    """MuJoCo offscreen renderer with managed lifecycle.

    Produces CAD-style renders: flat diffuse shading with dark edge
    outlines at part boundaries and depth discontinuities.

    Use as context manager:
        with Renderer3D(xml, 800, 800) as r:
            img = r.render_frame(azimuth=135, elevation=-30)
    """

    def __init__(self, xml: str, width: int = 800, height: int = 800):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width, height)
        mujoco.mj_forward(self.model, self.data)

        # Compute tight bounding box from mesh geoms only (ignoring debug
        # overlays like axis indicators and annotation spheres which inflate
        # model.stat.extent and shift model.stat.center).
        self._geom_center, self._geom_extent = _mesh_bounds(self.model, self.data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.renderer.close()

    def _setup_camera(
        self,
        azimuth: float,
        elevation: float,
        lookat: Sequence[float] | None,
        distance: float | None,
    ) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.orthographic = 1  # CAD-style orthographic projection
        if lookat is not None:
            cam.lookat[:] = lookat
        else:
            cam.lookat[:] = self._geom_center
        cam.distance = distance if distance is not None else self._geom_extent * 1.8
        cam.azimuth = azimuth
        cam.elevation = elevation
        return cam

    def render_frame(
        self,
        azimuth: float = 135,
        elevation: float = -30,
        lookat: Sequence[float] | None = None,
        distance: float | None = None,
    ) -> np.ndarray:
        """Render a single frame with CAD-style edges. Returns H*W*3 uint8."""
        cam = self._setup_camera(azimuth, elevation, lookat, distance)

        # Pass 1: color render
        self.renderer.update_scene(self.data, camera=cam)
        img = self.renderer.render().copy()
        white_background(img)

        # Pass 2: segmentation buffer
        self.renderer.update_scene(self.data, camera=cam)
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()

        # Pass 3: depth buffer
        self.renderer.update_scene(self.data, camera=cam)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        # Post-process: SSAO
        _apply_ssao(img, depth, strength=0.35)

        # Post-process: edge outlines
        edges = _detect_edges(seg, depth)
        bg_mask = np.all(img == 255, axis=2)
        img[edges & ~bg_mask] = [40, 40, 40]

        return img

    def render_frame_plain(
        self,
        azimuth: float = 135,
        elevation: float = -30,
        lookat: Sequence[float] | None = None,
        distance: float | None = None,
    ) -> np.ndarray:
        """Render without edge detection (for filmstrips where speed matters)."""
        cam = self._setup_camera(azimuth, elevation, lookat, distance)
        self.renderer.update_scene(self.data, camera=cam)
        img = self.renderer.render().copy()
        white_background(img)
        return img

    def render_views(
        self,
        views: dict[str, dict[str, float]],
        **kwargs,
    ) -> list[tuple[Image.Image, str]]:
        """Render multiple named views with CAD-style edges."""
        results = []
        for label, params in views.items():
            img = self.render_frame(
                azimuth=params["azimuth"],
                elevation=params["elevation"],
                **kwargs,
            )
            results.append((Image.fromarray(img), label))
        return results
