"""Cone/frustum mesh generation for MuJoCo scenes.

MuJoCo has no built-in cone geom type. This module generates vertex, face,
and normal data for truncated cones (frustums) that are injected as mesh
geoms into the compiled model.

Each geom slot gets a companion mesh asset created at spec time. At runtime,
the composer writes new vertex/normal data into the mesh arrays to reshape
the cone for whatever concept needs it.

Convention:
    - The cone is centered at the origin
    - Bottom ring at z = -half_h, radius = bottom_r
    - Top ring at z = +half_h, radius = top_r
    - Prim size tuple: (bottom_radius, half_height, top_radius)
"""

from __future__ import annotations

import numpy as np

# Polygon count — 16 sides is smooth enough while keeping data small.
N_SIDES = 16

# Total counts (same for all cones, since topology never changes)
N_VERTS = N_SIDES * 2 + 2  # bottom ring + top ring + 2 center points
N_FACES = N_SIDES * 4  # bottom cap + top cap + 2 side triangles per edge

# Pre-computed angles
_ANGLES = np.linspace(0, 2 * np.pi, N_SIDES, endpoint=False)
_COS = np.cos(_ANGLES)
_SIN = np.sin(_ANGLES)


def _build_faces() -> list[int]:
    """Face indices for the frustum topology (constant for all cones)."""
    faces: list[int] = []
    bc = N_SIDES * 2  # bottom center vertex index
    tc = N_SIDES * 2 + 1  # top center vertex index
    for i in range(N_SIDES):
        j = (i + 1) % N_SIDES
        # Bottom cap
        faces.extend([bc, j, i])
        # Top cap
        faces.extend([tc, i + N_SIDES, j + N_SIDES])
        # Side quad as 2 triangles
        faces.extend([i, j, j + N_SIDES])
        faces.extend([i, j + N_SIDES, i + N_SIDES])
    return faces


FACE_INDICES: list[int] = _build_faces()


def make_verts(bottom_r: float, half_h: float, top_r: float) -> np.ndarray:
    """Generate frustum vertices.

    Args:
        bottom_r: Radius of the bottom ring.
        half_h: Half the total height (bottom at -half_h, top at +half_h).
        top_r: Radius of the top ring.

    Returns:
        ndarray of shape (N_VERTS, 3).
    """
    verts = np.empty((N_VERTS, 3), dtype=np.float64)

    # Bottom ring
    verts[:N_SIDES, 0] = _COS * bottom_r
    verts[:N_SIDES, 1] = _SIN * bottom_r
    verts[:N_SIDES, 2] = -half_h

    # Top ring
    verts[N_SIDES : 2 * N_SIDES, 0] = _COS * top_r
    verts[N_SIDES : 2 * N_SIDES, 1] = _SIN * top_r
    verts[N_SIDES : 2 * N_SIDES, 2] = half_h

    # Cap centers
    verts[-2] = [0, 0, -half_h]
    verts[-1] = [0, 0, half_h]

    return verts


def make_normals(bottom_r: float, half_h: float, top_r: float) -> np.ndarray:
    """Generate per-vertex normals for smooth cone lighting.

    Side vertices get outward normals perpendicular to the cone surface.
    Cap centers get flat up/down normals.

    Returns:
        ndarray of shape (N_VERTS, 3).
    """
    normals = np.empty((N_VERTS, 3), dtype=np.float64)

    # The cone surface slope in the r-z plane:
    #   tangent = (R_t - R_b, H)  where H = 2 * half_h
    #   outward normal perpendicular to tangent = (H, R_b - R_t)
    h = 2 * half_h
    dr = bottom_r - top_r
    mag = np.sqrt(h * h + dr * dr)
    if mag < 1e-10:
        n_r, n_z = 1.0, 0.0
    else:
        n_r = h / mag
        n_z = dr / mag

    # Bottom ring — outward along slope
    normals[:N_SIDES, 0] = _COS * n_r
    normals[:N_SIDES, 1] = _SIN * n_r
    normals[:N_SIDES, 2] = n_z

    # Top ring — same outward direction (smooth shading)
    normals[N_SIDES : 2 * N_SIDES, 0] = _COS * n_r
    normals[N_SIDES : 2 * N_SIDES, 1] = _SIN * n_r
    normals[N_SIDES : 2 * N_SIDES, 2] = n_z

    # Cap centers
    normals[-2] = [0, 0, -1]  # bottom
    normals[-1] = [0, 0, 1]  # top

    return normals
