"""Visualization and export utilities for FEA results.

Exports Warp stress grids and masks as PLY files with vertex colors.
"""

from __future__ import annotations

import numpy as np
import trimesh


def export_voxel_mesh(vd, mask, color, path: str):
    """Export a mask of voxels as a single mesh of cubes."""
    grid = vd.grid
    res = grid.res
    bounds_lo = grid.bounds_lo

    # Calculate dx, dy, dz
    dx = (grid.bounds_hi[0] - grid.bounds_lo[0]) / float(res[0])
    dy = (grid.bounds_hi[1] - grid.bounds_lo[1]) / float(res[1])
    dz = (grid.bounds_hi[2] - grid.bounds_lo[2]) / float(res[2])
    cube_size = np.array([dx, dy, dz])

    mask_np = mask.numpy()
    indices = np.where(mask_np == 1)[0]

    if len(indices) == 0:
        print(f"Warning: Mask for {path} is empty.")
        return

    # Create one cube per voxel
    cubes = []
    for idx in indices:
        # Reconstruct ix, iy, iz from tid
        iz = idx % res[2]
        iy = (idx // res[2]) % res[1]
        ix = idx // (res[1] * res[2])

        pos = np.array(
            [
                bounds_lo[0] + (ix + 0.5) * dx,
                bounds_lo[1] + (iy + 0.5) * dy,
                bounds_lo[2] + (iz + 0.5) * dz,
            ]
        )

        cube = trimesh.creation.box(extents=cube_size * 0.9)  # gap for visibility
        cube.apply_translation(pos)
        cubes.append(cube)

    combined = trimesh.util.concatenate(cubes)
    combined.visual.vertex_colors = color
    combined.export(path)
    print(f"Exported voxel mesh to {path}")


def export_stress_mesh(solid, space, u_field, stress_array, path: str):
    """Interpolate stress onto CAD mesh and export as vertex-colored PLY."""
    verts, faces = solid.tessellate(tolerance=0.0005)  # high res for heatmap
    verts_np = np.array([(v.X, v.Y, v.Z) for v in verts], dtype=np.float32)
    faces_np = np.array(faces, dtype=np.int32)

    # Sample stress at every vertex using Warp interpolation
    # stress_array is element-indexed (Cells). To get vertex values,
    # we use fem.interpolate at arbitrary points.

    # For now, a simpler approach: find closest cell center for each vertex
    # because Warp arbitrary-point interpolation is another complex API.

    stress_vals = stress_array.numpy()
    grid = space.geometry
    res = grid.res
    lo = grid.bounds_lo
    hi = grid.bounds_hi

    dx = (hi[0] - lo[0]) / res[0]
    dy = (hi[1] - lo[1]) / res[1]
    dz = (hi[2] - lo[2]) / res[2]

    # Map vertices to cell indices
    ix = np.clip(((verts_np[:, 0] - lo[0]) / dx).astype(int), 0, res[0] - 1)
    iy = np.clip(((verts_np[:, 1] - lo[1]) / dy).astype(int), 0, res[1] - 1)
    iz = np.clip(((verts_np[:, 2] - lo[2]) / dz).astype(int), 0, res[2] - 1)

    cell_indices = ix * (res[1] * res[2]) + iy * res[2] + iz
    vertex_stresses = stress_vals[cell_indices]

    # Normalize stress for heatmap (0.0 to 1.0)
    # We use yield strength (40MPa) as the 1.0 mark
    max_range = 40e6  # 40 MPa
    normalized = np.clip(vertex_stresses / max_range, 0, 1)

    # Map to colors (Blue -> Cyan -> Green -> Yellow -> Red)
    from matplotlib import cm

    cmap = cm.get_cmap("jet")  # 'jet' is standard for FEA
    colors = (cmap(normalized)[:, :3] * 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, vertex_colors=colors)
    mesh.export(path)
    print(f"Exported stress heatmap mesh to {path}")
