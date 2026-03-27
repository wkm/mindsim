"""Voxelization utilities for CAD-to-FEA bridge.

Converts build123d solids into Warp Grid3D domains.
Uses ShapeScript tags to identify boundary condition and load voxels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
import warp.fem as fem


@dataclass
class VoxelDomain:
    grid: fem.Grid3D
    cell_count: int
    inside_mask: wp.array(dtype=int)  # 1 if cell center is inside CAD solid
    tag_masks: dict[
        str, wp.array(dtype=int)
    ]  # tag -> mask of cells near tagged primitive


@wp.kernel
def check_inside_kernel(
    mesh: wp.uint64,
    bounds_lo: wp.vec3,
    dx: wp.vec3,
    res: wp.vec3i,
    mask: wp.array(dtype=int),
):
    tid = wp.tid()

    # Compute cell center coordinate from tid
    iz = tid % res[2]
    iy = (tid // res[2]) % res[1]
    ix = tid // (res[1] * res[2])

    p = bounds_lo + wp.vec3(
        (float(ix) + 0.5) * dx[0], (float(iy) + 0.5) * dx[1], (float(iz) + 0.5) * dx[2]
    )

    # Warp SDF query
    query = wp.mesh_query_point(mesh, p, 10.0)
    if query.result and query.sign < 0.0:
        mask[tid] = 1


@wp.kernel
def check_tag_sdf_kernel(
    mesh: wp.uint64,
    bounds_lo: wp.vec3,
    dx: wp.vec3,
    res: wp.vec3i,
    mask: wp.array(dtype=int),
    threshold: float,
):
    tid = wp.tid()
    iz = tid % res[2]
    iy = (tid // res[2]) % res[1]
    ix = tid // (res[1] * res[2])

    p = bounds_lo + wp.vec3(
        (float(ix) + 0.5) * dx[0], (float(iy) + 0.5) * dx[1], (float(iz) + 0.5) * dx[2]
    )

    # Use large max_dist to find interior voxels too (not just surface-near)
    query = wp.mesh_query_point(mesh, p, 10.0)
    if query.result:
        # Inside the shape: always tag
        if query.sign < 0.0:
            mask[tid] = 1
        else:
            # Outside but within threshold of surface: tag (captures boundary voxels)
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            if wp.length(p - closest) <= threshold:
                mask[tid] = 1


def voxelize_solid(
    solid, res: tuple[int, int, int], tags=None, shapes=None, body=None
) -> VoxelDomain:
    """Voxelize a build123d solid into a Warp domain."""
    bb = solid.bounding_box()
    margin = 0.002
    bounds_lo = wp.vec3(bb.min.X - margin, bb.min.Y - margin, bb.min.Z - margin)
    bounds_hi = wp.vec3(bb.max.X + margin, bb.max.Y + margin, bb.max.Z + margin)

    res_wp = wp.vec3i(*res)
    grid = fem.Grid3D(res=res_wp, bounds_lo=bounds_lo, bounds_hi=bounds_hi)

    dx = wp.vec3(
        (bounds_hi[0] - bounds_lo[0]) / float(res[0]),
        (bounds_hi[1] - bounds_lo[1]) / float(res[1]),
        (bounds_hi[2] - bounds_lo[2]) / float(res[2]),
    )

    verts, faces = solid.tessellate(tolerance=0.001)
    verts_np = np.array([(v.X, v.Y, v.Z) for v in verts], dtype=np.float32)
    faces_np = np.array(faces, dtype=np.int32)
    wp_mesh = wp.Mesh(
        points=wp.array(verts_np, dtype=wp.vec3),
        indices=wp.array(faces_np.flatten(), dtype=int),
    )

    print(f"  Voxelizing mesh ({grid.cell_count()} cells)...")
    inside_mask = wp.zeros(grid.cell_count(), dtype=int)
    wp.launch(
        check_inside_kernel,
        dim=grid.cell_count(),
        inputs=(wp_mesh.id, bounds_lo, dx, res_wp, inside_mask),
    )

    tag_masks = {}
    if tags and shapes:
        print(f"  Processing {len(tags._declarations)} tags...")
        # Compute max voxel half-diagonal for proximity threshold
        voxel_diag = float(np.sqrt(dx[0] ** 2 + dx[1] ** 2 + dx[2] ** 2)) * 0.5

        for tag_name in tags._declarations:
            try:
                ref = tags.source_ref(tag_name)
                source_shape = shapes.get(ref.id)
                if not source_shape:
                    continue

                # Tessellate tagged shape and create Warp mesh for SDF query
                try:
                    t_verts, t_faces = source_shape.tessellate(tolerance=0.001)
                except Exception:
                    continue
                if len(t_verts) < 4 or len(t_faces) < 1:
                    continue

                t_verts_np = np.array(
                    [(v.X, v.Y, v.Z) for v in t_verts], dtype=np.float32
                )
                t_faces_np = np.array(t_faces, dtype=np.int32)
                t_wp_mesh = wp.Mesh(
                    points=wp.array(t_verts_np, dtype=wp.vec3),
                    indices=wp.array(t_faces_np.flatten(), dtype=int),
                )

                mask = wp.zeros(grid.cell_count(), dtype=int)
                wp.launch(
                    check_tag_sdf_kernel,
                    dim=grid.cell_count(),
                    inputs=(t_wp_mesh.id, bounds_lo, dx, res_wp, mask, voxel_diag),
                )
                n_tagged = int(np.sum(mask.numpy()))
                if n_tagged > 0:
                    print(f"    tag: {tag_name} ({n_tagged} voxels)")
                    tag_masks[tag_name] = mask
            except Exception as e:
                print(f"    Warning: failed to process tag {tag_name}: {e}")
                continue

    if "fastener_hole" not in tag_masks:
        print("  Warning: No 'fastener_hole' tag found. FEA will not have fixed BCs.")

    return VoxelDomain(grid, grid.cell_count(), inside_mask, tag_masks)
