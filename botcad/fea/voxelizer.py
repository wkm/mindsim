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
    if query.result:
        if query.sign < 0.0:
            mask[tid] = 1


@wp.kernel
def check_tag_kernel(
    bounds_lo: wp.vec3,
    dx: wp.vec3,
    res: wp.vec3i,
    mask: wp.array(dtype=int),
    t_lo: wp.vec3,
    t_hi: wp.vec3,
):
    tid = wp.tid()
    iz = tid % res[2]
    iy = (tid // res[2]) % res[1]
    ix = tid // (res[1] * res[2])

    p = bounds_lo + wp.vec3(
        (float(ix) + 0.5) * dx[0], (float(iy) + 0.5) * dx[1], (float(iz) + 0.5) * dx[2]
    )

    if (
        p[0] >= t_lo[0]
        and p[0] <= t_hi[0]
        and p[1] >= t_lo[1]
        and p[1] <= t_hi[1]
        and p[2] >= t_lo[2]
        and p[2] <= t_hi[2]
    ):
        mask[tid] = 1


def voxelize_solid(
    solid, res: tuple[int, int, int], tags=None, shapes=None
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
        for tag_name in tags._declarations.keys():
            try:
                print(f"    tag: {tag_name}")
                ref = tags.source_ref(tag_name)
                source_shape = shapes.get(ref.id)
                if not source_shape:
                    continue

                t_bb = source_shape.bounding_box()
                mask = wp.zeros(grid.cell_count(), dtype=int)

                t_margin = 0.001
                t_lo = wp.vec3(
                    t_bb.min.X - t_margin, t_bb.min.Y - t_margin, t_bb.min.Z - t_margin
                )
                t_hi = wp.vec3(
                    t_bb.max.X + t_margin, t_bb.max.Y + t_margin, t_bb.max.Z + t_margin
                )

                wp.launch(
                    check_tag_kernel,
                    dim=grid.cell_count(),
                    inputs=(bounds_lo, dx, res_wp, mask, t_lo, t_hi),
                )
                tag_masks[tag_name] = mask
            except Exception as e:
                print(f"    Warning: failed to process tag {tag_name}: {e}")
                continue

    return VoxelDomain(grid, grid.cell_count(), inside_mask, tag_masks)
