"""Component-level structural analysis.

Takes a CAD solid + ShapeScript execution result and runs a Warp FEM solve.
Automatically identifies BCs and Loads from tags.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.fem as fem
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import cg

from botcad.fea.voxelizer import voxelize_solid
from botcad.materials import PLA


@wp.func
def hooke_stress(strain: wp.mat33, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(
        n=3, dtype=float
    )


@fem.integrand
def hooke_elasticity_form(
    s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2, mask: wp.array(dtype=int)
):
    # Integrate CAD solid
    weight = 1.0
    if mask[s.element_index] == 0:
        weight = 1e-3  # Tiny penalty for air to prevent singular matrix
    return weight * wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def von_mises_integrand(s: fem.Sample, u: fem.Field, lame: wp.vec2):
    eps = fem.D(u, s)
    sigma = hooke_stress(eps, lame)
    trace_sigma = wp.trace(sigma)
    s_dev = sigma - (1.0 / 3.0) * trace_sigma * wp.identity(n=3, dtype=float)
    v_m = wp.sqrt(1.5 * wp.ddot(s_dev, s_dev))
    return v_m


def analyze_component(
    solid, result, torque_nm: float, res=(20, 20, 20), material=PLA, body=None
):
    """Run FEA on a single CAD component.

    Returns (u_field, stress_array, voxel_domain, timings_dict)
    """
    import time

    timings = {}

    t0 = time.monotonic()
    print(f"Voxelizing component at resolution {res}...")
    vd = voxelize_solid(solid, res, tags=result.tags, shapes=result.shapes, body=body)
    t1 = time.monotonic()
    timings["voxelization"] = round(t1 - t0, 2)

    # Material properties (infill-scaled via Gibson-Ashby)
    E = material.effective_youngs_modulus
    nu = material.poisson_ratio
    lame = wp.vec2(
        (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        E / (2.0 * (1.0 + nu)),
    )

    space = fem.make_polynomial_space(vd.grid, degree=1, dtype=wp.vec3)
    domain = fem.Cells(vd.grid)

    # 1. Identify Fixed and Load BCs from available tags
    #
    # Tag-to-BC mapping (priority order):
    #   Fixed (anchored) side:
    #     - "fastener_hole" — bolt holes (ideal, but rarely emitted currently)
    #     - "coupler" — where body connects to parent joint's horn
    #   Load (force) side:
    #     - "bracket_*" (excluding "bracket_env_*") — servo bracket interfaces
    #     - "front_plate", "pocket", "horn_hole" — bracket sub-features
    #
    # For base bodies (no coupler): largest bracket region = fixed, others = load.

    fixed_mask = None
    load_mask = None

    # Collect candidate masks
    FIXED_KEYWORDS = ["fastener_hole", "coupler"]
    LOAD_KEYWORDS = ["front_plate", "pocket", "horn_hole"]

    bracket_masks: list[tuple[str, int]] = []  # (tag_name, voxel_count)

    for tag, mask in vd.tag_masks.items():
        mask_count = int(np.sum(mask.numpy()))
        if mask_count == 0:
            continue

        # Skip envelope tags and wire tags — not structural
        if "env" in tag or "wire" in tag:
            continue

        if any(kw in tag for kw in FIXED_KEYWORDS):
            if fixed_mask is None:
                fixed_mask = mask
                print(f"  Fixed BC: {tag} ({mask_count} voxels)")
            continue

        if "bracket" in tag or any(kw in tag for kw in LOAD_KEYWORDS):
            bracket_masks.append((tag, mask_count))
            continue

    # Assign load from bracket masks
    if bracket_masks:
        # Sort by voxel count descending
        bracket_masks.sort(key=lambda x: x[1], reverse=True)

        if fixed_mask is None:
            # No coupler/fastener — base body. Use largest bracket as fixed.
            fixed_tag, fixed_count = bracket_masks[0]
            fixed_mask = vd.tag_masks[fixed_tag]
            print(f"  Fixed BC (fallback): {fixed_tag} ({fixed_count} voxels)")
            # Use next bracket as load
            if len(bracket_masks) > 1:
                load_tag, load_count = bracket_masks[1]
                load_mask = vd.tag_masks[load_tag]
                print(f"  Load BC: {load_tag} ({load_count} voxels)")
        else:
            # Have a proper fixed BC — use first bracket as load
            load_tag, load_count = bracket_masks[0]
            load_mask = vd.tag_masks[load_tag]
            print(f"  Load BC: {load_tag} ({load_count} voxels)")

    if fixed_mask is None:
        available = list(vd.tag_masks.keys()) if vd.tag_masks else []
        print(f"  No fixed BC tags found. Available tags: {available}")
        return None

    n_fixed = int(np.sum(fixed_mask.numpy()))
    if n_fixed == 0:
        return None

    if load_mask is None:
        available = list(vd.tag_masks.keys()) if vd.tag_masks else []
        print(f"  No load BC tags found. Available tags: {available}")
        return None

    # Build Dirichlet projector for fixed BCs
    fixed_subdomain = fem.Subdomain(domain, element_mask=fixed_mask)
    u_fixed_test = fem.make_test(space=space, domain=fixed_subdomain)
    u_fixed_trial = fem.make_trial(space=space, domain=fixed_subdomain)
    bd_matrix = fem.integrate(
        boundary_projector_form,
        fields={"u": u_fixed_trial, "v": u_fixed_test},
        output_dtype=wp.float32,
    )
    fem.normalize_dirichlet_projector(bd_matrix)

    # Final Solve
    t_asm_start = time.monotonic()
    u_test = fem.make_test(space=space)
    u_trial = fem.make_trial(space=space)
    stiffness = fem.integrate(
        hooke_elasticity_form,
        fields={"u": u_trial, "v": u_test},
        values={"lame": lame, "mask": vd.inside_mask},
        output_dtype=wp.float32,
    )
    t_asm_end = time.monotonic()
    timings["assembly"] = round(t_asm_end - t_asm_start, 2)

    # --- Build load vector from torque and load mask ---
    load_vec_np = np.zeros((space.node_count(), 3), dtype=np.float32)

    # Find centroids of fixed and loaded regions
    load_mask_np = load_mask.numpy()
    fixed_mask_np = fixed_mask.numpy()
    res_arr = [vd.grid.res[0], vd.grid.res[1], vd.grid.res[2]]
    lo = vd.grid.bounds_lo
    hi = vd.grid.bounds_hi
    dx_arr = [(hi[i] - lo[i]) / res_arr[i] for i in range(3)]

    def _cell_positions(mask_np):
        idxs = np.where(mask_np == 1)[0]
        if len(idxs) == 0:
            return np.zeros((0, 3))
        iz = idxs % res_arr[2]
        iy = (idxs // res_arr[2]) % res_arr[1]
        ix = idxs // (res_arr[1] * res_arr[2])
        return np.column_stack(
            [
                lo[0] + (ix + 0.5) * dx_arr[0],
                lo[1] + (iy + 0.5) * dx_arr[1],
                lo[2] + (iz + 0.5) * dx_arr[2],
            ]
        )

    fixed_pos = _cell_positions(fixed_mask_np)
    load_pos = _cell_positions(load_mask_np)

    if len(fixed_pos) == 0 or len(load_pos) == 0:
        return None

    fixed_centroid = fixed_pos.mean(axis=0)
    load_centroid = load_pos.mean(axis=0)

    # Arm direction: from fixed toward load region
    arm_dir = load_centroid - fixed_centroid
    arm_length = float(np.linalg.norm(arm_dir))
    if arm_length < 1e-9:
        return None
    arm_dir = arm_dir / arm_length

    # Force direction: PERPENDICULAR to arm (bending, not axial)
    # Cross arm with Z-up to get perpendicular; if arm is nearly vertical, use X
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(arm_dir, up)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    force_dir = np.cross(arm_dir, up)
    force_dir = force_dir / np.linalg.norm(force_dir)

    # Force magnitude from torque: F = T / r
    force_magnitude = torque_nm / arm_length

    # Select load nodes: nodes whose position falls within load region AABB
    node_positions = space.node_positions().numpy()
    load_aabb_lo = load_pos.min(axis=0) - np.array(dx_arr) * 0.5
    load_aabb_hi = load_pos.max(axis=0) + np.array(dx_arr) * 0.5
    in_load = (
        (node_positions[:, 0] >= load_aabb_lo[0])
        & (node_positions[:, 0] <= load_aabb_hi[0])
        & (node_positions[:, 1] >= load_aabb_lo[1])
        & (node_positions[:, 1] <= load_aabb_hi[1])
        & (node_positions[:, 2] >= load_aabb_lo[2])
        & (node_positions[:, 2] <= load_aabb_hi[2])
    )
    n_load_nodes = int(np.sum(in_load))
    if n_load_nodes == 0:
        print("  Warning: no nodes found in load region AABB")
        return None
    per_node_force = force_magnitude / n_load_nodes
    load_vec_np[in_load] = (force_dir * per_node_force).astype(np.float32)

    load_wp = wp.from_numpy(load_vec_np, dtype=wp.vec3f)

    t_proj_start = time.monotonic()
    fem.project_linear_system(stiffness, load_wp, bd_matrix)
    t_proj_end = time.monotonic()
    timings["projection"] = round(t_proj_end - t_proj_start, 2)

    data_raw = stiffness.values.numpy()
    cols_raw = stiffness.columns.numpy()
    offsets = stiffness.offsets.numpy()
    nnz = offsets[-1]

    data = data_raw[:nnz]
    cols = cols_raw[:nnz]

    if len(data.shape) == 1:
        data = data.reshape(-1, 3, 3)

    A_scipy = bsr_matrix(
        (data, cols, offsets), shape=(space.node_count() * 3, space.node_count() * 3)
    )
    b_scipy = load_wp.numpy().flatten().astype(np.float64)

    t_sol_start = time.monotonic()
    x_sol, info = cg(A_scipy, b_scipy, rtol=1e-8)
    t_sol_end = time.monotonic()
    timings["solve"] = round(t_sol_end - t_sol_start, 2)

    if info == 0:
        u_field = fem.make_discrete_field(space)
        u_field.dof_values = wp.from_numpy(
            x_sol.reshape(-1, 3).astype(np.float32), dtype=wp.vec3
        )
        stress_array = wp.zeros(domain.element_count(), dtype=float)

        t_heat_start = time.monotonic()
        fem.interpolate(
            von_mises_integrand,
            dest=stress_array,
            at=domain,
            fields={"u": u_field},
            values={"lame": lame},
        )
        t_heat_end = time.monotonic()
        timings["heatmap"] = round(t_heat_end - t_heat_start, 2)

        return u_field, stress_array, vd, timings
    return None
