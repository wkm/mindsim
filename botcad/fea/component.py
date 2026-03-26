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

    # Material properties
    lame = wp.vec2(
        (material.youngs_modulus * material.poisson_ratio)
        / ((1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio)),
        material.youngs_modulus / (2.0 * (1.0 + material.poisson_ratio)),
    )

    space = fem.make_polynomial_space(vd.grid, degree=1, dtype=wp.vec3)
    domain = fem.Cells(vd.grid)

    # 1. Identify Fixed BCs
    fixed_mask = vd.tag_masks.get("fastener_hole")
    n_fixed = np.sum(fixed_mask.numpy()) if fixed_mask is not None else 0

    if n_fixed == 0:
        return None

    fixed_subdomain = fem.Subdomain(domain, element_mask=fixed_mask)
    u_fixed_test = fem.make_test(space=space, domain=fixed_subdomain)
    u_fixed_trial = fem.make_trial(space=space, domain=fixed_subdomain)
    bd_matrix = fem.integrate(
        boundary_projector_form,
        fields={"u": u_fixed_trial, "v": u_fixed_test},
        output_dtype=wp.float32,
    )
    fem.normalize_dirichlet_projector(bd_matrix)

    # 2. Identify Loads
    load_mask = None
    for tag, mask in vd.tag_masks.items():
        if any(
            keyword in tag
            for keyword in ["front_plate", "pocket", "horn_hole", "bracket"]
        ):
            load_mask = mask
            break

    if not load_mask:
        return None

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

    # Simple test load
    load_vec_np = np.zeros((space.node_count(), 3), dtype=np.float32)
    load_vec_np[:, 2] = -1.0
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
