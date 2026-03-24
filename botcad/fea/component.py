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


def analyze_component(solid, result, torque_nm: float, res=(20, 20, 20), material=PLA):
    """Run FEA on a single CAD component."""
    print(f"Voxelizing component at resolution {res}...")
    vd = voxelize_solid(solid, res, tags=result.tags, shapes=result.shapes)

    # Material properties
    lame = wp.vec2(
        (material.youngs_modulus * material.poisson_ratio)
        / ((1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio)),
        material.youngs_modulus / (2.0 * (1.0 + material.poisson_ratio)),
    )

    space = fem.make_polynomial_space(vd.grid, degree=1, dtype=wp.vec3)
    domain = fem.Cells(vd.grid)

    # 1. Identify Fixed BCs (from fastener_hole tag)
    fixed_mask = vd.tag_masks.get("fastener_hole")
    n_fixed = np.sum(fixed_mask.numpy()) if fixed_mask else 0
    print(f"Fixed cells (tags): {n_fixed}")

    if n_fixed == 0:
        print("Warning: No 'fastener_hole' tag found. Simulation will likely fail.")
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

    # 2. Identify Loads (from horn_hole or front_plate or pocket)
    load_mask = (
        vd.tag_masks.get("front_plate")
        or vd.tag_masks.get("pocket")
        or vd.tag_masks.get("horn_hole")
    )
    n_load = np.sum(load_mask.numpy()) if load_mask else 0
    print(f"Load cells (tags): {n_load}")

    if not load_mask:
        print("Warning: No load-bearing tag found (front_plate, pocket, horn_hole).")
        return None

    # Final Solve
    print("Assembling system...")
    u_test = fem.make_test(space=space)
    u_trial = fem.make_trial(space=space)
    stiffness = fem.integrate(
        hooke_elasticity_form,
        fields={"u": u_trial, "v": u_test},
        values={"lame": lame, "mask": vd.inside_mask},
        output_dtype=wp.float32,
    )

    # Simple test load
    load_vec_np = np.zeros((space.node_count(), 3), dtype=np.float32)
    load_vec_np[:, 2] = -1.0
    load_wp = wp.from_numpy(load_vec_np, dtype=wp.vec3f)

    print("Projecting...")
    fem.project_linear_system(stiffness, load_wp, bd_matrix)

    print("Solving...")
    data = stiffness.values.numpy().reshape(-1, 3, 3)
    cols = stiffness.columns.numpy()
    offsets = stiffness.offsets.numpy()
    A_scipy = bsr_matrix(
        (data, cols, offsets), shape=(space.node_count() * 3, space.node_count() * 3)
    )
    b_scipy = load_wp.numpy().flatten().astype(np.float64)
    x_sol, info = cg(A_scipy, b_scipy, rtol=1e-8)
    print(f"Solver status: info={info}")

    if info == 0:
        u_field = fem.make_discrete_field(space)
        u_field.dof_values = wp.from_numpy(
            x_sol.reshape(-1, 3).astype(np.float32), dtype=wp.vec3
        )
        stress_array = wp.zeros(domain.element_count(), dtype=float)
        print("Computing stress heatmap...")
        fem.interpolate(
            von_mises_integrand,
            dest=stress_array,
            at=domain,
            fields={"u": u_field},
            values={"lame": lame},
        )
        return u_field, stress_array, vd
    return None
