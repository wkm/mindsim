"""Step 0 POC: Simple 3D FEA solver using warp.fem.

Simulates a cantilever beam under a point load.
Calculates displacement and von Mises stress.
"""

import numpy as np
import warp as wp
import warp.fem as fem
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import cg

# Set simulation parameters (matching our PLA material)
YOUNGS_MODULUS = 2.3e9  # Pa
POISSON_RATIO = 0.35

# Lame parameters for linear elasticity
LAME = wp.vec2(
    (YOUNGS_MODULUS * POISSON_RATIO)
    / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO)),
    YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO)),
)


@wp.func
def hooke_stress(strain: wp.mat33, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(
        n=3, dtype=float
    )


@fem.integrand
def hooke_elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def classify_boundaries(
    s: fem.Sample,
    domain: fem.Domain,
    left: wp.array(dtype=int),
    right: wp.array(dtype=int),
):
    x = fem.position(domain, s)
    if x[0] < 0.006:
        left[s.element_index] = 1
    if x[0] > 0.094:
        right[s.element_index] = 1


@fem.integrand
def von_mises_integrand(s: fem.Sample, u: fem.Field, lame: wp.vec2):
    eps = fem.D(u, s)
    sigma = hooke_stress(eps, lame)
    trace_sigma = wp.trace(sigma)
    s_dev = sigma - (1.0 / 3.0) * trace_sigma * wp.identity(n=3, dtype=float)
    v_m = wp.sqrt(1.5 * wp.ddot(s_dev, s_dev))
    return v_m


def solve_beam():
    print("Initializing Warp FEM 3D POC...")
    wp.init()

    # 1. Mesh
    res = wp.vec3i(20, 4, 4)
    bounds_lo = wp.vec3(0.0, 0.0, 0.0)
    bounds_hi = wp.vec3(0.1, 0.02, 0.02)
    grid = fem.Grid3D(res=res, bounds_lo=bounds_lo, bounds_hi=bounds_hi)

    # 2. Function Space
    space = fem.make_polynomial_space(grid, degree=1, dtype=wp.vec3)

    # 3. Identify boundaries
    domain = fem.Cells(grid)
    left_mask = wp.zeros(grid.cell_count(), dtype=int)
    right_mask = wp.zeros(grid.cell_count(), dtype=int)
    fem.interpolate(
        classify_boundaries, at=domain, values={"left": left_mask, "right": right_mask}
    )

    left_side = fem.Subdomain(domain, element_mask=left_mask)

    # 4. Dirichlet Projector
    u_left_test = fem.make_test(space=space, domain=left_side)
    u_left_trial = fem.make_trial(space=space, domain=left_side)
    bd_matrix = fem.integrate(
        boundary_projector_form,
        fields={"u": u_left_trial, "v": u_left_test},
        output_dtype=wp.float32,
    )
    fem.normalize_dirichlet_projector(bd_matrix)

    # 5. Global Assembly
    u_test = fem.make_test(space=space)
    u_trial = fem.make_trial(space=space)

    print("Assembling stiffness matrix...")
    stiffness = fem.integrate(
        hooke_elasticity_form,
        fields={"u": u_trial, "v": u_test},
        values={"lame": LAME},
        output_dtype=wp.float32,
    )

    print("Assembling load vector manually...")
    node_positions = space.node_positions().numpy()
    right_node_indices = np.where(node_positions[:, 0] > 0.099)[0]
    load_vec = np.zeros((space.node_count(), 3), dtype=np.float32)
    if len(right_node_indices) > 0:
        load_vec[right_node_indices, 2] = -100.0 / len(right_node_indices)
    load_wp = wp.from_numpy(load_vec, dtype=wp.vec3f)

    # 6. Apply BCs
    fem.project_linear_system(stiffness, load_wp, bd_matrix)

    # 7. Solve
    print(f"Solving linear system ({space.node_count() * 3} DOFs)...")
    data = stiffness.values.numpy().reshape(-1, 3, 3)
    cols = stiffness.columns.numpy()
    offsets = stiffness.offsets.numpy()
    A_scipy = bsr_matrix(
        (data, cols, offsets), shape=(space.node_count() * 3, space.node_count() * 3)
    )
    b_scipy = load_wp.numpy().flatten().astype(np.float64)
    x_sol, info = cg(A_scipy, b_scipy, rtol=1e-8)

    if info == 0:
        # 8. Compute von Mises stress
        print("Computing von Mises stress heatmap...")
        u_field = fem.make_discrete_field(space)
        u_field.dof_values = wp.from_numpy(
            x_sol.reshape(-1, 3).astype(np.float32), dtype=wp.vec3
        )

        stress_array = wp.zeros(domain.element_count(), dtype=float)
        fem.interpolate(
            von_mises_integrand,
            dest=stress_array,
            at=domain,
            fields={"u": u_field},
            values={"lame": LAME},
        )

        stress_numpy = stress_array.numpy()
        max_stress = np.max(stress_numpy)
        max_disp = np.max(np.linalg.norm(x_sol.reshape(-1, 3), axis=1))

        print("\n--- Final POC Results ---")
        print(f"Max Displacement: {max_disp * 1000:.4f} mm")
        print(f"Max von Mises Stress: {max_stress / 1e6:.2f} MPa")
        print("Analytical Max Stress: ~7.5 MPa")

        if max_stress > 40e6:
            print("STATUS: DESIGN WILL FAIL (Yielded)")
        else:
            print(f"STATUS: DESIGN SAFE (SF: {40e6 / max_stress:.2f})")

    else:
        print(f"Solver failed with info {info}")


if __name__ == "__main__":
    solve_beam()
