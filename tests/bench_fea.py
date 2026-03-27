"""FEA benchmark: cantilever beam at multiple resolutions.

Validates stress convergence toward analytical solution and tracks performance.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import warp as wp
import warp.fem as fem
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import cg

# PLA material
E = 2.3e9
NU = 0.35
LAME = wp.vec2(
    (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU)),
    E / (2.0 * (1.0 + NU)),
)

# Beam geometry: 100mm x 20mm x 20mm
BEAM_L = 0.1
BEAM_H = 0.02
BEAM_W = 0.02
TIP_FORCE = 100.0  # N

# Analytical max bending stress
ANALYTICAL_STRESS_MPA = 7.5


@wp.func
def hooke_stress(strain: wp.mat33, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(
        n=3, dtype=float
    )


@fem.integrand
def elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def classify_left(s: fem.Sample, domain: fem.Domain, mask: wp.array(dtype=int)):
    x = fem.position(domain, s)
    # Threshold must capture at least one cell at all resolutions.
    # At res_x=8, cell centers are at x=0.00625, 0.01875, ...
    # Use 0.008 to reliably capture the first cell layer.
    if x[0] < 0.008:
        mask[s.element_index] = 1


@fem.integrand
def von_mises_integrand(s: fem.Sample, u: fem.Field, lame: wp.vec2):
    eps = fem.D(u, s)
    sigma = hooke_stress(eps, lame)
    trace_sigma = wp.trace(sigma)
    s_dev = sigma - (1.0 / 3.0) * trace_sigma * wp.identity(n=3, dtype=float)
    return wp.sqrt(1.5 * wp.ddot(s_dev, s_dev))


def solve_cantilever(res_x: int) -> tuple[float, float, float]:
    """Solve cantilever beam. Returns (max_stress_mpa, max_disp_mm, wall_seconds)."""
    t0 = time.monotonic()

    # Aspect-ratio-aware resolution
    aspect = BEAM_L / BEAM_H
    res_yz = max(4, int(res_x / aspect))
    res = wp.vec3i(res_x, res_yz, res_yz)

    grid = fem.Grid3D(
        res=res,
        bounds_lo=wp.vec3(0.0, 0.0, 0.0),
        bounds_hi=wp.vec3(BEAM_L, BEAM_W, BEAM_H),
    )
    space = fem.make_polynomial_space(grid, degree=1, dtype=wp.vec3)
    domain = fem.Cells(grid)

    # Fixed BC on left face
    left_mask = wp.zeros(grid.cell_count(), dtype=int)
    fem.interpolate(classify_left, at=domain, values={"mask": left_mask})
    left_sub = fem.Subdomain(domain, element_mask=left_mask)
    u_left_test = fem.make_test(space=space, domain=left_sub)
    u_left_trial = fem.make_trial(space=space, domain=left_sub)
    bd_matrix = fem.integrate(
        boundary_projector_form,
        fields={"u": u_left_trial, "v": u_left_test},
        output_dtype=wp.float32,
    )
    fem.normalize_dirichlet_projector(bd_matrix)

    # Stiffness
    u_test = fem.make_test(space=space)
    u_trial = fem.make_trial(space=space)
    stiffness = fem.integrate(
        elasticity_form,
        fields={"u": u_trial, "v": u_test},
        values={"lame": LAME},
        output_dtype=wp.float32,
    )

    # Load on right-end nodes
    node_pos = space.node_positions().numpy()
    right_nodes = np.where(node_pos[:, 0] > BEAM_L - 0.002)[0]
    load_vec = np.zeros((space.node_count(), 3), dtype=np.float32)
    if len(right_nodes) > 0:
        load_vec[right_nodes, 2] = -TIP_FORCE / len(right_nodes)
    load_wp = wp.from_numpy(load_vec, dtype=wp.vec3f)

    fem.project_linear_system(stiffness, load_wp, bd_matrix)

    # Solve
    data = stiffness.values.numpy()
    cols = stiffness.columns.numpy()
    offsets = stiffness.offsets.numpy()
    nnz = offsets[-1]
    data = data[:nnz]
    cols = cols[:nnz]
    if len(data.shape) == 1:
        data = data.reshape(-1, 3, 3)

    A = bsr_matrix(
        (data, cols, offsets), shape=(space.node_count() * 3, space.node_count() * 3)
    )
    b = load_wp.numpy().flatten().astype(np.float64)
    x_sol, info = cg(A, b, rtol=1e-8)
    assert info == 0, f"CG solver failed with info={info}"

    # Von Mises stress
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

    max_stress_mpa = float(stress_array.numpy().max()) / 1e6
    max_disp_mm = float(np.max(np.linalg.norm(x_sol.reshape(-1, 3), axis=1))) * 1000
    wall_s = time.monotonic() - t0

    return max_stress_mpa, max_disp_mm, wall_s


# --- Tests ---


@pytest.mark.parametrize("res_x", [8, 16, 32])
def test_cantilever_convergence(res_x):
    """Stress should converge toward analytical 7.5 MPa as resolution increases."""
    wp.init()
    stress_mpa, disp_mm, wall_s = solve_cantilever(res_x)

    print(
        f"\n  res_x={res_x}: stress={stress_mpa:.2f} MPa, disp={disp_mm:.4f} mm, time={wall_s:.2f}s"
    )

    # At any resolution, stress should be in the right ballpark (within 5x)
    assert 1.0 < stress_mpa < 40.0, f"Stress {stress_mpa} MPa is unreasonable"


def test_cantilever_stress_order():
    """Higher resolution should give stress closer to analytical."""
    wp.init()
    stress_8, _, _ = solve_cantilever(8)
    stress_32, _, _ = solve_cantilever(32)

    # 32x should be closer to 7.5 MPa than 8x
    err_8 = abs(stress_8 - ANALYTICAL_STRESS_MPA)
    err_32 = abs(stress_32 - ANALYTICAL_STRESS_MPA)
    assert err_32 < err_8, (
        f"32x error ({err_32:.2f}) should be < 8x error ({err_8:.2f})"
    )


def test_cantilever_performance():
    """Benchmark: 32x should complete in under 30 seconds."""
    wp.init()
    _, _, wall_s = solve_cantilever(32)
    print(f"\n  32x solve: {wall_s:.2f}s")
    assert wall_s < 30.0, f"32x solve took {wall_s:.1f}s, expected < 30s"
