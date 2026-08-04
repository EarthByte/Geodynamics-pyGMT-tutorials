"""
Verification tests for geodynkit.

These are not smoke tests. Each one checks the solver against something
external — an analytic solution, a published benchmark, or an invariant that
must hold exactly — because a geodynamic code that runs without crashing and
produces a plausible-looking picture is the single most common way to be wrong.
"""

import numpy as np
import pytest

from geodynkit.advection import advect_1d, courant_dt
from geodynkit.convection import BLANKENBACH_1A, ConvectionModel
from geodynkit.diffusion import gaussian_analytic_1d, solve_1d_explicit, solve_1d_implicit
from geodynkit.markers import MarkerSet, markers_to_grid
from geodynkit.stokes import (
    StokesGrid,
    manufactured_solution,
    solve_stokes,
    velocity_to_centres,
)


# ---------------------------------------------------------------- diffusion
def test_diffusion_matches_analytic_gaussian():
    """A diffusing Gaussian must follow sqrt(w^2 + 4 kappa t)."""
    nx, L, kappa = 401, 20.0, 0.5
    x = np.linspace(-L / 2, L / 2, nx)
    dx = x[1] - x[0]
    T0 = gaussian_analytic_1d(x, 0.0, kappa, width=1.0)

    dt = 0.4 * dx**2 / (2 * kappa)
    nsteps = 400
    hist = solve_1d_explicit(T0, kappa, dx, dt, nsteps, bc=("neumann", "neumann"))
    expected = gaussian_analytic_1d(x, nsteps * dt, kappa, width=1.0)

    assert np.max(np.abs(hist[-1] - expected)) < 2e-3


def test_explicit_scheme_refuses_unstable_timestep():
    x = np.linspace(0, 1, 51)
    dx = x[1] - x[0]
    with pytest.raises(ValueError, match="stability limit"):
        solve_1d_explicit(np.sin(np.pi * x), 1.0, dx, dt=dx**2, nsteps=1)


def test_implicit_is_stable_at_large_timestep():
    """Backward Euler must stay bounded where the explicit scheme would blow up."""
    x = np.linspace(0, 1, 51)
    dx = x[1] - x[0]
    T0 = np.sin(np.pi * x)
    hist = solve_1d_implicit(T0, 1.0, dx, dt=50 * dx**2, nsteps=50)
    assert np.all(np.isfinite(hist))
    assert hist[-1].max() <= T0.max() + 1e-12       # no spurious growth
    assert hist[-1].max() < 0.5 * T0.max()          # it did actually diffuse


# ---------------------------------------------------------------- advection
def _square_wave(nx=200):
    dx = 1.0 / nx
    return dx, np.where(np.abs(np.arange(nx) * dx - 0.3) < 0.1, 1.0, 0.0)


def test_ftcs_is_unstable_even_below_the_courant_limit():
    """Accuracy and stability are different properties — the lesson of rung 03.

    FTCS is second-order accurate in space and unconditionally unstable for
    pure advection. At Courant 0.5 it grows by eight orders of magnitude in
    200 steps.
    """
    dx, C0 = _square_wave()
    dt = courant_dt(1.0, dx, cfl=0.5)
    ftcs = advect_1d(C0, 1.0, dx, dt, 200, scheme="ftcs")[-1]
    assert ftcs.max() > 1e3 * C0.max()


def test_upwind_is_monotone_but_lax_wendroff_overshoots():
    """First order buys monotonicity; second order buys ripples."""
    dx, C0 = _square_wave()
    dt = courant_dt(1.0, dx, cfl=0.5)
    up = advect_1d(C0, 1.0, dx, dt, 200, scheme="upwind")[-1]
    lw = advect_1d(C0, 1.0, dx, dt, 200, scheme="lax_wendroff")[-1]

    assert up.min() >= -1e-12 and up.max() <= 1.0 + 1e-12   # no new extrema
    assert lw.min() < -0.05                                  # undershoot
    assert lw.max() > 1.05                                   # overshoot


def test_upwind_is_diffusive():
    """A Gaussian loses ~18% of its peak in 400 upwind steps, purely numerically."""
    nx = 200
    dx = 1.0 / nx
    x = np.arange(nx) * dx
    C0 = np.exp(-((x - 0.3) ** 2) / 0.01)
    dt = courant_dt(1.0, dx, cfl=0.5)
    up = advect_1d(C0, 1.0, dx, dt, 400, scheme="upwind")[-1]
    assert 0.7 < up.max() < 0.9 * C0.max()


def test_semi_lagrangian_equals_upwind_below_courant_one():
    """Not a bug — a fact worth teaching.

    Semi-Lagrangian advection with linear interpolation and Courant <= 1 traces
    the departure point less than one cell back, so the interpolation weights
    are exactly the upwind coefficients. The two schemes are identical to
    machine precision. Their behaviour only diverges above Courant 1.
    """
    dx, C0 = _square_wave()
    dt = courant_dt(1.0, dx, cfl=0.5)
    up = advect_1d(C0, 1.0, dx, dt, 100, scheme="upwind")[-1]
    sl = advect_1d(C0, 1.0, dx, dt, 100, scheme="semi_lagrangian")[-1]
    assert np.allclose(up, sl, rtol=1e-12, atol=1e-14)


def test_semi_lagrangian_survives_above_courant_one():
    """This is why real geodynamic codes use it: no Courant restriction."""
    dx, C0 = _square_wave()
    dt = courant_dt(1.0, dx, cfl=3.0)
    sl = advect_1d(C0, 1.0, dx, dt, 200, scheme="semi_lagrangian")[-1]
    up = advect_1d(C0, 1.0, dx, dt, 200, scheme="upwind")[-1]

    assert sl.min() >= -1e-12 and sl.max() <= 1.0 + 1e-12    # still bounded
    assert up.max() > 1e6                                     # upwind has died


# ------------------------------------------------------------------- Stokes
@pytest.mark.parametrize("resolutions", [(16, 32)])
def test_stokes_converges_at_second_order(resolutions):
    """Manufactured solution: doubling resolution must cut the error ~4x."""
    errors = []
    for n in resolutions:
        grid = StokesGrid(n, n, 1.0, 1.0)
        exact = manufactured_solution(grid, eta=1.0)
        vx, vz, _ = solve_stokes(
            grid, np.ones((n, n)), np.zeros((n, n)), gz=0.0,
            fx_c=exact["fx"], fz_c=exact["fz"], bc="free-slip",
        )
        errors.append(np.sqrt(np.mean((vx - exact["vx"]) ** 2)))

    rate = np.log(errors[0] / errors[1]) / np.log(resolutions[1] / resolutions[0])
    assert rate > 1.9, f"observed convergence rate {rate:.2f}, expected ~2"


def test_hydrostatic_case_produces_no_flow():
    """Uniform density and viscosity: the exact answer is v = 0 everywhere.

    This is the test that catches matrix-scaling bugs. In SI units the pressure
    gradient must cancel gravity to ~10 significant figures, and if the matrix
    is poorly conditioned what survives the cancellation is a spurious flow of
    centimetres per year — which looks entirely plausible on a plot.
    """
    n = 32
    grid = StokesGrid(n, n, 500e3, 500e3)
    vx, vz, _ = solve_stokes(
        grid, np.full((n, n), 1e21), np.full((n, n), 3300.0), gz=9.81
    )
    v_cm_yr = max(np.abs(vx).max(), np.abs(vz).max()) * 365.25 * 24 * 3600 * 100
    assert v_cm_yr < 1e-6, f"spurious hydrostatic flow of {v_cm_yr:g} cm/yr"


def test_incompressibility_is_satisfied():
    n = 32
    grid = StokesGrid(n, n, 500e3, 500e3)
    rng = np.random.default_rng(0)
    eta = 10.0 ** rng.uniform(20, 23, (n, n))
    rho = rng.uniform(3200, 3400, (n, n))
    vx, vz, _ = solve_stokes(grid, eta, rho, gz=9.81)
    div = (vx[:, 1:] - vx[:, :-1]) / grid.dx + (vz[1:, :] - vz[:-1, :]) / grid.dz
    scale = max(np.abs(vx).max(), np.abs(vz).max()) / grid.dx
    assert np.abs(div).max() / scale < 1e-10


def test_stiff_block_behaves_rigidly():
    """A block 10^6 times stiffer than its surroundings must sink as one piece."""
    n, L = 40, 500e3
    grid = StokesGrid(n, n, L, L)
    X, Z = np.meshgrid(grid.xc, grid.zc)
    inside = (np.abs(X - L / 2) < L / 8) & (np.abs(Z - L / 4) < L / 8)

    eta = np.where(inside, 1e27, 1e21)
    rho = np.where(inside, 3300.0, 3200.0)
    vx, vz, _ = solve_stokes(grid, eta, rho, gz=9.81)
    vxc, vzc = velocity_to_centres(grid, vx, vz)

    inner = vzc[inside]
    assert inner.mean() > 0                                   # sinks (z is depth)
    assert inner.std() / abs(inner.mean()) < 0.1              # moves as one piece


# ---------------------------------------------------------------- markers
def test_averaging_schemes_agree_at_unit_contrast():
    """With one material, arithmetic/geometric/harmonic must be identical."""
    n, L = 20, 1.0
    grid = StokesGrid(n, n, L, L)
    swarm = MarkerSet.regular(L, L, n, n, per_cell=3, seed=0)
    values = np.full(len(swarm), 1e21)
    out = [
        markers_to_grid(swarm.x, swarm.z, values, grid.xc, grid.zc, average=a)
        for a in ("arithmetic", "geometric", "harmonic")
    ]
    assert np.allclose(out[0], out[1], rtol=1e-12)
    assert np.allclose(out[0], out[2], rtol=1e-12)


def test_harmonic_average_is_bounded_by_arithmetic():
    n, L = 20, 1.0
    grid = StokesGrid(n, n, L, L)
    swarm = MarkerSet.regular(L, L, n, n, per_cell=3, seed=0)
    swarm.set_phase_where(swarm.x > L / 2, 1)
    values = swarm.property_map([1e21, 1e27])
    ari = markers_to_grid(swarm.x, swarm.z, values, grid.xc, grid.zc, "arithmetic")
    har = markers_to_grid(swarm.x, swarm.z, values, grid.xc, grid.zc,
                          average="harmonic")
    assert np.all(har <= ari * (1 + 1e-9))


# ------------------------------------------------------------- convection
@pytest.mark.slow
def test_blankenbach_richardson_extrapolation():
    """Two resolutions, extrapolated, must land on the published benchmark.

    Neither resolution alone is close — that is the point, and the lesson of
    notebook 06. The error roughly halves as the grid doubles, so the
    Richardson estimate is worth far more than either raw number.
    """
    results = {}
    for n in (16, 32):
        model = ConvectionModel(nx=n, nz=n, Ra=BLANKENBACH_1A["Ra"])
        model.run(max_steps=4000, tol=1e-4)
        results[n] = model.v_rms()

    extrapolated = results[32] - (results[16] - results[32])
    rel = abs(extrapolated - BLANKENBACH_1A["v_rms"]) / BLANKENBACH_1A["v_rms"]
    assert rel < 0.05, (
        f"extrapolated v_rms {extrapolated:.3f} vs published "
        f"{BLANKENBACH_1A['v_rms']:.3f} ({rel:.1%} off)"
    )
