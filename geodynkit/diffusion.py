"""
Heat conduction: 1-D and 2-D, explicit and implicit.

These are the first rungs of the ladder. They exist to establish three ideas
that everything later depends on: discretisation of a second derivative, the
stability limit of an explicit scheme, and why you would ever bother with an
implicit one.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "diffusion_stability_dt",
    "solve_1d_explicit",
    "solve_1d_implicit",
    "solve_2d_explicit",
    "solve_2d_implicit",
    "gaussian_analytic_1d",
]


def diffusion_stability_dt(kappa, dx, dz=None, safety=0.9):
    """Largest stable timestep for an explicit (FTCS) diffusion scheme.

    In 1-D the classic limit is ``dt <= dx^2 / (2*kappa)``; in 2-D the two
    directions share the budget, giving ``dt <= 1/(2*kappa*(1/dx^2 + 1/dz^2))``.
    The safety factor keeps you off the knife edge.
    """
    if dz is None:
        return safety * dx**2 / (2.0 * kappa)
    return safety / (2.0 * kappa * (1.0 / dx**2 + 1.0 / dz**2))


# --------------------------------------------------------------------------
# 1-D
# --------------------------------------------------------------------------
def solve_1d_explicit(T0, kappa, dx, dt, nsteps, bc=("dirichlet", "dirichlet"),
                      check_stability=True):
    """Explicit FTCS solution of dT/dt = kappa d2T/dx2.

    Returns the full history, shape ``(nsteps + 1, nx)``, so notebooks can plot
    the evolution without re-running.

    Raises if ``dt`` exceeds the stability limit — silently producing garbage is
    a worse lesson than an error message. Pass ``check_stability=False`` to
    cross the limit deliberately, which is worth doing once: the failure is not
    a gentle loss of accuracy but an exponentially growing sawtooth at the grid
    scale, exactly as the von Neumann analysis predicts.
    """
    T0 = np.asarray(T0, dtype=float)
    dt_max = diffusion_stability_dt(kappa, dx, safety=1.0)
    if check_stability and dt > dt_max:
        raise ValueError(
            f"dt={dt:g} exceeds the explicit stability limit dx^2/(2*kappa)={dt_max:g}. "
            "Reduce dt, coarsen the grid, use solve_1d_implicit, or pass "
            "check_stability=False if you are crossing the limit on purpose."
        )

    hist = np.empty((nsteps + 1, T0.size))
    hist[0] = T0
    T = T0.copy()
    s = kappa * dt / dx**2
    for n in range(nsteps):
        Tnew = T.copy()
        Tnew[1:-1] = T[1:-1] + s * (T[2:] - 2.0 * T[1:-1] + T[:-2])
        _apply_bc_1d(Tnew, T, bc)
        T = Tnew
        hist[n + 1] = T
    return hist


def solve_1d_implicit(T0, kappa, dx, dt, nsteps, bc=("dirichlet", "dirichlet")):
    """Backward-Euler solution of the same equation — unconditionally stable.

    The point of this rung is that the matrix is built once and factorised
    once; only the right-hand side changes each step.
    """
    T0 = np.asarray(T0, dtype=float)
    nx = T0.size
    s = kappa * dt / dx**2

    main = np.full(nx, 1.0 + 2.0 * s)
    off = np.full(nx - 1, -s)
    A = sp.diags([off, main, off], [-1, 0, 1], format="lil")

    # Boundary rows: Dirichlet pins the value, Neumann mirrors the interior.
    for i, side in zip((0, nx - 1), bc):
        A[i, :] = 0.0
        if side == "dirichlet":
            A[i, i] = 1.0
        elif side == "neumann":
            A[i, i] = 1.0
            A[i, i + 1 if i == 0 else i - 1] = -1.0
        else:
            raise ValueError(f"unknown boundary condition {side!r}")

    solve = spla.factorized(A.tocsc())

    hist = np.empty((nsteps + 1, nx))
    hist[0] = T0
    T = T0.copy()
    for n in range(nsteps):
        rhs = T.copy()
        if bc[0] == "dirichlet":
            rhs[0] = T0[0]
        elif bc[0] == "neumann":
            rhs[0] = 0.0
        if bc[1] == "dirichlet":
            rhs[-1] = T0[-1]
        elif bc[1] == "neumann":
            rhs[-1] = 0.0
        T = solve(rhs)
        hist[n + 1] = T
    return hist


def _apply_bc_1d(Tnew, Told, bc):
    if bc[0] == "dirichlet":
        Tnew[0] = Told[0]
    elif bc[0] == "neumann":
        Tnew[0] = Tnew[1]
    if bc[1] == "dirichlet":
        Tnew[-1] = Told[-1]
    elif bc[1] == "neumann":
        Tnew[-1] = Tnew[-2]


def gaussian_analytic_1d(x, t, kappa, amplitude=1.0, width=1.0, x0=0.0):
    """Analytic diffusion of a Gaussian — the verification target for rung 01.

    A Gaussian of initial half-width ``width`` spreads as
    ``sqrt(width^2 + 4*kappa*t)`` while conserving its integral. Comparing the
    numerical solution against this is the first time students see a
    convergence test, and it is worth dwelling on.
    """
    w = np.sqrt(width**2 + 4.0 * kappa * t)
    return amplitude * (width / w) * np.exp(-((x - x0) ** 2) / w**2)


# --------------------------------------------------------------------------
# 2-D
# --------------------------------------------------------------------------
def solve_2d_explicit(T0, kappa, dx, dz, dt, nsteps, fixed=("top", "bottom")):
    """Explicit FTCS in 2-D. ``T0`` is shaped ``(nz, nx)``.

    Sides not listed in ``fixed`` get zero-flux (insulating) conditions, which
    is the usual choice for a convection box.
    """
    T = np.asarray(T0, dtype=float).copy()
    dt_max = diffusion_stability_dt(kappa, dx, dz, safety=1.0)
    if dt > dt_max:
        raise ValueError(
            f"dt={dt:g} exceeds the 2-D explicit stability limit {dt_max:g}."
        )
    sx, sz = kappa * dt / dx**2, kappa * dt / dz**2
    T_init = T.copy()
    for _ in range(nsteps):
        lap = (
            sx * (T[1:-1, 2:] - 2.0 * T[1:-1, 1:-1] + T[1:-1, :-2])
            + sz * (T[2:, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[:-2, 1:-1])
        )
        T[1:-1, 1:-1] += lap
        _apply_bc_2d(T, T_init, fixed)
    return T


def solve_2d_implicit(T0, kappa, dx, dz, dt, nsteps, fixed=("top", "bottom")):
    """Backward Euler in 2-D, assembled as a sparse system.

    Built with a five-point stencil in lexicographic (row-major) ordering, the
    same ordering the Stokes solver uses — so the indexing habit transfers.
    """
    T0 = np.asarray(T0, dtype=float)
    nz, nx = T0.shape
    A = _laplacian_2d_matrix(nz, nx, kappa, dx, dz, dt, fixed)
    solve = spla.factorized(A.tocsc())

    T = T0.copy()
    for _ in range(nsteps):
        rhs = T.ravel().copy()
        _pin_rhs_2d(rhs, T0, nz, nx, fixed)
        T = solve(rhs).reshape(nz, nx)
    return T


def _laplacian_2d_matrix(nz, nx, kappa, dx, dz, dt, fixed):
    n = nz * nx
    sx, sz = kappa * dt / dx**2, kappa * dt / dz**2
    A = sp.lil_matrix((n, n))
    idx = lambda i, j: i * nx + j  # noqa: E731

    for i in range(nz):
        for j in range(nx):
            k = idx(i, j)
            on_top, on_bot = i == 0, i == nz - 1
            on_left, on_right = j == 0, j == nx - 1

            if (on_top and "top" in fixed) or (on_bot and "bottom" in fixed) \
               or (on_left and "left" in fixed) or (on_right and "right" in fixed):
                A[k, k] = 1.0
                continue

            A[k, k] = 1.0 + 2.0 * sx + 2.0 * sz
            # Insulating sides fold the ghost node back onto the interior.
            A[k, idx(i, j - 1) if not on_left else idx(i, j + 1)] = -sx
            A[k, idx(i, j + 1) if not on_right else idx(i, j - 1)] = -sx
            A[k, idx(i - 1, j) if not on_top else idx(i + 1, j)] = -sz
            A[k, idx(i + 1, j) if not on_bot else idx(i - 1, j)] = -sz
    return A


def _pin_rhs_2d(rhs, T0, nz, nx, fixed):
    if "top" in fixed:
        rhs[0:nx] = T0[0, :]
    if "bottom" in fixed:
        rhs[(nz - 1) * nx : nz * nx] = T0[-1, :]
    if "left" in fixed:
        rhs[0::nx] = T0[:, 0]
    if "right" in fixed:
        rhs[nx - 1 :: nx] = T0[:, -1]


def _apply_bc_2d(T, T_init, fixed):
    T[0, :] = T_init[0, :] if "top" in fixed else T[1, :]
    T[-1, :] = T_init[-1, :] if "bottom" in fixed else T[-2, :]
    T[:, 0] = T_init[:, 0] if "left" in fixed else T[:, 1]
    T[:, -1] = T_init[:, -1] if "right" in fixed else T[:, -2]
