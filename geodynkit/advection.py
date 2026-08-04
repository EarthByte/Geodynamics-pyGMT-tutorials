"""
Advection schemes, and the numerical diffusion they smuggle in.

The pedagogical point of this module is comparative: the same initial
condition, advected once around a periodic domain by four different schemes,
looks dramatically different at the end. Upwind survives but smears; FTCS
blows up; Lax-Wendroff and semi-Lagrangian sit in between with different
personalities.
"""

import numpy as np

__all__ = [
    "courant_dt",
    "advect_1d",
    "advect_2d_semilagrangian",
    "SCHEMES",
]

SCHEMES = ("upwind", "ftcs", "lax_wendroff", "semi_lagrangian")


def courant_dt(vmax, dx, cfl=0.5):
    """Timestep from a target Courant number: ``dt = cfl * dx / |v|max``."""
    if vmax <= 0:
        return np.inf
    return cfl * dx / vmax


def advect_1d(C0, v, dx, dt, nsteps, scheme="upwind"):
    """Advect a 1-D field at constant velocity ``v`` on a periodic domain.

    Returns the full history, shape ``(nsteps + 1, nx)``.

    Schemes
    -------
    upwind
        First order. Unconditionally monotone at ``Co <= 1``, and unmistakably
        diffusive — the square wave rounds off within a few dozen steps.
    ftcs
        Centred in space, forward in time. Second-order accurate and
        *unconditionally unstable* for pure advection. Included on purpose:
        watching it explode is the most memorable way to learn that accuracy
        and stability are different properties.
    lax_wendroff
        Second order, stable for ``Co <= 1``, but dispersive — it produces the
        characteristic ripples behind a sharp front.
    semi_lagrangian
        Traces the characteristic backwards and interpolates. Stable at any
        Courant number, which is why it is the workhorse in real geodynamic
        codes; the cost is interpolation smoothing.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")

    C = np.asarray(C0, dtype=float).copy()
    nx = C.size
    co = v * dt / dx  # Courant number

    hist = np.empty((nsteps + 1, nx))
    hist[0] = C

    x = np.arange(nx) * dx
    for n in range(nsteps):
        if scheme == "upwind":
            if v >= 0:
                C = C - co * (C - np.roll(C, 1))
            else:
                C = C - co * (np.roll(C, -1) - C)
        elif scheme == "ftcs":
            C = C - 0.5 * co * (np.roll(C, -1) - np.roll(C, 1))
        elif scheme == "lax_wendroff":
            Cp, Cm = np.roll(C, -1), np.roll(C, 1)
            C = C - 0.5 * co * (Cp - Cm) + 0.5 * co**2 * (Cp - 2.0 * C + Cm)
        elif scheme == "semi_lagrangian":
            xd = (x - v * dt) % (nx * dx)          # departure points
            C = np.interp(xd, np.append(x, nx * dx), np.append(C, C[0]))
        hist[n + 1] = C
    return hist


def advect_2d_semilagrangian(F, vx, vz, x, z, dt):
    """One semi-Lagrangian advection step of a 2-D field.

    ``F``, ``vx`` and ``vz`` are all shaped ``(nz, nx)`` on the regular grid
    defined by ``x`` and ``z``. Departure points are traced back one step and
    the field is bilinearly interpolated there. Points that leave the domain
    are clamped to the boundary, which is the usual pragmatic choice.

    Stable at any Courant number — this is what makes it worth the extra
    machinery over upwind.
    """
    X, Z = np.meshgrid(x, z)
    xd = np.clip(X - vx * dt, x[0], x[-1])
    zd = np.clip(Z - vz * dt, z[0], z[-1])
    return _bilinear(F, x, z, xd, zd)


def _bilinear(F, x, z, xq, zq):
    """Bilinear interpolation of ``F`` at query points, no SciPy needed."""
    dx, dz = x[1] - x[0], z[1] - z[0]
    nz, nx = F.shape

    fi = np.clip((xq - x[0]) / dx, 0, nx - 1.000001)
    fj = np.clip((zq - z[0]) / dz, 0, nz - 1.000001)
    i0, j0 = fi.astype(int), fj.astype(int)
    i1, j1 = i0 + 1, j0 + 1
    tx, tz = fi - i0, fj - j0

    return (
        F[j0, i0] * (1 - tx) * (1 - tz)
        + F[j0, i1] * tx * (1 - tz)
        + F[j1, i0] * (1 - tx) * tz
        + F[j1, i1] * tx * tz
    )
