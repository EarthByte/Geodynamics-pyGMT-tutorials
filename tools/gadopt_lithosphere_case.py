#!/usr/bin/env python3
"""
Lithospheric shortening and extension, after Spiegelman, May & Wilson (2016).

This is G-ADOPT's `Drucker_Prager` demo with one thing changed: the sign of the
horizontal boundary velocity. That single flip turns the canonical shear-band
*shortening* benchmark into a *rifting* model, which is the point notebooks T13
and T14 are built on.

    left  ux = +1, right ux = -1   ->  material driven inward   ->  SHORTENING
    left  ux = -1, right ux = +1   ->  material driven outward  ->  EXTENSION

Physics, unchanged from the benchmark:
  * 120 x 30 km box (4 x 1 non-dimensional), driven at 5 mm/yr
  * strong upper layer (1e23 Pa s) over a weak substrate (1e21 Pa s)
  * pressure-dependent Drucker-Prager yielding, friction angle 30 deg,
    cohesion 100 MPa, harmonically combined with the ductile viscosity
  * a small notch in the layer interface to seed localisation
  * free upper surface, free-slip base

The problem is *instantaneous*: one nonlinear Stokes solve, no timestepping.
That is what makes it cheap enough to teach with.

Solve strategy is the benchmark's, and is the reason the demo exists: Newton
alone does not converge from a cold start, so we do an isoviscous Picard warm-up
(the strain-rate invariant is zero at u=0, which would divide by zero), then
Picard iterations, then Newton to polish.

Usage:
    python3 gadopt_lithosphere_case.py --mode extension --nx 128 --ny 64
"""

import argparse
import json
import time

import numpy as np
from gadopt import *


def build(mode, nx, ny):
    H = 30e3
    year = 86400 * 365
    U0 = Constant(5e-3 / year)
    mu0 = 1e22
    mu1 = Constant(1e23 / mu0)          # strong upper layer
    mu2 = Constant(1e21 / mu0)          # weak substrate

    g = Constant(9.81)
    rho0 = Constant(2700)
    phi = Constant(30 / 180 * pi)       # friction angle
    C = Constant(1e8 / (mu0 * U0 / H))  # cohesion
    A, B, alpha = Constant(C * cos(phi)), Constant(sin(phi)), Constant(1)

    mesh = RectangleMesh(nx, ny, 4, 1, quadrilateral=True)
    mesh.cartesian = True
    boundary = get_boundary_ids(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = MixedFunctionSpace([V, W])
    z = Function(Z)
    u, p = split(z)
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")

    # Layer interface, with a notch at the centre to seed localisation. Without
    # the seed the solution is translation-invariant and bands appear only from
    # discretisation noise — which is a bad lesson to teach.
    H1 = 0.75
    r, h, ww = 0.02, 1 / 12, 1 / 6
    X = SpatialCoordinate(mesh)
    x = X[0]
    d = 1 - X[1]                        # depth, 0 at top

    dx0 = abs(x - 2)
    interface_depth = H1 - conditional(dx0 < ww / 2, h, 0)
    dx1 = ww / 2 + r - dx0
    interface_depth += conditional(And(dx1 > 0, dx1 < r), sqrt(r**2 - dx1**2) - r, 0)
    dx2 = dx0 - (ww / 2 - r)
    interface_depth -= conditional(And(dx2 > 0, dx2 < r), sqrt(r**2 - dx2**2) - r, 0)

    plith = g * rho0 * d * H / (U0 * mu0 / H)

    switch = Constant(1.0)

    def eta(epsii, p_):
        mu_plast = (A + B * (plith + alpha * p_)) / (2 * epsii)
        mu1_eff = conditional(switch > 0.5, mu1 * mu_plast / (mu1 + mu_plast), mu1)
        mu1_eff = max_value(mu1_eff, mu2)
        return conditional(d < interface_depth, mu1_eff, mu2)

    # ---- the one line that distinguishes the two experiments ----
    sgn = 1.0 if mode == "shortening" else -1.0
    bcs = {
        boundary.left: {"ux": sgn * 1},
        boundary.right: {"ux": -sgn * 1},
        boundary.bottom: {"uy": 0},
    }

    eps = sym(grad(u))
    epsii = sqrt(0.5 * inner(eps, eps))

    z_picard = Function(Z)
    u_pic, p_pic = split(z_picard)
    eps_pic = sym(grad(u_pic))
    epsii_pic = sqrt(0.5 * inner(eps_pic, eps_pic))

    newton = StokesSolver(z, BoussinesqApproximation(0, mu=eta(epsii, p)), bcs=bcs)
    picard = StokesSolver(
        z, BoussinesqApproximation(0, mu=eta(epsii_pic, p_pic)), bcs=bcs
    )
    return dict(mesh=mesh, Z=Z, z=z, z_picard=z_picard, switch=switch,
                newton=newton, picard=picard, u=u, p=p,
                epsii=epsii, eta=eta, V=V, W=W, mu0=mu0, U0=U0, H=H)


def solve(m, picard_iters=50, tol=1e-5):
    z, zp, switch = m["z"], m["z_picard"], m["switch"]
    u, p = m["u"], m["p"]

    t0 = time.perf_counter()
    switch.assign(0.0)          # isoviscous warm-up: epsii = 0 at u = 0
    m["picard"].solve()

    switch.assign(1.0)
    hist = []
    for i in range(picard_iters):
        zp.assign(z)
        m["picard"].solve()
        du = norm(u - split(zp)[0])
        hist.append(du)
        if du < tol:
            break
    picard_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    converged = True
    try:
        m["newton"].solve()
    except ConvergenceError:
        converged = False
    newton_time = time.perf_counter() - t1

    return dict(picard_iterations=len(hist), picard_seconds=round(picard_time, 2),
                newton_converged=converged, newton_seconds=round(newton_time, 2),
                final_du=float(hist[-1]) if hist else None)


def sample(m, n=(321, 81)):
    """Sample onto a regular grid for pyGMT. Returns dimensional-ish fields."""
    nx, ny = n
    xs = np.linspace(0.0, 4.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    Q = FunctionSpace(m["mesh"], "CG", 2)
    eii = Function(Q).interpolate(m["epsii"])
    visc = Function(Q).interpolate(m["eta"](m["epsii"], m["p"]))
    uvec = m["z"].subfunctions[0]

    E = np.array(eii.at(pts, tolerance=1e-8)).reshape(ny, nx)
    M = np.array(visc.at(pts, tolerance=1e-8)).reshape(ny, nx)
    U = np.array(uvec.at(pts, tolerance=1e-8)).reshape(ny, nx, 2)
    return dict(x_km=xs * m["H"] / 1e3, depth_km=(1.0 - ys[::-1]) * m["H"] / 1e3,
                strain_rate=E[::-1], viscosity=M[::-1],
                vx=U[::-1, :, 0], vz=-U[::-1, :, 1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shortening", "extension"], required=True)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = build(args.mode, args.nx, args.ny)
    res = solve(m)
    res["mode"], res["nx"], res["ny"] = args.mode, args.nx, args.ny

    flds = sample(m)
    out = args.out or f"lithosphere_{args.mode}.npz"
    np.savez(out, **flds)
    res["output"] = out
    res["strain_rate_max"] = float(np.nanmax(flds["strain_rate"]))
    res["localisation_ratio"] = round(
        float(np.nanmax(flds["strain_rate"]) / np.nanmedian(flds["strain_rate"])), 1)

    print("RESULT " + json.dumps(res), flush=True)
