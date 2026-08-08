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


def build(mode, nx, ny, visc_floor=1e21, notch_km=5.0):
    H = 30e3
    year = 86400 * 365
    U0 = Constant(5e-3 / year)
    mu0 = 1e22
    mu1 = Constant(1e23 / mu0)          # strong upper layer
    mu2 = Constant(1e21 / mu0)          # weak substrate
    # Floor on the yielded viscosity inside the strong layer. The benchmark
    # sets this equal to the substrate viscosity; it is exposed here because it
    # is the model's only regularisation, and therefore the only thing setting
    # the width of a shear band. See the `--visc-floor` experiment in T09.
    mu_floor = Constant(visc_floor / mu0)

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
    r, h = 0.02, 1 / 12
    ww = notch_km * 1e3 / H            # notch width, non-dimensionalised by H
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
        mu1_eff = max_value(mu1_eff, mu_floor)
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
                newton=newton, picard=picard, u=u, p=p, nx=nx, ny=ny,
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


def sample(m, n=None):
    """Sample onto a regular grid for pyGMT. Returns non-dimensional fields.

    Three things here are not incidental.

    **The sampling grid follows the mesh.** A fixed export grid is a trap in a
    localisation problem: refine the mesh, the band narrows, and a grid that
    does not refine with it starts stepping over the peak. Measured on a fixed
    321x81 grid, the peak strain-rate invariant of this benchmark went 2.93 ->
    3.41 -> 4.92 -> 3.78 for nx = 64, 128, 192, 256 -- the fall at the end is
    the sampler missing the ridge, not the model relaxing. The grid below is
    tied to the mesh, and `strain_rate_max_nodal` in the RESULT line is taken
    from the finite-element nodes directly, with no sampling grid at all.

    **Export through CG1, not CG2.** The strain-rate invariant and the effective
    viscosity both have ridges a couple of elements wide, and a quadratic
    interpolant through a ridge overshoots on one side and undershoots on the
    other. Sampled from CG2 this run reported a minimum strain-rate invariant of
    -0.385 against a maximum of 4.2 -- a quantity that is a square root, and so
    cannot be negative -- and a minimum viscosity of 0.084 where the substrate
    floor is exactly 0.1. Neither is a modelling error; both are the export
    lying about the solution. CG1 cannot overshoot a monotone ridge in the same
    way, and the residual clipping below is reported rather than silent.

    **One PointEvaluator for all three fields.** `Function.at` is deprecated,
    and it also re-does the point location for every call. Locating 26,001
    points once and reusing the map is both current API and several times
    faster.
    """
    if n is None:                       # two sample points per element, per axis
        n = (2 * m["nx"] + 1, 2 * m["ny"] + 1)
    nx, ny = n
    xs = np.linspace(0.0, 4.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    Q = FunctionSpace(m["mesh"], "CG", 1)
    eii = Function(Q).interpolate(m["epsii"])
    visc = Function(Q).interpolate(m["eta"](m["epsii"], m["p"]))
    uvec = m["z"].subfunctions[0]

    ev = PointEvaluator(m["mesh"], pts, tolerance=1e-8)
    E = np.asarray(ev.evaluate(eii)).reshape(ny, nx)
    M = np.asarray(ev.evaluate(visc)).reshape(ny, nx)
    U = np.asarray(ev.evaluate(uvec)).reshape(ny, nx, 2)

    # Report, then clip. A diagnostic that quietly repairs itself teaches the
    # wrong lesson about how much to trust an exported field.
    neg = int((E < 0).sum()) + int((M <= 0).sum())
    if neg:
        print(f"NOTE {neg} non-physical values from interpolation, clipped "
              f"(eps_min={E.min():.3g}, eta_min={M.min():.3g})", flush=True)
    E = np.maximum(E, 0.0)
    M = np.maximum(M, 1e-6)

    return dict(x_km=xs * m["H"] / 1e3, depth_km=(1.0 - ys[::-1]) * m["H"] / 1e3,
                strain_rate=E[::-1], viscosity=M[::-1],
                vx=U[::-1, :, 0], vz=-U[::-1, :, 1],
                strain_rate_max_nodal=float(eii.dat.data_ro.max()))


def band_dip_deg(flds, zmin=2.0, zmax=7.5):
    """Dip of the strain-rate ridge on the left side of the notch, in degrees.

    This is the quantity the Spiegelman, May & Wilson (2016) benchmark is
    really about. Classical soil mechanics offers three candidate angles for a
    Drucker-Prager band with friction angle phi and dilatancy angle psi:

        Coulomb  45 - phi/2      = 30.0 deg   (phi = 30, the stress-based answer)
        Roscoe   45 - psi/2      = 45.0 deg   (psi = 0, the kinematic answer)
        Arthur   45 - (phi+psi)/4 = 37.5 deg  (the compromise)

    A non-dilatant model has no reason to pick the Coulomb angle, and measuring
    which one it does pick is a check on the *solution*, not on the code that
    printed it. Unlike a max/median ratio, this number cannot be flattered by a
    run in which nothing deforms.
    """
    E, x, z = flds["strain_rate"], flds["x_km"], flds["depth_km"]
    rows = np.where((z > zmin) & (z < zmax))[0]
    if rows.size < 3:
        return None
    # Restrict to the left half so the two conjugate bands are not mixed.
    mask = x < 0.5 * (x[0] + x[-1])
    pts = [(x[np.nanargmax(np.where(mask, E[i], -np.inf))], z[i]) for i in rows]
    pts = np.asarray(pts)
    slope = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
    return round(float(np.degrees(np.arctan(abs(slope)))), 1)


def band_width_km(flds, depth_km=5.0):
    """Full width at half maximum of the strain-rate ridge, in km.

    Taken across a horizontal profile through the strong layer.

    It is worth being clear about what this number does and does not measure,
    because the obvious expectation is wrong. Drucker-Prager plasticity is
    usually described as having no length scale, so one expects the band to
    thin towards the cell size. It does not: for nx = 64, 128, 192, 256 the
    extension case gives 10.31, 9.38, 9.38, 9.38 km while the cell width falls
    from 1.875 to 0.469 km. Nor is it the viscosity floor holding it open --
    dropping the floor from 1e21 to 3e19 Pa s changes the answer not at all,
    because the floor never binds.

    What sets it is the *seed*. Widening the notch from 2.5 to 20 km widens the
    band from 8.9 to 13.6 km and drops the peak strain rate from 5.31 to 3.45.
    The reason is that this solve is instantaneous and the rheology has no
    memory: nothing rewards a band for having formed, so there is no positive
    feedback to drive it to collapse. Strain weakening -- which needs a
    *time-dependent* model to accumulate anything to weaken on -- is what
    supplies that feedback, and is where localisation stops being easy.
    """
    E, x, z = flds["strain_rate"], flds["x_km"], flds["depth_km"]
    i = int(np.argmin(np.abs(z - depth_km)))
    row = E[i]
    half = x < 0.5 * (x[0] + x[-1])
    j = int(np.nanargmax(np.where(half, row, -np.inf)))
    peak, base = row[j], np.nanmedian(row)
    if not np.isfinite(peak) or peak <= base:
        return None
    thr = base + 0.5 * (peak - base)
    a = j
    while a > 0 and row[a] > thr:
        a -= 1
    b = j
    while b < row.size - 1 and row[b] > thr:
        b += 1
    return round(float(x[b] - x[a]), 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shortening", "extension"], required=True)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--notch-km", type=float, default=5.0,
                    help="width of the seeding notch in km (benchmark value 5)")
    ap.add_argument("--visc-floor", type=float, default=1e21,
                    help="lower bound on yielded viscosity in the strong layer (Pa s)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = build(args.mode, args.nx, args.ny, visc_floor=args.visc_floor,
              notch_km=args.notch_km)
    res = solve(m)
    res["mode"], res["nx"], res["ny"] = args.mode, args.nx, args.ny

    flds = sample(m)
    out = args.out or f"lithosphere_{args.mode}.npz"
    np.savez(out, **flds)
    res["output"] = out
    res["strain_rate_max"] = round(float(np.nanmax(flds["strain_rate"])), 4)
    res["strain_rate_median"] = round(float(np.nanmedian(flds["strain_rate"])), 4)
    res["peak_over_background"] = round(
        float(np.nanmax(flds["strain_rate"]) / np.nanmedian(flds["strain_rate"])), 2)
    res["strain_rate_max_nodal"] = round(float(flds["strain_rate_max_nodal"]), 4)
    res["band_dip_deg"] = band_dip_deg(flds)
    res["band_width_km"] = band_width_km(flds)
    res["cell_width_km"] = round(120.0 / args.nx, 3)
    res["visc_floor"] = args.visc_floor
    res["notch_km"] = args.notch_km

    print("RESULT " + json.dumps(res), flush=True)
