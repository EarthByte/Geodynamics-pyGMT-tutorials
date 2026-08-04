#!/usr/bin/env python3
"""
Time-dependent lithospheric extension with three advected layers and
strain-weakening plasticity.

This is the step beyond `gadopt_lithosphere_case.py`. That one solves the
Spiegelman benchmark *instantaneously* — one nonlinear Stokes solve, so nothing
moves and no strain accumulates. A rift is a feedback loop, and you cannot see a
feedback loop in a snapshot:

    yielding -> plastic strain -> weaker rock -> more yielding

So this script wires the two pieces that close that loop:

* **`LevelSetSolver`** advects two conservative level sets, giving three
  materials — upper crust, lower crust, mantle lithosphere — that keep their
  identity through large finite strain.
* **`GenericTransportSolver`** advects a scalar plastic-strain field with a
  source term equal to the plastic strain rate, and that field feeds back into
  the yield stress via the Naliboff & Buiter (2015) weakening law.

Layer structure, flow laws and weakening parameters come from
`geodynkit.lithosphere`; see that module for provenance.

Non-dimensionalisation follows the Spiegelman convention used by G-ADOPT's
Drucker-Prager demo: lengths by the domain depth H, velocities by the boundary
velocity U0, viscosities by a reference mu0, so stress scales as mu0*U0/H.

Usage:
    python3 gadopt_rift_case.py --steps 20 --nx 128 --ny 64
"""

import argparse
import json
import sys
import time

import numpy as np
from gadopt import *

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from geodynkit import lithosphere as LI  # noqa: E402

# --- scales -------------------------------------------------------------
H = 100e3                       # domain depth, m
YEAR = 86400 * 365.25
U0 = 0.25e-2 / YEAR             # 0.25 cm/yr, the cookbook extension rate
MU0 = 1e22                      # reference viscosity, Pa s
STRESS = MU0 * U0 / H           # stress scale, Pa
SR = U0 / H                     # strain-rate scale, 1/s
T_SCALE = H / U0                # time scale, s


def build(nx, ny, aspect=2.0):
    """Assemble mesh, fields, rheology and solvers. Returns a dict of handles."""
    mesh = RectangleMesh(nx, ny, aspect, 1.0, quadrilateral=True)
    mesh.cartesian = True
    boundary = get_boundary_ids(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 2)          # plastic strain
    K = FunctionSpace(mesh, "Q", 2)           # level sets
    Z = MixedFunctionSpace([V, W])

    z = Function(Z)
    u_, p_ = split(z)
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")
    u_fn = z.subfunctions[0]

    X = SpatialCoordinate(mesh)
    depth_nd = 1.0 - X[1]                     # 0 at surface, 1 at base

    # ---- three materials from two level sets ---------------------------
    # psi_uc: 1 above the upper/lower-crust interface (20 km depth)
    # psi_lc: 1 above the crust/mantle interface       (40 km depth)
    y_uc = 1.0 - 20e3 / H
    y_lc = 1.0 - 40e3 / H

    psi_uc, psi_lc = Function(K, name="psi_uc"), Function(K, name="psi_lc")
    epsilon = interface_thickness(K, min_cell_edge_length=True)
    assign_level_set_values(psi_uc, epsilon, X[1] - y_uc)
    assign_level_set_values(psi_lc, epsilon, X[1] - y_lc)

    # ORDERING MATTERS, and not in the obvious way. `material_field` recurses by
    # popping from the END of both lists, and `material_interface(ls, a, b)`
    # gives `a` where ls = 1. So the LAST level set pairs with the LAST value on
    # its 1-side, and the list must run from the DEEPEST interface to the
    # shallowest. Listing psi_uc first instead makes the outermost conditional
    # "upper-crust value wherever depth < 40 km", which silently swallows the
    # lower crust — a wrong model that runs perfectly happily.
    layers = [LI.MANTLE_LITHOSPHERE, LI.LOWER_CRUST, LI.UPPER_CRUST]
    ls = [psi_lc, psi_uc]           # deepest interface first

    def mat(attr, how="arithmetic"):
        return material_field(ls, [getattr(l, attr) for l in layers], interface=how)

    rho = mat("density")
    coh = mat("cohesion")
    fric = mat("friction_deg")

    # ---- prescribed conductive geotherm --------------------------------
    # Temperature is held fixed here. Coupling EnergySolver in is the next
    # step; keeping T frozen isolates the mechanics so the level-set and
    # strain-weakening machinery can be verified on its own.
    zc = np.linspace(0.0, H / 1e3, 400)
    Tc = np.atleast_1d(LI.geotherm(zc))
    Tfield = Function(Q, name="Temperature")
    Tfield.interpolate(Constant(0.0))
    Tfield.dat.data[:] = np.interp(
        (1.0 - Function(Q).interpolate(X[1]).dat.data_ro) * H / 1e3, zc, Tc)

    # ---- plastic strain, advected with a source ------------------------
    strain = Function(Q, name="PlasticStrain")
    strain.interpolate(Constant(0.0))
    # A small random seed in the centre, as the cookbook does: without it the
    # problem is translation-invariant and localises only on grid noise.
    xc = Function(Q).interpolate(X[0]).dat.data_ro
    yc = Function(Q).interpolate(X[1]).dat.data_ro
    rng = np.random.default_rng(0)
    seed = ((np.abs(xc - aspect / 2) < 0.25) & (yc > y_lc))
    strain.dat.data[seed] = rng.uniform(0.5, 1.5, seed.sum())

    # ---- rheology ------------------------------------------------------
    # Visco-plastic Stokes does not converge under Newton from a cold start —
    # that is the whole point of G-ADOPT's Drucker-Prager demo. Two devices are
    # needed, and both are expressed here as functions of (u, p) so the same
    # algebra can be built twice: once for Newton, once with a LAGGED solution
    # for Picard.
    #
    #   `switch` = 0 : plastic branch off, linear viscosity. Used for the very
    #                  first solve, because at u = 0 the strain-rate invariant
    #                  is zero and the plastic viscosity divides by it.
    #   `switch` = 1 : the real rheology.
    switch = Constant(1.0)
    plith = rho * 9.81 * depth_nd * H / STRESS       # non-dimensional

    # Naliboff & Buiter linear weakening, in UFL. `strain` is updated between
    # timesteps, so it is lagged by construction and does not enter the
    # nonlinear solve.
    w = conditional(strain < 0.5, 1.0,
                    conditional(strain > 1.5, 0.25,
                                1.0 + (0.25 - 1.0) * (strain - 0.5) / 1.0))

    def creep(lay, epsii):
        n, A = lay.stress_exponent, lay.prefactor
        return (0.5 * A ** (-1.0 / n) * (epsii * SR) ** ((1.0 - n) / n)
                * exp(lay.activation_energy / (n * LI.R_GAS * Tfield))) / MU0

    def rheology(uu, pp):
        """(mu, mu_creep, mu_plast, epsii) for a given velocity/pressure pair."""
        e = sym(grad(uu))
        epsii = sqrt(0.5 * inner(e, e) + 1e-10)      # guards the cold start
        mu_c = material_field(ls, [creep(l, epsii) for l in layers],
                              interface="geometric")
        phi = fric * pi / 180.0 * w
        sigma_y = coh * w / STRESS * cos(phi) + (plith + pp) * sin(phi)
        mu_p = sigma_y / (2 * epsii)
        # switch = 0 disables the plastic branch entirely
        mu_eff = conditional(switch > 0.5, min_value(mu_c, mu_p), mu_c)
        return (max_value(min_value(mu_eff, 1e26 / MU0), 1e18 / MU0),
                mu_c, mu_p, epsii)

    mu, mu_creep, mu_plast, epsii = rheology(u_, p_)

    # ---- solvers -------------------------------------------------------
    dt = Constant(2e-3)
    bcs = {boundary.left: {"ux": -1}, boundary.right: {"ux": 1},
           boundary.bottom: {"uy": 0}}          # top is stress-free

    stokes = StokesSolver(z, BoussinesqApproximation(0, mu=mu), bcs=bcs)

    # Picard: identical problem, but the viscosity is evaluated at the PREVIOUS
    # iterate, making each solve linear. Slow but robust, where Newton is fast
    # but only converges once it is already close.
    z_pic = Function(Z)
    u_pic, p_pic = split(z_pic)
    mu_pic = rheology(u_pic, p_pic)[0]
    picard = StokesSolver(z, BoussinesqApproximation(0, mu=mu_pic), bcs=bcs)

    ls_solver = [LevelSetSolver(psi, adv_kwargs={"u": u_fn, "timestep": dt},
                                reini_kwargs={"epsilon": epsilon})
                 for psi in ls]

    # Plastic strain: advected, with a source equal to the plastic strain rate
    # where the plastic branch governs. This is the feedback that makes a fault
    # keep slipping once it has formed.
    yielding = conditional(mu_plast < mu_creep, 1.0, 0.0)
    strain_solver = GenericTransportSolver(
        ["advection", "mass", "source"], strain, dt, DIRK33,
        eq_attrs={"u": u_fn, "source": yielding * epsii},
        su_advection=True,
    )

    return dict(mesh=mesh, z=z, u=u_fn, dt=dt, stokes=stokes, ls=ls,
                ls_solver=ls_solver, strain=strain, strain_solver=strain_solver,
                mu=mu, epsii=epsii, Q=Q, K=K, w=w, aspect=aspect, rho=rho,
                mu_creep=mu_creep, mu_plast=mu_plast, X=X, H=H,
                picard=picard, z_pic=z_pic, switch=switch)


def check_layering(m):
    """Assert the three materials land at the depths they should.

    Density is the cleanest probe: 2700 / 2900 / 3300 kg/m3 are far apart, so a
    mis-ordered `material_field` shows up immediately. This is cheap and it
    catches the one bug in this script that would otherwise be invisible.
    """
    Q = m["Q"]
    rho_fn = Function(Q).interpolate(m["rho"])
    y = Function(Q).interpolate(m["X"][1]).dat.data_ro
    depth_km = (1.0 - y) * H / 1e3
    rho = rho_fn.dat.data_ro

    out = {}
    for label, lo, hi, expect in (("upper crust", 2, 18, 2700.0),
                                  ("lower crust", 22, 38, 2900.0),
                                  ("mantle lithosphere", 45, 95, 3300.0)):
        sel = (depth_km > lo) & (depth_km < hi)
        got = float(rho[sel].mean())
        out[label] = dict(expected=expect, got=round(got, 1),
                          ok=abs(got - expect) < 25.0)
    return out


def solve_stokes(m, picard_iters, tol=1e-4, cold=False):
    """Robust visco-plastic Stokes solve: isoviscous -> Picard -> Newton.

    Returns (picard_iterations_used, newton_converged).
    """
    if cold:
        m["switch"].assign(0.0)          # linear viscosity; breaks the 1/epsii
        m["picard"].solve()              # singularity at u = 0
        m["switch"].assign(1.0)

    used = 0
    for _ in range(picard_iters):
        m["z_pic"].assign(m["z"])
        m["picard"].solve()
        used += 1
        du = norm(split(m["z"])[0] - split(m["z_pic"])[0])
        if du < tol:
            break

    try:
        m["stokes"].solve()              # Newton polish, now close enough
        return used, True
    except ConvergenceError:
        return used, False               # Picard result stands; keep going


def run(m, steps):
    """Step: Stokes -> advect level sets -> advect plastic strain."""
    hist = []
    t0 = time.perf_counter()
    for n in range(steps):
        pic, newton_ok = solve_stokes(m, picard_iters=40 if n == 0 else 8,
                                      cold=(n == 0))
        for s in m["ls_solver"]:
            s.solve()
        m["strain_solver"].solve()

        sd = m["strain"].dat.data_ro
        hist.append(dict(step=n, picard=pic, newton=newton_ok,
                         strain_max=float(sd.max()),
                         strain_mean=float(sd.mean()),
                         weak_fraction=float((sd > 1.5).mean())))
        print(f"  step {n:3d}  picard {pic:3d}  newton {'ok' if newton_ok else 'FAILED'}"
              f"  strain max {sd.max():.3f} mean {sd.mean():.4f}", flush=True)
    return hist, time.perf_counter() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="/tmp/rift.npz")
    args = ap.parse_args()

    m = build(args.nx, args.ny)

    layering = check_layering(m)
    for k, v in layering.items():
        print(f"  {k:20s} expected {v['expected']:.0f}  got {v['got']:7.1f}  "
              f"{'OK' if v['ok'] else 'MISMATCH'}", flush=True)
    if not all(v["ok"] for v in layering.values()):
        raise SystemExit("layer ordering wrong — check the material_field list order")

    hist, secs = run(m, args.steps)

    # Level sets must stay in [0, 1]: that is the conservative-level-set
    # invariant, and its violation is the first sign advection has gone wrong.
    ls_range = [(float(p.dat.data_ro.min()), float(p.dat.data_ro.max()))
                for p in m["ls"]]

    Q = m["Q"]
    out = dict(
        strain=Function(Q).interpolate(m["strain"]).dat.data_ro.copy(),
        viscosity=Function(Q).interpolate(m["mu"]).dat.data_ro.copy(),
        strain_rate=Function(Q).interpolate(m["epsii"]).dat.data_ro.copy(),
    )
    np.savez(args.out, **out)

    print("RESULT " + json.dumps(dict(
        steps=args.steps, nx=args.nx, ny=args.ny, seconds=round(secs, 2),
        level_set_range=ls_range,
        strain_max_initial=round(hist[0]["strain_max"], 3),
        strain_max_final=round(hist[-1]["strain_max"], 3),
        strain_mean_final=round(hist[-1]["strain_mean"], 5),
        weak_fraction_final=round(hist[-1]["weak_fraction"], 5),
        output=args.out,
    )), flush=True)
