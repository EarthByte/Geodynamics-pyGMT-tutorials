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


def build(nx, ny, aspect=2.0, damper=1e21, seed_halfwidth_km=25.0,
          heat_flow=0.055):
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
    Tc = np.atleast_1d(LI.geotherm(zc, surface_heat_flow=heat_flow))
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
    # Seed width matters. 25 km half-width on a 200 km domain is not a seed,
    # it is a weak province occupying a quarter of the model — and a province
    # cannot localise, because there is no gradient for strain to concentrate
    # into. A real seed is a few km across.
    sw = seed_halfwidth_km * 1e3 / H
    seed = ((np.abs(xc - aspect / 2) < sw) & (yc > y_lc))
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
        # PLASTIC DAMPER, after Duretz et al. (2020) and used in ASPECT's
        # continental_extension cookbook ("Plastic damper viscosity = 1e21").
        # Without it the plastic branch can drive the viscosity arbitrarily low
        # wherever the strain rate is large; the linearised Picard problem then
        # has near-zero-viscosity regions where the velocity is essentially
        # unconstrained, and the iteration converges happily to |u| ~ 1e4 times
        # the boundary velocity. Adding a damper in series puts a floor under
        # the plastic viscosity that scales with the physics rather than being
        # an arbitrary clip.
        mu_damp = damper / MU0
        mu_p = sigma_y / (2 * epsii) + mu_damp

        # switch = 0 disables the plastic branch entirely
        mu_eff = conditional(switch > 0.5, min_value(mu_c, mu_p), mu_c)
        # Viscosity contrast capped at 1e6. The cookbook's 1e18-1e26 range is
        # 1e8, which is solvable in ASPECT's SI formulation but not here.
        return (max_value(min_value(mu_eff, 1e26 / MU0), 1e20 / MU0),
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

    # Reinitialisation pseudo-timestep MUST scale with the interface thickness.
    # G-ADOPT's default is a fixed 0.02, while `interface_thickness` returns
    # ~0.35 * h_min — so the default is stable on a coarse mesh and unstable on
    # a fine one. At 32x16 epsilon ~ 0.022 and it works; at 96x48 epsilon ~
    # 0.0073 and reinitialisation returns DIVERGED_FUNCTION_NANORINF. Tying the
    # step to epsilon makes the model resolution-independent, which is the
    # difference between a demo and something you can refine.
    eps_min = float(epsilon.dat.data_ro.min())
    reini = {"epsilon": epsilon, "timestep": 0.5 * eps_min, "steps": 2}
    ls_solver = [LevelSetSolver(psi, adv_kwargs={"u": u_fn, "timestep": dt},
                                reini_kwargs=reini)
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
                h_min=aspect / nx,
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


class PicardDiverged(RuntimeError):
    """Raised when the Picard iteration is going backwards.

    This exists because the alternative is worse. Picard can diverge silently:
    it returns a velocity field, the code carries on, and the first sign of
    trouble is the level-set advection exploding several steps later with an
    error that points at the wrong component entirely. Ask me how I know.
    """


def solve_stokes(m, picard_iters, tol=1e-4, cold=False, u_sane=1e3):
    """Robust visco-plastic Stokes solve: isoviscous -> Picard -> Newton.

    Returns (picard_iterations_used, newton_converged, residual_history).
    """
    if cold:
        m["switch"].assign(0.0)          # linear viscosity; breaks the 1/epsii
        m["picard"].solve()              # singularity at u = 0
        m["switch"].assign(1.0)

    # Damped (under-relaxed) Picard. Undamped, this diverges above ~48x24:
    # each iteration overshoots, the viscosity evaluated at the overshoot is
    # worse, and the next overshoot is bigger. Taking only a fraction `omega`
    # of each update tames that, at the cost of more iterations.
    #
    # `omega` adapts: halve it whenever the update grows, and creep back up
    # while it shrinks. Fixed damping either converges slowly everywhere or not
    # at all where the rheology is stiffest.
    # Picard on this system OSCILLATES rather than converging monotonically:
    # the residual falls, rises, and falls again. An earlier version of this
    # code ran a fixed 40 iterations and reported success purely because it
    # happened to stop on a downswing — the same run stopped at iteration 19
    # would have looked like divergence.
    #
    # So: keep the BEST iterate seen, restore it at the end, and only call it
    # divergence if a long window passes with no improvement. This is the
    # standard safeguard for a non-monotone fixed-point iteration, and it makes
    # the outcome independent of where you happen to stop.
    used, hist = 0, []
    omega = m.get("omega0", 0.5)
    Zs = m["z"].function_space()
    z_prev, z_raw = Function(Zs), Function(Zs)
    z_best = Function(Zs)
    best_du, best_at = float("inf"), -1

    for i in range(picard_iters):
        z_prev.assign(m["z"])
        m["z_pic"].assign(m["z"])
        m["picard"].solve()
        z_raw.assign(m["z"])            # raw fixed-point image, kept separate

        # Measure the UNDAMPED update. This is the residual of the fixed-point
        # map and is independent of omega — essential, because omega changes
        # between iterations, so a damped step norm is not comparable with the
        # previous one and would corrupt both the convergence test and the
        # adaptation below.
        # RELATIVE residual. An absolute one is meaningless when the solution
        # magnitude is itself in question: at |u| ~ 5e4 an absolute update of
        # 1e-3 reads as converged to eight digits, while the answer is wrong by
        # four orders of magnitude.
        unorm = max(float(norm(split(z_raw)[0])), 1e-12)
        du = float(norm(split(z_raw)[0] - split(z_prev)[0])) / unorm
        hist.append(du)
        used += 1

        if du < best_du:
            best_du, best_at = du, i
            z_best.assign(z_raw)

        # Relax through an explicit temporary. Writing
        #   z.assign(z_prev + omega*(z - z_prev))
        # puts z on both sides of its own assignment, which aliases.
        m["z"].assign(z_prev + omega * (z_raw - z_prev))

        if du < tol:
            break

        if len(hist) > 1:
            if hist[-1] > hist[-2]:
                omega = max(0.05, 0.5 * omega)      # back off
            elif hist[-1] < 0.5 * hist[-2]:
                omega = min(1.0, 1.3 * omega)       # converging well, push on

        # Genuine stagnation: a long window with no new best. Not merely "the
        # residual went up", which happens routinely here.
        if i - best_at > 25:
            break

    # Restore the best iterate found, not whichever one we stopped on.
    m["z"].assign(z_best)
    hist.append(best_du)

    # The boundary velocity is 1 by construction, so a converged solution is
    # O(1-10). Anything vastly larger is not a solution, whatever the solver
    # reported.
    umax = float(np.abs(m["u"].dat.data_ro).max())
    if not np.isfinite(umax) or umax > u_sane:
        raise PicardDiverged(
            f"|u|max = {umax:.3e}, but the boundary velocity is 1 — "
            "the Stokes solution is not physical")

    try:
        m["stokes"].solve()              # Newton polish, now close enough
        return used, True, hist
    except ConvergenceError:
        return used, False, hist         # Picard result stands; keep going


def run(m, steps):
    """Step: Stokes -> advect level sets -> advect plastic strain."""
    hist = []
    t0 = time.perf_counter()
    for n in range(steps):
        pic, newton_ok, res = solve_stokes(m, picard_iters=120 if n == 0 else 30,
                                           cold=(n == 0))

        # CFL-limit the timestep from the ACTUAL velocity, not a guess. A fixed
        # timestep is a resolution-dependent bug waiting to happen: halve the
        # mesh and the Courant number doubles.
        umax = max(float(np.abs(m["u"].dat.data_ro).max()), 1e-12)
        dt_cfl = 0.4 * m["h_min"] / umax
        m["dt"].assign(min(2e-3, dt_cfl))

        for s in m["ls_solver"]:
            s.solve()
        m["strain_solver"].solve()

        sd = m["strain"].dat.data_ro
        hist.append(dict(step=n, picard=pic, newton=newton_ok,
                         strain_max=float(sd.max()),
                         strain_mean=float(sd.mean()),
                         weak_fraction=float((sd > 1.5).mean())))
        print(f"  step {n:3d}  picard {pic:3d} (res {res[-1]:.2e})  "
              f"newton {'ok' if newton_ok else 'FAILED'}  |u|max {umax:.2f}  "
              f"dt {float(m['dt']):.2e}  strain max {sd.max():.3f}", flush=True)
    return hist, time.perf_counter() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="/tmp/rift.npz")
    ap.add_argument("--heatflow", type=float, default=0.055,
                    help="surface heat flow W/m2; low = cold, strong, coupled crust")
    ap.add_argument("--seed-km", type=float, default=25.0,
                    help="seed half-width in km")
    ap.add_argument("--damper", type=float, default=1e21,
                    help="plastic damper viscosity, Pa s (cookbook: 1e21)")
    args = ap.parse_args()

    m = build(args.nx, args.ny, damper=args.damper,
              seed_halfwidth_km=args.seed_km,
              heat_flow=args.heatflow)

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

    # Sample onto a REGULAR grid. Saving raw dof arrays is useless for
    # plotting: CG2 dofs sit at vertices, edge midpoints and cell interiors in
    # an ordering that is not a reshape of the mesh, so there is no way to
    # recover a picture from them afterwards.
    Q, aspect = m["Q"], m["aspect"]
    nxs, nys = 4 * args.nx + 1, 4 * args.ny + 1
    xs = np.linspace(0.0, aspect, nxs)
    ys = np.linspace(0.0, 1.0, nys)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Export through CG1, not CG2. Viscosity and strain rate have near-
    # discontinuous transitions where the plastic branch takes over, and a
    # quadratic basis OVERSHOOTS at those jumps: interpolating into CG2 gave
    # 487 points of NEGATIVE viscosity (down to -4e24) and 313 of negative
    # strain rate, from a UFL expression bounded below at 1e20 and by a square
    # root respectively. The model was fine; the diagnostic was lying.
    # CG1 cannot overshoot a monotone jump, and positive fields are clipped as
    # a second line of defence.
    P1 = FunctionSpace(m["mesh"], "CG", 1)

    def grid(expr, positive=False):
        f = Function(P1).interpolate(expr)
        g = np.array(f.at(pts, tolerance=1e-8)).reshape(nys, nxs)[::-1]
        return np.maximum(g, 0.0) if positive else g

    out = dict(
        x_km=xs * H / 1e3,
        depth_km=(1.0 - ys[::-1]) * H / 1e3,
        strain=grid(m["strain"], positive=True),
        viscosity=grid(m["mu"], positive=True) * MU0,
        strain_rate=grid(m["epsii"], positive=True) * SR,
        density=grid(m["rho"], positive=True),
        weakening=grid(m["w"], positive=True),
    )
    uu = np.array(m["u"].at(pts, tolerance=1e-8)).reshape(nys, nxs, 2)[::-1]
    out["vx"], out["vz"] = uu[..., 0], -uu[..., 1]
    np.savez(args.out, **out)

    # Localisation metric. A rift concentrates strain at the seed and unloads
    # its surroundings; diffuse thinning raises both equally. The ratio is the
    # number to watch, not the absolute strain.
    xk, zk, st = out["x_km"], out["depth_km"], out["strain"]
    inseed = np.abs(xk[None, :] - 100.0) < 25.0
    crust = (zk[:, None] > 5.0) & (zk[:, None] < 40.0)
    s_in = float(st[np.broadcast_to(inseed, st.shape) & np.broadcast_to(crust, st.shape)].mean())
    s_out = float(st[(~np.broadcast_to(inseed, st.shape)) & np.broadcast_to(crust, st.shape)].mean())

    print("RESULT " + json.dumps(dict(
        damper=args.damper, seed_km=args.seed_km, heat_flow=args.heatflow,
        strain_in_seed=round(s_in, 4),
        strain_outside=round(s_out, 4),
        localisation=round(s_in / max(s_out, 1e-9), 3),
        steps=args.steps, nx=args.nx, ny=args.ny, seconds=round(secs, 2),
        level_set_range=ls_range,
        strain_max_initial=round(hist[0]["strain_max"], 3),
        strain_max_final=round(hist[-1]["strain_max"], 3),
        strain_mean_final=round(hist[-1]["strain_mean"], 5),
        weak_fraction_final=round(hist[-1]["weak_fraction"], 5),
        output=args.out,
    )), flush=True)
