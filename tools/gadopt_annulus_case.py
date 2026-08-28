#!/usr/bin/env python3
"""
Thermal convection in a 2-D cylindrical annulus, after G-ADOPT's `2d_cylindrical`
demo, with the knobs the demo hardcodes exposed so the cost can be measured.

The annulus is the first geometry in this suite that is not a box, and three
things change with it:

* **There are no side walls.** A Cartesian box needs boundary conditions on four
  sides and the two vertical ones are a fiction -- the mantle has no edges. The
  annulus is periodic by construction, so that fiction goes away.
* **The two boundaries have different areas.** Surface area scales with radius,
  so with rmin = 1.22 and rmax = 2.22 the top boundary is 1.8x the bottom. Heat
  entering the base spreads over a larger surface, which changes the Nusselt
  number's meaning and is why the annulus is not just a bent box.
* **The velocity nullspace grows.** A closed box with free-slip walls admits no
  rigid-body motion; a closed annulus admits a **rotation**. It has to be told
  about, or the pressure solve is singular.

Diagnostics follow the demo: RMS velocity and Nusselt number at both boundaries.

Usage:
    python3 gadopt_annulus_case.py --ncells 128 --nlayers 32 --tol 1e-5
"""

import argparse
import json
import sys
import time

import numpy as np

# PETSc parses sys.argv on import; hide our flags from it (see the rift driver).
_ARGV = sys.argv[:]
sys.argv = sys.argv[:1]
from gadopt import *  # noqa: E402
sys.argv = _ARGV

RMIN, RMAX = 1.22, 2.22
# Only rank 0 prints, or an mpiexec run emits one copy of every line per rank.
ROOT = COMM_WORLD.rank == 0


def build(ncells=128, nlayers=32, ra=1e5, plate_profile=None,
          su_advection=False):
    mesh1d = CircleManifoldMesh(ncells, radius=RMIN, degree=2)
    mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type="radial")
    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 2)
    Z = MixedFunctionSpace([V, W])
    z = Function(Z)
    u, p = split(z)
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")

    approximation = BoussinesqApproximation(Constant(ra))
    delta_t = Constant(1e-7)
    t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1,
                              increase_tolerance=1.5)

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    T = Function(Q, name="Temperature")
    # Conductive profile plus a wavenumber-4 perturbation. The perturbation is
    # what breaks the rotational symmetry; without it the initial state is a
    # fixed point and nothing ever happens.
    T.interpolate(RMAX - r + 0.02 * cos(4 * atan2(X[1], X[0]))
                  * sin((r - RMIN) * pi))

    # THE ROTATIONAL NULLSPACE. A closed annulus with free-slip boundaries can
    # spin as a rigid body at no cost, so the discrete Stokes operator is
    # singular in that direction and the solve will not converge until the
    # nullspace is declared. This is the one thing about the annulus that bites
    # immediately if you forget it, and the Cartesian box never taught you to
    # expect it.
    # With the surface velocity prescribed the annulus can no longer spin
    # freely, so the rotational nullspace disappears -- declaring it anyway
    # would remove a component of the solution that is now physical.
    rotational = plate_profile is None
    ns = create_stokes_nullspace(Z, closed=True, rotational=rotational)
    near_ns = create_stokes_nullspace(Z, closed=False, rotational=True,
                                      translations=[0, 1])

    # SURFACE BOUNDARY CONDITION.
    #
    # Free slip (`un = 0`, tangential stress zero) is the default and is what
    # T14 uses: the surface is free to move however the interior wants.
    #
    # With a plate profile the surface instead becomes *kinematic* -- the full
    # velocity vector is prescribed, normal component zero and tangential
    # component taken from the reconstruction. That is the difference between a
    # convecting shell that happens to have a top and a mantle being driven by
    # plates.
    u_surf = None
    if plate_profile is not None:
        th_prof, v_prof = plate_profile           # degrees, non-dimensional
        u_surf = Function(V, name="PlateVelocity")
        # dof coordinates in the vector space's own layout
        Xv = Function(V).interpolate(X).dat.data_ro
        th = np.degrees(np.arctan2(Xv[:, 1], Xv[:, 0])) % 360.0
        # periodic interpolation onto the sampled profile
        vt = np.interp(th, th_prof, v_prof, period=360.0)
        u_surf.dat.data[:, 0] = -np.sin(np.radians(th)) * vt
        u_surf.dat.data[:, 1] = np.cos(np.radians(th)) * vt

    stokes_bcs = {boundary.bottom: {"un": 0}}
    stokes_bcs[boundary.top] = ({"un": 0} if u_surf is None else {"u": u_surf})
    temp_bcs = {boundary.bottom: {"T": 1.0}, boundary.top: {"T": 0.0}}

    # STREAMLINE-UPWIND STABILISATION, and when it stops being optional.
    #
    # Free-slip convection at Ra = 1e5 has u_rms ~ 200 and a cell size of about
    # 1/16, so the cell Peclet number is around 12 -- borderline, and plain
    # Galerkin copes. Prescribe plate velocities and u_rms jumps to ~2000,
    # taking the cell Peclet to about 125, and it does not: the 0 Ma case
    # undershoots to T = -0.44 with 2.5% of the domain outside [0, 1].
    #
    # This is T03's lesson arriving in a real model. The cure is the same, and
    # so is the price -- streamline-upwind stabilisation adds numerical
    # diffusion along the flow direction, which is exactly what suppresses the
    # oscillation and exactly what smears a sharp thermal front.
    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint,
                                 bcs=temp_bcs, su_advection=su_advection)
    stokes_solver = StokesSolver(
        z, approximation, T, bcs=stokes_bcs, solver_parameters="iterative",
        nullspace=ns, transpose_nullspace=ns, near_nullspace=near_ns,
    )
    gd = GeodynamicalDiagnostics(z, T, boundary.bottom, boundary.top,
                                 quad_degree=6)
    return dict(mesh=mesh, z=z, T=T, delta_t=delta_t, t_adapt=t_adapt,
                energy=energy_solver, stokes=stokes_solver, gd=gd, Q=Q,
                ncells=ncells, nlayers=nlayers, ra=ra, u_surf=u_surf,
                su_advection=su_advection)


def run(m, tol=1e-5, max_steps=20000, report_every=50, min_steps=100):
    """Step to steady state. Returns (history, seconds, converged)."""
    T_old = Function(m["Q"])
    hist, t0, converged = [], time.perf_counter(), False
    time_now, taken = 0.0, 0
    for step in range(max_steps):
        dt = float(m["delta_t"]) if step == 0 else m["t_adapt"].update_timestep()
        time_now += dt
        m["stokes"].solve()
        T_old.assign(m["T"])
        m["energy"].solve()

        # Steady state, measured as G-ADOPT's demo does: the L2 change in
        # temperature over one step.
        #
        # `min_steps` is not cosmetic. The timestep starts at 1e-7 and is grown
        # by the adaptor towards 0.1, so the change over the first step is
        # small for a reason that has nothing to do with steadiness -- the very
        # first evaluation gives 7.6e-6, which passes any tolerance looser than
        # that and declares a conductive initial condition "converged". The
        # rate of change is reported alongside for the same reason: it is the
        # dt-independent version of the same quantity.
        maxchange = float(np.sqrt(assemble((m["T"] - T_old) ** 2 * dx)))
        steady = maxchange < tol and step >= min_steps
        # Always record the final step, or a run that stops on `max_steps`
        # reports diagnostics from whenever the last periodic sample happened.
        if step % report_every == 0 or steady or step == max_steps - 1:
            rec = dict(step=step, time=round(time_now, 8), dt=round(dt, 10),
                       maxchange=maxchange, change_rate=maxchange / dt,
                       u_rms=float(m["gd"].u_rms()),
                       nu_top=float(m["gd"].Nu_top()),
                       nu_base=float(m["gd"].Nu_bottom()))
            hist.append(rec)
            if ROOT:
                print(f"  step {step:5d}  t {time_now:.5f}  dt {dt:.2e}  "
                      f"dT {maxchange:.3e}  u_rms {rec['u_rms']:8.3f}  "
                      f"Nu {rec['nu_top']:6.3f}/{rec['nu_base']:6.3f}", flush=True)
        # `taken` is the real step count. `hist` only records every
        # `report_every` step, so reading the count off its last entry
        # undercounts badly -- a 30-step run reported 1.
        taken = step + 1
        if steady:
            converged = True
            break
    return hist, time.perf_counter() - t0, converged, taken


def sample(m, ntheta=361, nr=61):
    """Sample onto a regular (theta, r) grid — the form GMT's polar projection wants."""
    # Inset the sampling radii. `CircleManifoldMesh(ncells)` is a regular
    # ncells-gon, so the true circle of radius RMIN lies slightly OUTSIDE the
    # mesh and points placed on it are silently dropped -- 352 of them, the
    # first time this was run. The sagitta r(1 - cos(pi/ncells)) is the gap;
    # take twice it, which is negligible for plotting.
    pad = 2.0 * RMAX * (1.0 - np.cos(np.pi / m["ncells"]))
    th = np.linspace(0.0, 360.0, ntheta)
    rr = np.linspace(RMIN + pad, RMAX - pad, nr)
    TH, RR = np.meshgrid(np.radians(th), rr)
    pts = np.column_stack([(RR * np.cos(TH)).ravel(), (RR * np.sin(TH)).ravel()])

    P1 = FunctionSpace(m["mesh"], "CG", 1)
    ev = PointEvaluator(m["mesh"], pts, tolerance=1e-8,
                        missing_points_behaviour="warn")
    Tg = np.asarray(ev.evaluate(Function(P1).interpolate(m["T"]))).reshape(nr, ntheta)
    uu = np.asarray(ev.evaluate(m["z"].subfunctions[0])).reshape(nr, ntheta, 2)
    # Cartesian velocity -> radial and tangential components.
    ur = uu[..., 0] * np.cos(TH) + uu[..., 1] * np.sin(TH)
    ut = -uu[..., 0] * np.sin(TH) + uu[..., 1] * np.cos(TH)
    return dict(theta_deg=th, radius=rr, temperature=Tg, u_r=ur, u_theta=ut,
                vx=uu[..., 0], vy=uu[..., 1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncells", type=int, default=128)
    ap.add_argument("--nlayers", type=int, default=32)
    ap.add_argument("--ra", type=float, default=1e5)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="steady-state tolerance. The demo uses 1e-7, which "
                         "costs over an hour; see T14 for the trade-off")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--min-steps", type=int, default=100,
                    help="steps before the steady-state test is allowed to pass")
    ap.add_argument("--su", action="store_true",
                    help="streamline-upwind stabilisation in the energy "
                         "equation. Needed once the cell Peclet number is "
                         "large, which plate driving makes it")
    ap.add_argument("--plate-dir", default=None,
                    help="directory holding a reconstruction (rotation model "
                         "plus topological plate boundaries). Without it the "
                         "surface is free-slip, as in T14")
    ap.add_argument("--plate-age", type=float, default=0.0,
                    help="reconstruction age in Ma")
    ap.add_argument("--pole-lat", type=float, default=90.0,
                    help="pole of the great circle sampled; 90 gives the equator")
    ap.add_argument("--pole-lon", type=float, default=0.0)
    ap.add_argument("--out", default="annulus.npz")
    args = ap.parse_args()

    profile = None
    if args.plate_dir:
        import glob
        sys.path.insert(0, "..")
        sys.path.insert(0, ".")
        from geodynkit import plates as PL
        rot = sorted(glob.glob(f"{args.plate_dir}/**/*.rot", recursive=True))
        top = sorted(glob.glob(f"{args.plate_dir}/**/*.gpml", recursive=True)
                     + glob.glob(f"{args.plate_dir}/**/*.gpmlz", recursive=True))
        if not rot or not top:
            raise SystemExit(f"no rotation/topology files under {args.plate_dir}")
        th = np.linspace(0.0, 360.0, 181)
        prof = PL.surface_velocity_profile(rot, top, args.plate_age, th,
                                           pole_lat=args.pole_lat,
                                           pole_lon=args.pole_lon)
        v_nd = PL.nondimensionalise_velocity(prof["velocity_cm_yr"])
        profile = (th, v_nd)
        if ROOT:
            print(f"plate model: {len(rot)} rotation, {len(top)} topology files; "
                  f"age {args.plate_age} Ma; |v| up to "
                  f"{np.abs(prof['velocity_cm_yr']).max():.2f} cm/yr "
                  f"({np.abs(v_nd).max():.0f} non-dimensional)", flush=True)

    m = build(args.ncells, args.nlayers, args.ra, plate_profile=profile,
              su_advection=args.su)
    hist, secs, converged, taken = run(m, tol=args.tol,
                                       max_steps=args.max_steps,
                                       min_steps=args.min_steps)
    flds = sample(m)
    if profile is not None:
        flds["plate_theta_deg"] = profile[0]
        flds["plate_velocity_nd"] = profile[1]
        flds["plate_velocity_cm_yr"] = prof["velocity_cm_yr"]
        flds["plate_id"] = prof["plate_id"]
    np.savez(args.out, **flds)

    last = hist[-1]
    T_arr = flds["temperature"]
    t_min, t_max = float(np.nanmin(T_arr)), float(np.nanmax(T_arr))
    if not ROOT:
        sys.exit(0)
    print("RESULT " + json.dumps(dict(
        ncells=args.ncells, nlayers=args.nlayers, ra=args.ra, tol=args.tol,
        plate_age=(args.plate_age if args.plate_dir else None),
        plate_driven=bool(args.plate_dir), su_advection=args.su,
        converged=converged, steps=taken, seconds=round(secs, 1),
        procs=COMM_WORLD.size,
        model_time=last["time"], final_maxchange=last["maxchange"],
        final_change_rate=round(last["change_rate"], 6),
        u_rms=round(last["u_rms"], 4),
        T_min=round(t_min, 4), T_max=round(t_max, 4),
        T_out_of_bounds=round(float(np.mean((T_arr < -1e-6)
                                            | (T_arr > 1 + 1e-6))), 5),
        nu_top=round(last["nu_top"], 4), nu_base=round(last["nu_base"], 4),
        # NOT an imbalance. At steady state the heat entering the base must
        # leave the top, but the two boundaries have different areas, so it is
        # the *fluxes* that balance and not the Nusselt numbers:
        #     Nu_base * 2 pi rmin = Nu_top * 2 pi rmax
        # so Nu_base / Nu_top should equal rmax / rmin = 1.820. A box would give
        # 1. This ratio is therefore a verification of the converged state that
        # needs no reference solution at all -- see T14.
        nu_ratio=round(last["nu_base"] / max(last["nu_top"], 1e-12), 4),
        nu_ratio_expected=round(RMAX / RMIN, 4),
        nu_ratio_error=round(abs(last["nu_base"] / max(last["nu_top"], 1e-12)
                                 - RMAX / RMIN) / (RMAX / RMIN), 5),
        output=args.out,
    )), flush=True)
