#!/usr/bin/env python3
"""
A fixed-cost G-ADOPT convection case, for scaling measurements.

Notebook T08 needs to time the *same amount of work* on different numbers of
processes. G-ADOPT's own `base_case.py` runs until it reaches steady state,
which is exactly what you want for physics and exactly what you must not do for
a scaling study: the number of timesteps varies slightly between runs, so any
speed-up you measure is contaminated by a different amount of work.

So this script takes a **fixed** number of timesteps and reports the wall-clock
time spent in the time loop. Setup and I/O are excluded, because they are not
the part that parallelises and including them would flatter or flatten the
result depending on how much of it there is.

Usage:
    mpiexec -n 4 python3 gadopt_scaling_case.py --nx 80 --steps 30

Output: one line of JSON on rank 0, so the notebook can parse it without
scraping human-readable text.
"""

import argparse
import json
import time

from gadopt import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=40, help="cells per side")
    ap.add_argument("--steps", type=int, default=30, help="fixed timestep count")
    ap.add_argument("--Ra", type=float, default=1e4)
    args = ap.parse_args()

    nx = ny = args.nx
    mesh = UnitSquareMesh(nx, ny, quadrilateral=True)
    mesh.cartesian = True

    left, right, bottom, top = 1, 2, 3, 4

    V = VectorFunctionSpace(mesh, "CG", 2)   # velocity
    W = FunctionSpace(mesh, "CG", 1)         # pressure
    Q = FunctionSpace(mesh, "CG", 2)         # temperature
    Z = MixedFunctionSpace([V, W])

    z = Function(Z)
    u, p = split(z)
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")

    T = Function(Q, name="Temperature")
    X = SpatialCoordinate(mesh)
    T.interpolate(
        (1.0 - X[1])
        + 0.05 * cos(pi * X[0]) * sin(pi * X[1])
    )

    Ra = Constant(args.Ra)
    approximation = BoussinesqApproximation(Ra)

    delta_t = Constant(1e-6)
    t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1,
                              increase_tolerance=1.5)

    stokes_bcs = {
        bottom: {"uy": 0}, top: {"uy": 0},
        left: {"ux": 0}, right: {"ux": 0},
    }
    temp_bcs = {bottom: {"T": 1.0}, top: {"T": 0.0}}

    energy_solver = EnergySolver(T, u, approximation, delta_t,
                                 ImplicitMidpoint, bcs=temp_bcs)
    # Note the argument order: (z, approximation, T). Passing T second raises
    # "'CoordinatelessFunction' object has no attribute 'rho_continuity'",
    # which is a confusing way to be told the arguments are the wrong way round.
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
    stokes_solver = StokesSolver(
        z, approximation, T, bcs=stokes_bcs,
        nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
    )

    # One solve outside the timed loop. The first Stokes solve pays for symbolic
    # setup, JIT compilation of the generated kernels and matrix preallocation —
    # costs that are paid once and would otherwise be misattributed to the
    # parallel part.
    stokes_solver.solve()

    comm = mesh.comm
    comm.Barrier()
    t0 = time.perf_counter()

    for _ in range(args.steps):
        t_adapt.update_timestep()
        stokes_solver.solve()
        energy_solver.solve()

    comm.Barrier()
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        print("RESULT " + json.dumps({
            "nx": args.nx,
            "steps": args.steps,
            "nprocs": comm.size,
            "seconds": round(elapsed, 3),
            "seconds_per_step": round(elapsed / args.steps, 4),
            "velocity_dofs": V.dim(),
            "total_dofs": Z.dim() + Q.dim(),
        }), flush=True)


if __name__ == "__main__":
    main()
