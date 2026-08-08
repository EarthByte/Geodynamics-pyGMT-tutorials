#!/usr/bin/env python3
"""
Why do the level sets leave [0, 1]? Measure the incompressibility of the
velocity field that advects them.

A conservative level set is a smoothed indicator function, and the advection
equation

    d(psi)/dt + u . grad(psi) = 0

preserves its bounds ONLY if the advecting field is divergence-free. Stokes
guarantees that at the solution; it guarantees nothing about an iterate. If the
Picard iteration stops short, ``div u`` is not zero, the advection is no longer
a pure transport, and psi drifts out of [0, 1] a little every step.

That is a hypothesis, not a fact, so this script measures it: the relative
divergence of the velocity field as a function of how many Picard iterations it
was given. If the two fall together, the fix is in the solver, not the advection
scheme.

    python3 tools/rift_divergence_probe.py --nx 64 --ny 32
"""

import argparse
import json
import sys

import numpy as np

_ARGV = sys.argv[:]                    # keep PETSc from parsing our flags
sys.argv = sys.argv[:1]
from gadopt import *  # noqa: E402
sys.argv = _ARGV

sys.path.insert(0, ".")
sys.path.insert(0, "..")
import gadopt_rift_case as R  # noqa: E402


def relative_divergence(m):
    """||div u|| / ||grad u||, both in L2. Dimensionless; 0 at the solution.

    Normalising by the velocity gradient rather than by the velocity itself
    matters: divergence has units of a gradient, so ||div u|| / ||u|| would
    carry a length scale and could not be compared between meshes.
    """
    u = m["u"]
    d = sqrt(assemble(div(u) ** 2 * dx))
    g = sqrt(assemble(inner(grad(u), grad(u)) * dx))
    return float(d / max(g, 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--seed-km", type=float, default=10.0)
    ap.add_argument("--heatflow", type=float, default=0.055)
    ap.add_argument("--iters", type=int, nargs="+",
                    default=[10, 30, 60, 120, 240])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for k in args.iters:
        # Rebuild each time: a fresh cold start, so every entry gets the same
        # initial condition and the comparison is about iteration count alone.
        m = R.build(args.nx, args.ny, seed_halfwidth_km=args.seed_km,
                    heat_flow=args.heatflow)
        used, newton_ok, hist = R.solve_stokes(m, picard_iters=k, cold=True)
        umax = float(np.abs(m["u"].dat.data_ro).max())
        rows.append(dict(picard_cap=k, picard_used=used,
                         residual=round(hist[-1], 8),
                         newton=newton_ok,
                         rel_div=relative_divergence(m),
                         umax=round(umax, 4)))
        r = rows[-1]
        print(f"  cap {k:>4}  used {r['picard_used']:>4}  res {r['residual']:.3e}  "
              f"newton {'ok' if newton_ok else 'FAILED':6s}  "
              f"|u|max {r['umax']:7.3f}  rel div {r['rel_div']:.4e}", flush=True)

    print("PROBE " + json.dumps(rows), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
