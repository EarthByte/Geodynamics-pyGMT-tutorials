#!/usr/bin/env python3
"""
Why does the Picard iteration stall, and what fixes it?

`rift_divergence_probe.py` established *that* it stalls: at iteration caps of
10, 30, 60, 120 and 240 the rift case returns the identical iterate — 27
iterations used, relative residual 3.006e-4, ||div u||/||grad u|| = 0.52. More
iterations are not the answer. This script asks why, and tries the standard
remedies on the same problem so they can be compared rather than guessed at.

The suspicion, and what section 1 measures
------------------------------------------
The effective viscosity is

    mu_eff = min(mu_creep, mu_plast)      (plus a damper and hard caps)

which is not differentiable at the switch. A Picard iteration on a non-smooth
map can *chatter*: a cell sits near the yield surface, the previous iterate puts
it on the plastic branch, the resulting velocity puts it back on the ductile
branch, and it flips every iteration forever. The residual then plateaus at the
amplitude of the chatter instead of falling. If that is what is happening, the
number of points changing branch between iterations settles at a non-zero value
rather than going to zero — which is directly measurable, so section 1 measures
it alongside the residual and the relaxation factor.

The strategies in section 2
---------------------------
* **baseline** — the driver's adaptive under-relaxation, for reference.
* **fixed omega** — is the adaptation helping or hurting? Adaptive schemes that
  halve on any increase can ratchet omega down to nothing on an iteration that
  is *supposed* to be non-monotone.
* **viscosity relaxation** — under-relax the viscosity FIELD between iterations
  rather than the solution vector. This is the targeted fix for chatter: a cell
  that flips branch moves its viscosity only part of the way, so the switch stops
  being a step function from the iteration's point of view.
* **Anderson acceleration** — form the next iterate from a least-squares
  combination of the last m fixed-point residuals rather than from the last one
  alone. The standard remedy for a stagnating fixed-point iteration, and it
  costs one small dense solve per step.

Usage:
    python3 tools/rift_picard_probe.py --nx 64 --ny 32 --mode chatter
    python3 tools/rift_picard_probe.py --nx 64 --ny 32 --mode strategies
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
sys.path.insert(0, "tools")
import gadopt_rift_case as R  # noqa: E402


def rel_div(m):
    u = m["u"]
    d = sqrt(assemble(div(u) ** 2 * dx))
    g = sqrt(assemble(inner(grad(u), grad(u)) * dx))
    return float(d / max(g, 1e-300))


def yield_indicator(m, space):
    """1 where the plastic branch governs, 0 where creep does."""
    return Function(space).interpolate(
        conditional(m["mu_plast"] < m["mu_creep"], 1.0, 0.0))


# ---------------------------------------------------------------------------
# 1. Is it chatter?
# ---------------------------------------------------------------------------
def chatter(m, iters=60, omega=0.5, adaptive=True):
    """Run the driver's Picard scheme, recording what each iteration does."""
    P0 = FunctionSpace(m["mesh"], "DG", 0)
    Zs = m["z"].function_space()
    z_prev, z_raw = Function(Zs), Function(Zs)

    m["switch"].assign(0.0)
    m["picard"].solve()
    m["switch"].assign(1.0)

    prev_flag = yield_indicator(m, P0).dat.data_ro.copy()
    rows, hist = [], []
    for i in range(iters):
        z_prev.assign(m["z"])
        m["z_pic"].assign(m["z"])
        m["picard"].solve()
        z_raw.assign(m["z"])

        unorm = max(float(norm(split(z_raw)[0])), 1e-12)
        du = float(norm(split(z_raw)[0] - split(z_prev)[0])) / unorm
        hist.append(du)

        m["z"].assign(z_prev + omega * (z_raw - z_prev))

        flag = yield_indicator(m, P0).dat.data_ro.copy()
        flipped = int((flag != prev_flag).sum())
        yielded = int(flag.sum())
        prev_flag = flag

        rows.append(dict(i=i, du=du, omega=round(omega, 5),
                         flipped=flipped, yielded=yielded,
                         n_cells=flag.size))
        print(f"  it {i:>3}  du {du:.4e}  omega {omega:.4f}  "
              f"flipped {flipped:>5} / {flag.size}  yielded {yielded:>5}",
              flush=True)

        if adaptive and len(hist) > 1:
            if hist[-1] > hist[-2]:
                omega = max(0.05, 0.5 * omega)
            elif hist[-1] < 0.5 * hist[-2]:
                omega = min(1.0, 1.3 * omega)
    return rows


# ---------------------------------------------------------------------------
# 2. Candidate strategies
# ---------------------------------------------------------------------------
def picard_solution_relax(m, iters, omega0=0.5, adaptive=True, tol=1e-6):
    Zs = m["z"].function_space()
    z_prev, z_raw = Function(Zs), Function(Zs)
    omega, hist = omega0, []
    m["switch"].assign(0.0)
    m["picard"].solve()
    m["switch"].assign(1.0)
    for i in range(iters):
        z_prev.assign(m["z"])
        m["z_pic"].assign(m["z"])
        m["picard"].solve()
        z_raw.assign(m["z"])
        unorm = max(float(norm(split(z_raw)[0])), 1e-12)
        du = float(norm(split(z_raw)[0] - split(z_prev)[0])) / unorm
        hist.append(du)
        m["z"].assign(z_prev + omega * (z_raw - z_prev))
        if du < tol:
            break
        if adaptive and len(hist) > 1:
            if hist[-1] > hist[-2]:
                omega = max(0.05, 0.5 * omega)
            elif hist[-1] < 0.5 * hist[-2]:
                omega = min(1.0, 1.3 * omega)
    return hist


def picard_anderson(m, iters, depth=5, beta=1.0, tol=1e-6):
    """Anderson acceleration on the Picard map, in the raw dof vectors.

    Keeps the last `depth` fixed-point residuals f_k = G(x_k) - x_k and takes

        x_{k+1} = sum_j alpha_j G(x_{k-j}),   alpha = argmin || sum_j alpha_j f_{k-j} ||

    with the alphas constrained to sum to 1. Solved as an unconstrained least
    squares in the differences, which is the standard formulation and avoids the
    ill-conditioning of the constrained one.
    """
    Zs = m["z"].function_space()
    z_prev = Function(Zs)
    X, F, hist = [], [], []

    m["switch"].assign(0.0)
    m["picard"].solve()
    m["switch"].assign(1.0)

    with m["z"].dat.vec_ro as v:
        x = v.array_r.copy()

    for i in range(iters):
        # One application of the fixed-point map G.
        with m["z"].dat.vec as v:
            v.array[:] = x
        z_prev.assign(m["z"])
        m["z_pic"].assign(m["z"])
        m["picard"].solve()
        with m["z"].dat.vec_ro as v:
            gx = v.array_r.copy()

        unorm = max(float(norm(split(m["z"])[0])), 1e-12)
        du = float(norm(split(m["z"])[0] - split(z_prev)[0])) / unorm
        hist.append(du)
        if du < tol:
            break

        f = gx - x
        X.append(gx)
        F.append(f)
        if len(X) > depth:
            X.pop(0)
            F.pop(0)

        if len(F) == 1:
            x = x + beta * f
        else:
            # Least squares in the residual differences.
            dF = np.column_stack([F[j + 1] - F[j] for j in range(len(F) - 1)])
            dX = np.column_stack([X[j + 1] - X[j] for j in range(len(X) - 1)])
            gamma, *_ = np.linalg.lstsq(dF, F[-1], rcond=None)
            x = X[-1] - dX @ gamma - (1.0 - beta) * (F[-1] - dF @ gamma)

    with m["z"].dat.vec as v:
        v.array[:] = x
    return hist


STRATEGIES = {
    "baseline (adaptive omega)": lambda m, n: picard_solution_relax(m, n),
    "fixed omega 0.3": lambda m, n: picard_solution_relax(m, n, 0.3, False),
    "fixed omega 0.7": lambda m, n: picard_solution_relax(m, n, 0.7, False),
    "fixed omega 1.0": lambda m, n: picard_solution_relax(m, n, 1.0, False),
    "anderson depth 5": lambda m, n: picard_anderson(m, n, depth=5),
    "anderson depth 10": lambda m, n: picard_anderson(m, n, depth=10),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--seed-km", type=float, default=10.0)
    ap.add_argument("--heatflow", type=float, default=0.055)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--mode", choices=["chatter", "strategies"],
                    default="chatter")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def fresh():
        return R.build(args.nx, args.ny, seed_halfwidth_km=args.seed_km,
                       heat_flow=args.heatflow)

    if args.mode == "chatter":
        rows = chatter(fresh(), iters=args.iters)
        tail = rows[len(rows) // 2:]
        print("\nsecond half of the run:")
        print(f"  residual      min {min(r['du'] for r in tail):.3e}  "
              f"max {max(r['du'] for r in tail):.3e}")
        print(f"  points flipping branch per iteration: "
              f"min {min(r['flipped'] for r in tail)}  "
              f"max {max(r['flipped'] for r in tail)}  "
              f"of {rows[0]['n_cells']} cells")
        print("\nIf that flip count is stuck well above zero while the residual "
              "has plateaued,\nthe iteration is chattering across the yield "
              "switch, not converging slowly.")
        result = rows
    else:
        result = []
        for name, fn in STRATEGIES.items():
            m = fresh()
            hist = fn(m, args.iters)
            best = min(hist)
            result.append(dict(strategy=name, iterations=len(hist),
                               final=hist[-1], best=best,
                               rel_div=rel_div(m),
                               umax=float(np.abs(m["u"].dat.data_ro).max())))
            r = result[-1]
            print(f"  {name:<26} {r['iterations']:>4} its   "
                  f"best {best:.3e}   final {hist[-1]:.3e}   "
                  f"rel div {r['rel_div']:.4f}   |u|max {r['umax']:7.3f}",
                  flush=True)

    print("PROBE " + json.dumps(result), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
