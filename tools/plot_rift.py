#!/usr/bin/env python3
"""
Plot the output of `gadopt_rift_case.py` with pyGMT.

    python3 tools/plot_rift.py /tmp/rift_long.npz --prefix rift_long

Produces a strain panel, a viscosity panel and a strain-rate panel, plus a
one-line summary of the localisation diagnostics.

Run this in the container (it needs pyGMT), or in any environment with the
suite's conda environment active:

    docker run --rm -v $PWD:/work -w /work \\
      ghcr.io/earthbyte/geodyn-pygmt:0.1.1 \\
      python3 tools/plot_rift.py rift_long.npz
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "..")
from geodynkit import plotting  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--prefix", default="rift")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    d = np.load(args.npz)
    x, z = d["x_km"], d["depth_km"]

    # Diagnostics first — a picture of a model you have not sanity-checked is
    # just a picture.
    st = d["strain"]
    inseed = np.broadcast_to(np.abs(x[None, :] - 100.0) < 25.0, st.shape)
    crust = np.broadcast_to((z[:, None] > 5.0) & (z[:, None] < 40.0), st.shape)
    print(f"strain   in-seed {st[inseed & crust].mean():.3f}   "
          f"outside {st[(~inseed) & crust].mean():.3f}   "
          f"max {st.max():.3f}   fully weakened {(st > 1.5).mean():.4%}")
    print(f"|u|max   {np.hypot(d['vx'], d['vz']).max():.2f} (boundary velocity = 1)")
    if (d["viscosity"] <= 0).any():
        print(f"WARNING: {(d['viscosity'] <= 0).sum()} non-positive viscosity values "
              "— regenerate with the CG1 export")

    panels = [
        ("strain", d["strain"], "strain_rate", "Accumulated plastic strain",
         "plastic strain", None),
        ("viscosity", np.log10(np.maximum(d["viscosity"], 1e18)), "viscosity",
         "Effective viscosity", "log@-10@- viscosity (Pa s)", None),
        ("strainrate", np.log10(np.maximum(d["strain_rate"], 1e-20)), "strain_rate",
         "Strain rate", "log@-10@- strain rate (s@+-1@+)", None),
    ]

    for name, field, kind, title, label, series in panels:
        fig = plotting.field_panel(
            field, x, z, kind=kind, title=title, label=label, series=series,
            vx=d["vx"], vz=d["vz"], every=40, width_cm=17.0,
            xlabel="distance (km)", ylabel="depth (km)",
        )
        out = f"{args.prefix}_{name}.png"
        fig.savefig(out, dpi=args.dpi)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
