# The long rift run — instructions

Everything below is parameterised and tested at 20 steps. What has *not* been
done is a run long enough for strain weakening to propagate and a fault to grow.

## Why q0 = 0.040

The heat-flow sweep at 64x32, 20 steps, damper 1e21, seed +/-10 km:

| q0 (W/m2) | in-seed | outside | ratio | max strain | mean | weak frac |
|---|---|---|---|---|---|---|
| **0.040** | 0.77 | 0.39 | 1.95 | **1.53 (grew)** | **0.60** | **6e-4** |
| 0.048 | 0.66 | 0.23 | 2.93 | 1.40 (decayed) | 0.18 | 0 |
| 0.055 | 0.72 | 0.28 | 2.52 | 1.48 (decayed) | 0.16 | 0 |

Cold crust is the only case where mean strain is large (0.60 vs ~0.17), where
`strain_max` grows past the seeded maximum instead of decaying, and where any
material at all crosses the weakening threshold.

**Ignore the `ratio` column.** It rewards runs where nothing happens anywhere,
because a small denominator inflates it. It misled us twice: once when a damper
of 1e22 scored 15.9 with essentially no deformation, and again here where 0.048
looks best while accumulating a third of the strain. Judge by strain *growth*
and by `weak_fraction`.

Physical basis: the strength envelope puts the deepest brittle-ductile
transition at 78.9 km for q0 = 0.040 and 24.1 km for q0 = 0.080. Cold means the
crust is coupled to the mantle and transmits stress; warm means a ductile lower
crust decouples them and extension spreads out. That is the textbook narrow vs
wide rift control, and we had been sitting on the wide-rift side.

## The run

```bash
docker run --rm -v $PWD:/work -w /work ghcr.io/earthbyte/geodyn-pygmt:0.1.1 \
  python3 tools/gadopt_rift_case.py \
    --nx 96 --ny 48 --steps 400 \
    --heatflow 0.040 --seed-km 10 --damper 1e21 \
    --out rift_long.npz | tee rift_long.log
```

Expect roughly **6-8 hours**: about 50-60 s per step at 96x48, and cold rock is
~55% slower to converge than warm (511 s vs 330 s for 20 steps at 64x32).
Start it and leave it.

If you want a first look sooner, run 150 steps as well — that is ~2.5 h and
should already show whether weakening is spreading.

## What to watch in the log

Each step prints picard count, residual, |u|max, dt and strain max.

* **`|u|max` must stay of order 1-10.** The boundary velocity is 1. If it
  climbs past 1e3 the run aborts by design (`PicardDiverged`) rather than
  producing plausible nonsense.
* **`strain max` should climb past 1.5.** That is the threshold at which
  weakening is complete; below it the feedback loop is only partly engaged.
* **`newton FAILED` is expected and harmless.** Picard carries the solution.
* **`dt` shrinking** is the CFL limiter responding to faster flow — normal, and
  a sign something is localising.

## Success criteria

The run has produced a rift if:

1. `weak_fraction_final` is percent-level, not 1e-4 — a real volume of rock has
   fully weakened;
2. strain growth *inside* the seed clearly exceeds growth outside (compare
   against `strain_max_initial`, do not use the ratio);
3. the strain field shows a **dipping band** reaching the surface, rather than
   the horizontal layer-parallel band we have been getting.

Failure mode to expect: strain still accumulating in a flat lower-crustal band
across the whole domain. If that happens after 400 steps, the next lever is
thermal coupling (`EnergySolver`), so upwelling advects heat into the axis and
weakens it preferentially — the feedback that actually selects a rift axis.

## Plotting

```bash
docker run --rm -v $PWD:/work -w /work ghcr.io/earthbyte/geodyn-pygmt:0.1.1 \
  python3 tools/plot_rift.py rift_long.npz --prefix rift_long
```

Prints the diagnostics and writes three pyGMT panels: strain, viscosity and
strain rate. It warns if any viscosity is non-positive, which would mean the
output predates the CG1 export fix.

## Known caveats

* Temperature is **frozen** at the initial conductive geotherm. No thermal
  feedback yet.
* The top boundary is stress-free but the mesh does not deform, so there is no
  evolving topography.
* Newton fails most steps; Picard converges to ~1e-3 relative, which is enough
  to be physical but not tight enough to quote a number from.
* Level sets drift to about [-0.008, 1.06] over 40 steps, so conservation
  degrades slowly. Worth re-checking at 400.
