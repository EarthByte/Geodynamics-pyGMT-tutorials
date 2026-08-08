# Running the rift case

**Revised 7 August 2026**, after the 400-step run of 6 August produced an
uninterpretable result. This supersedes the previous version of this file, whose
central recommendation was wrong.

## What happened last time

The run completed — 400 steps, 96×48, 2 h 22 min — and every plot from it was
meaningless. The cause was in its own output:

```
"level_set_range": [[-1.572, 2.722], [-1.720, 3.023]]
```

A conservative level set is a smoothed indicator function; its value *is* the
volume fraction of material above the interface, so it lives in **[0, 1]** by
definition. `material_field` blends layer properties by interpolating in psi, so
once psi is 3.0 it extrapolates properties no rock has. The exported density had
3057 kg/m³ at 5 km depth and 2703 kg/m³ at 90 km — the column was inverted in
places. Everything downstream of that (viscosity, strain, strain rate) was
computed from it.

The range was being measured. It was printed once, at the end, and the run was
allowed to continue for four hundred steps past the point where it stopped
meaning anything.

## What changed

**1. The level-set bound is now enforced every step.** `--ls-tol` (default 0.05)
sets how far psi may stray outside [0, 1]; beyond that the run raises
`AdvectionBroke`, saves the state at failure, and exits non-zero. A broken run
now costs a minute.

**2. Reinitialisation raised from 2 steps to 12.** This is the actual fix, and it
was measured, not guessed. At 64×32, 20 steps, varying nothing else:

| reinitialisation steps | psi excursion beyond [0,1] | wall clock |
|---|---|---|
| 2 (old default) | 0.0277 | 481 s |
| 6 | 0.0148 | 493 s |
| 12 (new default) | 0.0048 | 506 s |

Six-fold less drift for 5% more wall-clock — the Stokes solve dominates the cost
and reinitialisation is nearly free beside it. At the old setting the excursion
grew roughly exponentially: 1.02 by step 11, 1.17 by step 30, 1.36 by step 40.

**3. Use q₀ = 0.055 W/m², not 0.040.** The previous version of this file
recommended 0.040 and that was an error. It puts 740 °C at 100 km depth, when the
base of a 100 km lithosphere should be about 1300 °C:

| q₀ | T(20 km) | T(40 km) | T(100 km) |
|---|---|---|---|
| 0.040 | 240 °C | 380 °C | **740 °C** |
| 0.048 | 304 °C | 508 °C | 1060 °C |
| **0.055** | 360 °C | 620 °C | **1340 °C** |

At 0.040 the mantle lithosphere sits pinned at the 10²⁶ Pa s cap and cannot
deform, so all the extension is forced into the crust, which then fails
everywhere: 74% of the domain fully weakened, mean strain 2.28 against a
weakening saturation of 1.5. The heat-flow sweep chose 0.040 because it scored
highest on total accumulated strain — which a rigid mantle maximises. 0.055 is
the `geodynkit.lithosphere` default and is right.

**4. No more localisation ratio.** It has been removed from the RESULT line. It
was reported for several rounds and misled every time, because it is maximised by
a run in which nothing deforms outside the seed. Absolute in-seed and outside
strains are reported instead — but read them only after the invariants below are
clean.

## Read the RESULT line in this order

```
"failure": null                 <- anything else and the rest is void
"level_set_excursion": 0.0047   <- must be small; > ls_tol aborts the run
"volume_drift": 0.0897          <- reported, not enforced; see below
```

Only then look at `strain_in_seed`, `strain_outside`, `strain_max_final` and
`weak_fraction_final`.

**`volume_drift` is a diagnostic, not a pass/fail.** In a closed box ∫psi dx is
conserved exactly and drift would be an error. This domain is open: the walls are
driven at ux = ∓1, so material genuinely leaves through the sides and enters
through the stress-free top. Measured drift is a steady ~0.8% per step from step
0, and its constancy is what says it is flux rather than error. What to watch for
is a *change in slope*.

## The command

```bash
cd /Users/dietmar/Documents/GPlates/Geodynamics-pyGMT_tutorials

docker run --rm -v $PWD:/work -w /work \
  ghcr.io/earthbyte/geodyn-pygmt:0.1.1 \
  python3 tools/gadopt_rift_case.py \
    --nx 96 --ny 48 --steps 400 \
    --heatflow 0.055 --seed-km 10 --damper 1e21 \
    --reini-steps 12 --ls-tol 0.05 \
    --history rift_long_history.json \
    --out rift_long.npz 2>&1 | tee rift_long.log
```

**Before committing to 400 steps, run 40 first.** It takes about 12 minutes at
64×32 and tells you whether the invariants hold. If step 40 comes back with an
excursion under about 0.01 and no `**` lines, the long run is worth starting.

```bash
docker run --rm -v $PWD:/work -w /work \
  ghcr.io/earthbyte/geodyn-pygmt:0.1.1 \
  python3 tools/gadopt_rift_case.py \
    --nx 64 --ny 32 --steps 40 --heatflow 0.055 --seed-km 10 \
    --reini-steps 12 --out /tmp/probe.npz
```

Plot with:

```bash
docker run --rm -v $PWD:/work -w /work \
  ghcr.io/earthbyte/geodyn-pygmt:0.1.1 \
  python3 tools/plot_rift.py rift_long.npz --prefix rift_long
```

## Geometry — it is a full rift, not one flank

200 km wide, 100 km deep, seed at x = 100 km, `left: ux = -1` and
`right: ux = +1`. So a correct result is roughly **mirror-symmetric about
x = 100 km**, with necking in the middle. A corner-to-corner gradient is a broken
run, not an asymmetric rift. (Real rifts do go asymmetric, and a random seed can
select a flank — but that shows up as a *sharper* structure on one side, not as
loss of structure everywhere.)

## Known limitations, unchanged

* **Temperature is frozen.** No `EnergySolver` yet, so the geotherm does not
  respond to advection or thinning. This is deliberate — it isolates the
  mechanics while the level-set and strain-weakening machinery is being verified —
  but it means the model cannot produce the thermal weakening that a real rift
  depends on.
* **No free surface.** The top is stress-free but flat; no topography develops.
* **Newton fails at essentially every step**, and Picard stalls at a relative
  residual of about 3×10⁻⁴ rather than reaching its 10⁻⁴ tolerance. Raising the
  iteration cap does not help — the iteration genuinely stagnates, verified at
  caps of 10, 30, 60, 120 and 240, all returning the identical iterate. The
  velocity field is therefore an approximation, and its relative divergence
  ‖∇·u‖/‖∇u‖ is 0.52 at 64×32 against 0.09 for the well-converged instantaneous
  case at the same resolution. `tools/rift_divergence_probe.py` measures this.
  Fixing the stagnation is the next substantive piece of work, and it is probably
  what stands between this model and a rift that localises.
