# Changelog

Notable changes to the suite. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); version numbers track
the container image tags on `ghcr.io/earthbyte/geodyn-pygmt`.

## Unreleased

### Added

- **T15 — Driving the mantle with plate reconstructions.** The annulus surface
  becomes kinematic, with velocities read from the Müller et al. (2022) 1 Ga
  model via pyGPlates along a great circle. 8 min at 64x16. Includes the
  scaling reality check (1 cm/yr = 916 non-dimensional, so plates out-drive
  Ra = 1e5 convection 35 to 1) and the cell-Péclet failure it exposes.
- **T14 — The annulus: no side walls, unequal boundaries, and a nullspace.**
  Ends with a verification needing no reference solution: at steady state the
  Nusselt ratio must equal r_max/r_min = 1.820, and it measures 1.810. 8 min.
- `geodynkit/plates.py` — great-circle sampling of a plate reconstruction and
  the non-dimensionalisation, with the in-plane-only decision documented.
- `plotting.annulus_panel` — GMT's polar projection, the suite's last new
  plotting idiom.
- `tools/gadopt_annulus_case.py`, with `--plate-dir`, `--plate-age`, `--su`.
- **Container**: pyGPlates, `libgl1` (the wheel links against libGL), and the
  Müller 2022 reconstruction baked in at `/opt/plate-model` so the notebooks
  need no network.
- **T13 — The other sign: mountains, and why extension is the harder
  direction.** `--mode shortening` completes 40 steps to 16% strain and builds a
  doubly-vergent pop-up with +27 km of surface uplift; the matched extension run
  trips the level-set invariant at step 23 and 9.2%. The asymmetry is geometric:
  extension thins layers towards the representational limit, shortening thickens
  them away from it. 9 min at 48x24.
- **T12 — What you seed is what you get: mode selection.** Set out to reproduce
  Buck's (1991) narrow/wide diagram and does not; the four failed attempts are
  the notebook. 11 min at 48x24.
- `--seed-mode random` / `--seed-amp`, `--crust-km` / `--t-base`,
  `--rate-cm-yr`, `--dt-max` in the rift driver, plus the W50 concentration
  metric in its output.
- `LI.heat_flow_for_base_temperature` and `LI.scaled_column` — pin the mantle
  adiabat and derive the surface flux, instead of the reverse.
- **T11 — A free surface: letting the rift subside, and two traps on the way.**
  The rift now develops topography: a 4.5 km graben with flexural shoulders on
  both flanks. 25 min at 64×32.
- **T10 — A rift, and the point at which it stops being one.** Closes the
  feedback loop T09 lacks (advected materials, accumulated plastic strain,
  strain weakening) and the model localises: in-seed strain 14× the
  surroundings, peak strain growing past the seeded maximum, 0.8% of the domain
  fully weakened. 13 min at 64×32.
- **T09 — Visco-plasticity and shear bands.** Spiegelman, May & Wilson (2016) in
  shortening *and* extension from one script. Its subject is which diagnostics
  survive mesh refinement — the band dip converges to the Roscoe angle for
  non-dilatant plasticity (45°, against the Coulomb 30°) while the peak strain
  rate does not converge at all.
- `--free-surface` in `tools/gadopt_rift_case.py`, with the buoyancy this
  requires, correctly non-dimensionalised.
- Per-step invariant checks in the rift driver: the run aborts and saves state
  when a conservative level set leaves [0, 1].
- `--fs-bug {time-level,lithostatic}` to reproduce two free-surface errors on
  purpose, so T11 can demonstrate rather than assert them.
- `tools/rift_divergence_probe.py` and `tools/rift_picard_probe.py` — measure
  ‖∇·u‖/‖∇u‖ and compare Picard relaxation strategies.
- `tools/plot_rift.py`, `tools/LONG-RUN.md`, `CLAUDE.md`, this file.

### Changed

- Streamline-upwind stabilisation available in the annulus energy equation.
  Plate driving takes u_rms from ~193 to ~2000 and the cell Péclet number from
  12 to 125; plain Galerkin then undershoots temperature to −0.44 over 2.5% of
  the domain, and SU removes it entirely at no measurable cost.
- **Thermal coupling done.** `EnergySolver` wired in behind `--thermal`:
  Peclet 10.5, layered radiogenic heating, temperature pinned at both ends.
  Verified stationary — with u = 0 the analytic geotherm drifts 0.31 K over
  20 steps against a 1340 K range.
- **Part 3 renumbered twice.** Lithospheric deformation took the lower numbers
  (T09–T13) because it is what the project needs; then T11 became the free-surface
  notebook and the annulus moved to T14–T15. Cross-references updated with each.
- Level-set reinitialisation default raised from 2 sweeps per step to 12.
  Measured at 64×32 over 20 steps: ψ excursion beyond [0, 1] falls 0.0277 →
  0.0148 → 0.0048 for 2 / 6 / 12 sweeps, at 5% more wall-clock.
- Picard iteration cap exposed as `--picard-iters`, default raised 30 → 40.
- Under-relaxation made gentler: back off only above a 1.5× rise in the residual,
  by 0.7, with a floor of 0.25. The old rule (halve on any rise, floor 0.05) hit
  its floor by iteration 6 and never recovered.
- Localisation ratio removed from the rift driver's output. It misled the
  parameter search repeatedly; absolute in-seed and outside strains are reported
  instead.
- `check_layering` demoted to an initialisation check. Fixed depth windows cannot
  be a runtime invariant in a model that extends.
- Field export moved from CG2 to CG1 throughout, with the sampling grid tied to
  the mesh; `Function.at` → `PointEvaluator`.

### Fixed

- **The Picard "stagnation" was three self-inflicted problems**, all in
  `solve_stokes`. The best-iterate safeguard minimised the change between
  successive iterates — smallest at iteration 0 for reasons unrelated to
  accuracy — and so selected the least incompressible field available
  (‖∇·u‖/‖∇u‖ of 0.52, against 0.07–0.12 for simply keeping the last iterate).
  A failed Newton solve left PETSc's diverged iterate in place despite a comment
  claiming otherwise. And the under-relaxation strangled itself. With these
  fixed, at 64×32 over 40 steps: Picard 30 iterations at 1e-2 → 15–21 at 9e-5;
  Newton from failing every step to succeeding on most; ψ excursion 0.36 → 0.010.
  **The model localises for the first time.**
- Free-surface displacement was advancing once per Picard iteration rather than
  once per timestep, giving 64 km of subsidence in five steps instead of 2.
- Lithostatic pressure double-counted once buoyancy is enabled.
- `h_min` was hardcoded as `aspect / nx`, which is the *average* cell width on a
  graded mesh, so the CFL limiter used a timestep several times too large for the
  smallest cells while reporting a comfortable Courant number.
- Level-set reinitialisation pseudo-timestep now scales with the interface
  thickness rather than being fixed, which is what made it stable on a coarse
  mesh and divergent on a fine one.
- PETSc's "options you set that were not used" warning suppressed; it was burying
  the real output of long runs.

### Not done, deliberately

- **Metric-based adaptive mesh refinement.** `adapt` drives Mmg through PETSc;
  the Firedrake image's PETSc is built without `--download-mmg` or
  `--download-parmmg`, `animate` is not installed, and Mmg adapts simplex meshes
  only while this suite uses quadrilaterals. Enabling it means rebuilding PETSc
  and Firedrake from source on two architectures. A statically graded mesh was
  tried instead and made things worse: abort at step 18 rather than 40, at twice
  the cost per step. T11 section 5 has the measurements.
- **Narrow versus wide rifting**, in the sense of Buck (1991). Four attempts,
  all measured and all in T12: a centred seed imposes a nucleation site and gives
  a narrow rift at every crustal thickness from 20 to 50 km; distributed noise
  below the weakening onset of 0.5 leaves the feedback loop switched off;
  raising the extension rate cannot buy finite strain, because U0 is the velocity
  scale and cancels out of `steps x dt`; and pushing the timestep to reach 30%
  stretching hits the level-set representational ceiling at 15-25%. Reaching the
  transition needs finer cells, several hundred steps and probably a wider
  domain — a research run, not a notebook cell.
- **Critical taper**, in the sense of Davis, Suppe & Dahlen. Symmetric
  convergence with a central seed gives a bivergent pop-up, which is a different
  structure. A taper needs a basal decollement, a backstop and a one-sided feed —
  a change of geometry rather than of physics, and all three are expressible in
  what the driver already does. T13 section 4 sets it out.

## 0.1.1 — 6 August 2026

### Fixed

- BinderHub launch failed with `exec: "jupyterhub-singleuser": executable file
  not found` — jupyterlab and notebook were installed but not jupyterhub.
- `NB_USER` was an `ARG` rather than an `ENV`, producing an `UndefinedVar`
  warning at build time.
- CI gained a binder-readiness gate so neither can regress.

## 0.1.0 — 4 August 2026

### Added

- First public release. Repository at `EarthByte/Geodynamics-pyGMT-tutorials`,
  container at `ghcr.io/earthbyte/geodyn-pygmt` — multi-arch (linux/amd64 and
  linux/arm64), with provenance and SBOM attestations.
- **Notebooks T00–T08.** Part 1 (T00–T06) needs no installation beyond NumPy,
  SciPy and pyGMT; Part 2 (T07–T08) adds G-ADOPT and Firedrake in the container.
- `geodynkit` — 1-D and 2-D diffusion, advection schemes, a staggered-grid
  variable-viscosity Stokes solver, markers, thermal convection, and the pyGMT
  plotting layer the whole suite shares.
- Verification chain: manufactured solution for Stokes (second order in velocity
  *and* pressure), machine-zero velocity for a hydrostatic case, and Blankenbach
  1a via Richardson extrapolation (v_rms 42.85 against 42.865 published), with
  G-ADOPT independently returning 42.865 on the same problem.
- CI: verification tests, notebook execution inside the stated runtime budgets,
  and a check that the README's notebook table matches what is on disk.
- CI asserts the published image carries no proxy CA in its trust store.
