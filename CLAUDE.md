# Working notes for this repository

A teaching suite of Jupyter notebooks for numerical geodynamics, plotted with
pyGMT. Read `README.md` first for the curriculum; this file is the operational
detail — how to build things, and the conventions and traps that have already
cost time here.

## Layout

```
tools/sources/T*.py      jupytext percent-format SOURCES — the source of truth
Notebooks/T*.ipynb       GENERATED, executed, committed with outputs
geodynkit/               pure NumPy/SciPy library (Part 1) + lithosphere scaffolding
tools/gadopt_*_case.py   G-ADOPT drivers, run as subprocesses from the notebooks
tools/*_probe.py         diagnostic scripts; not part of the curriculum
tests/                   19 verification tests, pure Python, no container needed
tools/LONG-RUN.md        how to run the rift case, and every measured number
```

## Commands

```bash
# build + execute notebooks (from the repo root, inside the container)
python3 tools/build_notebooks.py            # all
python3 tools/build_notebooks.py T09 T10    # note the T prefix

python3 tools/check_manifest.py             # README table vs Notebooks/ on disk
python3 -m pytest tests/ -q                 # 19 tests, ~80 s, no container

docker run --rm -v $PWD:/work -w /work ghcr.io/earthbyte/geodyn-pygmt:0.1.2 <cmd>
```

`build_notebooks.py` executes with `Notebooks/` as the working directory, which
is what nbmake does in CI and what a student gets in JupyterLab. Every notebook
therefore does `sys.path.insert(0, "..")` and uses `../tools/...` paths.

## Conventions

- **Sources are `.py`, notebooks are generated.** Edit `tools/sources/`, never the
  `.ipynb`. Outputs are committed so figures render on GitHub without installing
  anything.
- **Runtimes in notebook headers are measured, not estimated**, and CI enforces
  the budget. If you change a notebook's cost, re-measure and update the header
  *and* the README table.
- **`check_manifest.py` must pass.** A README table row is a claim that a
  notebook exists; anything marked "planned" is a roadmap entry and is allowed.
- **No ASPECT or GPL code.** Parameter *values* from the literature are facts and
  are reproduced with citation (see `geodynkit/lithosphere.py`); code is not.
  This suite is BSD-3 with CC BY 4.0 text.

## Rules learned the hard way

**Never run git in this working copy through the Cowork device bridge.** The
mount cannot unlink, so a stale `.git/index.lock` survives and blocks the user's
next command. It happened twice. Git stays in a terminal on the Mac.

**A diagnostic that only prints is not a check.** The first production run of the
rift case completed 400 steps in 2 h 22 min and was uninterpretable; its own
output said so (`level_set_range` at `[-1.72, 3.02]` when a conservative level
set is in [0, 1] by definition) and nothing acted on it. Invariants belong inside
the loop, and they must raise.

**Beware any metric that is maximised by the failure it is meant to detect.**
This model defeated three:

- *localisation ratio* (in-seed / outside strain) — maximised when nothing
  deforms outside the seed, including when nothing deforms at all;
- *total accumulated strain*, used to choose the geotherm — picked a lithosphere
  so cold the mantle could not deform, which forces all strain into the crust;
- *layering checked in fixed depth windows* — a model that extends moves its
  layers out of the windows, so a successful run must fail the check.

When designing a check, ask what it does on a *successful* run.

**Export finite-element fields through CG1, not CG2**, and tie the sampling grid
to the mesh. A quadratic interpolant overshoots at a ridge: CG2 gave a negative
strain-rate invariant (a square root) and a viscosity below its own floor. A
fixed export grid steps over a narrowing shear band and makes a non-convergent
peak look convergent.

**PETSc parses `sys.argv` on import** and then warns about every flag it does not
recognise. The drivers hide their arguments for the duration of the import.

**`Function.at` is deprecated** — use `PointEvaluator`, built once and reused.

## G-ADOPT specifics

**`StokesSolver.solve()` assigns `solution_old` on every call.** That is right
when one solve is one timestep, and wrong for the visco-plastic Picard loop,
which calls it forty times per step: any time-integrated boundary condition
(the free surface) then advances once per *iteration*. Reset the stored time
level before each Picard solve and advance it once per step.

**A free surface needs buoyancy**, and buoyancy changes which pressure the yield
criterion should see. With no body force the returned pressure is purely dynamic
and the lithostatic part is supplied analytically; with buoyancy on, `p` already
contains everything but the *reference* lithostatic pressure. Keeping both counts
it twice — a quiet error, about 3% in strain growth.

**Metric-based mesh adaptivity is not available in this container.** `adapt`
drives Mmg through PETSc; the Firedrake image's PETSc is configured without
`--download-mmg` or `--download-parmmg`, `animate` is not installed, and Mmg
adapts simplex meshes only while this suite uses quadrilaterals. A statically
graded mesh was tried as a substitute and made things worse (abort at step 18
rather than 40, at twice the cost per step). Do not spend time on this again
without a reason to expect a different answer; see T11 section 5.

**Level-set reinitialisation must scale with the interface thickness.** G-ADOPT's
default pseudo-timestep is a fixed 0.02 while `interface_thickness` returns
~0.35 h_min, so the default is stable coarse and divergent fine. 12 sweeps per
step, not 2.

## Part 4 specifics

**The annulus needs its rotational nullspace declared.** A closed annulus with
free-slip boundaries can spin as a rigid body at no cost, so the Stokes operator
is singular in that direction. `create_stokes_nullspace(Z, closed=True,
rotational=True)`. Prescribe the surface velocity (T15) and the freedom
disappears — declare it then and you project out part of a physical solution.

**Nusselt numbers do not match in an annulus, and should not.** The boundaries
have different areas, so at steady state `Nu_base / Nu_top = rmax / rmin =
1.820`. Measured 1.810. This is a verification that needs no reference solution,
and it fails loudly on an unconverged run.

**`CircleManifoldMesh(n)` is an n-gon**, so the circle of radius `rmin` lies
slightly *outside* the mesh and sample points placed on it are silently dropped.
Inset by twice the sagitta, `r(1 - cos(pi/n))`.

**pyGPlates needs `libgl1`** even though nothing draws; without it the import
fails on `libGL.so.1`. Wheels exist for cp312 on manylinux x86_64 *and* aarch64,
which is what keeps the image multi-arch. The Müller 2022 reconstruction (57 MB)
is baked into the image at `/opt/plate-model` via `plate-model-manager`, so
notebooks need no network — `ensure_reconstruction` only checks paths, it does
not download.

**G-ADOPT's `GplatesVelocityFunction` is 3-D only.** It seeds a Fibonacci sphere
and works in lat/lon; an annulus has one angular coordinate. `geodynkit.plates`
samples a great circle instead and keeps only the in-plane velocity component.

**Plate driving makes the energy equation advection-dominated.** 1 cm/yr is 916
non-dimensional, so plates reach ~6800 against free-slip convection's 193 — cell
Péclet goes from 12 to 125 and plain Galerkin undershoots temperature to −0.44.
Pass `su_advection=True`.

## Editing the big drivers

`gadopt_rift_case.py` has been broken **four times** by blind string-replacement
edits inserting a keyword argument twice. Any scripted edit to it should assert
an expected occurrence count before substituting and `ast.parse` the result
before writing. And **smoke-test two steps before launching a detached sweep** —
one batch of three runs died instantly on a `SyntaxError` and the failure was
silent for twenty minutes.

## Container and publishing

`ghcr.io/earthbyte/geodyn-pygmt` — multi-arch (amd64 + arm64), public. See
`PUBLISHING.md`. One gotcha worth repeating: publishing the first *public*
package from the org is blocked by an org-wide policy, not by your role, and the
switch is at Organisation settings → Packages → Package creation → enable
Public. Web UI only — no REST API, no `gh` command.

CI asserts the image carries no proxy CA in its trust store (`SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE`, `PIP_CERT` must be absent) and that `jupyterhub-singleuser`
exists, because a BinderHub launch fails without it.
