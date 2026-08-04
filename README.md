# Geodynamics-pyGMT tutorials

An open, reproducible tutorial suite of Jupyter notebooks for **numerical geodynamic
modelling**, with every model output visualised using
[pyGMT](https://www.pygmt.org). A sibling to
[GPlately-pyGMT-tutorials](https://github.com/EarthByte/GPlately-pyGMT-tutorials),
following the same philosophy: a teaching ladder from undergraduate primer to
research-grade workflow, with executed outputs preserved so every figure renders
on GitHub without installing anything.

**Design constraint, enforced by CI:** no notebook takes longer than 30–60 minutes
on a modern laptop. Most take under a minute.

---

## Why this exists

As of August 2026, no repository anywhere combines a geodynamic solver with pyGMT
plotting. The geodynamics community plots with ParaView, matplotlib or MATLAB;
the pyGMT community plots observational data.

The need for a shared, modular geodynamics teaching-notebook library is
well recognised — the Computational Infrastructure for Geodynamics (CIG)
community has set out a thoughtful design for exactly this, including a
per-notebook template of concepts and equations, an analytical treatment, and a
numerical one. Building out that content is a large collective effort still in
progress. This suite is one contribution towards it, and borrows gratefully from
that design.

## The ladder

Notebooks are numbered so that the sort order *is* the curriculum, and the
dimensional escalation is encoded in the number. Runtimes are measured, not
estimated, and are stated in every notebook header.

### Part 1 — no installation required
These run on Google Colab, in JupyterLite, or anywhere with NumPy and SciPy.
Only the plotting needs pyGMT.

| # | Notebook | Runtime |
|---|---|---|
| 00 | Environment check, and pyGMT idioms for model output | seconds |
| 01 | 1-D advection and heat conduction: explicit, implicit, and why stability is not accuracy | seconds |
| 02 | 2-D diffusion: a cooling sill and a rising plume head | seconds |
| 03 | Advection schemes and the numerical diffusion they smuggle in | seconds |
| 04 | 2-D Stokes flow, and verifying it against a manufactured solution | ~1 min |
| 05 | Markers, a falling block, and why viscosity averaging matters | ~1 min |
| 06 | Thermal convection and the Blankenbach benchmark | ~5 min |

### Part 2 — needs the container
These use [G-ADOPT](https://gadopt.org) (Firedrake) and run in the project's
Docker image or on a BinderHub.

| # | Notebook | Runtime |
|---|---|---|
| 07 | G-ADOPT: finite elements, and the same plots | **2 min 41 s (measured)** |
| 08 | Parallelism, measured rather than assumed | **95 s (measured)** |

Planned, not yet built:

* **T09** — visco-plastic rheology and strain localisation
* **T10** — the 2-D cylindrical annulus (see the note below)
* **T11** — driving the annulus with pyGPlates surface velocities

### Part 3 — lithospheric deformation

The mantle-scale ladder above is only half the picture. Lithospheric extension
and shortening exercise a different part of the physics: visco-plastic strain
localisation, a free surface, and large finite strain.

**This turns out to be far cheaper than expected.** G-ADOPT already ships a
kinematically-driven, visco-plastic lithospheric model — it is simply not
labelled as one. The `Drucker_Prager` demo implements
[Spiegelman, May & Wilson (2016)](https://doi.org/10.1002/2015GC006228): a
120 x 30 km box, driven at 5 mm/yr, with pressure-dependent Drucker-Prager
yielding and a free upper surface:

```python
boundary.left:   {'ux':  1}      # driven inward
boundary.right:  {'ux': -1}      #   -> shortening
boundary.bottom: {'uy':  0}      # top free
```

That is a lithospheric **shortening** model, filed under mantle convection and
framed as a tutorial on nonlinear solver strategy. **Extension is the same model
with those two signs flipped.** The rheology is user-supplied UFL rather than
library-internal, so any lithospheric rheology - temperature-dependent creep,
strain softening, layered crust - is writable in the notebook itself.

| # | Notebook | Basis | Status |
|---|---|---|---|
| 12 | Pure and simple shear: the kinematic frame for everything that follows | `geodynkit` markers | planned |
| 13 | **Lithospheric shortening**: shear-band localisation, after Spiegelman et al. (2016) | wrap G-ADOPT `Drucker_Prager` | planned |
| 14 | **Lithospheric extension**: necking and boudinage - the same model, signs reversed | as T13 | planned |
| 15 | **Rifting modes**: narrow vs wide vs core complex, set by crustal strength | + layered rheology, free surface | planned |
| 16 | **Orogenic wedges**: critical taper and thrust sequences | + surface processes | planned |

**Prototyped and working.** `tools/gadopt_lithosphere_case.py` runs both modes
from one script — the sign of the boundary velocity is the only difference. At
96x48 on two slow cores, shortening converges in 18 Picard iterations (5.8 s)
and extension in 34 (10.3 s), both then polished by Newton in ~1 s. Conjugate
shear bands form from the seed notch in both. Extension localises *more*
strongly (peak/median strain rate 9.1 vs 6.9), which is what you expect: necking
is self-amplifying, thickening is not.

The Spiegelman case is **instantaneous** — one nonlinear solve, no advection, no
evolving topography, no temperature, no strain weakening. A research-grade rift
model needs all of those. Every ingredient exists in G-ADOPT already; what does
not exist is the combination:

| Requirement | G-ADOPT capability | Status |
|---|---|---|
| Layered crust + mantle lithosphere | `conditional()` on depth, or level sets | available |
| Temperature-dependent viscosity | `mu_lin = exp(-gamma_T*T + gamma_Z*d)`, harmonic with plastic | in `viscoplastic_case` |
| Material advection through large strain | `LevelSetSolver`, `material_field`, `material_entrainment` | `multi_material` demos |
| Evolving topography | `free_surface_equation`, incl. with multi-material | `free_surface` demos |
| Plastic strain weakening | `GenericTransportSolver` to advect accumulated strain | machinery exists, **we write the source term** |
| Refined shear zones | `animate` RiemannianMetric + `adapt`, re-adapting every N steps | `adaptive_base_case` |

So the work is assembly and geological judgement, not solver development. What we
write ourselves is the scaffolding — layered crust and mantle lithosphere, a weak
seed, defensible rheological parameters, a strain-weakening law — not the
machinery. Worth mining ASPECT's `continental_extension` cookbook and the
archived UWGeodynamics rifting tutorials for **model design**; not for code.

Following the GemPy tutorials' "units of three" grammar, each rung adds exactly
one structural complication rather than several at once.

> **Note on the planned T10.** The annulus at G-ADOPT's default settings
> (128×32, Ra = 1e5, steady-state tolerance 1e-7) exceeded 50 minutes serial on a
> 2-core cloud container and projects to roughly 100 minutes. On a modern laptop
> that is ~30 min serial, and comfortably less under MPI. The notebook relaxes the
> tolerance and shows the MPI route.

## Installation

### Part 1 only
```bash
conda env create -f environment.yml
conda activate geodyn-pygmt
jupyter lab
```
`pip install pygmt` is **not** sufficient — pyGMT is a wrapper around the GMT C
library, which pip will not install for you. Use conda/mamba, or the container.

### Everything, including G-ADOPT
```bash
docker build -f Dockerfile -t geodyn-pygmt .
docker run --rm -p 8888:8888 geodyn-pygmt jupyter lab --ip=0.0.0.0
```
The image is built on `firedrakeproject/firedrake-vanilla-default`, which is
Ubuntu 24.04 — whose apt GMT is 6.5.0, exactly pyGMT's minimum. That means
Firedrake, G-ADOPT and pyGMT coexist in one image with **no conda at all**.

### Publishing the image

Don't push an image built on a laptop. Tag a release and let CI do it:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

`.github/workflows/publish-image.yml` then builds `linux/amd64` and `linux/arm64`
on **native** runners (so Apple Silicon gets a native image rather than an
emulated one), publishes `ghcr.io/earthbyte/geodyn-pygmt`, and smoke-tests the
result. It needs `packages: write`, which the workflow requests, and for the
`EarthByte` org to allow Actions to publish packages.

The workflow ends with a check that the image does not set `SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE` or `PIP_CERT`. That is not paranoia: anyone building behind
a TLS-intercepting proxy — a university network, a corporate VPN, a cloud
sandbox — can silently bake their own CA into the image and make every student
who pulls it trust that certificate authority. A clean CI runner avoids it; the
check proves it.

Once published, `.binder/Dockerfile` resolves and the BinderHub launcher works.

## Verification

Every solver in `geodynkit/` is verified against something external, and the
verification is part of the teaching material rather than hidden in a test suite:

| Solver | Verified against | Result |
|---|---|---|
| 2-D Stokes | Manufactured solution from a stream function | **2nd-order convergence in velocity and pressure** |
| 2-D Stokes | Hydrostatic case (uniform density and viscosity) | spurious velocity at machine zero |
| MPI scaling | Repeated timings, median of 3 | 1.17x at 13,764 dofs, 1.33x at 54,148 dofs on 2 cores — efficiency 59% and 66% |
| Thermal convection | Blankenbach et al. (1989) case 1a | Richardson extrapolation from n=24, 48 gives v_rms **42.85** vs published **42.865** |
| G-ADOPT (independent code) | Blankenbach case 1a | Nu **4.9187**, v_rms **42.865** — agrees with our extrapolation |
| Diffusion | Analytic spreading of a Gaussian | 2nd order in space |
| Marker projection | Viscosity averaging comparison | arithmetic, geometric and harmonic agree exactly at unit contrast |

Run them all with `pytest`.

## Licence

Following the [CFDPython](https://github.com/barbagroup/CFDPython) precedent, the
two kinds of content in a teaching notebook are licensed separately:

* **Code** (`geodynkit/`, code cells) — BSD 3-Clause, matching pyGMT. See `LICENSE`.
* **Prose and figures** (markdown cells, documentation) — CC BY 4.0. See `LICENSE-TEXT`.

## Acknowledgements

The numerical ladder in Part 1 follows the structure of
[GeoModBox.jl](https://github.com/LukasFuchs/GeoModBox.jl) (Lukas Fuchs, Goethe
Universität Frankfurt, MIT), reimplemented in Python. Part 2 wraps the demo
notebooks of [G-ADOPT](https://github.com/g-adopt/g-adopt) (ANU / ARDC / AuScope,
MIT). Benchmark values are from Blankenbach et al. (1989). Colour maps are Fabio
Crameri's scientific colour maps, which ship with GMT.
