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
the pyGMT community plots observational data. CIG designed a geodynamics teaching
notebook library and shipped an empty repository. This suite fills that gap.

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

Planned, not yet built:

* **T08** — parallelism as a topic: the same case at 1, 2 and 4 processes, with a scaling plot
* **T09** — visco-plastic rheology and strain localisation
* **T10** — the 2-D cylindrical annulus (see the note below)
* **T11** — driving the annulus with pyGPlates surface velocities

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
