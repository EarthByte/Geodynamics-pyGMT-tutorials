# Notebook clusters

The notebooks are flat and numbered, so the sort order *is* the curriculum.
Grouping is declared here rather than with directories — cheap to reorganise, and
it never breaks a relative data path or a Binder link.

These `.ipynb` files are **generated** from the sources in `tools/sources/` and
committed with their outputs, so every figure renders on GitHub without anyone
installing anything. Edit the `.py` source, then:

```bash
python tools/build_notebooks.py T04     # rebuild and re-execute one
python tools/build_notebooks.py         # rebuild everything
```

---

## Cluster A — Foundations

| Notebook | What it establishes | Runtime |
|---|---|---|
| `T00_pyGMT_for_model_output` | The plotting vocabulary the whole suite uses: NumPy → xarray → GMT, depth-downwards sections, arrows, streamlines, colour maps, animations, and where pyGMT is the wrong tool | ~10 s |

## Cluster B — The numerical ladder

Pure NumPy and SciPy. No installation beyond `environment.yml`; runs on Colab.
Each rung adds one process, and each ends with a verification against something
external.

| Notebook | Adds | Verified against | Runtime |
|---|---|---|---|
| `T01_heat_conduction_and_stability` | 1-D diffusion, explicit and implicit | analytic Gaussian spreading | ~3 s |
| `T02_two_dimensional_diffusion` | 2-D diffusion, mixed boundary conditions | conductive timescale a²/κ | ~3 s |
| `T03_advection_schemes` | four advection schemes | exact translation after one revolution | ~3 s |
| `T04_stokes_flow_and_verification` | variable-viscosity Stokes | manufactured solution, 2nd order | ~4 s |
| `T05_markers_and_the_falling_block` | marker-in-cell transport | rigid-body motion of a stiff block | ~5 s |
| `T06_thermal_convection_blankenbach` | Stokes coupled to energy | **Blankenbach et al. (1989) case 1a** | ~4.5 min |

## Cluster C — Research-grade tools

Requires the container (Firedrake + G-ADOPT + pyGMT, no conda).

| Notebook | Adds | Runtime |
|---|---|---|
| `T07_gadopt_base_case` | finite elements, and the FE → regular grid → pyGMT bridge | **2 min 41 s** |

Planned: T08 parallel scaling · T09 visco-plastic rheology · T10 cylindrical
annulus · T11 pyGPlates-driven surface velocities.

---

## Per-notebook conventions

Every notebook carries, in this order:

1. a **three-part header** — cluster and motivation, learning objectives, then
   prerequisites *and a stated runtime*;
2. a `# === USER CONFIGURATION ===` block exposing every tunable as a named
   constant, immediately after the imports;
3. a **verification** step — an analytic solution, a published benchmark, or an
   invariant that must hold exactly;
4. a closing **"Extend this"** section with follow-on questions;
5. a one-line pointer to the next notebook.

Runtimes are measured on a 2-core cloud container, which is a pessimistic proxy
for a laptop. CI enforces them: `--nbmake-timeout=300` for Cluster A and B.
