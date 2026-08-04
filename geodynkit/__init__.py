"""
geodynkit — a small, readable library of 2-D geodynamic solvers and pyGMT
plotting helpers for the Geodynamics-pyGMT teaching notebook suite.

Design rules, in priority order:

1. **Readable beats fast.** Every solver should be followable line by line
   against the governing equation. Where a vectorised form obscures the
   physics, the loop stays.
2. **Pure NumPy/SciPy.** No compiled extensions, no MPI, no conda-only
   dependencies. The notebooks that use this module must run on Google Colab
   and in JupyterLite with nothing installed.
3. **Plain arrays out.** Every solver returns numpy arrays on a regular grid,
   because that is exactly what pyGMT wants.

Conventions used throughout:
    * fields are shaped ``(nz, nx)``
    * ``x`` is horizontal, increasing right
    * ``z`` is DEPTH, positive DOWNWARDS, ``z = 0`` at the surface
    * velocities ``(vx, vz)`` share that frame, so ``vz > 0`` is downwards
"""

__version__ = "0.1.0.dev0"

from . import plotting  # noqa: F401
from . import diffusion  # noqa: F401
from . import advection  # noqa: F401
from . import stokes  # noqa: F401
from . import markers  # noqa: F401
from . import convection  # noqa: F401

__all__ = [
    "plotting",
    "diffusion",
    "advection",
    "stokes",
    "markers",
    "convection",
]
