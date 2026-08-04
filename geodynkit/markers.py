"""
Marker-in-cell (particle-in-cell) transport.

Advecting material properties on Lagrangian markers instead of on the Eulerian
grid is what lets a geodynamic code carry sharp compositional interfaces
through large finite strain without smearing them into mush. It is the single
idea that separates a toy convection code from something you could do
lithosphere dynamics with.

The cycle each timestep:

    1. interpolate marker properties -> grid   (for the Stokes solve)
    2. solve Stokes on the grid
    3. interpolate grid velocity -> markers
    4. advect markers (Runge-Kutta)

Steps 1 and 3 are where accuracy is won or lost, so they are written out
plainly rather than hidden behind a library call.
"""

import numpy as np

__all__ = ["MarkerSet", "markers_to_grid", "grid_to_markers"]


class MarkerSet:
    """A swarm of Lagrangian markers carrying material properties.

    Parameters
    ----------
    x, z : arrays
        Marker positions.
    phase : integer array
        Material index per marker. Properties are looked up per phase, which
        keeps the common case (a handful of rock types) simple.
    """

    def __init__(self, x, z, phase):
        self.x = np.asarray(x, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.phase = np.asarray(phase, dtype=int)
        self.strain = np.zeros_like(self.x)  # accumulated finite strain

    @classmethod
    def regular(cls, lx, lz, nx_cells, nz_cells, per_cell=4, jitter=0.5, seed=0):
        """Fill a box with ``per_cell**? `` markers per cell, randomly jittered.

        ``per_cell`` is markers per cell **per direction**, so ``per_cell=4``
        gives 16 markers per cell — a common choice. Jitter breaks the regular
        lattice, which otherwise produces visible artefacts in the projection
        back to the grid.
        """
        rng = np.random.default_rng(seed)
        dx, dz = lx / nx_cells, lz / nz_cells
        sub = (np.arange(per_cell) + 0.5) / per_cell
        gx = (np.arange(nx_cells)[:, None] + sub[None, :]).ravel() * dx
        gz = (np.arange(nz_cells)[:, None] + sub[None, :]).ravel() * dz
        X, Z = np.meshgrid(gx, gz)
        X = X.ravel() + jitter * (rng.random(X.size) - 0.5) * dx / per_cell
        Z = Z.ravel() + jitter * (rng.random(Z.size) - 0.5) * dz / per_cell
        X = np.clip(X, 0.0, lx)
        Z = np.clip(Z, 0.0, lz)
        return cls(X, Z, np.zeros(X.size, dtype=int))

    def __len__(self):
        return self.x.size

    def set_phase_where(self, mask, phase):
        """Assign a phase to every marker satisfying ``mask``."""
        self.phase[mask] = phase
        return self

    def property_map(self, values):
        """Map per-phase ``values`` onto per-marker values."""
        return np.asarray(values, dtype=float)[self.phase]

    def advect(self, vx_c, vz_c, x, z, dt, lx, lz, order=2):
        """Advect markers through the cell-centred velocity field.

        ``order=1`` is forward Euler; ``order=2`` is midpoint Runge-Kutta,
        which is the usual choice and visibly better for rotating flows.
        Markers that would leave the box are clamped to the wall.
        """
        if order == 1:
            ux = grid_to_markers(vx_c, x, z, self.x, self.z)
            uz = grid_to_markers(vz_c, x, z, self.x, self.z)
            nx_, nz_ = self.x + ux * dt, self.z + uz * dt
        elif order == 2:
            ux = grid_to_markers(vx_c, x, z, self.x, self.z)
            uz = grid_to_markers(vz_c, x, z, self.x, self.z)
            xm = np.clip(self.x + 0.5 * ux * dt, 0.0, lx)
            zm = np.clip(self.z + 0.5 * uz * dt, 0.0, lz)
            uxm = grid_to_markers(vx_c, x, z, xm, zm)
            uzm = grid_to_markers(vz_c, x, z, xm, zm)
            nx_, nz_ = self.x + uxm * dt, self.z + uzm * dt
        else:
            raise ValueError("order must be 1 or 2")

        self.x = np.clip(nx_, 0.0, lx)
        self.z = np.clip(nz_, 0.0, lz)
        return self


def markers_to_grid(mx, mz, values, x, z, fill=None, average="arithmetic"):
    """Project marker values onto cell centres by bilinear (area) weighting.

    This is the direction that matters for the Stokes solve: viscosity and
    density must be known on the grid. Cells that catch no markers are filled
    with ``fill`` (or the swarm mean); if that happens often the swarm is too
    sparse, which is worth telling students to watch for.

    Parameters
    ----------
    average : {"arithmetic", "geometric", "harmonic"}
        How to combine marker values within a cell. **This choice matters a
        great deal for viscosity.** With a six-decade contrast between a stiff
        block and its weak surroundings, arithmetic averaging lets the stiff
        phase dominate any cell it touches, which stiffens the interface and
        produces spurious velocity spikes there. Geometric (equivalently, the
        arithmetic mean of log eta) or harmonic averaging behaves far better,
        and harmonic is the standard choice for viscosity in this situation.
        Use arithmetic for density, where it is the physically correct mean.
    """
    values = np.asarray(values, dtype=float)
    if average not in ("arithmetic", "geometric", "harmonic"):
        raise ValueError("average must be arithmetic, geometric or harmonic")

    # Transform, average linearly in the transformed variable, transform back.
    if average == "geometric":
        work = np.log(np.clip(values, 1e-300, None))
    elif average == "harmonic":
        work = 1.0 / np.clip(values, 1e-300, None)
    else:
        work = values

    dx, dz = x[1] - x[0], z[1] - z[0]
    nx, nz = x.size, z.size

    fi = np.clip((mx - x[0]) / dx, 0.0, nx - 1.000001)
    fj = np.clip((mz - z[0]) / dz, 0.0, nz - 1.000001)
    i0, j0 = fi.astype(int), fj.astype(int)
    tx, tz = fi - i0, fj - j0

    num = np.zeros((nz, nx))
    den = np.zeros((nz, nx))
    for di, wx in ((0, 1.0 - tx), (1, tx)):
        for dj, wz in ((0, 1.0 - tz), (1, tz)):
            w = wx * wz
            np.add.at(num, (j0 + dj, i0 + di), w * work)
            np.add.at(den, (j0 + dj, i0 + di), w)

    empty = den <= 0.0
    out = np.where(empty, 0.0, num / np.where(empty, 1.0, den))

    if average == "geometric":
        out = np.exp(out)
    elif average == "harmonic":
        out = 1.0 / np.clip(out, 1e-300, None)

    if empty.any():
        out[empty] = float(np.mean(values)) if fill is None else fill
    return out


def grid_to_markers(field, x, z, mx, mz):
    """Bilinear interpolation of a cell-centred field onto marker positions."""
    dx, dz = x[1] - x[0], z[1] - z[0]
    nx, nz = x.size, z.size

    fi = np.clip((mx - x[0]) / dx, 0.0, nx - 1.000001)
    fj = np.clip((mz - z[0]) / dz, 0.0, nz - 1.000001)
    i0, j0 = fi.astype(int), fj.astype(int)
    i1, j1 = i0 + 1, j0 + 1
    tx, tz = fi - i0, fj - j0

    return (
        field[j0, i0] * (1 - tx) * (1 - tz)
        + field[j0, i1] * tx * (1 - tz)
        + field[j1, i0] * (1 - tx) * tz
        + field[j1, i1] * tx * tz
    )
