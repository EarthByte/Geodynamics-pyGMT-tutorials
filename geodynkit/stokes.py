"""
2-D incompressible Stokes flow with variable viscosity, on a staggered grid.

This is the heart of the library. Everything above it (falling block,
Rayleigh-Taylor, thermal convection) is this solver plus a transport equation.

Governing equations, in the infinite-Prandtl limit that applies to the mantle:

    d/dx_j ( 2 eta eps_ij ) - dP/dx_i + rho g_i = 0        (momentum)
    dv_i/dx_i = 0                                          (continuity)

Staggered ("MAC") layout for a domain of ``nx`` by ``nz`` cells. Staggering is
not decoration: colocating all unknowns at the same points admits a
checkerboard pressure mode, and this arrangement is the standard cure.

      vz(i,j)          vz at horizontal faces, shape (nz+1, nx)
    +----|----+
    |         |
 -- vx   P    vx --    vx at vertical faces, shape (nz, nx+1)
    |         |        P  at cell centres,    shape (nz, nx)
    +----|----+

Viscosity lives in two places, because the two terms of the deviatoric stress
tensor are evaluated at different points:

    eta_c : (nz, nx)     normal viscosity, at cell centres
    eta_n : (nz+1, nx+1) shear viscosity, at cell corners

Coordinate convention, as everywhere in geodynkit: ``z`` is depth, positive
downwards, so gravity is ``+g`` in z.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "StokesGrid",
    "solve_stokes",
    "centre_to_corner",
    "velocity_to_centres",
    "strain_rate_second_invariant",
    "manufactured_solution",
]


class StokesGrid:
    """Geometry and index bookkeeping for the staggered grid.

    Keeping every index calculation in one place is the difference between a
    solver a student can read and one they cannot.
    """

    def __init__(self, nx, nz, lx, lz):
        self.nx, self.nz = int(nx), int(nz)
        self.lx, self.lz = float(lx), float(lz)
        self.dx = self.lx / self.nx
        self.dz = self.lz / self.nz

        self.n_vx = self.nz * (self.nx + 1)
        self.n_vz = (self.nz + 1) * self.nx
        self.n_p = self.nz * self.nx
        self.ndof = self.n_vx + self.n_vz + self.n_p

        # Cell-centre coordinates (where P, eta_c, rho and T live).
        self.xc = (np.arange(self.nx) + 0.5) * self.dx
        self.zc = (np.arange(self.nz) + 0.5) * self.dz
        # Corner / basic-node coordinates (where eta_n lives).
        self.xn = np.arange(self.nx + 1) * self.dx
        self.zn = np.arange(self.nz + 1) * self.dz

    # -- global degree-of-freedom indices --------------------------------
    def ivx(self, i, j):
        """Index of vx at row i (0..nz-1), face j (0..nx)."""
        return i * (self.nx + 1) + j

    def ivz(self, i, j):
        """Index of vz at face i (0..nz), column j (0..nx-1)."""
        return self.n_vx + i * self.nx + j

    def ip(self, i, j):
        """Index of P at cell (i, j)."""
        return self.n_vx + self.n_vz + i * self.nx + j

    def unpack(self, sol):
        """Split a solution vector into (vx, vz, P) arrays."""
        vx = sol[: self.n_vx].reshape(self.nz, self.nx + 1)
        vz = sol[self.n_vx : self.n_vx + self.n_vz].reshape(self.nz + 1, self.nx)
        p = sol[self.n_vx + self.n_vz :].reshape(self.nz, self.nx)
        return vx, vz, p

    def __repr__(self):
        return (f"StokesGrid(nx={self.nx}, nz={self.nz}, lx={self.lx:g}, "
                f"lz={self.lz:g}, ndof={self.ndof})")


def centre_to_corner(field_c):
    """Average a cell-centred field (nz, nx) onto corners (nz+1, nx+1).

    Used to get the shear viscosity from the normal viscosity. Note this is an
    *arithmetic* mean; for viscosity contrasts of many orders of magnitude the
    harmonic or geometric mean behaves better, which is a good discussion point
    in the notebook. Edges are extended by nearest-neighbour.
    """
    nz, nx = field_c.shape
    padded = np.pad(field_c, 1, mode="edge")
    return 0.25 * (
        padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]
    )[: nz + 1, : nx + 1]


def solve_stokes(grid, eta_c, rho_c, gz=9.81, bc="free-slip",
                 eta_n=None, fx_c=None, fz_c=None):
    """Assemble and solve the variable-viscosity Stokes system.

    Parameters
    ----------
    grid : StokesGrid
    eta_c : (nz, nx) array
        Viscosity at cell centres.
    rho_c : (nz, nx) array
        Density at cell centres. Buoyancy enters the z-momentum equation as
        ``rho * gz``, with z positive downwards.
    gz : float
        Gravitational acceleration, positive downwards.
    bc : {"free-slip", "no-slip"}
        Applied on all four walls. Walls are impermeable in both cases.
    eta_n : (nz+1, nx+1) array, optional
        Shear viscosity at corners. Defaults to ``centre_to_corner(eta_c)``.
    fx_c, fz_c : arrays, optional
        Extra body-force components, sampled at vx and vz points respectively.
        Used by the manufactured-solution verification.

    Returns
    -------
    vx : (nz, nx+1)
    vz : (nz+1, nx)
    p  : (nz, nx)

    Notes
    -----
    Only the wall-*normal* velocities sit on the boundary in this layout, so
    only those are pinned. The tangential components live half a cell inside
    the domain and are handled with ghost-node relations folded into the
    interior stencil. Pinning whole boundary rows instead — a tempting
    shortcut — decouples the corner pressures and makes the matrix singular.
    """
    nx, nz, dx, dz = grid.nx, grid.nz, grid.dx, grid.dz
    eta_c = np.asarray(eta_c, dtype=float)
    rho_c = np.asarray(rho_c, dtype=float)
    if eta_n is None:
        eta_n = centre_to_corner(eta_c)
    if bc not in ("free-slip", "no-slip"):
        raise ValueError("bc must be 'free-slip' or 'no-slip'")
    noslip = bc == "no-slip"

    # ------------------------------------------------------------------
    # Scaling. This is not cosmetic: in SI units a lithospheric problem has
    # viscous coefficients of order eta/h^2 ~ 1e13 sitting in the same matrix
    # as pressure-gradient coefficients of order 1/h ~ 1e-4. Seventeen orders
    # of magnitude defeats a direct sparse solve and returns confident
    # nonsense. So we solve for a scaled pressure P' = P / pscale, scale the
    # momentum rows to O(1), and scale the continuity rows to match.
    #
    # The geometric mean is the right reference viscosity here — with contrasts
    # of many decades the arithmetic mean is dominated by the stiffest phase.
    # ------------------------------------------------------------------
    h = min(dx, dz)
    eta_ref = float(np.exp(np.mean(np.log(np.clip(eta_c, 1e-300, None)))))
    pscale = eta_ref / h          # column scaling: P = pscale * P'

    rows, cols, vals = [], [], []
    rhs = np.zeros(grid.ndof)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # ------------------------------------------------------------------
    # x-momentum, at vx points.
    # vx sits ON the left/right walls (j = 0, nx) -> pinned (impermeable).
    # vx rows i = 0 and nz-1 are half a cell inside the top/bottom walls.
    # ------------------------------------------------------------------
    for i in range(nz):
        for j in range(nx + 1):
            k = grid.ivx(i, j)

            if j == 0 or j == nx:                    # impermeable side walls
                add(k, k, 1.0)
                rhs[k] = 0.0
                continue

            # d/dx( 2 eta_c dvx/dx )
            eL, eR = eta_c[i, j - 1], eta_c[i, j]
            add(k, grid.ivx(i, j - 1), 2.0 * eL / dx**2)
            add(k, grid.ivx(i, j + 1), 2.0 * eR / dx**2)
            add(k, k, -2.0 * (eL + eR) / dx**2)

            # d/dz( eta_n (dvx/dz + dvz/dx) ), assembled as (sigma_b - sigma_t)/dz
            eT, eB = eta_n[i, j], eta_n[i + 1, j]

            if i == 0:
                # Top wall. vz == 0 all along it, so dvz/dx vanishes there.
                if noslip:
                    # vx = 0 at the wall, dz/2 away: sigma_t = eT * 2 vx / dz
                    add(k, k, -2.0 * eT / dz**2)
                # free slip: sigma_t = 0, nothing to add
            else:
                # sigma_t = eT * ( (vx[i,j]-vx[i-1,j])/dz + (vz[i,j]-vz[i,j-1])/dx )
                # and it enters as -sigma_t/dz
                add(k, grid.ivx(i - 1, j), eT / dz**2)
                add(k, k, -eT / dz**2)
                add(k, grid.ivz(i, j), -eT / (dx * dz))
                add(k, grid.ivz(i, j - 1), eT / (dx * dz))

            if i == nz - 1:
                if noslip:
                    add(k, k, -2.0 * eB / dz**2)
            else:
                # sigma_b enters as +sigma_b/dz
                add(k, grid.ivx(i + 1, j), eB / dz**2)
                add(k, k, -eB / dz**2)
                add(k, grid.ivz(i + 1, j), eB / (dx * dz))
                add(k, grid.ivz(i + 1, j - 1), -eB / (dx * dz))

            # -dP/dx  (column scaled: the unknown is P' = P / pscale)
            add(k, grid.ip(i, j - 1), pscale / dx)
            add(k, grid.ip(i, j), -pscale / dx)

            if fx_c is not None:
                rhs[k] = -fx_c[i, j]

    # ------------------------------------------------------------------
    # z-momentum, at vz points.
    # vz sits ON the top/bottom walls (i = 0, nz) -> pinned (impermeable).
    # ------------------------------------------------------------------
    for i in range(nz + 1):
        for j in range(nx):
            k = grid.ivz(i, j)

            if i == 0 or i == nz:                    # impermeable top/bottom
                add(k, k, 1.0)
                rhs[k] = 0.0
                continue

            # d/dz( 2 eta_c dvz/dz )
            eT, eB = eta_c[i - 1, j], eta_c[i, j]
            add(k, grid.ivz(i - 1, j), 2.0 * eT / dz**2)
            add(k, grid.ivz(i + 1, j), 2.0 * eB / dz**2)
            add(k, k, -2.0 * (eT + eB) / dz**2)

            # d/dx( eta_n (dvz/dx + dvx/dz) ), as (sigma_r - sigma_l)/dx
            eL, eR = eta_n[i, j], eta_n[i, j + 1]

            if j == 0:
                # Left wall. vx == 0 all along it, so dvx/dz vanishes there.
                if noslip:
                    add(k, k, -2.0 * eL / dx**2)
            else:
                # sigma_l = eL * ( (vz[i,j]-vz[i,j-1])/dx + (vx[i,j]-vx[i-1,j])/dz )
                # and it enters as -sigma_l/dx
                add(k, grid.ivz(i, j - 1), eL / dx**2)
                add(k, k, -eL / dx**2)
                add(k, grid.ivx(i, j), -eL / (dx * dz))
                add(k, grid.ivx(i - 1, j), eL / (dx * dz))

            if j == nx - 1:
                if noslip:
                    add(k, k, -2.0 * eR / dx**2)
            else:
                # sigma_r enters as +sigma_r/dx
                add(k, grid.ivz(i, j + 1), eR / dx**2)
                add(k, k, -eR / dx**2)
                add(k, grid.ivx(i, j + 1), eR / (dx * dz))
                add(k, grid.ivx(i - 1, j + 1), -eR / (dx * dz))

            # -dP/dz  (column scaled)
            add(k, grid.ip(i - 1, j), pscale / dz)
            add(k, grid.ip(i, j), -pscale / dz)

            # buoyancy: z is depth-positive, so gravity acts in +z
            rho_face = 0.5 * (rho_c[i - 1, j] + rho_c[i, j])
            rhs[k] = -rho_face * gz
            if fz_c is not None:
                rhs[k] -= fz_c[i, j]

    # ------------------------------------------------------------------
    # continuity, at cell centres
    # ------------------------------------------------------------------
    for i in range(nz):
        for j in range(nx):
            k = grid.ip(i, j)
            if i == 0 and j == 0:
                # Pressure is defined only up to a constant with impermeable
                # walls, so pin one cell to make the system non-singular.
                add(k, k, 1.0)
                rhs[k] = 0.0
                continue
            add(k, grid.ivx(i, j + 1), 1.0 / dx)
            add(k, grid.ivx(i, j), -1.0 / dx)
            add(k, grid.ivz(i + 1, j), 1.0 / dz)
            add(k, grid.ivz(i, j), -1.0 / dz)
            rhs[k] = 0.0

    A = sp.coo_matrix((vals, (rows, cols)), shape=(grid.ndof, grid.ndof)).tocsr()

    # Row equilibration: divide every row by its largest entry. Doing this by
    # equation *type* instead is tempting and wrong — it rescales the pinned
    # boundary rows along with the physical ones, leaving entries spanning
    # 1e-13 to 1e4 and destroying the conditioning that the column scaling
    # above was meant to buy.
    row_max = np.maximum(np.abs(A).max(axis=1).toarray().ravel(), 1e-300)
    A = sp.diags(1.0 / row_max) @ A
    rhs = rhs / row_max

    lu = spla.splu(A.tocsc())
    sol = lu.solve(rhs)

    # One step of iterative refinement. Cheap (the factorisation is reused) and
    # worth it: for a hydrostatic problem the pressure gradient has to cancel
    # gravity almost exactly, and what survives that cancellation is the flow
    # we actually care about.
    sol -= lu.solve(A @ sol - rhs)

    if not np.all(np.isfinite(sol)):
        raise RuntimeError(
            "Stokes solve produced non-finite values. This usually means the "
            "viscosity field contains zeros or NaNs, or an extreme contrast."
        )
    vx, vz, p = grid.unpack(sol)
    return vx, vz, p * pscale          # undo the pressure column scaling


def velocity_to_centres(grid, vx, vz):
    """Interpolate staggered velocities onto cell centres, for plotting.

    pyGMT wants both components on the same regular grid, so this is the last
    step before any quiver plot.
    """
    vxc = 0.5 * (vx[:, :-1] + vx[:, 1:])
    vzc = 0.5 * (vz[:-1, :] + vz[1:, :])
    return vxc, vzc


def strain_rate_second_invariant(grid, vx, vz):
    """Second invariant of the strain-rate tensor at cell centres.

    ``eps_II = sqrt(eps_xx^2 + eps_xz^2)`` for the incompressible 2-D case,
    where ``eps_zz = -eps_xx``. This is the field to plot when you want to see
    where deformation is localising — shear bands, necking, plate boundaries.
    """
    dx, dz = grid.dx, grid.dz
    exx = (vx[:, 1:] - vx[:, :-1]) / dx                       # (nz, nx)

    vx_pad = np.pad(vx, ((1, 1), (0, 0)), mode="edge")
    vz_pad = np.pad(vz, ((0, 0), (1, 1)), mode="edge")
    dvxdz = (vx_pad[2:, :] - vx_pad[:-2, :]) / (2.0 * dz)     # (nz, nx+1)
    dvzdx = (vz_pad[:, 2:] - vz_pad[:, :-2]) / (2.0 * dx)     # (nz+1, nx)

    exz = 0.5 * (
        0.5 * (dvxdz[:, :-1] + dvxdz[:, 1:])
        + 0.5 * (dvzdx[:-1, :] + dvzdx[1:, :])
    )
    return np.sqrt(exx**2 + exz**2)


# --------------------------------------------------------------------------
# Verification: a manufactured solution
# --------------------------------------------------------------------------
def manufactured_solution(grid, eta=1.0):
    """Analytic Stokes solution and the body force that produces it.

    Built from the stream function ``psi = sin(pi x / lx) sin(pi z / lz)``, so
    the velocity field is divergence-free by construction:

        vx =  d(psi)/dz,    vz = -d(psi)/dx

    with pressure ``P = cos(pi x / lx) cos(pi z / lz)``. For constant
    viscosity, ``lap(v) = -(kx^2 + kz^2) v``, so the required body force is

        f_i = dP/dx_i - eta lap(v_i)

    This particular stream function is chosen so the solution satisfies
    **free-slip and impermeable conditions on all four walls** — ``vx = 0`` and
    ``dvz/dx = 0`` on the sides, ``vz = 0`` and ``dvx/dz = 0`` on top and
    bottom. That means it can be verified against the solver's ordinary
    boundary conditions, with no special Dirichlet machinery.

    Feed ``fx``/``fz`` back into ``solve_stokes`` with ``gz=0`` and the
    computed field should converge to ``vx``/``vz`` at second order. This is
    the verification cell for the Stokes notebook — a real convergence test,
    not a plausible-looking picture. Compare pressure only after removing the
    mean from both fields, since the solver pins one cell arbitrarily.

    Returns a dict with keys ``vx``, ``vz``, ``p``, ``fx``, ``fz`` sampled at
    the correct staggered locations.
    """
    lx, lz = grid.lx, grid.lz
    kx, kz = np.pi / lx, np.pi / lz
    lap = -(kx**2 + kz**2)

    # vx points: x on faces, z at centres
    Xvx, Zvx = np.meshgrid(grid.xn, grid.zc)
    # vz points: x at centres, z on faces
    Xvz, Zvz = np.meshgrid(grid.xc, grid.zn)
    # P points: both at centres
    Xp, Zp = np.meshgrid(grid.xc, grid.zc)

    vx = kz * np.sin(kx * Xvx) * np.cos(kz * Zvx)
    vz = -kx * np.cos(kx * Xvz) * np.sin(kz * Zvz)
    p = np.cos(kx * Xp) * np.cos(kz * Zp)

    dpdx = -kx * np.sin(kx * Xvx) * np.cos(kz * Zvx)
    dpdz = -kz * np.cos(kx * Xvz) * np.sin(kz * Zvz)

    fx = dpdx - eta * lap * vx
    fz = dpdz - eta * lap * vz
    return {"vx": vx, "vz": vz, "p": p, "fx": fx, "fz": fz}
