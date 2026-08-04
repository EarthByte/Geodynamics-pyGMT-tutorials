"""
Thermal convection: Stokes coupled to the energy equation.

Non-dimensionalised in the standard way for the infinite-Prandtl Boussinesq
problem, so the whole system is governed by one number:

    Ra = rho g alpha Delta_T d^3 / (eta kappa)

In these units the domain is the unit square, viscosity and thermal
diffusivity are 1, temperature runs 0 (top) to 1 (bottom), and buoyancy enters
as ``Ra * T``. That makes the Blankenbach benchmark directly comparable, which
is the point: this module exists so students can reproduce a published number.
"""

import numpy as np

from .advection import advect_2d_semilagrangian
from .stokes import StokesGrid, solve_stokes, velocity_to_centres

__all__ = ["ConvectionModel", "nusselt_number", "rms_velocity",
           "BLANKENBACH_1A"]

#: Published steady-state values for Blankenbach et al. (1989) case 1a —
#: isoviscous, base-heated, Ra = 1e4, unit aspect ratio. The verification
#: target for the convection notebook.
BLANKENBACH_1A = {"Ra": 1e4, "Nu": 4.884409, "v_rms": 42.864947}


class ConvectionModel:
    """Non-dimensional 2-D thermal convection in a unit box.

    Isoviscous by default; pass a callable ``viscosity(T)`` for a
    temperature-dependent rheology (Frank-Kamenetskii is one line).

    Boundary conditions follow the Blankenbach convention: isothermal top and
    bottom, insulating sides, free slip everywhere.
    """

    def __init__(self, nx=48, nz=48, Ra=1e4, viscosity=None, bc="free-slip",
                 seed=0):
        self.grid = StokesGrid(nx, nz, 1.0, 1.0)
        self.Ra = float(Ra)
        self.viscosity = viscosity
        self.bc = bc
        self.nx, self.nz = nx, nz

        self.x = self.grid.xc
        self.z = self.grid.zc
        self.time = 0.0
        self.step = 0
        self.history = []          # (time, Nu, v_rms)

        self.T = self.initial_temperature(seed=seed)
        self.vxc = np.zeros((nz, nx))
        self.vzc = np.zeros((nz, nx))

    def initial_temperature(self, amplitude=0.05, seed=0):
        """Conductive profile plus a single-mode perturbation.

        A clean sinusoidal seed (rather than noise) makes the run reproducible
        and converges to the expected single cell, which is what the benchmark
        assumes.
        """
        X, Z = np.meshgrid(self.x, self.z)
        T = Z.copy()                      # z is depth: 0 at top, 1 at bottom
        T += amplitude * np.cos(np.pi * X) * np.sin(np.pi * Z)
        return np.clip(T, 0.0, 1.0)

    # ------------------------------------------------------------------
    def _eta(self):
        if self.viscosity is None:
            return np.ones((self.nz, self.nx))
        return np.asarray(self.viscosity(self.T), dtype=float)

    def solve_flow(self):
        """One Stokes solve for the current temperature field.

        In non-dimensional form the buoyancy term is ``-Ra * T`` acting
        upwards; with z positive downwards we pass ``rho = -Ra * T`` and
        ``gz = 1``.
        """
        eta = self._eta()
        rho = -self.Ra * self.T
        vx, vz, p = solve_stokes(self.grid, eta, rho, gz=1.0, bc=self.bc)
        self.vxc, self.vzc = velocity_to_centres(self.grid, vx, vz)
        self.p = p
        return self.vxc, self.vzc

    def timestep(self, cfl=0.5, diffusion_safety=0.4):
        """Advance one step: Stokes solve, then advection, then diffusion.

        Operator splitting keeps each piece recognisable. The timestep is the
        smaller of the advective (Courant) and diffusive limits.
        """
        from .diffusion import solve_2d_explicit

        self.solve_flow()
        dx, dz = self.grid.dx, self.grid.dz
        vmax = max(float(np.abs(self.vxc).max()), float(np.abs(self.vzc).max()), 1e-30)
        dt_adv = cfl * min(dx, dz) / vmax
        dt_dif = diffusion_safety * min(dx, dz) ** 2 / 2.0
        dt = min(dt_adv, dt_dif)

        self.T = advect_2d_semilagrangian(self.T, self.vxc, self.vzc,
                                          self.x, self.z, dt)
        self.T[0, :] = 0.0
        self.T[-1, :] = 1.0
        self.T = solve_2d_explicit(self.T, kappa=1.0, dx=dx, dz=dz,
                                   dt=dt, nsteps=1, fixed=("top", "bottom"))

        self.time += dt
        self.step += 1
        self.history.append((self.time, self.nusselt(), self.v_rms()))
        return dt

    def run(self, max_steps=2000, tol=1e-5, report_every=None, verbose=False):
        """Integrate until steady state or ``max_steps``.

        Steady state is declared when the relative change in temperature per
        unit time falls below ``tol``. Returns the number of steps taken.
        """
        for n in range(max_steps):
            Tprev = self.T.copy()
            dt = self.timestep()
            change = float(np.abs(self.T - Tprev).max()) / max(dt, 1e-30)
            if verbose and report_every and self.step % report_every == 0:
                print(f"step {self.step:5d}  t={self.time:.5f}  "
                      f"Nu={self.nusselt():.4f}  v_rms={self.v_rms():.3f}  "
                      f"dT/dt={change:.3e}")
            if change < tol and self.step > 20:
                return self.step
        return self.step

    # ------------------------------------------------------------------
    def nusselt(self):
        return nusselt_number(self.T, self.grid.dz)

    def v_rms(self):
        return rms_velocity(self.vxc, self.vzc)


def nusselt_number(T, dz):
    """Surface Nusselt number: mean non-dimensional surface heat flux.

    Uses a one-sided second-order difference at the top boundary, which
    matters — a first-order estimate is visibly off against the benchmark.
    """
    dTdz_top = (-3.0 * T[0, :] + 4.0 * T[1, :] - T[2, :]) / (2.0 * dz)
    return float(np.mean(dTdz_top))


def rms_velocity(vxc, vzc):
    """Root-mean-square velocity over the domain."""
    return float(np.sqrt(np.mean(vxc**2 + vzc**2)))
