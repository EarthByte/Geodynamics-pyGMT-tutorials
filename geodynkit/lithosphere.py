"""
Rheological and thermal scaffolding for lithospheric extension and shortening.

Part 3 of the suite needs a *geologically defensible* lithosphere, not just a
solver. This module holds that: layer structure, dislocation-creep flow laws,
Drucker-Prager yielding with strain weakening, and a conductive geotherm.

Everything here is pure NumPy. That is deliberate — it means the setup can be
inspected, plotted and tested before any finite-element code runs, and the
classic first check on a lithospheric model (the strength envelope) needs no
solver at all.

Parameter provenance
--------------------
The values follow the standard continental-extension setup of

    Naliboff, J. and Buiter, S. (2015), Rift reactivation and migration during
    multiphase extension, Earth Planet. Sci. Lett. 421, 58-67.
    https://doi.org/10.1016/j.epsl.2015.03.050

as configured in ASPECT's `continental_extension` cookbook. The flow laws
themselves come from the experimental rock-mechanics literature:

    wet quartzite   — Rutter & Brodie (2004), J. Struct. Geol. 26, 2011-2023
    wet anorthite   — Rybacki et al. (2006), J. Geophys. Res. 111, B03203
    dry olivine     — Hirth & Kohlstedt (2004), AGU Geophys. Monogr. 138, 83-105

Numerical *parameter values* are facts from the published literature and are
reproduced here with citation. No code is taken from ASPECT, which is GPL-2 and
therefore incompatible with this suite's BSD-3 licence.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Layer", "UPPER_CRUST", "LOWER_CRUST", "MANTLE_LITHOSPHERE", "ASTHENOSPHERE",
    "DEFAULT_COLUMN", "R_GAS",
    "geotherm", "dislocation_creep_viscosity", "drucker_prager_yield_stress",
    "strain_weakening_factor", "effective_viscosity", "strength_envelope",
]

#: Universal gas constant, J/(mol K).
R_GAS = 8.314


@dataclass(frozen=True)
class Layer:
    """One lithospheric layer: geometry, density, flow law and yield parameters.

    The dislocation-creep law is the usual form

        eta = 0.5 * A^(-1/n) * epsII^((1-n)/n) * exp((E + P V) / (n R T))

    with ``A`` the prefactor, ``n`` the stress exponent, ``E`` the activation
    energy and ``V`` the activation volume.
    """

    name: str
    thickness_km: float
    density: float             # kg/m3
    prefactor: float           # A, Pa^-n s^-1
    stress_exponent: float     # n
    activation_energy: float   # E, J/mol
    activation_volume: float   # V, m3/mol
    heat_production: float     # W/m3
    conductivity: float = 2.5  # W/(m K)
    friction_deg: float = 30.0
    cohesion: float = 20e6     # Pa


# Naliboff & Buiter (2015) column, as in ASPECT's continental_extension cookbook.
UPPER_CRUST = Layer("upper crust", 20.0, 2700.0,
                    prefactor=1.37e-26, stress_exponent=4.0,
                    activation_energy=223e3, activation_volume=0.0,
                    heat_production=1.00e-6)

LOWER_CRUST = Layer("lower crust", 20.0, 2900.0,
                    prefactor=5.71e-23, stress_exponent=3.0,
                    activation_energy=345e3, activation_volume=0.0,
                    heat_production=0.25e-6)

MANTLE_LITHOSPHERE = Layer("mantle lithosphere", 60.0, 3300.0,
                           prefactor=7.37e-15, stress_exponent=3.5,
                           activation_energy=530e3, activation_volume=18e-6,
                           heat_production=0.0)

#: Same flow law as the mantle lithosphere; separated because it is
#: thermally, not compositionally, distinct.
ASTHENOSPHERE = Layer("asthenosphere", 0.0, 3300.0,
                      prefactor=7.37e-15, stress_exponent=3.5,
                      activation_energy=530e3, activation_volume=18e-6,
                      heat_production=0.0)

#: 20 km upper crust / 20 km lower crust / 60 km mantle lithosphere = 100 km.
DEFAULT_COLUMN = (UPPER_CRUST, LOWER_CRUST, MANTLE_LITHOSPHERE)


# --------------------------------------------------------------------------
# Thermal structure
# --------------------------------------------------------------------------
def geotherm(depth_km, column=DEFAULT_COLUMN, surface_T=273.0,
             surface_heat_flow=0.055):
    """Steady-state conductive geotherm through a layered column.

    Integrates downwards through each layer, carrying the heat flux and
    reducing it by the radiogenic production of the layer above:

        dT/dz = q(z) / k        q(z) = q_top - A z

    Returns temperature in K at each ``depth_km``.

    The alternative — a linear geotherm — is a common shortcut and a bad one
    for extension models: it puts the brittle-ductile transition in the wrong
    place, because the crustal heat production that steepens the shallow
    geotherm is precisely what weakens the lower crust.
    """
    depth_km = np.atleast_1d(np.asarray(depth_km, dtype=float))
    T = np.empty_like(depth_km)

    boundaries, acc = [], 0.0
    for lay in column:
        acc += lay.thickness_km
        boundaries.append(acc)

    # Temperature and flux at the top of each layer.
    T_top, q_top, z_top = [surface_T], [surface_heat_flow], [0.0]
    for lay, z_bot in zip(column, boundaries):
        h = lay.thickness_km * 1e3
        q_b = q_top[-1] - lay.heat_production * h
        T_b = (T_top[-1] + q_top[-1] * h / lay.conductivity
               - lay.heat_production * h**2 / (2 * lay.conductivity))
        T_top.append(T_b)
        q_top.append(q_b)
        z_top.append(z_bot)

    for i, d in enumerate(depth_km):
        k = min(np.searchsorted(boundaries, d, side="right"), len(column) - 1)
        lay = column[k]
        dz = (d - z_top[k]) * 1e3
        T[i] = (T_top[k] + q_top[k] * dz / lay.conductivity
                - lay.heat_production * dz**2 / (2 * lay.conductivity))
    return T if T.size > 1 else float(T[0])


def lithostatic_pressure(depth_km, column=DEFAULT_COLUMN, g=9.81):
    """Lithostatic pressure (Pa) at depth, integrating layer densities."""
    depth_km = np.atleast_1d(np.asarray(depth_km, dtype=float))
    P = np.zeros_like(depth_km)
    for i, d in enumerate(depth_km):
        remaining, acc = d, 0.0
        for lay in column:
            t = min(remaining, lay.thickness_km)
            if t <= 0:
                break
            acc += lay.density * g * t * 1e3
            remaining -= t
        if remaining > 0:                       # below the column
            acc += column[-1].density * g * remaining * 1e3
        P[i] = acc
    return P if P.size > 1 else float(P[0])


# --------------------------------------------------------------------------
# Rheology
# --------------------------------------------------------------------------
def dislocation_creep_viscosity(layer, T, strain_rate, pressure=0.0):
    """Effective viscosity (Pa s) for dislocation creep.

        eta = 0.5 A^(-1/n) epsII^((1-n)/n) exp((E + P V) / (n R T))

    Strongly non-Newtonian: with n ~ 3-4, doubling the strain rate drops the
    viscosity by roughly a factor of two, which is what allows shear zones to
    run away.
    """
    n = layer.stress_exponent
    return (0.5 * layer.prefactor ** (-1.0 / n)
            * np.asarray(strain_rate, dtype=float) ** ((1.0 - n) / n)
            * np.exp((layer.activation_energy + pressure * layer.activation_volume)
                     / (n * R_GAS * np.asarray(T, dtype=float))))


def strain_weakening_factor(plastic_strain, start=0.5, end=1.5, final=0.25):
    """Linear plastic strain weakening, after Naliboff & Buiter (2015).

    Friction angle and cohesion are scaled by 1 at strain <= ``start``, falling
    linearly to ``final`` at strain >= ``end``.

    Without this a model localises briefly and then re-hardens; strain
    weakening is what lets a fault, once formed, keep slipping — and therefore
    what produces a rift rather than diffuse thinning.
    """
    e = np.asarray(plastic_strain, dtype=float)
    f = 1.0 + (final - 1.0) * (e - start) / (end - start)
    return np.clip(f, min(final, 1.0), max(final, 1.0))


def drucker_prager_yield_stress(pressure, friction_deg=30.0, cohesion=20e6,
                                weakening=1.0):
    """Drucker-Prager yield stress (Pa), with optional strain weakening.

        sigma_y = C cos(phi) + P sin(phi)

    Both cohesion and friction are scaled by ``weakening``.
    """
    phi = np.radians(friction_deg) * np.asarray(weakening, dtype=float)
    return (np.asarray(cohesion) * weakening * np.cos(phi)
            + np.asarray(pressure, dtype=float) * np.sin(phi))


def effective_viscosity(layer, T, strain_rate, pressure, plastic_strain=0.0,
                        eta_min=1e18, eta_max=1e26):
    """Viscosity actually used by the solver: the weaker of creep and yielding.

    The plastic branch is expressed as a viscosity, ``eta_plast = sigma_y /
    (2 epsII)``, and combined with the ductile branch by taking the minimum —
    the material deforms by whichever mechanism is easier. Bounds follow the
    cookbook: 1e18 to 1e26 Pa s.
    """
    eta_creep = dislocation_creep_viscosity(layer, T, strain_rate, pressure)
    w = strain_weakening_factor(plastic_strain)
    sy = drucker_prager_yield_stress(pressure, layer.friction_deg,
                                     layer.cohesion, w)
    eta_plast = sy / (2.0 * np.asarray(strain_rate, dtype=float))
    return np.clip(np.minimum(eta_creep, eta_plast), eta_min, eta_max)


def strength_envelope(depth_km=None, strain_rate=1e-15, column=DEFAULT_COLUMN,
                      plastic_strain=0.0, **kw):
    """Differential stress supported at each depth — the classic sanity check.

    Returns a dict with depth, temperature, pressure, the ductile and brittle
    stresses separately, and the governing (weaker) one. Plotting this is the
    first thing to do with any lithospheric setup: if the brittle-ductile
    transitions are in the wrong place, nothing downstream will be right.
    """
    if depth_km is None:
        depth_km = np.linspace(0.0, sum(l.thickness_km for l in column), 400)
    depth_km = np.asarray(depth_km, dtype=float)

    T = np.atleast_1d(geotherm(depth_km, column, **kw))
    P = np.atleast_1d(lithostatic_pressure(depth_km, column))

    boundaries, acc = [], 0.0
    for lay in column:
        acc += lay.thickness_km
        boundaries.append(acc)

    ductile = np.empty_like(depth_km)
    brittle = np.empty_like(depth_km)
    for i, d in enumerate(depth_km):
        k = min(np.searchsorted(boundaries, d, side="right"), len(column) - 1)
        lay = column[k]
        eta = dislocation_creep_viscosity(lay, T[i], strain_rate, P[i])
        ductile[i] = 2.0 * eta * strain_rate
        w = strain_weakening_factor(plastic_strain)
        brittle[i] = 2.0 * drucker_prager_yield_stress(
            P[i], lay.friction_deg, lay.cohesion, w)

    return dict(depth_km=depth_km, temperature=T, pressure=P,
                ductile_MPa=ductile / 1e6, brittle_MPa=brittle / 1e6,
                strength_MPa=np.minimum(ductile, brittle) / 1e6,
                strain_rate=strain_rate)
