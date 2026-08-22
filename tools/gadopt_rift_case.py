#!/usr/bin/env python3
"""
Time-dependent lithospheric extension with three advected layers and
strain-weakening plasticity.

This is the step beyond `gadopt_lithosphere_case.py`. That one solves the
Spiegelman benchmark *instantaneously* — one nonlinear Stokes solve, so nothing
moves and no strain accumulates. A rift is a feedback loop, and you cannot see a
feedback loop in a snapshot:

    yielding -> plastic strain -> weaker rock -> more yielding

So this script wires the two pieces that close that loop:

* **`LevelSetSolver`** advects two conservative level sets, giving three
  materials — upper crust, lower crust, mantle lithosphere — that keep their
  identity through large finite strain.
* **`GenericTransportSolver`** advects a scalar plastic-strain field with a
  source term equal to the plastic strain rate, and that field feeds back into
  the yield stress via the Naliboff & Buiter (2015) weakening law.

Layer structure, flow laws and weakening parameters come from
`geodynkit.lithosphere`; see that module for provenance.

Non-dimensionalisation follows the Spiegelman convention used by G-ADOPT's
Drucker-Prager demo: lengths by the domain depth H, velocities by the boundary
velocity U0, viscosities by a reference mu0, so stress scales as mu0*U0/H.

Usage:
    python3 gadopt_rift_case.py --steps 20 --nx 128 --ny 64

Invariants, and why they are checked every step
-----------------------------------------------
A conservative level set is a smoothed indicator function: its value IS the
volume fraction of material above the interface, so it lives in [0, 1] by
definition. `material_field` blends layer properties by interpolating in psi,
which means it will happily extrapolate a density no rock has if psi leaves
that interval.

The first production run of this script (400 steps, 96x48, 2 h 22 min) ended
with level sets spanning [-1.72, 3.02] and a density field with mantle at the
surface and crustal density at 90 km depth. Every plot made from it was
uninterpretable. The range was already being measured -- and only printed, once,
at the end. Both invariants are now checked inside the time loop and raise.

Reinitialisation, measured
--------------------------
The drift is a reinitialisation deficit, not an advection-scheme failure. At
64x32, q0 = 0.055, 20 steps, varying only the number of reinitialisation steps
per timestep:

    reinitialisation steps   psi excursion beyond [0,1]   wall clock
             2 (old default)          0.0277                481 s
             6                        0.0148                493 s
            12 (new default)          0.0048                506 s

A six-fold reduction in drift for 5% more wall-clock, because the Stokes solve
dominates the cost and reinitialisation is nearly free beside it. At the old
default the excursion grew roughly exponentially and the layering check failed
by step 34.

The solve path mattered more
----------------------------
The "stagnation" the Picard iteration appeared to suffer was three separate
self-inflicted problems, all in `solve_stokes`; see the long comments there.
Briefly: the best-iterate safeguard minimised the change between successive
iterates, which is smallest at iteration 0 for reasons unrelated to accuracy and
selected the least incompressible field available; a failed Newton solve left
PETSc's diverged iterate in place; and the under-relaxation halved omega on any
rise, so it hit its floor by iteration 6 and stayed there.

At 64x32 over 40 steps, before and after:

    Picard              30 its (the cap) at 1e-2   ->  15-21 its at 9e-5
    Newton              failed every step          ->  succeeds most steps
    ||div u||/||grad u||  0.52                     ->  0.11  (control: 0.09)
    psi excursion       0.36                       ->  0.010
    in-seed / outside strain  1.31 / 0.55          ->  1.31 / 0.10

With those fixed the model localises: a symmetric conjugate pair dipping inward
from ~45 km at x = 75 and 125 km, converging beneath the axis at ~30 km, steep
faults through the brittle upper crust, and peak strain growing past the seeded
maximum for the first time.

How long a run is meaningful
----------------------------
Not as long as you would like, and the limit is the missing free surface rather
than anything numerical.

The mesh is Eulerian and fixed. Both walls are driven outward, the base is
no-flux, so by incompressibility the material leaving through the sides must
enter through the top: measured, the vertical velocity on the top boundary is
0.999 in non-dimensional units -- the full boundary velocity, 0.25 cm/yr,
directed downwards, essentially uniformly across the domain. The mass balance
closes to 0.1% (200.0 out the sides against 199.8 in the top).

That inflow is fictitious. A real rift subsides; this one imports rock from
nowhere through a flat lid. One timestep at dt = 2e-3 is 0.08 Myr, so 200 m
descends per step, and the 20 km upper crust is entirely replaced by imported
material after **about 100 steps**. At 40 steps 8 km has come in -- 40% of the
upper crust -- and the rift structure is still clean.

So treat ~80 steps (6.4 Myr, 16 km of import) as the horizon for the current
physics, and read anything past ~100 steps as a statement about the boundary
condition. A free surface is not a refinement here; it is what the next order of
magnitude in run length depends on.
"""

import argparse
import json
import sys
import time

import numpy as np

# PETSc parses `sys.argv` when it initialises, which happens on import, and then
# complains at exit about every flag it did not recognise -- all of ours:
#     WARNING! There are options you set that were not used!
#     Option left: name:--heatflow value: 0.040 source: command line
# Harmless, but it buries the real output of a long run. Hide our arguments from
# PETSc for the duration of the import and put them back afterwards.
_ARGV = sys.argv[:]
sys.argv = sys.argv[:1]
from gadopt import *  # noqa: E402
sys.argv = _ARGV

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from geodynkit import lithosphere as LI  # noqa: E402

# --- scales -------------------------------------------------------------
H = 100e3                       # domain depth, m
YEAR = 86400 * 365.25
MU0 = 1e22                      # reference viscosity, Pa s
RATE_CM_YR = 0.25               # half-rate on each wall; the cookbook value
U0 = RATE_CM_YR * 1e-2 / YEAR
STRESS = MU0 * U0 / H           # stress scale, Pa
SR = U0 / H                     # strain-rate scale, 1/s
T_SCALE = H / U0                # time scale, s


def set_rate(cm_per_year):
    """Change the extension rate, and every scale that depends on it.

    U0 is the velocity scale, so it is not a free knob -- it sets the stress
    scale, the strain-rate scale, the time scale, the buoyancy group and the
    Peclet number all at once, and every one of those changes for a physical
    reason:

      * `STRESS = mu0 U0 / H` rises, so the non-dimensional yield stress
        `sigma_y / STRESS` FALLS: faster extension yields more easily.
      * `SR = U0 / H` rises, so the dimensional strain rate in the creep law
        rises and the dislocation-creep viscosity falls with it (n ~ 3-4).
      * `kappa / (U0 H)` falls, so the Peclet number rises: less time for
        conduction to erase the thermal anomaly that advection creates.
      * `g H^2 / (mu0 U0)` falls, so buoyancy matters less relative to the
        driving stress.

    The first two say a faster rift should localise more readily, the third
    that it should weaken thermally more readily, and the fourth that the free
    surface should matter less. They do not have to agree, which is why the
    experiment is worth running rather than arguing about.

    Module globals rather than parameters because `build()` reads them at call
    time; set the rate before building.
    """
    global RATE_CM_YR, U0, STRESS, SR, T_SCALE
    RATE_CM_YR = float(cm_per_year)
    U0 = RATE_CM_YR * 1e-2 / YEAR
    STRESS = MU0 * U0 / H
    SR = U0 / H
    T_SCALE = H / U0


def build(nx, ny, aspect=2.0, damper=1e21, seed_halfwidth_km=25.0,
          reini_steps=12, reini_factor=0.5,
          free_surface=False, cluster_x=0.0, cluster_y=0.0,
          fs_bug="none", thermal=False, crust_km=40.0, t_base=1613.0,
          seed_mode="centre", seed_amp=1.0, dt_max=2e-3):
    """Assemble mesh, fields, rheology and solvers. Returns a dict of handles."""
    mesh = RectangleMesh(nx, ny, aspect, 1.0, quadrilateral=True)

    # MESH GRADING, in place of adaptive refinement.
    #
    # T10's run ends when the necking lower crust thins to about 3.5 cells and
    # the two level sets bounding it start to overlap. The textbook answer is
    # metric-based adaptivity -- `animate`'s RiemannianMetric plus `adapt` --
    # and it is not available here. `adapt` drives Mmg through PETSc; the
    # Firedrake image's PETSc is configured without --download-mmg or
    # --download-parmmg (checked: no MMG symbols in petscconf.h, no mmg in the
    # recorded CONFIGURE_OPTIONS), and `animate` is not installed. Enabling it
    # means rebuilding PETSc and Firedrake from source on two architectures, and
    # Mmg adapts only simplex meshes, so it would additionally mean giving up
    # the quadrilateral discretisation the rest of the suite uses. That is not a
    # reasonable price for a teaching container.
    #
    # It is also not needed here, because we know in advance where the
    # resolution has to go: the rift nucleates at the seed and the necking
    # happens in the crust. Adaptivity earns its keep when you do NOT know.
    #
    # The map matters more than it looks. The obvious choice, s -> sign(s)|s|^p
    # with p > 1, has zero derivative at s = 0, so cells at the axis collapse:
    # at p = 1.6 the widest cell was 10.6x the narrowest, the aspect ratios went
    # with it, and the level-set excursion after three steps was 0.0139 against
    # 0.0002 on a uniform mesh -- seventy times worse, for a mesh that was
    # supposed to help. (More reinitialisation did not touch it: 12 and 48
    # sweeps gave byte-identical results.)
    #
    # This map instead has bounded derivative everywhere:
    #
    #     s' = arctanh(s * tanh(b)) / b        s in [-1, 1]
    #
    # At b = 1 the cell-size ratio across the domain is about 2.4, and cells at
    # the centre are 0.76 of uniform. b = 0 is the identity.
    def cluster(t, b):
        """Map [-1, 1] to itself, concentrating points near 0. b = 0 is uniform."""
        if b <= 0:
            return t
        return np.arctanh(np.clip(t, -1.0, 1.0) * np.tanh(b)) / b

    if cluster_x > 0 or cluster_y > 0:
        xy = mesh.coordinates.dat.data
        c = aspect / 2.0
        xy[:, 0] = c + c * cluster((xy[:, 0] - c) / c, cluster_x)
        # Vertical: one-sided, concentrating rows towards the surface at y = 1,
        # which is where the crust and every brittle-ductile transition are.
        if cluster_y > 0:
            d = np.clip(1.0 - xy[:, 1], 0.0, 1.0)
            xy[:, 1] = 1.0 - np.arctanh(d * np.tanh(cluster_y)) / cluster_y

    mesh.cartesian = True
    boundary = get_boundary_ids(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 2)          # plastic strain
    K = FunctionSpace(mesh, "Q", 2)           # level sets
    # A free surface adds one scalar unknown per free-surface boundary: eta, the
    # surface displacement. It lives in the mixed space alongside velocity and
    # pressure because the surface load and the flow are solved together --
    # explicit coupling of a free surface to Stokes is notoriously unstable at
    # timesteps anywhere near the ones we want.
    Z = MixedFunctionSpace([V, W, W] if free_surface else [V, W])

    z = Function(Z)
    u_, p_ = split(z)[:2]
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")
    u_fn = z.subfunctions[0]

    X = SpatialCoordinate(mesh)
    depth_nd = 1.0 - X[1]                     # 0 at surface, 1 at base

    # ---- three materials from two level sets ---------------------------
    # psi_uc: 1 above the upper/lower-crust interface (20 km depth)
    # psi_lc: 1 above the crust/mantle interface       (40 km depth)
    # CRUSTAL THICKNESS is the experiment's control variable, and the basal
    # temperature is deliberately NOT. Pinning T at the base of the column to the
    # mantle adiabat and letting the surface heat flow follow keeps every case on
    # the same planet; specifying q0 instead lets the mantle potential
    # temperature drift from 1060 to 1940 C across a sweep, which is not a
    # geotherm experiment.
    column = LI.scaled_column(crust_km)
    heat_flow = LI.heat_flow_for_base_temperature(t_base, column)
    y_uc = 1.0 - (crust_km / 2.0) * 1e3 / H
    y_lc = 1.0 - crust_km * 1e3 / H

    psi_uc, psi_lc = Function(K, name="psi_uc"), Function(K, name="psi_lc")
    epsilon = interface_thickness(K, min_cell_edge_length=True)
    assign_level_set_values(psi_uc, epsilon, X[1] - y_uc)
    assign_level_set_values(psi_lc, epsilon, X[1] - y_lc)

    # ORDERING MATTERS, and not in the obvious way. `material_field` recurses by
    # popping from the END of both lists, and `material_interface(ls, a, b)`
    # gives `a` where ls = 1. So the LAST level set pairs with the LAST value on
    # its 1-side, and the list must run from the DEEPEST interface to the
    # shallowest. Listing psi_uc first instead makes the outermost conditional
    # "upper-crust value wherever depth < 40 km", which silently swallows the
    # lower crust — a wrong model that runs perfectly happily.
    layers = [column[2], column[1], column[0]]   # deepest first
    ls = [psi_lc, psi_uc]           # deepest interface first

    def mat(attr, how="arithmetic"):
        return material_field(ls, [getattr(l, attr) for l in layers], interface=how)

    rho = mat("density")
    rho_ref_buoy = float(LI.MANTLE_LITHOSPHERE.density)   # Boussinesq reference
    coh = mat("cohesion")
    fric = mat("friction_deg")

    # ---- geotherm ------------------------------------------------------
    # Initialised from the analytic steady conductive profile in either case.
    # With `thermal=False` it stays there for the whole run; with `thermal=True`
    # an EnergySolver advects and diffuses it, and the crust generates its own
    # heat.
    zc = np.linspace(0.0, H / 1e3, 400)
    Tc = np.atleast_1d(LI.geotherm(zc, column=column,
                                   surface_heat_flow=heat_flow))
    Tfield = Function(Q, name="Temperature")
    Tfield.interpolate(Constant(0.0))
    Tfield.dat.data[:] = np.interp(
        (1.0 - Function(Q).interpolate(X[1]).dat.data_ro) * H / 1e3, zc, Tc)
    T_surface, T_base = float(Tc[0]), float(Tc[-1])

    # THERMAL SCALING.
    #
    # Temperature stays in kelvin; only length, velocity and time are scaled.
    # The energy equation
    #     dT/dt + u.grad T = kappa lap T + A / (rho cp)
    # then non-dimensionalises with lengths H and velocity U0 to give a
    # diffusivity kappa * / (U0 H) -- an inverse Peclet number -- and a source
    # (A / (rho cp)) * (H / U0), in kelvin per unit non-dimensional time.
    #
    #   kappa = k / (rho cp) = 2.5 / (3300 * 1000) = 7.6e-7 m2/s
    #   kappa' = 7.6e-7 / (7.92e-11 * 1e5) = 0.096,  so Pe = 10.4
    #
    # Advection and diffusion are within an order of magnitude of each other,
    # which is the physically interesting regime and the reason a prescribed
    # geotherm is a real approximation rather than a harmless one.
    #
    # Conductivity and heat capacity are taken uniform (k = 2.5 W/m/K,
    # cp = 1000 J/kg/K, rho_ref); only the radiogenic production is layered,
    # since that is what actually differs between crust and mantle here.
    CP = 1000.0
    kappa_dim = 2.5 / (rho_ref_buoy * CP)
    KAPPA_ND = kappa_dim / (U0 * H)
    heat_nd = material_field(
        ls, [l.heat_production / (rho_ref_buoy * CP) * (H / U0) for l in layers],
        interface="arithmetic")

    # ---- plastic strain, advected with a source ------------------------
    strain = Function(Q, name="PlasticStrain")
    strain.interpolate(Constant(0.0))
    # A small random seed in the centre, as the cookbook does: without it the
    # problem is translation-invariant and localises only on grid noise.
    xc = Function(Q).interpolate(X[0]).dat.data_ro
    yc = Function(Q).interpolate(X[1]).dat.data_ro
    rng = np.random.default_rng(0)
    # Seed width matters. 25 km half-width on a 200 km domain is not a seed,
    # it is a weak province occupying a quarter of the model — and a province
    # cannot localise, because there is no gradient for strain to concentrate
    # into. A real seed is a few km across.
    # SEEDING, and why the mode matters for a narrow-versus-wide experiment.
    #
    # `centre` puts a weak patch at the middle of the domain with strain already
    # at or past the weakening onset of 0.5. That is right for asking *how* a
    # rift localises once it has a nucleation site, which is T10's question.
    #
    # It is wrong for asking *whether* it localises to one site, which is T12's.
    # A wide rift is precisely the regime in which no single location dominates,
    # and a central weak patch guarantees that one does. The first sweep with
    # this seeding produced four narrow rifts at every crustal thickness from 20
    # to 50 km, with the deformation never reaching the side walls -- the answer
    # was imposed by the initial condition.
    #
    # `random` instead spreads low-amplitude noise through the whole crust. The
    # amplitude is deliberately below the weakening onset, so nothing starts
    # pre-weakened: the noise only breaks the translational symmetry, and any
    # localisation that appears has to be earned.
    if seed_mode == "random":
        # AMPLITUDE MATTERS MORE THAN IT LOOKS. The weakening law does nothing
        # below a plastic strain of 0.5, so noise capped under that threshold
        # leaves the feedback loop switched off: a first attempt with
        # uniform(0, 0.3) never localised at any crustal thickness, reaching a
        # peak strain of only 0.37 after 40 steps with W50 stuck near the
        # uniform value of 0.43. The noise has to STRADDLE the onset, so that
        # some material is already weakening and can compete to capture the
        # deformation. uniform(0, 1) puts about half the crust past it, which is
        # what ASPECT's continental_extension cookbook does.
        crust = yc > y_lc
        strain.dat.data[crust] = rng.uniform(0.0, seed_amp, crust.sum())
    else:
        sw = seed_halfwidth_km * 1e3 / H
        seed = ((np.abs(xc - aspect / 2) < sw) & (yc > y_lc))
        strain.dat.data[seed] = rng.uniform(0.5, 1.5, seed.sum())

    # ---- rheology ------------------------------------------------------
    # Visco-plastic Stokes does not converge under Newton from a cold start —
    # that is the whole point of G-ADOPT's Drucker-Prager demo. Two devices are
    # needed, and both are expressed here as functions of (u, p) so the same
    # algebra can be built twice: once for Newton, once with a LAGGED solution
    # for Picard.
    #
    #   `switch` = 0 : plastic branch off, linear viscosity. Used for the very
    #                  first solve, because at u = 0 the strain-rate invariant
    #                  is zero and the plastic viscosity divides by it.
    #   `switch` = 1 : the real rheology.
    switch = Constant(1.0)
    # LITHOSTATIC PRESSURE, and the trap in adding buoyancy later.
    #
    # Without buoyancy the Stokes body force is zero, so the pressure the solver
    # returns is purely dynamic and carries no lithostatic part. The
    # Drucker-Prager yield stress needs the total pressure, so the lithostatic
    # part is supplied analytically here and added to p.
    #
    # Turn buoyancy on for the free surface and that stops being true: the body
    # force is -(rho - rho_ref) g, so p now contains everything except the
    # REFERENCE lithostatic pressure rho_ref * g * z, which is absorbed into the
    # reference state. Keeping the layered `plith` as well double-counts, roughly
    # doubling the yield stress at depth. The symptom is unmistakable once you
    # know it: the model stops yielding, peak strain decays instead of growing,
    # and the rift quietly fails to form.
    # `fs_bug='lithostatic'` deliberately keeps the layered plith with buoyancy
    # on, reproducing the double-count for T11 to demonstrate.
    plith_rho = (rho_ref_buoy if (free_surface and fs_bug != "lithostatic")
                 else rho)
    plith = plith_rho * 9.81 * depth_nd * H / STRESS

    # Naliboff & Buiter linear weakening, in UFL. `strain` is updated between
    # timesteps, so it is lagged by construction and does not enter the
    # nonlinear solve.
    w = conditional(strain < 0.5, 1.0,
                    conditional(strain > 1.5, 0.25,
                                1.0 + (0.25 - 1.0) * (strain - 0.5) / 1.0))

    def creep(lay, epsii):
        n, A = lay.stress_exponent, lay.prefactor
        return (0.5 * A ** (-1.0 / n) * (epsii * SR) ** ((1.0 - n) / n)
                * exp(lay.activation_energy / (n * LI.R_GAS * Tfield))) / MU0

    def rheology(uu, pp):
        """(mu, mu_creep, mu_plast, epsii) for a given velocity/pressure pair."""
        e = sym(grad(uu))
        epsii = sqrt(0.5 * inner(e, e) + 1e-10)      # guards the cold start
        mu_c = material_field(ls, [creep(l, epsii) for l in layers],
                              interface="geometric")
        phi = fric * pi / 180.0 * w
        sigma_y = coh * w / STRESS * cos(phi) + (plith + pp) * sin(phi)
        # PLASTIC DAMPER, after Duretz et al. (2020) and used in ASPECT's
        # continental_extension cookbook ("Plastic damper viscosity = 1e21").
        # Without it the plastic branch can drive the viscosity arbitrarily low
        # wherever the strain rate is large; the linearised Picard problem then
        # has near-zero-viscosity regions where the velocity is essentially
        # unconstrained, and the iteration converges happily to |u| ~ 1e4 times
        # the boundary velocity. Adding a damper in series puts a floor under
        # the plastic viscosity that scales with the physics rather than being
        # an arbitrary clip.
        mu_damp = damper / MU0
        mu_p = sigma_y / (2 * epsii) + mu_damp

        # switch = 0 disables the plastic branch entirely
        mu_eff = conditional(switch > 0.5, min_value(mu_c, mu_p), mu_c)
        # Viscosity contrast capped at 1e6. The cookbook's 1e18-1e26 range is
        # 1e8, which is solvable in ASPECT's SI formulation but not here.
        return (max_value(min_value(mu_eff, 1e26 / MU0), 1e20 / MU0),
                mu_c, mu_p, epsii)

    mu, mu_creep, mu_plast, epsii = rheology(u_, p_)

    # ---- solvers -------------------------------------------------------
    dt = Constant(dt_max)
    bcs = {boundary.left: {"ux": -1}, boundary.right: {"ux": 1},
           boundary.bottom: {"uy": 0}}          # top is stress-free

    # BUOYANCY, and why it only appears with the free surface.
    #
    # Without a free surface this model has none at all: it is driven purely
    # kinematically, the layered density does nothing, and the flat lid supplies
    # whatever normal stress it likes. A free surface cannot work that way --
    # there has to be something for the topography to push back against, and
    # that something is the weight of the rock.
    #
    # The equations are non-dimensionalised by the Spiegelman convention
    # (lengths by H, velocities by U0, stress by mu0*U0/H), so a dimensional
    # body force rho*g becomes rho * g * H^2 / (mu0 * U0). That group is BUOY
    # below; it is about 0.124 per kg/m3, so a mantle density of 3300 gives a
    # non-dimensional body force of ~409. Equivalently: a kilometre of
    # topography weighs about 26 MPa against a driving stress of mu0*U0/H ~
    # 8 MPa, which is why the surface matters at all.
    BUOY = 9.81 * H**2 / (MU0 * U0)
    def approx(viscosity):
        if not free_surface:
            return BoussinesqApproximation(0, mu=viscosity)
        return BoussinesqApproximation(
            0, mu=viscosity, g=1.0, RaB=BUOY, delta_rho=rho - rho_ref_buoy)

    if free_surface:
        # delta_rho_fs is the contrast across the surface: rock against nothing.
        bcs[boundary.top] = {"free_surface": {"RaFS": BUOY, "delta_rho_fs": rho}}

    kw = dict(bcs=bcs)
    if free_surface:
        kw["dt"] = dt                    # the free-surface balance needs it

    stokes = StokesSolver(z, approx(mu), **kw)

    # Picard: identical problem, but the viscosity is evaluated at the PREVIOUS
    # iterate, making each solve linear. Slow but robust, where Newton is fast
    # but only converges once it is already close.
    z_pic = Function(Z)
    u_pic, p_pic = split(z_pic)[:2]
    mu_pic = rheology(u_pic, p_pic)[0]
    picard = StokesSolver(z, approx(mu_pic), **kw)

    # Reinitialisation pseudo-timestep MUST scale with the interface thickness.
    # G-ADOPT's default is a fixed 0.02, while `interface_thickness` returns
    # ~0.35 * h_min — so the default is stable on a coarse mesh and unstable on
    # a fine one. At 32x16 epsilon ~ 0.022 and it works; at 96x48 epsilon ~
    # 0.0073 and reinitialisation returns DIVERGED_FUNCTION_NANORINF. Tying the
    # step to epsilon makes the model resolution-independent, which is the
    # difference between a demo and something you can refine.
    eps_min = float(epsilon.dat.data_ro.min())
    reini = {"epsilon": epsilon, "timestep": reini_factor * eps_min,
             "steps": reini_steps}
    ls_solver = [LevelSetSolver(psi, adv_kwargs={"u": u_fn, "timestep": dt},
                                reini_kwargs=reini)
                 for psi in ls]

    # Energy: advection + diffusion + radiogenic heating, with the temperature
    # pinned at both ends. The basal value is the analytic geotherm's own base,
    # so with u = 0 the initial condition is already the steady state and the
    # field should not move -- which is the verification test in `check_thermal`.
    energy = None
    if thermal:
        thermal_approx = BoussinesqApproximation(
            0, rho=1.0, kappa=KAPPA_ND, H=heat_nd)
        energy = EnergySolver(
            Tfield, u_fn, thermal_approx, dt, DIRK33,
            bcs={boundary.top: {"T": T_surface},
                 boundary.bottom: {"T": T_base}},
            su_advection=True,
        )

    # Plastic strain: advected, with a source equal to the plastic strain rate
    # where the plastic branch governs. This is the feedback that makes a fault
    # keep slipping once it has formed.
    yielding = conditional(mu_plast < mu_creep, 1.0, 0.0)
    strain_solver = GenericTransportSolver(
        ["advection", "mass", "source"], strain, dt, DIRK33,
        eq_attrs={"u": u_fn, "source": yielding * epsii},
        su_advection=True,
    )

    return dict(mesh=mesh, z=z, u=u_fn, dt=dt, stokes=stokes, ls=ls,
                ls_solver=ls_solver, strain=strain, strain_solver=strain_solver,
                mu=mu, epsii=epsii, Q=Q, K=K, w=w, aspect=aspect, rho=rho,
                mu_creep=mu_creep, mu_plast=mu_plast, X=X, H=H,
                # Read the true minimum cell width from the mesh. On a graded
                # mesh `aspect / nx` is the *average*, not the minimum, so the
                # CFL limiter would be using a timestep several times too large
                # for the smallest cells while reporting a comfortable Courant
                # number.
                fs_bug=fs_bug, thermal=thermal, energy=energy,
                heat_flow=heat_flow, crust_km=crust_km, column=column,
                dt_max=dt_max,
                strain_initial=Function(Q).assign(strain),
                Tfield=Tfield, T_surface=T_surface, T_base=T_base,
                h_min=float(np.diff(np.unique(np.round(
                    mesh.coordinates.dat.data[:, 0], 9))).min()),
                free_surface=free_surface,
                z_old=Function(Z),
                picard=picard, z_pic=z_pic, switch=switch)


def check_layering(m):
    """Assert the three materials land at the depths they should.

    Density is the cleanest probe: 2700 / 2900 / 3300 kg/m3 are far apart, so a
    mis-ordered `material_field` shows up immediately. This is cheap and it
    catches the one bug in this script that would otherwise be invisible.
    """
    Q = m["Q"]
    rho_fn = Function(Q).interpolate(m["rho"])
    y = Function(Q).interpolate(m["X"][1]).dat.data_ro
    depth_km = (1.0 - y) * H / 1e3
    rho = rho_fn.dat.data_ro

    out = {}
    ck = m["crust_km"]
    for label, lo, hi, expect in (
            ("upper crust", 0.1 * ck, 0.4 * ck, 2700.0),
            ("lower crust", 0.6 * ck, 0.9 * ck, 2900.0),
            ("mantle lithosphere", ck + 5, 95.0, 3300.0)):
        sel = (depth_km > lo) & (depth_km < hi)
        got = float(rho[sel].mean())
        out[label] = dict(expected=expect, got=round(got, 1),
                          ok=abs(got - expect) < 25.0)
    return out


class AdvectionBroke(RuntimeError):
    """Raised when a conservative level set leaves [0, 1].

    This is not a warning. A conservative level set is a smoothed indicator
    function: its value IS the volume fraction of the material above the
    interface, so it is in [0, 1] by definition. Once it is not, every field
    derived from it is meaningless, because `material_field` blends layer
    properties by linear interpolation in psi -- feed it psi = 2.7 and it
    happily extrapolates a density no rock has.

    The 400-step production run of 6 August ended with level sets spanning
    [-1.72, 3.02] and a density field with mantle at the surface and crust at
    90 km depth. That run cost 2 h 22 min and its output was uninterpretable.
    The range was already being measured -- and only printed, at the end. Hence
    this exception, and the per-step check that raises it.
    """


class MaterialNotConserved(RuntimeError):
    """Raised when the volume of a material changes during the run.

    For incompressible flow and a conservative level set, ``\\int psi dx`` is
    the volume of the material above that interface and is exactly conserved.
    Drift in it means the advection is creating or destroying material, which is
    the failure mode that ruined the first production run.

    This replaces an earlier per-step check that compared mean density inside
    *fixed depth windows* against 2700 / 2900 / 3300. That check was wrong by
    construction, and instructively so: in a model that extends, the layers
    move, so a window fixed at 22-38 km starts sampling upper crust and the mean
    density falls whether or not anything has gone wrong. It duly failed at step
    39 -- at the same step and to the same digit -- both with the old
    reinitialisation settings and with settings that reduced the level-set drift
    tenfold, which is what gave it away. Volume conservation is invariant under
    deformation; a depth window is not.
    """


class PicardDiverged(RuntimeError):
    """Raised when the Picard iteration is going backwards.

    This exists because the alternative is worse. Picard can diverge silently:
    it returns a velocity field, the code carries on, and the first sign of
    trouble is the level-set advection exploding several steps later with an
    error that points at the wrong component entirely. Ask me how I know.
    """


def solve_stokes(m, picard_iters, tol=1e-4, cold=False, u_sane=1e3):
    """Robust visco-plastic Stokes solve: isoviscous -> Picard -> Newton.

    Returns (picard_iterations_used, newton_converged, residual_history).
    """
    # THE FREE SURFACE AND THE PICARD LOOP DO NOT COMPOSE BY DEFAULT.
    #
    # G-ADOPT's `StokesSolver.solve()` ends with
    #     self.solution_old.assign(self.solution)
    # which is exactly right when one solve is one timestep. Our Picard loop
    # calls solve() forty times per timestep, and the free-surface equation is
    # (eta - eta_old)/dt = u.n -- so eta was being advanced once per ITERATION.
    # Measured: 10 steps produced 80 km of subsidence instead of 2, a factor of
    # 40, which is the iteration count.
    #
    # Every Picard iterate must solve the same time-discrete problem, so
    # `solution_old` is reset to the start-of-step state before each solve. The
    # two solver objects share `z` but each keeps its own `solution_old`, so both
    # need it.
    fs = m.get("free_surface", False)
    # `fs_bug='time-level'` skips the rewind, so the free-surface equation
    # advances once per Picard iteration instead of once per timestep. T11 runs
    # it deliberately to show the factor-of-40 error it produces.
    if fs and m.get("fs_bug") == "time-level":
        fs = False
    if fs:
        z_old = m["z_old"]

        def rewind():
            m["picard"].solution_old.assign(z_old)
            m["stokes"].solution_old.assign(z_old)
    else:
        def rewind():
            pass

    if cold:
        m["switch"].assign(0.0)          # linear viscosity; breaks the 1/epsii
        rewind()
        m["picard"].solve()              # singularity at u = 0
        m["switch"].assign(1.0)

    # Damped (under-relaxed) Picard. Undamped, this diverges above ~48x24:
    # each iteration overshoots, the viscosity evaluated at the overshoot is
    # worse, and the next overshoot is bigger. Taking only a fraction `omega`
    # of each update tames that, at the cost of more iterations.
    #
    # `omega` adapts: halve it whenever the update grows, and creep back up
    # while it shrinks. Fixed damping either converges slowly everywhere or not
    # at all where the rheology is stiffest.
    # Picard on this system OSCILLATES rather than converging monotonically:
    # the residual falls, rises, and falls again. An earlier version of this
    # code ran a fixed 40 iterations and reported success purely because it
    # happened to stop on a downswing — the same run stopped at iteration 19
    # would have looked like divergence.
    #
    # So: keep the BEST iterate seen, restore it at the end, and only call it
    # divergence if a long window passes with no improvement. This is the
    # standard safeguard for a non-monotone fixed-point iteration, and it makes
    # the outcome independent of where you happen to stop.
    used, hist = 0, []
    omega = m.get("omega0", 0.7)
    Zs = m["z"].function_space()
    z_prev, z_raw = Function(Zs), Function(Zs)
    z_best = Function(Zs)
    best_du, best_at = float("inf"), -1

    for i in range(picard_iters):
        z_prev.assign(m["z"])
        m["z_pic"].assign(m["z"])
        rewind()
        m["picard"].solve()
        z_raw.assign(m["z"])            # raw fixed-point image, kept separate

        # Measure the UNDAMPED update. This is the residual of the fixed-point
        # map and is independent of omega — essential, because omega changes
        # between iterations, so a damped step norm is not comparable with the
        # previous one and would corrupt both the convergence test and the
        # adaptation below.
        # RELATIVE residual. An absolute one is meaningless when the solution
        # magnitude is itself in question: at |u| ~ 5e4 an absolute update of
        # 1e-3 reads as converged to eight digits, while the answer is wrong by
        # four orders of magnitude.
        unorm = max(float(norm(split(z_raw)[0])), 1e-12)
        du = float(norm(split(z_raw)[0] - split(z_prev)[0])) / unorm
        hist.append(du)
        used += 1

        if du < best_du:
            best_du, best_at = du, i
            z_best.assign(z_raw)

        # Relax through an explicit temporary. Writing
        #   z.assign(z_prev + omega*(z - z_prev))
        # puts z on both sides of its own assignment, which aliases.
        m["z"].assign(z_prev + omega * (z_raw - z_prev))

        if du < tol:
            break

        # Adapt omega GENTLY, and with a floor high enough to still move.
        #
        # The previous rule halved omega on any increase in the residual, with a
        # floor of 0.05. Because this iteration is genuinely non-monotone, the
        # residual rises most steps, so omega ratcheted to the floor within six
        # iterations and stayed there for the rest of the run — measured, in the
        # 45-iteration trace: omega hit 0.05 at iteration 6 and never recovered
        # above 0.065. At that damping each iterate is a 5% step and the
        # iteration cannot go anywhere, so the residual plateaued at ~1e-3 and
        # the whole thing looked like stagnation. It was self-inflicted.
        #
        # Back off only on a substantial rise, back off less far, and never
        # below a fraction that can still make progress.
        if len(hist) > 1:
            if hist[-1] > 1.5 * hist[-2]:
                omega = max(0.25, 0.7 * omega)
            elif hist[-1] < 0.8 * hist[-2]:
                omega = min(1.0, 1.2 * omega)

    # KEEP THE LAST ITERATE, not the one with the smallest update.
    #
    # An earlier version restored the iterate that minimised `du`, on the
    # standard reasoning that a non-monotone fixed-point iteration should not be
    # judged by wherever you happened to stop. That reasoning is right and the
    # implementation was wrong, because `du` is the change BETWEEN successive
    # iterates, not a measure of nonlinear error, and it is smallest at
    # iteration 0 for a reason that has nothing to do with accuracy: the
    # isoviscous warm-up hands over a smooth field and the first plastic solve
    # barely moves it. Every strategy tried -- adaptive omega, fixed omega at
    # 0.3, 0.7 and 1.0, Anderson acceleration at depth 5 and 10 -- reported the
    # identical "best" of 3.006e-4, at iteration 0, every time.
    #
    # Judged by a measure that is independent of this bookkeeping -- the
    # relative divergence ||div u|| / ||grad u||, which is zero for any true
    # Stokes solution -- iteration 0 is the WORST iterate available:
    #
    #     iterate 0 (the old "best")            rel div 0.52
    #     iterate 40 (simply the last)          rel div 0.07 - 0.12
    #     converged instantaneous case, same nx rel div 0.09
    #
    # So the safeguard was selecting a field five times less incompressible than
    # the one it discarded, and that field is what advected the level sets
    # through every production run.
    hist.append(du)

    # The boundary velocity is 1 by construction, so a converged solution is
    # O(1-10). Anything vastly larger is not a solution, whatever the solver
    # reported. `z_best` is still the fallback for that case -- not because it
    # is accurate, but because it is bounded.
    umax = float(np.abs(m["u"].dat.data_ro).max())
    if not np.isfinite(umax) or umax > u_sane:
        m["z"].assign(z_best)
        umax = float(np.abs(m["u"].dat.data_ro).max())
        if not np.isfinite(umax) or umax > u_sane:
            raise PicardDiverged(
                f"|u|max = {umax:.3e}, but the boundary velocity is 1 — "
                "the Stokes solution is not physical")

    # A FAILED Newton solve does not leave the Picard result in place. PETSc has
    # already written its last, diverged iterate into `z` by the time the
    # exception is raised, so the old comment here -- "Picard result stands" --
    # was wrong, and the |u|max of 5.7 and rel div of 0.52 seen in the
    # production runs came partly from keeping it. Snapshot before trying, and
    # roll back if it fails.
    z_pre_newton = Function(m["z"].function_space())
    z_pre_newton.assign(m["z"])
    try:
        rewind()
        m["stokes"].solve()              # Newton polish, now close enough
    except ConvergenceError:
        m["z"].assign(z_pre_newton)
        return used, False, hist
    # Even a "converged" Newton step is only worth keeping if it did not move
    # the answer somewhere unphysical.
    umax = float(np.abs(m["u"].dat.data_ro).max())
    if not np.isfinite(umax) or umax > u_sane:
        m["z"].assign(z_pre_newton)
        return used, False, hist
    return used, True, hist


def level_set_range(m):
    """(min, max) of every level set. The invariant is [0, 1]."""
    return [(float(p.dat.data_ro.min()), float(p.dat.data_ro.max()))
            for p in m["ls"]]


def material_volumes(m):
    """``\\int psi dx`` for each level set — the volume of material above it."""
    return [float(assemble(p * dx)) for p in m["ls"]]


def volume_drift(m):
    """Largest relative change in any material volume since t = 0."""
    v0 = m["v0"]
    return max(abs(v - r) / max(abs(r), 1e-300)
               for v, r in zip(material_volumes(m), v0))


def run(m, steps, ls_tol=0.05, vol_tol=None, strict=True):
    """Step: Stokes -> advect level sets -> advect plastic strain.

    Two invariants are checked *inside* the loop, because both of the things
    that went wrong on the first production run were invisible from outside it:

    * every level set stays within ``ls_tol`` of [0, 1];
    Material volume drift is *reported* every step but not enforced, because
    this domain is open at the sides — see :class:`MaterialNotConserved`.
    Pass ``vol_tol`` explicitly if you have a closed-box variant.

    With ``strict`` (the default) a violation raises and the run stops there,
    which is the point -- a broken run should cost you a minute, not an
    afternoon. Pass ``strict=False`` and a large ``ls_tol`` to survey how a
    quantity degrades rather than to stop at the first sign of it.
    """
    # Kept on `m` rather than local, so the caller still has the per-step record
    # when one of the invariant checks below raises out of this loop.
    hist = m.setdefault("_hist", [])
    m.setdefault("v0", material_volumes(m))
    # 30 was enough at 64x32, where Picard reaches 9e-5 in 15-21 iterations. At
    # 96x48 it was still at 1.7e-4 after 30 -- close to the 1e-4 tolerance but
    # not there, so it burned the full cap every step. A slightly larger budget
    # costs nothing when the iteration converges early and buys convergence when
    # it does not.
    m.setdefault("picard_cap", 40)
    t0 = time.perf_counter()
    broke_at = None
    for n in range(steps):
        pic, newton_ok, res = solve_stokes(
            m, picard_iters=4 * m["picard_cap"] if n == 0 else m["picard_cap"],
            cold=(n == 0))

        if m.get("free_surface", False):
            m["z_old"].assign(m["z"])    # one time level per STEP, not per solve

        # CFL-limit the timestep from the ACTUAL velocity, not a guess. A fixed
        # timestep is a resolution-dependent bug waiting to happen: halve the
        # mesh and the Courant number doubles.
        umax = max(float(np.abs(m["u"].dat.data_ro).max()), 1e-12)
        dt_cfl = 0.4 * m["h_min"] / umax
        m["dt"].assign(min(m["dt_max"], dt_cfl))

        for s in m["ls_solver"]:
            s.solve()
        m["strain_solver"].solve()
        if m.get("energy") is not None:
            m["energy"].solve()

        lsr = level_set_range(m)
        lo = min(r[0] for r in lsr)
        hi = max(r[1] for r in lsr)
        excursion = max(-lo, hi - 1.0, 0.0)
        vdrift = volume_drift(m)

        sd = m["strain"].dat.data_ro
        rec = dict(step=n, picard=pic, newton=newton_ok,
                   strain_max=float(sd.max()),
                   strain_mean=float(sd.mean()),
                   weak_fraction=float((sd > 1.5).mean()),
                   ls_min=round(lo, 5), ls_max=round(hi, 5),
                   ls_excursion=round(excursion, 5),
                   volume_drift=round(vdrift, 6),
                   courant=round(umax * float(m["dt"]) / m["h_min"], 4))
        print(f"  step {n:3d}  picard {pic:3d} (res {res[-1]:.2e})  "
              f"newton {'ok' if newton_ok else 'FAILED'}  |u|max {umax:.2f}  "
              f"dt {float(m['dt']):.2e}  Co {rec['courant']:.3f}  "
              f"psi [{lo:+.4f}, {hi:+.4f}]  dV {vdrift:.2e}  "
              f"strain max {sd.max():.3f}", flush=True)

        if excursion > ls_tol and broke_at is None:
            broke_at = n
            print(f"  ** level set left [0, 1] by {excursion:.4f} at step {n} **",
                  flush=True)
            if strict:
                hist.append(rec)
                raise AdvectionBroke(
                    f"level set range [{lo:.4f}, {hi:.4f}] at step {n}: the "
                    f"conservative level set must stay in [0, 1]. Everything "
                    f"downstream of the material field is now meaningless.")

        if vol_tol is not None and vdrift > vol_tol:
            print(f"  ** material volume drifted {vdrift:.3%} at step {n} **",
                  flush=True)
            if strict:
                hist.append(rec)
                raise MaterialNotConserved(
                    f"material volume changed by {vdrift:.3%} at step {n}; "
                    f"incompressible flow conserves it exactly.")

        hist.append(rec)
    return hist, time.perf_counter() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="/tmp/rift.npz")
    ap.add_argument("--dt-max", type=float, default=2e-3,
                    help="cap on the non-dimensional timestep. THIS, not the "
                         "extension rate, is what buys finite strain: total "
                         "stretching is steps * dt, and the rate cancels out of "
                         "the non-dimensionalisation entirely. CFL usually "
                         "permits several times the historical 2e-3 cap")
    ap.add_argument("--rate-cm-yr", type=float, default=0.25,
                    help="half-rate on each wall in cm/yr; rescales stress, "
                         "strain rate, time, buoyancy and Peclet together")
    ap.add_argument("--seed-amp", type=float, default=1.0,
                    help="upper bound of the random initial plastic strain. "
                         "Must straddle the weakening onset of 0.5 or nothing "
                         "localises")
    ap.add_argument("--seed-mode", choices=["centre", "random"], default="centre",
                    help="centre: one weak patch at the axis (T10). random: "
                         "low-amplitude noise through the crust, so localisation "
                         "has to be earned (T12)")
    ap.add_argument("--crust-km", type=float, default=40.0,
                    help="total crustal thickness; thin = cold Moho and a "
                         "coupled, strong column, thick = hot Moho and a weak "
                         "lower crust. The surface heat flow follows from this "
                         "and --t-base rather than being set independently")
    ap.add_argument("--t-base", type=float, default=1613.0,
                    help="temperature at the base of the column, K (the mantle "
                         "adiabat). Held fixed across a sweep")
    ap.add_argument("--seed-km", type=float, default=25.0,
                    help="seed half-width in km")
    ap.add_argument("--damper", type=float, default=1e21,
                    help="plastic damper viscosity, Pa s (cookbook: 1e21)")
    ap.add_argument("--thermal", action="store_true",
                    help="solve the energy equation instead of holding the "
                         "geotherm fixed")
    ap.add_argument("--fs-bug", choices=["none", "time-level", "lithostatic"],
                    default="none",
                    help="reintroduce a free-surface bug on purpose, for teaching")
    ap.add_argument("--cluster-x", type=float, default=0.0,
                    help="mesh clustering towards the rift axis (0 = uniform, "
                         "1 gives a cell-size ratio of about 2.4)")
    ap.add_argument("--cluster-y", type=float, default=0.0,
                    help="mesh clustering towards the surface (0 = uniform)")
    ap.add_argument("--free-surface", action="store_true",
                    help="let the top boundary deform instead of holding it flat. "
                         "Turns on buoyancy, which the kinematic model does not have")
    ap.add_argument("--picard-iters", type=int, default=40,
                    help="Picard iterations per step (the first step gets 4x)")
    ap.add_argument("--reini-steps", type=int, default=12,
                    help="level-set reinitialisation steps per timestep. The "
                         "old default of 2 let psi drift out of [0, 1]; see the "
                         "measurement in the module docstring")
    ap.add_argument("--reini-factor", type=float, default=0.5,
                    help="reinitialisation pseudo-timestep, in units of epsilon")
    ap.add_argument("--ls-tol", type=float, default=0.05,
                    help="how far a level set may stray outside [0, 1] before "
                         "the run is abandoned")
    ap.add_argument("--vol-tol", type=float, default=None,
                    help="abort if a material's volume drifts by more than this. "
                         "Off by default: the domain is open at the sides, so "
                         "volume is not conserved for legitimate reasons")
    ap.add_argument("--no-strict", action="store_true",
                    help="log invariant violations but keep going; for surveying "
                         "how a quantity degrades, not for production")
    ap.add_argument("--history", default=None,
                    help="write the per-step record to this JSON file")
    args = ap.parse_args()

    set_rate(args.rate_cm_yr)          # before build(): it reads the scales

    m = build(args.nx, args.ny, damper=args.damper,
              seed_halfwidth_km=args.seed_km,
              crust_km=args.crust_km, t_base=args.t_base,
              seed_mode=args.seed_mode, seed_amp=args.seed_amp,
              dt_max=args.dt_max,
              reini_steps=args.reini_steps,
              reini_factor=args.reini_factor,
              free_surface=args.free_surface,
              cluster_x=args.cluster_x, cluster_y=args.cluster_y,
              fs_bug=args.fs_bug, thermal=args.thermal)
    m["picard_cap"] = args.picard_iters

    layering = check_layering(m)
    for k, v in layering.items():
        print(f"  {k:20s} expected {v['expected']:.0f}  got {v['got']:7.1f}  "
              f"{'OK' if v['ok'] else 'MISMATCH'}", flush=True)
    if not all(v["ok"] for v in layering.values()):
        raise SystemExit("layer ordering wrong — check the material_field list order")
    # This is an INITIALISATION check only. It compares mean density in fixed
    # depth windows, which is exactly right for catching a mis-ordered
    # `material_field` list at t = 0 and exactly wrong as a runtime invariant,
    # because a model that extends moves its layers out of those windows. The
    # runtime invariants are in `run()`.

    # Run, but always write the output — the state at the moment an invariant
    # broke is the most informative snapshot there is, and throwing it away
    # would mean re-running to see it.
    failure = None
    t0 = time.perf_counter()
    try:
        hist, secs = run(m, args.steps, ls_tol=args.ls_tol,
                         vol_tol=args.vol_tol, strict=not args.no_strict)
    except (AdvectionBroke, MaterialNotConserved, PicardDiverged) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        hist, secs = m.setdefault("_hist", []), time.perf_counter() - t0
        print(f"\nRUN ABANDONED — {failure}\n"
              "Saving the state at failure anyway; the fields below are the "
              "last ones computed.", flush=True)

    ls_range = level_set_range(m)

    # Sample onto a REGULAR grid. Saving raw dof arrays is useless for
    # plotting: CG2 dofs sit at vertices, edge midpoints and cell interiors in
    # an ordering that is not a reshape of the mesh, so there is no way to
    # recover a picture from them afterwards.
    Q, aspect = m["Q"], m["aspect"]
    nxs, nys = 4 * args.nx + 1, 4 * args.ny + 1
    xs = np.linspace(0.0, aspect, nxs)
    ys = np.linspace(0.0, 1.0, nys)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Export through CG1, not CG2. Viscosity and strain rate have near-
    # discontinuous transitions where the plastic branch takes over, and a
    # quadratic basis OVERSHOOTS at those jumps: interpolating into CG2 gave
    # 487 points of NEGATIVE viscosity (down to -4e24) and 313 of negative
    # strain rate, from a UFL expression bounded below at 1e20 and by a square
    # root respectively. The model was fine; the diagnostic was lying.
    # CG1 cannot overshoot a monotone jump, and positive fields are clipped as
    # a second line of defence.
    P1 = FunctionSpace(m["mesh"], "CG", 1)
    ev = PointEvaluator(m["mesh"], pts, tolerance=1e-8)   # `Function.at` is deprecated

    def grid(expr, positive=False):
        f = Function(P1).interpolate(expr)
        g = np.asarray(ev.evaluate(f)).reshape(nys, nxs)[::-1]
        return np.maximum(g, 0.0) if positive else g

    out = dict(
        x_km=xs * H / 1e3,
        depth_km=(1.0 - ys[::-1]) * H / 1e3,
        strain=grid(m["strain"], positive=True),
        viscosity=grid(m["mu"], positive=True) * MU0,
        strain_rate=grid(m["epsii"], positive=True) * SR,
        density=grid(m["rho"], positive=True),
        weakening=grid(m["w"], positive=True),
        temperature=grid(m["Tfield"]),
    )
    # The seed is the initial condition, so W50 needs the initial strain on the
    # same grid to difference against. Recomputed here from the stored copy.
    m["strain_initial_grid"] = np.asarray(
        ev.evaluate(Function(P1).interpolate(m["strain_initial"]))
    ).reshape(nys, nxs)[::-1]

    uu = np.asarray(ev.evaluate(m["u"])).reshape(nys, nxs, 2)[::-1]
    out["vx"], out["vz"] = uu[..., 0], -uu[..., 1]

    if m["free_surface"]:
        # eta is the surface displacement, carried as a third component of the
        # mixed space. It is defined over the whole domain but pinned to zero in
        # the interior, so only its trace on the top boundary means anything.
        # Non-dimensionalised by H, so multiply by H for metres.
        eta = np.asarray(ev.evaluate(m["z"].subfunctions[2])).reshape(nys, nxs)
        out["topography_km"] = eta[-1] * H / 1e3      # top row, +ve = uplift
        out["eta_field"] = eta[::-1]
    np.savez(args.out, **out)

    # Diagnostics.
    #
    # There is deliberately NO localisation ratio here any more. A max/mean or
    # in-seed/outside ratio was reported for several rounds of this model and
    # misled every time, because it is maximised by a run in which nothing
    # deforms outside the seed -- including runs where nothing deforms at all,
    # and the run of 6 August, where the "strain" being ratioed was an artefact
    # of a corrupted material field. Absolute in-seed and outside strains are
    # reported instead; judge them together, and only after `layering_ok` and
    # `level_set_range` say the fields mean anything.
    xk, zk, st = out["x_km"], out["depth_km"], out["strain"]
    inseed = np.abs(xk[None, :] - 100.0) < args.seed_km
    crust = (zk[:, None] > 5.0) & (zk[:, None] < 40.0)
    sel = np.broadcast_to(inseed, st.shape) & np.broadcast_to(crust, st.shape)
    out_sel = (~np.broadcast_to(inseed, st.shape)) & np.broadcast_to(crust, st.shape)
    s_in, s_out = float(st[sel].mean()), float(st[out_sel].mean())

    # W50 -- the fraction of the domain width carrying half the NEW strain.
    #
    # Narrow versus wide is a statement about the WIDTH of the deforming zone,
    # so the metric has to measure a distribution rather than an amount. Take
    # the column-integrated increment in plastic strain, normalise it to a
    # probability, sort descending, and ask what fraction of the domain is
    # needed to reach half the total.
    #
    #   all deformation in one column   ->  W50 -> 0      (narrow)
    #   deformation spread uniformly    ->  W50 = 0.5     (wide)
    #
    # Two properties earn it a place after three metrics that misled here. It is
    # computed on the INCREMENT, so it reports where deformation happened during
    # the run rather than where the seed was placed. And a run in which nothing
    # deforms cannot score as narrow -- a flat field is maximally spread and
    # gives 0.5, while an exactly zero field returns None rather than a number.
    # Absolute strain is reported beside it, because a distribution says nothing
    # about magnitude.
    def w50(field):
        col = np.maximum(field, 0.0).sum(axis=0)
        tot = col.sum()
        if tot <= 0:
            return None
        p = np.sort(col / tot)[::-1]
        return round(float((np.searchsorted(np.cumsum(p), 0.5) + 1) / p.size), 4)

    increment = out["strain"] - m["strain_initial_grid"]

    excursion = max(max(-lo for lo, _ in ls_range),
                    max(hi - 1.0 for _, hi in ls_range), 0.0)

    if args.history:
        with open(args.history, "w") as fh:
            json.dump(hist, fh, indent=1)
        print(f"wrote per-step history to {args.history}", flush=True)

    print("RESULT " + json.dumps(dict(
        failure=failure,
        steps_completed=len(hist), steps_requested=args.steps,
        damper=args.damper, seed_km=args.seed_km,
        crust_km=args.crust_km, t_base=args.t_base,
        seed_mode=args.seed_mode, seed_amp=args.seed_amp,
        rate_cm_yr=RATE_CM_YR, dt_max=args.dt_max,
        stretch_percent=round(100 * args.steps * args.dt_max * 2.0 / 2.0, 2),
        surface_heat_flow=round(float(m["heat_flow"]), 5),
        reini_steps=args.reini_steps, reini_factor=args.reini_factor,
        free_surface=args.free_surface,
        cluster_x=args.cluster_x, cluster_y=args.cluster_y,
        fs_bug=args.fs_bug, thermal=args.thermal,
        nx=args.nx, ny=args.ny, seconds=round(secs, 2),
        # Trust nothing below this line unless both of these are clean.
        level_set_range=ls_range,
        level_set_excursion=round(excursion, 5),
        volume_drift=round(volume_drift(m), 6),
        topography_km=(None if not m["free_surface"] else
                       [round(float(out["topography_km"].min()), 4),
                        round(float(out["topography_km"].max()), 4)]),
        w50=w50(increment),
        w50_all=w50(out["strain"]),
        strain_increment_total=round(float(np.maximum(increment, 0).sum()), 1),
        strain_in_seed=round(s_in, 4),
        strain_outside=round(s_out, 4),
        strain_max_initial=round(hist[0]["strain_max"], 3) if hist else None,
        strain_max_final=round(hist[-1]["strain_max"], 3) if hist else None,
        strain_mean_final=round(hist[-1]["strain_mean"], 5) if hist else None,
        weak_fraction_final=round(hist[-1]["weak_fraction"], 5) if hist else None,
        output=args.out,
    )), flush=True)
    if failure:
        raise SystemExit(1)
