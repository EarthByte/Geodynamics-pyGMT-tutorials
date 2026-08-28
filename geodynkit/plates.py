"""
Surface velocities from a plate reconstruction, sampled along a great circle.

A 2-D cylindrical annulus is a cross-section through the Earth, so driving one
with plate motions means sampling a plate model along a **great circle** and
resolving the surface velocity onto the tangent of that circle. That is what
this module does.

Why not use G-ADOPT's `GplatesVelocityFunction` directly? Because it assumes a
three-dimensional spherical shell: it seeds a Fibonacci sphere, works in
(latitude, longitude) from 3-D Cartesian coordinates, and hands velocities to a
2-D spherical *surface*. An annulus has one angular coordinate, not two, and the
component of plate motion perpendicular to the section has nowhere to go. The
choice of what to do with it is a modelling decision, not a detail, and it
belongs in the open:

**Only the in-plane component is kept.** Motion perpendicular to the section is
discarded rather than projected or averaged. A cross-section through a plate
moving obliquely to it therefore shows less convergence than the plate really
has. Choose the great circle so the tectonics you care about is roughly in-plane,
and read any comparison with the 3-D case in that light.

Requires `pygplates`, and a reconstruction (rotation model plus topological plate
boundaries). Both are in the project container; see `PUBLISHING.md`.
"""

import numpy as np

__all__ = ["great_circle_latlon", "surface_velocity_profile",
           "nondimensionalise_velocity"]

#: G-ADOPT's mantle-convection convention: velocity scale is kappa / L.
NONDIM_LENGTH_M = 2890e3
NONDIM_KAPPA = 1e-6
SECONDS_PER_YEAR = 365 * 24 * 60 * 60


def great_circle_latlon(theta_deg, pole_lat=90.0, pole_lon=0.0, origin_lon=0.0):
    """Latitude and longitude of points on a great circle, at angles ``theta_deg``.

    The circle is the one whose pole is at (``pole_lat``, ``pole_lon``); the
    default pole at the north pole gives the **equator**, with ``theta_deg``
    measured as longitude east from ``origin_lon``.

    Returns ``(lat, lon, tangent)`` — degrees, degrees, and the unit tangent to
    the circle at each point as a 3-vector, which is what the velocity has to be
    projected onto.
    """
    th = np.radians(np.asarray(theta_deg, dtype=float) + origin_lon)
    plat, plon = np.radians(pole_lat), np.radians(pole_lon)

    # Orthonormal frame: p is the pole, (a, b) span the plane of the circle.
    p = np.array([np.cos(plat) * np.cos(plon),
                  np.cos(plat) * np.sin(plon),
                  np.sin(plat)])
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed, p)) > 0.99:                 # pole near the z axis
        seed = np.array([1.0, 0.0, 0.0])
    a = seed - np.dot(seed, p) * p
    a /= np.linalg.norm(a)
    b = np.cross(p, a)

    pts = np.cos(th)[:, None] * a + np.sin(th)[:, None] * b       # (n, 3)
    tang = -np.sin(th)[:, None] * a + np.cos(th)[:, None] * b     # d/dtheta
    lat = np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return lat, lon, tang


def surface_velocity_profile(rotation_files, topology_files, age_ma, theta_deg,
                             pole_lat=90.0, pole_lon=0.0, origin_lon=0.0,
                             delta_t=1.0):
    """In-plane surface velocity along a great circle, in cm/yr.

    Positive is in the direction of increasing ``theta_deg``.

    Returns a dict with the velocity, the plate id at each sample (useful for
    marking plate boundaries on a plot) and the sample coordinates.
    """
    import pygplates

    theta_deg = np.asarray(theta_deg, dtype=float)
    lat, lon, tang = great_circle_latlon(theta_deg, pole_lat, pole_lon, origin_lon)

    rotation_model = pygplates.RotationModel(list(rotation_files))
    resolved = []
    pygplates.resolve_topologies(list(topology_files), rotation_model, resolved,
                                 float(age_ma))
    partitioner = pygplates.PlatePartitioner(resolved, rotation_model)

    v_par = np.zeros(theta_deg.size)
    plate_ids = np.full(theta_deg.size, -1, dtype=int)

    for i in range(theta_deg.size):
        point = pygplates.PointOnSphere(float(lat[i]), float(lon[i]))
        found = partitioner.partition_point(point)
        if found is None:                       # a gap in the topologies
            continue
        pid = found.get_feature().get_reconstruction_plate_id()
        plate_ids[i] = pid
        stage = rotation_model.get_rotation(float(age_ma), pid,
                                            float(age_ma) + delta_t)
        vec = pygplates.calculate_velocities(
            [point], stage, delta_t, pygplates.VelocityUnits.cms_per_yr)[0]
        # pygplates returns the velocity in the global 3-D Cartesian frame, so
        # the in-plane component is a plain dot product with the tangent.
        v_par[i] = float(np.dot(np.asarray(vec.to_xyz()), tang[i]))

    return dict(theta_deg=theta_deg, lat=lat, lon=lon,
                velocity_cm_yr=v_par, plate_id=plate_ids)


def nondimensionalise_velocity(v_cm_yr, length_m=NONDIM_LENGTH_M,
                               kappa=NONDIM_KAPPA):
    """cm/yr to the non-dimensional velocity a Boussinesq annulus expects.

    G-ADOPT's convention is ``u* = u L / kappa`` with L = 2890 km and
    kappa = 1e-6 m^2/s, so 1 cm/yr maps to about 916. That number deserves a
    moment: the annulus at Ra = 1e5 convects with u_rms of order 200, so
    imposing Earth-like plate speeds of a few cm/yr makes the **surface** the
    dominant driver by more than an order of magnitude. That is not a bug — it
    is what "plate-driven mantle flow" means — but it does mean buoyancy in such
    a model is a passenger unless Ra is raised towards Earth's 1e7-1e8.
    """
    return (np.asarray(v_cm_yr, dtype=float) * 1e-2 / SECONDS_PER_YEAR
            * length_m / kappa)
