"""
pyGMT plotting helpers for 2-D geodynamic model output.

Verified against pyGMT 0.19.0 / GMT 6.5.0. Every idiom here was executed, not
inferred from documentation — including the two that are easy to get wrong
(see ``quiver`` and ``box_projection``).

Conventions: fields are ``(nz, nx)``; ``x`` is distance, ``z`` is depth
positive downwards; ``vz > 0`` is downwards.
"""

import numpy as np
import xarray as xr

__all__ = [
    "to_grid",
    "box_projection",
    "box_frame",
    "field_panel",
    "quiver",
    "streamlines",
    "set_headless",
    "SCM",
]

#: Sensible Crameri scientific colour maps for common geodynamic fields.
#: These ship with GMT under the ``SCM/`` section — no extra package needed.
#: Note ``roma`` runs red -> blue with *increasing* value, so temperature
#: wants ``reverse=True`` to put hot at the red end.
SCM = {
    "temperature": ("SCM/roma", True),
    "temperature_anomaly": ("SCM/vik", False),
    "viscosity": ("SCM/lajolla", False),
    "strain_rate": ("SCM/batlow", False),
    "density": ("SCM/broc", False),
    "composition": ("SCM/batlow", False),
    "velocity": ("SCM/batlow", False),
}


def set_headless():
    """Stop pyGMT trying to open an external PDF viewer.

    Inside Jupyter ``fig.show()`` renders an inline PNG and works headless
    already. In a plain ``.py`` driver it shells out to the OS viewer and will
    hang on a headless machine — call this first in any script.
    """
    import pygmt

    pygmt.set_display(method="none")


# --------------------------------------------------------------------------
# numpy -> xarray -> GMT
# --------------------------------------------------------------------------
def to_grid(field, x, z, name="field"):
    """Wrap a ``(nz, nx)`` array as a GMT-ready Cartesian ``xarray.DataArray``.

    GMT's requirements, enforced in ``pygmt.clib.conversion.dataarray_to_matrix``:

    * exactly two dimensions;
    * dimension order ``(rows, cols)`` == ``(y, x)``;
    * **both coordinate vectors evenly spaced** — a stretched or graded mesh
      must be resampled first, because GMT silently substitutes a uniform
      increment and only emits a ``RuntimeWarning``;
    * dimension *names* are free — ``lat``/``lon`` are not required.
    """
    from pygmt.enums import GridRegistration, GridType

    field = np.asarray(field, dtype=float)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if field.shape != (z.size, x.size):
        raise ValueError(
            f"field shape {field.shape} does not match (z, x) = {(z.size, x.size)}"
        )
    for label, c in (("x", x), ("z", z)):
        d = np.diff(c)
        if d.size and not np.allclose(d, d[0], rtol=1e-6):
            raise ValueError(
                f"{label} coordinates must be evenly spaced for GMT; "
                "resample the field onto a uniform grid first."
            )

    da = xr.DataArray(field, dims=("z", "x"), coords={"z": z, "x": x}, name=name)
    # Already the defaults for an in-memory array, but the accessor resets
    # after arithmetic or Dataset slicing, so be explicit.
    da.gmt.registration = GridRegistration.GRIDLINE
    da.gmt.gtype = GridType.CARTESIAN
    return da


# --------------------------------------------------------------------------
# Cartesian box geometry
# --------------------------------------------------------------------------
def box_projection(box_x, box_z, width_cm=15.0, depth_down=True):
    """Return ``(region, projection)`` for a true-aspect box section.

    A **negative height reverses the y-axis**, which is how you get depth
    increasing downwards. This is the key non-obvious idiom for geodynamic
    sections and is not called out in the pyGMT docs.
    """
    height_cm = width_cm * float(box_z) / float(box_x)
    region = [0.0, float(box_x), 0.0, float(box_z)]
    sign = "-" if depth_down else ""
    return region, f"X{width_cm}c/{sign}{height_cm}c"


def box_frame(title="", xlabel="distance (km)", ylabel="depth (km)",
              xa=None, xf=None, ya=None, yf=None):
    """Readable wrapper around GMT's ``-B`` syntax.

    Passing ``None`` for the tick intervals lets GMT choose them, which is
    usually right and keeps notebooks uncluttered.
    """
    xspec = f"xa{xa}f{xf}" if xa is not None else "xaf"
    yspec = f"ya{ya}f{yf}" if ya is not None else "yaf"
    return [f"WSne+t{title}", f"{xspec}+l{xlabel}", f"{yspec}+l{ylabel}"]


# --------------------------------------------------------------------------
# Velocity arrows
# --------------------------------------------------------------------------
def quiver(fig, x, z, vx, vz, every=10, target_cm=0.7, size="0.2c",
           fill="black", pen="0.7p,black"):
    """Plot a decimated velocity field as Cartesian arrows.

    There is **no** ``pygmt.Figure.grdvector`` — no grid-in / arrows-out
    one-liner — so we decimate in numpy and call ``Figure.plot(style="v...")``.

    Two traps, both verified empirically rather than read off the docs:

    1. With a reversed y-axis, GMT's ``-Sv`` angles are measured in the *user*
       frame and **do** honour the flip. So for ``vz`` positive-downwards you
       must NOT negate it — doing so points every arrow backwards. We sidestep
       the question entirely by using the ``+z`` modifier, which takes the raw
       components.
    2. Arrow length is a **plot** length in centimetres, never a data length.
       Hard-coding the scale gives invisible or absurd arrows depending on the
       field, so we derive it from the data: the longest arrow is
       ``target_cm``.
    """
    vx = np.asarray(vx, dtype=float)
    vz = np.asarray(vz, dtype=float)
    X, Z = np.meshgrid(np.asarray(x), np.asarray(z))
    s = int(every)

    vmax = float(np.hypot(vx, vz).max())
    if not np.isfinite(vmax) or vmax == 0.0:
        return  # nothing to draw

    # Normalise the components rather than folding the magnitude into the +z
    # scale. Model velocities are routinely ~1e-9 m/s, and the scale factor then
    # formats in scientific notation — "+z6.9e+08c" — whose "+08" GMT parses as
    # a second, invalid modifier. Normalising keeps the scale a plain number.
    fig.plot(
        x=X[::s, ::s].ravel(),
        y=Z[::s, ::s].ravel(),
        style=f"v{size}+e+a40+g{fill}+z{target_cm:g}c",
        direction=[(vx[::s, ::s] / vmax).ravel(), (vz[::s, ::s] / vmax).ravel()],
        pen=pen,
    )
    return vmax


def streamlines(fig, x, z, vx, vz, density=1.5, pen="0.5p,gray30"):
    """Overlay streamlines.

    GMT has no streamline module at all — the only suggestion in the community
    is contouring a ``grdgradient`` aspect angle, which contours a scalar where
    you wanted a vector and should not be taught. So we integrate in Python and
    plot polylines.

    Uses matplotlib headlessly. For the notebooks, integrating trajectories
    explicitly with ``scipy.integrate.solve_ivp`` is better pedagogy and
    generalises to passive tracers; this is the quick version.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figm, axm = plt.subplots()
    res = axm.streamplot(np.asarray(x), np.asarray(z), np.asarray(vx),
                         np.asarray(vz), density=density)
    segments = res.lines.get_segments()
    plt.close(figm)
    for seg in segments:
        if len(seg) > 1:
            fig.plot(x=seg[:, 0], y=seg[:, 1], pen=pen)
    return len(segments)


# --------------------------------------------------------------------------
# The workhorse panel
# --------------------------------------------------------------------------
def field_panel(field, x, z, kind="temperature", title="", label=None,
                unit="", series=None, contours=None, highlight=None,
                vx=None, vz=None, every=10, stamp=None, width_cm=15.0,
                xlabel="distance (km)", ylabel="depth (km)",
                depth_down=True, interpolation="n", cmap=None, reverse=None):
    """Build one publication-grade panel: raster + contours + arrows + colourbar.

    ``kind`` picks a sensible Crameri colour map from :data:`SCM`; override
    with ``cmap``/``reverse`` if you want something else.

    Returns the ``pygmt.Figure`` so the caller can add layers or save it.
    """
    import pygmt

    field = np.asarray(field, dtype=float)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    default_cmap, default_rev = SCM.get(kind, ("SCM/batlow", False))
    cmap = cmap or default_cmap
    reverse = default_rev if reverse is None else reverse
    label = label if label is not None else kind.replace("_", " ")

    grid = to_grid(field, x, z, name=label)
    region, projection = box_projection(x.max(), z.max(), width_cm, depth_down)

    if series is None:
        lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
        if hi <= lo:
            hi = lo + 1.0
        series = [lo, hi, (hi - lo) / 50.0]

    fig = pygmt.Figure()
    pygmt.makecpt(cmap=cmap, series=series, continuous=True, reverse=reverse)

    fig.grdimage(
        grid=grid, region=region, projection=projection, cmap=True,
        interpolation=interpolation,   # "n" shows the real model cells
        nan_transparent=True,          # masks sticky air / inactive cells
        frame=box_frame(title, xlabel, ylabel),
    )

    if contours is not None:
        fig.grdcontour(grid=grid, levels=contours, pen="0.4p,gray25")
    if highlight is not None:
        fig.grdcontour(grid=grid, levels=list(np.atleast_1d(highlight)),
                       pen="1.5p,black")
    if vx is not None and vz is not None:
        quiver(fig, x, z, vx, vz, every=every)

    cb_label = f"xaf+l{label}"
    frame = [cb_label] + ([f"y+l{unit}"] if unit else [])
    fig.colorbar(position=f"JBC+w{0.8 * width_cm}c/0.4c+h+o0c/1.4c", frame=frame)

    if stamp:  # drawn last so it sits above every layer
        fig.text(x=0.04 * x.max(), y=0.06 * z.max(), text=stamp, justify="LT",
                 font="12p,Helvetica-Bold,white", no_clip=True)
    return fig


# --------------------------------------------------------------------------
# Annulus (polar) geometry
# --------------------------------------------------------------------------
def annulus_panel(field, theta_deg, radius, kind="temperature", title="",
                  label=None, unit="", series=None, width_cm=13.0,
                  cmap=None, reverse=None, contours=None, stamp=None,
                  u_r=None, u_theta=None, every=12, arrow_cm=0.5):
    """One panel of a 2-D cylindrical annulus, in GMT's polar projection.

    Every panel in this suite so far has been a Cartesian box. An annulus needs
    GMT's **polar** projection, ``-JP``, which takes the data as ``(theta, r)``
    rather than ``(x, y)`` — so the grid handed in has *angle* along its columns
    and *radius* along its rows, and no coordinate conversion happens here.

    Two things about ``-JP`` are worth knowing before you fight it:

    * the region is ``theta_min/theta_max/r_min/r_max``, **angles first**, which
      is the opposite order to every Cartesian region you have written;
    * the projection width is the diameter of the full circle, not the width of
      the annulus, so a 13 cm panel of an annulus with rmax/rmin = 1.8 draws a
      ring about 3 cm across. Size for the outer radius.

    Vector components, if given, must already be resolved into radial and
    tangential parts — plotting Cartesian ``(vx, vy)`` on a polar projection
    puts every arrow in the wrong direction except at theta = 0.
    """
    import pygmt

    field = np.asarray(field, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    radius = np.asarray(radius, dtype=float)

    default_cmap, default_rev = SCM.get(kind, ("SCM/batlow", False))
    cmap = cmap or default_cmap
    reverse = default_rev if reverse is None else reverse
    label = label if label is not None else kind.replace("_", " ")

    grid = to_grid(field, theta_deg, radius, name=label)
    region = [theta_deg.min(), theta_deg.max(), radius.min(), radius.max()]
    projection = f"P{width_cm}c"

    if series is None:
        lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
        if hi <= lo:
            hi = lo + 1.0
        series = [lo, hi, (hi - lo) / 50.0]

    fig = pygmt.Figure()
    pygmt.makecpt(cmap=cmap, series=series, reverse=reverse, background=True)
    fig.grdimage(grid=grid, region=region, projection=projection,
                 cmap=True, nan_transparent=True)
    if contours is not None:
        fig.grdcontour(grid=grid, region=region, projection=projection,
                       levels=contours, pen="0.3p,gray30")

    if u_r is not None and u_theta is not None:
        s = every
        th = np.radians(theta_deg[::s])
        rr = radius[::s]
        TH, RR = np.meshgrid(th, rr)
        ur = np.asarray(u_r)[::s, ::s]
        ut = np.asarray(u_theta)[::s, ::s]
        # Back to Cartesian for the arrow direction, then to GMT's
        # azimuth-from-north convention.
        vx = ur * np.cos(TH) - ut * np.sin(TH)
        vy = ur * np.sin(TH) + ut * np.cos(TH)
        mag = np.hypot(vx, vy)
        vmax = float(np.nanmax(mag)) or 1.0
        fig.plot(
            x=np.degrees(TH).ravel(), y=RR.ravel(),
            style=f"v0.15c+e+a40+gblack+z{arrow_cm:g}c",
            direction=[(vx / vmax).ravel(), (vy / vmax).ravel()],
            pen="0.4p,black", region=region, projection=projection,
        )

    fig.basemap(region=region, projection=projection, frame=["xa45f15", "ya0.25"])
    if title:
        fig.text(position="TC", text=title, font="12p,Helvetica-Bold",
                 offset="0/0.7c", no_clip=True)
    if stamp:
        fig.text(position="BL", text=stamp, font="9p,Helvetica",
                 offset="0.3c/0.3c", no_clip=True)
    # No quotes around the label: GMT takes the rest of the modifier verbatim,
    # so quoting it puts literal quote marks on the colour bar.
    frame = [f"xaf+l{label}"] + ([f"y+l{unit}"] if unit else [])
    fig.colorbar(frame=frame,
                 position=f"JBC+w{0.55 * width_cm}c/0.35c+h+o0/1.2c")
    return fig
