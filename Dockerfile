# Combined image: Firedrake + G-ADOPT + pyGMT, no conda.
# Verified 3 Aug 2026 — all three import cleanly in one interpreter.
FROM firedrakeproject/firedrake-vanilla-default:2026.4.1

# These labels must be present at PUBLISH time, not added later. The
# org.opencontainers.image.source label is what links the package to the repo,
# and GitHub only grants the package the repo's access permissions if the link
# exists *before* the first publish. Linking afterwards keeps whatever
# permissions the package already had.
LABEL org.opencontainers.image.source="https://github.com/EarthByte/Geodynamics-pyGMT-tutorials" \
      org.opencontainers.image.description="Firedrake + G-ADOPT + pyGMT for the Geodynamics-pyGMT teaching notebooks" \
      org.opencontainers.image.licenses="BSD-3-Clause"

USER root
# Ubuntu 24.04 apt ships GMT 6.5.0 == pyGMT 0.19's minimum.
# Debian/Ubuntu provide only libgmt.so.6, so pyGMT needs the .so symlink.
# Resolve the multiarch directory rather than hardcoding x86_64-linux-gnu —
# this image is built for arm64 as well, where the path is aarch64-linux-gnu.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gmt gmt-dcw gmt-gshhg ghostscript \
 && rm -rf /var/lib/apt/lists/* \
 && lib="$(find /usr/lib -name 'libgmt.so.6*' -type f -print -quit)" \
 && ln -sf "$lib" "$(dirname "$lib")/libgmt.so" \
 && test -e "$(dirname "$lib")/libgmt.so"

# jupyterhub is NOT optional for a BinderHub image: it provides the
# `jupyterhub-singleuser` executable that the hub exec's to start the server.
# Without it the image builds, pushes and runs perfectly by hand, and then
# fails on Binder with
#   exec: "jupyterhub-singleuser": executable file not found in $PATH
# which says nothing about the actual omission.
RUN pip install --no-cache-dir \
        gadopt pygmt xarray netCDF4 imageio imageio-ffmpeg \
        jupyterhub jupyterlab notebook nbgitpuller

# BinderHub requires a UID-1000 user that owns $HOME, and forbids running as root.
ARG NB_USER=jovyan
ARG NB_UID=1000
# Ubuntu 24.04 base images already ship a UID-1000 'ubuntu' user; remove it first.
RUN if id -u ${NB_UID} >/dev/null 2>&1; then userdel -r "$(id -un ${NB_UID})" || true; fi \
 && useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER}
ENV NB_USER=${NB_USER} \
    HOME=/home/${NB_USER} \
    PYGMT_USE_EXTERNAL_DISPLAY=false
WORKDIR ${HOME}
USER ${NB_USER}
