# Binder entry point. Pinned to a frozen tag so the Binder cache stays warm:
# course notebooks are injected at launch via nbgitpuller, so editing a notebook
# never triggers an image rebuild. (Pattern borrowed from
# underworldcode/uw3-binder-launcher.)
#
# BUMP THIS WITH EVERY IMAGE RELEASE. It was left at 0.1.0 through the whole of
# 0.1.1, which means Binder kept building from the image that lacked
# `jupyterhub-singleuser` -- the very thing 0.1.1 was released to fix. The
# published image being correct is not enough if this line still points at the
# broken one, and that is almost certainly why the Nectar retest never worked.
FROM ghcr.io/earthbyte/geodyn-pygmt:0.1.2

USER ${NB_USER}
WORKDIR ${HOME}
