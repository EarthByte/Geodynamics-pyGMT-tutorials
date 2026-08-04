# Binder entry point. Pinned to a frozen tag so the Binder cache stays warm:
# course notebooks are injected at launch via nbgitpuller, so editing a notebook
# never triggers an image rebuild. (Pattern borrowed from
# underworldcode/uw3-binder-launcher.)
FROM ghcr.io/earthbyte/geodyn-pygmt:0.1.0

USER ${NB_USER}
WORKDIR ${HOME}
