# Binder entry point. Pinned to a frozen tag so the Binder cache stays warm:
# course notebooks are injected at launch via nbgitpuller, so editing a notebook
# never triggers an image rebuild. (Pattern borrowed from
# underworldcode/uw3-binder-launcher.)
#
# The base image already sets USER, HOME and WORKDIR, and exports NB_USER as an
# ENV so it is visible here. Referencing an ARG from the parent image instead
# yields an empty string and a build warning:
#   UndefinedVar: Usage of undefined variable '$NB_USER'
FROM ghcr.io/earthbyte/geodyn-pygmt:0.1.1
