ARG BASE_IMAGE=ghcr.io/loke-60000/dsv41-1xgb10:v0.1.1-kvfix4
FROM ${BASE_IMAGE}
COPY --chmod=755 tensorfold-dsv41 /opt/tensorfold/bin/tensorfold-dsv41
