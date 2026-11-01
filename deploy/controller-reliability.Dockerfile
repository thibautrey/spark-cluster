# Backend-only update: retain the deployed frontend and installed dependency versions.
ARG BASE_IMAGE=spark-cluster:strata-telemetry-20261030b
FROM ${BASE_IMAGE}
USER root
COPY src/spark_cluster/*.py /usr/local/lib/python3.12/site-packages/spark_cluster/
USER spark
