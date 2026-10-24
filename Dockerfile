FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SPARK_CLUSTER_DATABASE_PATH=/data/spark-cluster.db SPARK_CLUSTER_MODEL_CACHE_PATH=/models SPARK_CLUSTER_SETUP_STATE_PATH=/data/setup-state SPARK_CLUSTER_HEALTH_URL=http://localhost:8080/healthz
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client skopeo && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[agent]'
RUN useradd --create-home --uid 10001 spark && mkdir /data /models && chown spark:spark /data /models
USER spark
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 CMD python -c "import os,urllib.request; urllib.request.urlopen(os.environ['SPARK_CLUSTER_HEALTH_URL'])"
ENTRYPOINT ["spark-cluster"]
CMD ["serve"]
