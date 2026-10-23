FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SPARK_DEEPSEEK_DATABASE_PATH=/data/spark-deepseek.db SPARK_DEEPSEEK_MODEL_CACHE_PATH=/models
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY deploy/helm/spark-deepseek ./deploy/helm/spark-deepseek
RUN pip install --no-cache-dir '.[agent]'
RUN useradd --create-home --uid 10001 spark && mkdir /data /models && chown spark:spark /data /models
USER spark
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')"
ENTRYPOINT ["spark-deepseek"]
CMD ["serve"]
