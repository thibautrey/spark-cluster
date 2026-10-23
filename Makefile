.PHONY: install dev test lint docker helm

install:
	python -m pip install -e '.[test,agent]'

dev:
	SPARK_CLUSTER_MODE=demo spark-cluster serve --port 8080

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

docker:
	docker build -t spark-cluster:local .

helm:
	helm lint deploy/helm/spark-cluster
