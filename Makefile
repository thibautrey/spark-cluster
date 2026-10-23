.PHONY: install dev test lint docker helm

install:
	python -m pip install -e '.[test,agent]'

dev:
	SPARK_DEEPSEEK_MODE=demo spark-deepseek serve --port 8080

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

docker:
	docker build -t spark-deepseek:local .

helm:
	helm lint deploy/helm/spark-deepseek
