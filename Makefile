CURRENT_DIR = $(shell pwd)

# ID of currently running 'dev-run' container (used for exec, etc.)
DEV_ENV = $(shell docker ps --filter "name=dev-run" --format "{{.ID}}")

# ID of any 'dev-run' containers (running or stopped), used for cleanup
DEV_ENV_ALL = $(shell docker ps -a --filter "name=dev-run" --format "{{.ID}}")

LOCALSTACK_ENV = $(shell docker ps --filter "name=localstack" --format "{{.ID}}")

# Docker registry settings (change to your own Docker Hub username)
DOCKER_USERNAME := sokaa2011
DOCKER_IMAGE_NAME := delivery_time_prediction
DOCKER_TAG := latest

include .env
export

prepare-dirs: ## Create local directories for mounted volumes (Postgres, MinIO, MLflow, dataset, Grafana)
	mkdir -p ${CURRENT_DIR}/data_store/postgres_data || true && \
	mkdir -p ${CURRENT_DIR}/data_store/minio || true && \
	mkdir -p ${CURRENT_DIR}/data_store/mlflow || true && \
	mkdir -p ${CURRENT_DIR}/data_store/dataset || true && \
	mkdir -p ${CURRENT_DIR}/data_store/prefect || true && \
	mkdir -p ${CURRENT_DIR}/data_store/grafana || true

run-dev: ## Run dev container in detached mode
	docker compose run -d dev

download-data: ## Download dataset from Kaggle
	docker exec -it ${DEV_ENV} python3 src/download_dataset.py

prepare-data: ## Preprocess data and split into train/validation sets
	docker exec -it ${DEV_ENV} python3 src/prepare_data.py /srv/src/config.yml

params-search: ## Run hyperparameter search with Hyperopt for CatBoost
	docker exec -it ${DEV_ENV} python3 src/hyperopt_params_search.py /srv/src/config.yml

register-model: ## Train and register model with best hyperparameters in MLflow
	docker exec -it ${DEV_ENV} python3 src/register_model.py /srv/src/config.yml

test: ## Run unit test for data preparation
	docker exec -it ${DEV_ENV} pytest src/tests/test_prepare_data.py

integration-tests: ## Run integration tests for batch prediction and S3 interaction via LocalStack
	docker exec -it ${LOCALSTACK_ENV} awslocal --endpoint-url=http://localhost:4566 s3 mb s3://delivery-prediction && \
	docker exec -it ${DEV_ENV} pytest src/integration_tests/test_predict_batch.py

install-local-reqs:  ## Install local-only developer dependencies (e.g. httpx for testing FastAPI)
	python3 -m pip install --upgrade pip && python3 -m pip install --no-cache-dir -r local-requirements.txt

check:  ## Run linters in check mode (black, isort, pylint)
	isort --check-only . || true
	black --check --diff . || true
	pylint src/ --rcfile=pyproject.toml || true

format: ## Format Python code using isort and black
	isort .
	black .

build-prod: ## Build prod Docker image using dedicated Dockerfile
	docker build -f ${CURRENT_DIR}/services/production/Dockerfile -t $(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .

run-prod: ## Run prod container locally on port 8090 and test API
	docker run -p 8090:8090 -it --rm $(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

test-prod: ## Run integration test for prod FastAPI API
	pytest src/integration_tests/test_api.py

prepare-prod: ## Push prod image to Docker Hub (requires `docker login`)
	docker push $(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

backfill: ## Run batch prediction and store monitoring metrics to database
	docker exec -it ${DEV_ENV} python3 src/batch_prediction_backfill.py /srv/src/config.yml

setup-commit-hook: ## Install Git commit-msg hook that enforces allowed prefixes
	echo '#!/bin/sh' > .git/hooks/commit-msg && \
	echo 'start_check=$$(head -1 "$$1" | grep -qiE "^(Feature|Fix|Refactor|Docs|Test|Chore|Style|Perf|Revert|WIP):")' >> .git/hooks/commit-msg && \
	echo 'if [ $$? -ne 0 ]; then' >> .git/hooks/commit-msg && \
	echo '  echo "❌ Commit message must start with one of: Feature:, Fix:, Refactor:, Docs:, Test:, Chore:, Style:, Perf:, Revert:" 1>&2' >> .git/hooks/commit-msg && \
	echo '  exit 1' >> .git/hooks/commit-msg && \
	echo 'fi' >> .git/hooks/commit-msg && \
	chmod +x .git/hooks/commit-msg
	@echo "✅ commit-msg hook updated successfully."

test-bad-commit: ## Test hook with invalid commit message (should fail)
	echo 'Test content' > dummy-test-commit.txt && \
	git add dummy-test-commit.txt && \
	git commit -m 'Add code linters' || echo "❌ Commit rejected as expected."

test-good-commit: ## Test hook with valid commit message (should pass)
	echo 'Test content' > dummy-test-commit.txt && \
	git add dummy-test-commit.txt && \
	git commit -m 'Feature: code linters'

build-ci: ## Build CI Docker image used for testing and linting
	docker build -f ${CURRENT_DIR}/services/ci/Dockerfile -t ci:latest .

pipfreeze: ## Show installed Python packages inside dev container
	docker exec -it ${DEV_ENV} pip freeze

run-mlflow: ## Run only MLflow service using docker-compose
	docker compose --env-file .env up mlflow

run-postgres: ## Run only PostgreSQL service using docker-compose
	docker compose --env-file .env up postgres

run-jupyter: ## Run Jupyter Notebook container in detached mode
	docker compose --env-file .env up -d jupyter

run-dagster: ## Start Dagster webserver for local orchestration
	dagster-webserver -m services.dagster_code

prefect-prepare-data: ## Run Prefect flow for data preparation (merging & splitting datasets)
	python src/prefect_prepare_data.py

run-prefect: ## Start Prefect server locally (runs until stopped)
	prefect server start

prefect-all: ## Create Prefect work pool and start worker (runs until stopped)
	export PREFECT_API_URL=http://127.0.0.1:4200/api && \
	prefect work-pool create --type process default-agent-pool && \
	prefect worker start --pool default-agent-pool

run-prefect-deploy: ## Create Prefect deployment and schedule it (runs until stopped)
	python src/prefect_deploy_prepare.py

clear-db-dirs: ## Remove local Postgres data directory (clears DB data)
	rm -rf ${CURRENT_DIR}/data_store/postgres_data || true

build-dev: ## Build Docker image for dev
	docker compose build dev

up-dev: ## Start dev container with up in detached mode
	docker compose up -d dev

down-dev: ## Stop and remove dev container and clean up any stale dev-run containers
	docker compose down || true
	@if [ -n "$(DEV_ENV_ALL)" ]; then \
		echo "Removing stale dev-run container(s): $(DEV_ENV_ALL)"; \
		docker rm -f $(DEV_ENV_ALL); \
	else \
		echo "No dev-run containers to remove."; \
	fi

clean-network:  ## Remove custom Docker network if not in use
	@docker network rm delivery-time-prediction-mlops_prj_network || echo "Network could not be removed (still in use or already deleted)."

down:  ## Down dev and clean-network and rebuild
	$(MAKE) down-dev
	$(MAKE) clean-network
	$(MAKE) build-dev

stop-all:  ## Stop all running Docker containers (use with caution!)
	@docker ps -q | xargs -r docker stop

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

