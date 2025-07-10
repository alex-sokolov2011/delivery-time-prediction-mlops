CURRENT_DIR = $(shell pwd)

# ID of the currently running 'dev-run' container (used for exec, etc.)
DEV_ENV = $(shell docker ps --filter "name=dev-run" --format "{{.ID}}")

# ID of any 'dev-run' containers (running or stopped), used for cleanup
DEV_ENV_ALL = $(shell docker ps -a --filter "name=dev-run" --format "{{.ID}}")

LOCALSTACK_ENV = $(shell docker ps --filter "name=localstack" --format "{{.ID}}")

# Automatically detect the running Jupyter container by name
JUPYTER_CONTAINER_ID := $(shell docker ps -qf "name=jupyter")

# Docker registry settings (change to your own Docker Hub username)
DOCKER_USERNAME := sokaa2011
DOCKER_IMAGE_NAME := delivery_time_prediction
DOCKER_TAG := latest

include .env
export

prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data_store/postgres_data || true && \
	mkdir -p ${CURRENT_DIR}/data_store/minio || true && \
	mkdir -p ${CURRENT_DIR}/data_store/mlflow || true && \
	mkdir -p ${CURRENT_DIR}/data_store/dataset || true && \
	mkdir -p ${CURRENT_DIR}/data_store/grafana || true

clear-db-dirs:
	rm -rf ${CURRENT_DIR}/data_store/postgres_data || true

run-dev:
	docker compose run -d dev

build-dev:
	docker compose build dev

up-dev:
	docker compose up -d dev

download-data:
	docker exec -it ${DEV_ENV} python3 src/download_dataset.py

prepare-data:
	docker exec -it ${DEV_ENV} python3 src/prepare_data.py /srv/src/config.yml

params-search:
	docker exec -it ${DEV_ENV} python3 src/hyperopt_params_search.py /srv/src/config.yml

register-model:
	docker exec -it ${DEV_ENV} python3 src/register_model.py /srv/src/config.yml

tests:
	docker exec -it ${DEV_ENV} pytest src/tests/

integration-tests:
	docker exec -it ${LOCALSTACK_ENV} awslocal --endpoint-url=http://localhost:4566 s3 mb s3://delivery-prediction && \
	docker exec -it ${DEV_ENV} pytest src/integration_tests/

check:  ## Run linters in check mode (black, isort, pylint)
	isort --check-only . || true
	black --check --diff . || true
	pylint src/ --rcfile=pyproject.toml || true

format:
	isort .
	black .

build-prod: ## Build the production Docker image using the dedicated Dockerfile
	docker build -f ${CURRENT_DIR}/services/production/Dockerfile -t $(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .

run-prod: ## Run the production container locally on port 8090 and test the API
	docker run -p 8090:8090 -it --rm $(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

prepare-prod: ## Push the production image to Docker Hub (requires `docker login`)
	docker push $(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

backfill:
	docker exec -it ${DEV_ENV} python3 src/batch_prediction_backfill.py /srv/src/config.yml

setup-commit-hook: ## Install Git commit-msg hook that enforces allowed prefixes
	echo '#!/bin/sh' > .git/hooks/commit-msg && \
	echo 'start_check=$$(head -1 "$$1" | grep -qiE "^(Feature|Fix|Refactor|Docs|Test|Chore|Style|Perf|Revert):")' >> .git/hooks/commit-msg && \
	echo 'if [ $$? -ne 0 ]; then' >> .git/hooks/commit-msg && \
	echo '  echo "❌ Commit message must start with one of: Feature:, Fix:, Refactor:, Docs:, Test:, Chore:, Style:, Perf:, Revert:" 1>&2' >> .git/hooks/commit-msg && \
	echo '  exit 1' >> .git/hooks/commit-msg && \
	echo 'fi' >> .git/hooks/commit-msg && \
	chmod +x .git/hooks/commit-msg
	@echo "✅ commit-msg hook updated successfully."

test-bad-commit: ## Test hook with invalid commit message (should fail)
	echo 'Test content' > dummy.txt && \
	git add dummy.txt && \
	git commit -m 'Add code linters' || echo "❌ Commit rejected as expected."

test-good-commit: ## Test hook with valid commit message (should pass)
	echo 'Test content' > dummy.txt && \
	git add dummy.txt && \
	git commit -m 'Feature: code linters'

clean-commit-test: ## Clean up dummy file and staged changes after commit hook tests
	@if ! git diff --quiet || ! git diff --cached --quiet; then \
		echo "❗ Uncommitted changes found. Please commit or stash before cleaning."; \
		exit 1; \
	fi
	@echo "🧼 Cleaning up dummy commit test artifacts..."
	@rm -f dummy.txt
	@git reset
	@git checkout -- .
	@echo "✅ Clean complete."

pipfreeze:
	docker exec -it ${DEV_ENV} pip freeze

run-prefect:
	prefect server start

run-mlflow:
	docker compose --env-file .env up mlflow

run-postgres:
	docker compose --env-file .env up postgres

run-jupyter:
	docker compose --env-file .env up -d jupyter

run-dagster:
	dagster-webserver -m services.dagster_code

build-ci:
	docker build -f ${CURRENT_DIR}/services/ci/Dockerfile -t ci:latest .

down-dev:
	docker compose down || true
	@if [ -n "$(DEV_ENV_ALL)" ]; then \
		echo "Removing stale dev-run container(s): $(DEV_ENV_ALL)"; \
		docker rm -f $(DEV_ENV_ALL); \
	else \
		echo "No dev-run containers to remove."; \
	fi

clean-network:  ## Remove the custom Docker network if not in use
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

