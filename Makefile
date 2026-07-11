# RoboDojo workflow shortcuts.

SELF := $(firstword $(MAKEFILE_LIST))

-include .env
export

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Tooling
UV ?= uv
UV_RUN ?= $(UV) run --locked
ROBODOJO ?= $(UV_RUN) bash scripts/robodojo.sh
STORAGE ?= bash scripts/robodojo_storage.sh
DOCKER ?= docker

# Evaluation defaults (override on the command line or in .env)
DATASET ?= RoboDojo
TASK ?= stack_bowls
ENV_CFG ?= arx_x5
SEED ?= 0
ACTION_TYPE ?= ee
EXPERT_NUM ?= 100
EVAL_NUM ?= 1
POLICY_GPU ?= 0
ENV_GPU ?= 0
POLICY_DIR ?=
POLICY_NAME ?=
POLICY_ENV ?=
CKPT ?=
POLICY_HOST ?= 127.0.0.1
POLICY_PORT ?=
BIND_HOST ?= 0.0.0.0

# Downloads, results, storage, and containers
DATA_TYPE ?=
EVAL_ROOT ?= eval_result/RoboDojo
STORAGE_SOURCE ?=
STORAGE_DESTINATION ?=
STORAGE_RELATIVE ?=
STORAGE_KIND ?=
STORAGE_POLICY ?=
STORAGE_CHECKPOINT ?=
IMAGE ?= robodojo:cuda12.8
ONLY ?=
ARGS ?=

define require
$(if $(strip $($(1))),,$(error $(1) is required. Pass it as `make $@ $(1)=...` or set it in .env))
endef

define require_client_policy
$(if $(or $(strip $(POLICY_DIR)),$(strip $(POLICY_NAME))),,$(error POLICY_DIR or POLICY_NAME is required. Pass one on the make command line or set it in .env))
endef

# These are Make-only validation helpers; exporting them would evaluate the
# functions without arguments while constructing each recipe's environment.
unexport require require_client_policy

POLICY_FLAGS = \
	--policy-dir "$(POLICY_DIR)" \
	--task "$(TASK)" \
	--ckpt "$(CKPT)" \
	--policy-env "$(POLICY_ENV)" \
	--env-cfg "$(ENV_CFG)" \
	--action-type "$(ACTION_TYPE)" \
	--seed "$(SEED)" \
	--policy-gpu "$(POLICY_GPU)"

EVAL_FLAGS = \
	$(POLICY_FLAGS) \
	--dataset "$(DATASET)" \
	--expert-num "$(EXPERT_NUM)" \
	--env-gpu "$(ENV_GPU)" \
	--eval-num "$(EVAL_NUM)"

CLIENT_POLICY_FLAG = $(if $(strip $(POLICY_DIR)),--policy-dir "$(POLICY_DIR)",--policy-name "$(POLICY_NAME)")
CLIENT_FLAGS = \
	--task "$(TASK)" \
	$(CLIENT_POLICY_FLAG) \
	--policy-host "$(POLICY_HOST)" \
	--policy-port "$(POLICY_PORT)" \
	--ckpt "$(CKPT)" \
	--env-cfg "$(ENV_CFG)" \
	--action-type "$(ACTION_TYPE)" \
	--seed "$(SEED)" \
	--env-gpu "$(ENV_GPU)" \
	--eval-num "$(EVAL_NUM)"

SWEEP_FLAGS = \
	--policy-dir "$(POLICY_DIR)" \
	--ckpt "$(CKPT)" \
	--policy-env "$(POLICY_ENV)" \
	--env-cfg "$(ENV_CFG)" \
	--action-type "$(ACTION_TYPE)" \
	--seed "$(SEED)" \
	--policy-gpu "$(POLICY_GPU)" \
	--dataset "$(DATASET)" \
	--expert-num "$(EXPERT_NUM)" \
	--env-gpu "$(ENV_GPU)" \
	--eval-num "$(EVAL_NUM)" \
	$(if $(strip $(ONLY)),--only "$(ONLY)",--all)

.PHONY: \
	help tasks tasks-check \
	install sync assets data-list data \
	lint lint-fix format format-check test pre-commit check \
	doctor \
	eval eval-dry-run server server-dry-run client client-dry-run \
	smoke smoke-dry-run benchmark benchmark-dry-run summarize \
	storage-status storage-doctor storage-publish storage-publish-dry-run \
	storage-hydrate storage-link \
	docker-install docker-build docker-smoke docker-monitor docker-clean

##@ Discovery
help: ## Show targets and common configuration variables
	@printf 'RoboDojo local workflow\n\n'
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5); next} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' "$(SELF)"
	@printf '\nCommon variables:\n'
	@printf '  TASK=%s ENV_CFG=%s SEED=%s EVAL_NUM=%s\n' "$(TASK)" "$(ENV_CFG)" "$(SEED)" "$(EVAL_NUM)"
	@printf '  POLICY_DIR=... POLICY_ENV=... CKPT=... POLICY_GPU=%s ENV_GPU=%s\n' "$(POLICY_GPU)" "$(ENV_GPU)"
	@printf '  STORAGE_SOURCE=... STORAGE_DESTINATION=... STORAGE_RELATIVE=...\n'
	@printf '  STORAGE_KIND=assets|datasets|checkpoint STORAGE_POLICY=... STORAGE_CHECKPOINT=...\n'
	@printf '  ARGS="..." forwards additional arguments to the selected command.\n'

tasks: ## List the canonical RoboDojo tasks
	$(ROBODOJO) tasks $(ARGS)

tasks-check: ## Validate that every registered task has code and configuration
	$(ROBODOJO) tasks --check $(ARGS)

##@ Setup and downloads
install: ## Install system dependencies, submodules, and the locked environment
	bash scripts/install.sh --install $(ARGS)

sync: ## Synchronize the locked uv environment
	$(UV) sync --locked $(ARGS)

assets: ## Download RoboDojo assets
	bash scripts/init_assets.sh $(ARGS)

data-list: ## List available dataset formats and sizes
	bash scripts/RoboDojo/download_data.sh

data: ## Download DATA_TYPE (demo, lerobot_v3.0, lerobot_v2.1, hdf5, or real)
	$(call require,DATA_TYPE)
	bash scripts/RoboDojo/download_data.sh "$(DATA_TYPE)" $(ARGS)

##@ Code quality
lint: ## Run Ruff lint checks
	$(UV_RUN) ruff check . $(ARGS)

lint-fix: ## Apply Ruff's safe lint fixes
	$(UV_RUN) ruff check --fix . $(ARGS)

format: ## Format Python files with Ruff
	$(UV_RUN) ruff format . $(ARGS)

format-check: ## Check Ruff formatting without modifying files
	$(UV_RUN) ruff format --check . $(ARGS)

test: ## Run the test suite
	$(UV_RUN) pytest $(ARGS)

pre-commit: ## Run all configured pre-commit hooks
	$(UV_RUN) pre-commit run --all-files $(ARGS)

check: lint format-check test tasks-check ## Run non-mutating quality checks

##@ Diagnostics and evaluation
doctor: ## Validate the environment and optionally POLICY_DIR/CKPT/POLICY_ENV
	$(ROBODOJO) doctor \
		--task "$(TASK)" \
		--env-cfg "$(ENV_CFG)" \
		$(if $(strip $(POLICY_DIR)),--policy-dir "$(POLICY_DIR)",--skip-policy) \
		$(if $(strip $(CKPT)),--ckpt "$(CKPT)") \
		$(if $(strip $(POLICY_ENV)),--policy-env "$(POLICY_ENV)") \
		$(ARGS)

eval: ## Run a local policy server and simulator evaluation
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO) eval $(EVAL_FLAGS) $(ARGS)

eval-dry-run: ## Print the resolved local evaluation command
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO) eval $(EVAL_FLAGS) --dry-run $(ARGS)

server: ## Start a policy server for split or multi-machine evaluation
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO) server $(POLICY_FLAGS) \
		--bind-host "$(BIND_HOST)" \
		$(if $(strip $(POLICY_PORT)),--policy-port "$(POLICY_PORT)") \
		$(ARGS)

server-dry-run: ## Print the resolved standalone policy-server command
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO) server $(POLICY_FLAGS) \
		--bind-host "$(BIND_HOST)" \
		$(if $(strip $(POLICY_PORT)),--policy-port "$(POLICY_PORT)") \
		--dry-run $(ARGS)

client: ## Run the simulator client against an external policy server
	$(call require_client_policy)
	$(call require,POLICY_PORT)
	$(call require,CKPT)
	$(ROBODOJO) client $(CLIENT_FLAGS) $(ARGS)

client-dry-run: ## Print the resolved external simulator-client command
	$(call require_client_policy)
	$(call require,POLICY_PORT)
	$(call require,CKPT)
	$(ROBODOJO) client $(CLIENT_FLAGS) --dry-run $(ARGS)

smoke: ## Run one episode per selected task by default
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO) smoke $(SWEEP_FLAGS) $(ARGS)

smoke-dry-run: ## Resolve a smoke sweep without launching evaluations
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO) smoke $(SWEEP_FLAGS) --dry-run $(ARGS)

benchmark: ## Run a benchmark sweep using EVAL_NUM episodes per task
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO) benchmark $(SWEEP_FLAGS) $(ARGS)

benchmark-dry-run: ## Resolve a benchmark sweep without launching evaluations
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO) benchmark $(SWEEP_FLAGS) --dry-run $(ARGS)

summarize: ## Aggregate evaluation results into a Markdown report
	ROBODOJO_EVAL_ROOT="$(EVAL_ROOT)" $(ROBODOJO) summarize $(ARGS)

##@ Storage (S3 + local scratch)
storage-status: ## Check the configured read mount, local scratch, and AWS CLI
	$(STORAGE) status $(ARGS)

storage-doctor: ## Validate the configured storage environment
	$(STORAGE) doctor $(ARGS)

storage-publish: ## Publish STORAGE_SOURCE to STORAGE_RELATIVE
	$(call require,STORAGE_SOURCE)
	$(call require,STORAGE_RELATIVE)
	$(STORAGE) publish "$(STORAGE_SOURCE)" "$(STORAGE_RELATIVE)" $(ARGS)

storage-publish-dry-run: ## Preview publishing STORAGE_SOURCE to STORAGE_RELATIVE
	$(call require,STORAGE_SOURCE)
	$(call require,STORAGE_RELATIVE)
	$(STORAGE) publish "$(STORAGE_SOURCE)" "$(STORAGE_RELATIVE)" --dry-run $(ARGS)

storage-hydrate: ## Copy and verify STORAGE_SOURCE at STORAGE_DESTINATION
	$(call require,STORAGE_SOURCE)
	$(call require,STORAGE_DESTINATION)
	$(STORAGE) hydrate "$(STORAGE_SOURCE)" "$(STORAGE_DESTINATION)" $(ARGS)

storage-link: ## Link STORAGE_KIND into STORAGE_DESTINATION
	$(call require,STORAGE_KIND)
	$(call require,STORAGE_DESTINATION)
	$(if $(filter checkpoint,$(STORAGE_KIND)),$(call require,STORAGE_POLICY))
	$(if $(filter checkpoint,$(STORAGE_KIND)),$(call require,STORAGE_CHECKPOINT))
	$(STORAGE) link "$(STORAGE_KIND)" "$(STORAGE_DESTINATION)" \
		$(if $(filter checkpoint,$(STORAGE_KIND)),--policy "$(STORAGE_POLICY)" --checkpoint "$(STORAGE_CHECKPOINT)") \
		$(ARGS)

##@ Docker
docker-install: ## Install Docker Engine and the NVIDIA Container Toolkit
	sudo bash docker/install_docker_nvidia.sh $(ARGS)

docker-build: ## Build IMAGE (default: robodojo:cuda12.8)
	$(DOCKER) build $(ARGS) -t "$(IMAGE)" .

docker-smoke: ## Run the end-to-end Docker GPU smoke test
	ROBODOJO_IMAGE="$(IMAGE)" bash docker/smoke_docker.sh run $(ARGS)

docker-monitor: ## Monitor the latest Docker smoke test
	bash docker/smoke_docker.sh monitor $(ARGS)

docker-clean: ## Stop leftover Docker smoke processes and remove smoke logs
	bash docker/smoke_docker.sh clean $(ARGS)
