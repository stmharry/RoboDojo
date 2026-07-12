# RoboDojo workflow shortcuts. The Python CLI owns all workflow logic.

SELF := $(firstword $(MAKEFILE_LIST))
-include .env
export

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
# Keep lightweight workflows on the base environment; simulator workflows opt
# into the large `sim` dependency extra explicitly.
UV_RUN_SIM ?= $(UV) run --extra sim --locked
ROBODOJO_BASE ?= $(UV) run --locked robodojo
ROBODOJO_SIM ?= $(UV_RUN_SIM) robodojo
OMNI_KIT_ACCEPT_EULA ?= yes

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
DATA_TYPE ?=
STORAGE_SOURCE ?=
STORAGE_RELATIVE ?=
IMAGE ?= robodojo:cuda12.8
ONLY ?=
ARGS ?=

define require
$(if $(strip $($(1))),,$(error $(1) is required. Pass it as `make $@ $(1)=...` or set it in .env))
endef
unexport require

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

.PHONY: \
	help \
	tasks \
	tasks-check \
	install \
	sync \
	assets \
	data-list \
	data \
	lint \
	lint-fix \
	format \
	format-check \
	test \
	pre-commit \
	check \
	doctor \
	eval \
	eval-dry-run \
	server \
	server-dry-run \
	client \
	client-dry-run \
	smoke \
	smoke-dry-run \
	benchmark \
	benchmark-dry-run \
	summarize \
	storage-status \
	storage-doctor \
	storage-publish \
	storage-publish-dry-run \
	storage-pull \
	storage-pull-dry-run \
	docker-install \
	docker-build \
	docker-smoke \
	docker-monitor \
	docker-clean

help: ## Show targets and common configuration variables
	@printf \
		'RoboDojo local workflow\n\n'
	@awk \
		'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' \
		"$(SELF)"
	@printf \
		'\nTASK=%s ENV_CFG=%s SEED=%s EVAL_NUM=%s\n' \
		"$(TASK)" \
		"$(ENV_CFG)" \
		"$(SEED)" \
		"$(EVAL_NUM)"

tasks: ## List canonical tasks
	$(ROBODOJO_BASE) \
		tasks \
		$(ARGS)

tasks-check: ## Validate task code/config pairs
	$(ROBODOJO_BASE) \
		tasks \
		--check \
		$(ARGS)

install: ## Install system dependencies, submodules, and simulator environment
	$(ROBODOJO_BASE) \
		install \
		$(ARGS)

sync: ## Synchronize the locked simulator environment
	$(UV) \
		sync \
		--extra sim \
		--locked \
		$(ARGS)

assets: ## Download benchmark assets
	$(ROBODOJO_BASE) \
		assets \
		download \
		$(ARGS)

data-list: ## List dataset formats
	$(ROBODOJO_BASE) \
		data \
		list

data: ## Download DATA_TYPE
	$(call require,DATA_TYPE)
	$(ROBODOJO_BASE) \
		data \
		download \
		"$(DATA_TYPE)" \
		$(ARGS)

lint: ## Run Ruff lint checks
	$(UV_RUN_SIM) \
		ruff \
		check \
		. \
		$(ARGS)

lint-fix: ## Apply Ruff safe fixes
	$(UV_RUN_SIM) \
		ruff \
		check \
		--fix \
		. \
		$(ARGS)

format: ## Format Python code
	$(UV_RUN_SIM) \
		ruff \
		format \
		. \
		$(ARGS)

format-check: ## Check Python formatting
	$(UV_RUN_SIM) \
		ruff \
		format \
		--check \
		. \
		$(ARGS)

test: ## Run tests
	$(UV_RUN_SIM) \
		pytest \
		$(ARGS)

pre-commit: ## Run all pre-commit hooks
	$(UV_RUN_SIM) \
		pre-commit \
		run \
		--all-files \
		$(ARGS)

check: lint format-check test tasks-check ## Run all non-mutating checks

doctor: ## Validate installation and configuration
	$(ROBODOJO_SIM) \
		doctor \
		--task "$(TASK)" \
		--env-cfg "$(ENV_CFG)" \
		$(if $(strip $(POLICY_DIR)),--policy-dir "$(POLICY_DIR)",--skip-policy) \
		$(ARGS)

eval: ## Run local server + simulator evaluation
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO_SIM) \
		eval \
		$(EVAL_FLAGS) \
		$(ARGS)

eval-dry-run: ## Print resolved local evaluation commands
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO_SIM) \
		eval \
		$(EVAL_FLAGS) \
		--dry-run \
		$(ARGS)

server: ## Start only the policy server
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO_BASE) \
		server \
		$(POLICY_FLAGS) \
		--bind-host "$(BIND_HOST)" \
		$(if $(strip $(POLICY_PORT)),--policy-port "$(POLICY_PORT)") \
		$(ARGS)

server-dry-run: ## Print the resolved server command
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO_BASE) \
		server \
		$(POLICY_FLAGS) \
		--bind-host "$(BIND_HOST)" \
		$(if $(strip $(POLICY_PORT)),--policy-port "$(POLICY_PORT)") \
		--dry-run \
		$(ARGS)

client: ## Run simulator client against an external server
	$(call require,POLICY_PORT)
	$(call require,CKPT)
	$(ROBODOJO_SIM) \
		client \
		--task "$(TASK)" \
		$(if $(strip $(POLICY_DIR)),--policy-dir "$(POLICY_DIR)",--policy-name "$(POLICY_NAME)") \
		--policy-host "$(POLICY_HOST)" \
		--policy-port "$(POLICY_PORT)" \
		--ckpt "$(CKPT)" \
		--env-cfg "$(ENV_CFG)" \
		--action-type "$(ACTION_TYPE)" \
		--seed "$(SEED)" \
		--env-gpu "$(ENV_GPU)" \
		--eval-num "$(EVAL_NUM)" \
		$(ARGS)

client-dry-run: ## Print the resolved client command
	$(MAKE) \
		client \
		ARGS="--dry-run $(ARGS)"

smoke: ## Run selected/all tasks with one episode
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO_SIM) \
		smoke \
		--policy-dir "$(POLICY_DIR)" \
		--ckpt "$(CKPT)" \
		--policy-env "$(POLICY_ENV)" \
		--env-cfg "$(ENV_CFG)" \
		--action-type "$(ACTION_TYPE)" \
		--seed "$(SEED)" \
		--policy-gpu "$(POLICY_GPU)" \
		--env-gpu "$(ENV_GPU)" \
		$(if $(strip $(ONLY)),--only "$(ONLY)") \
		$(ARGS)

smoke-dry-run: ## Dry-run a smoke sweep
	$(MAKE) \
		smoke \
		ARGS="--dry-run $(ARGS)"

benchmark: ## Run a benchmark sweep
	$(call require,POLICY_DIR)
	$(call require,POLICY_ENV)
	$(call require,CKPT)
	$(ROBODOJO_SIM) \
		benchmark \
		--policy-dir "$(POLICY_DIR)" \
		--ckpt "$(CKPT)" \
		--policy-env "$(POLICY_ENV)" \
		--eval-num "$(EVAL_NUM)" \
		--env-cfg "$(ENV_CFG)" \
		--action-type "$(ACTION_TYPE)" \
		--seed "$(SEED)" \
		--policy-gpu "$(POLICY_GPU)" \
		--env-gpu "$(ENV_GPU)" \
		$(if $(strip $(ONLY)),--only "$(ONLY)") \
		$(ARGS)

benchmark-dry-run: ## Dry-run a benchmark sweep
	$(MAKE) \
		benchmark \
		ARGS="--dry-run $(ARGS)"

summarize: ## Aggregate results into Markdown
	$(ROBODOJO_BASE) \
		summarize \
		$(ARGS)

storage-status: ## Check storage configuration
	$(ROBODOJO_BASE) \
		storage \
		status \
		$(ARGS)

storage-doctor: ## Validate storage configuration
	$(ROBODOJO_BASE) \
		storage \
		doctor \
		$(ARGS)

storage-publish: ## Publish STORAGE_SOURCE to STORAGE_RELATIVE
	$(call require,STORAGE_SOURCE)
	$(call require,STORAGE_RELATIVE)
	$(ROBODOJO_BASE) \
		storage \
		publish \
		"$(STORAGE_SOURCE)" \
		"$(STORAGE_RELATIVE)" \
		$(ARGS)

storage-publish-dry-run: ## Preview storage publication
	$(MAKE) \
		storage-publish \
		ARGS="--dry-run $(ARGS)"

storage-pull: ## Pull and verify STORAGE_RELATIVE into local storage
	$(call require,STORAGE_RELATIVE)
	$(ROBODOJO_BASE) \
		storage \
		pull \
		"$(STORAGE_RELATIVE)" \
		$(ARGS)

storage-pull-dry-run: ## Preview storage pull
	$(MAKE) \
		storage-pull \
		ARGS="--dry-run $(ARGS)"

docker-install: ## Install Docker and NVIDIA runtime
	$(ROBODOJO_BASE) \
		docker \
		install \
		$(ARGS)

docker-build: ## Build simulator image
	$(ROBODOJO_BASE) \
		docker \
		build \
		--image "$(IMAGE)" \
		$(ARGS)

docker-smoke: ## Run Docker GPU smoke test
	$(call require,POLICY_PORT)
	$(ROBODOJO_BASE) \
		docker \
		smoke \
		--image "$(IMAGE)" \
		--policy-port "$(POLICY_PORT)" \
		$(ARGS)

docker-monitor: ## Monitor Docker smoke logs
	$(ROBODOJO_BASE) \
		docker \
		monitor \
		$(ARGS)

docker-clean: ## Clean Docker smoke state
	$(ROBODOJO_BASE) \
		docker \
		clean \
		$(ARGS)
