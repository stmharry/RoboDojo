# Opinionated local workflow shortcuts. Python owns workflow behavior; Make
# accepts experiment selection from arguments or the process environment and
# translates it to CLI flags.

SELF := $(firstword $(MAKEFILE_LIST))

SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
UV_RUN_SIM ?= $(UV) run --extra sim --locked
ROBODOJO_SETUP ?= $(UV) run --locked robodojo
ROBODOJO_BASE ?= $(UV) run --locked --no-sync robodojo
ROBODOJO_SIM ?= $(UV) run --extra sim --locked --no-sync robodojo
OMNI_KIT_ACCEPT_EULA ?= yes
export ROBODOJO_STORAGE_ROOT ROBODOJO_S3_URI AWS_PROFILE ROBODOJO_LOG_LEVEL OMNI_KIT_ACCEPT_EULA

# Stable benchmark default and Make-only workflow controls.
DATASET ?= RoboDojo
ACTION_TYPE ?= joint
SEED ?= 0
ENV_GPU ?= 1
POLICY_GPU ?= 0
EVAL_NUM ?= 1
PUBLISH ?= true
DEEP ?= false
DRY_RUN ?= false
ONLY ?=
ARGS ?=

REQUIRED_EXPERIMENT_VARS := TASK ENV_CFG POLICY_DIR POLICY_ENV CKPT

define require_experiment
$(foreach name,$(REQUIRED_EXPERIMENT_VARS),$(if $(strip $($(name))),,$(error $(name) is required; pass $(name)=... to make or export it)))
endef

define boolean_flag
$(if $(filter true,$(strip $($(1)))),$(2),$(if $(filter false,$(strip $($(1)))),,$(error $(1) must be true or false, got '$($(1))')))
endef

PUBLISH_FLAG = $(call boolean_flag,PUBLISH,--publish)
DEEP_FLAG = $(call boolean_flag,DEEP,--deep)
DRY_RUN_FLAG = $(call boolean_flag,DRY_RUN,--dry-run)
SCENE_FLAG = $(if $(strip $(SCENE)),--scene "$(SCENE)")
ONLY_FLAG = $(if $(strip $(ONLY)),--only "$(ONLY)")

POLICY_ARGS = \
	--policy-dir "$(POLICY_DIR)" \
	--task "$(TASK)" \
	--ckpt "$(CKPT)" \
	--policy-env "$(POLICY_ENV)" \
	--dataset "$(DATASET)" \
	--env-cfg "$(ENV_CFG)" \
	--action-type "$(ACTION_TYPE)" \
	--seed "$(SEED)" \
	--policy-gpu "$(POLICY_GPU)"

EXPERIMENT_ARGS = \
	$(POLICY_ARGS) \
	--env-gpu "$(ENV_GPU)" \
	$(SCENE_FLAG)

SWEEP_ARGS = \
	--policy-dir "$(POLICY_DIR)" \
	--ckpt "$(CKPT)" \
	--policy-env "$(POLICY_ENV)" \
	--env-cfg "$(ENV_CFG)" \
	--action-type "$(ACTION_TYPE)" \
	--seed "$(SEED)" \
	--policy-gpu "$(POLICY_GPU)" \
	--env-gpu "$(ENV_GPU)" \
	$(SCENE_FLAG) \
	$(ONLY_FLAG)

.PHONY: help setup preflight eval smoke benchmark tasks doctor results \
	lint lint-fix format format-check test pre-commit check _config-check

##@ Workflow
help: ## Show the supported local workflow
	@printf 'RoboDojo local workflow\n\n  select an experiment -> make setup -> make preflight -> make eval\n'
	@awk \
		'BEGIN {FS = ":.*## "} \
		/^##@ / {printf "\n%s\n", substr($$0, 5); next} \
		/^[a-zA-Z0-9_.-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' \
		"$(SELF)"
	@printf \
		'\nConfigured experiment\n  TASK=%s ENV_CFG=%s SCENE=%s SEED=%s EVAL_NUM=%s PUBLISH=%s\n' \
		"$(TASK)" "$(ENV_CFG)" "$(SCENE)" "$(SEED)" "$(EVAL_NUM)" "$(PUBLISH)"
_config-check:
	$(call require_experiment)
	@case "$(SEED):$(ENV_GPU):$(POLICY_GPU):$(EVAL_NUM)" in *[!0-9:]*|::*|:*:|*::*) printf 'SEED, ENV_GPU, POLICY_GPU, and EVAL_NUM must be nonnegative integers.\n' >&2; exit 2;; esac
	@$(call boolean_flag,DEEP,true)
	@$(call boolean_flag,DRY_RUN,true)

setup: _config-check ## Prepare submodules, locked env, assets, policy runtime, and checkpoint
	$(ROBODOJO_SETUP) setup $(POLICY_ARGS) $(SCENE_FLAG) $(ARGS)

preflight: _config-check ## Validate the configured experiment; DEEP=true checks policy readiness
	$(ROBODOJO_SIM) preflight $(EXPERIMENT_ARGS) $(DEEP_FLAG) $(ARGS)

eval: _config-check ## Evaluate locally and publish when PUBLISH=true
	$(ROBODOJO_SIM) eval $(EXPERIMENT_ARGS) --eval-num "$(EVAL_NUM)" $(PUBLISH_FLAG) $(DRY_RUN_FLAG) $(ARGS)

smoke: _config-check ## Run one local episode for each selected task
	$(ROBODOJO_SIM) smoke $(SWEEP_ARGS) $(DRY_RUN_FLAG) $(ARGS)

benchmark: _config-check ## Run the configured local benchmark sweep
	$(ROBODOJO_SIM) benchmark $(SWEEP_ARGS) --eval-num "$(EVAL_NUM)" $(DRY_RUN_FLAG) $(ARGS)

##@ Inspection
tasks: ## List canonical tasks
	$(ROBODOJO_BASE) tasks $(ARGS)

doctor: _config-check ## Inspect the configured simulator and policy adapter
	$(ROBODOJO_SIM) doctor --task "$(TASK)" --env-cfg "$(ENV_CFG)" $(SCENE_FLAG) --policy-dir "$(POLICY_DIR)" $(ARGS)

results: ## Summarize local evaluation results
	$(ROBODOJO_BASE) results summarize $(ARGS)

##@ Development
lint: ## Run Ruff lint checks
	$(UV_RUN_SIM) ruff check . $(ARGS)

lint-fix: ## Apply Ruff safe fixes
	$(UV_RUN_SIM) ruff check --fix . $(ARGS)

format: ## Format Python code
	$(UV_RUN_SIM) ruff format . $(ARGS)

format-check: ## Check Python formatting
	$(UV_RUN_SIM) ruff format --check . $(ARGS)

test: ## Run the test suite
	$(UV_RUN_SIM) pytest $(ARGS)

pre-commit: ## Run all pre-commit hooks
	$(UV_RUN_SIM) pre-commit run --all-files $(ARGS)

check: lint format-check test ## Run all non-mutating repository checks
	$(UV) run --locked robodojo tasks --format json --check >/dev/null
