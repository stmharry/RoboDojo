# Opinionated local workflow shortcuts. Python owns workflow behavior; Make
# accepts experiment selection from arguments, the process environment, or an
# optional machine-local .env and translates it to CLI flags.

SELF := $(firstword $(MAKEFILE_LIST))
ROOT_DIR := $(patsubst %/,%,$(dir $(abspath $(SELF))))

# Make-only machine defaults. Use ?= assignments so explicit Make arguments
# and exported process variables retain precedence.
-include $(ROOT_DIR)/.env

SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
UV_RUN_SIM ?= $(UV) run --extra sim --locked
ROBODOJO_SETUP ?= $(UV) run --locked robodojo --log-level "$(VERBOSITY)"
ROBODOJO_BASE ?= $(UV) run --locked --no-sync robodojo --log-level "$(VERBOSITY)"
ROBODOJO_SIM ?= $(UV) run --extra sim --locked --no-sync robodojo --log-level "$(VERBOSITY)"
OMNI_KIT_ACCEPT_EULA ?= yes
export ROBODOJO_STORAGE_ROOT ROBODOJO_S3_URI AWS_PROFILE ROBODOJO_LOG_LEVEL OMNI_KIT_ACCEPT_EULA

# Stable benchmark default and Make-only workflow controls.
RECIPE ?=
RECIPES ?= $(RECIPE)
SEED ?= 0
ENV_GPU ?= auto
POLICY_GPU ?= auto
EVAL_NUM ?= native
VERBOSITY ?= INFO
PUBLISH ?= false
EXPORT_SCENE ?= false
DEEP ?= false
DRY_RUN ?= false
ONLY ?=
ARGS ?=

ifneq ($(strip $(PRESET)),)
$(error PRESET has been removed; use RECIPE=<name> and run 'make recipes' to list valid names)
endif

define require_experiment
$(if $(strip $(RECIPE)),,$(error RECIPE is required; run 'make recipes' to list valid recipes))
endef

define require_recipes
$(if $(strip $(RECIPES)),,$(error RECIPES is required; run 'make recipes' to list valid recipes))
endef

define boolean_flag
$(if $(filter true,$(strip $($(1)))),$(2),$(if $(filter false,$(strip $($(1)))),,$(error $(1) must be true or false, got '$($(1))')))
endef

PUBLISH_FLAG = $(call boolean_flag,PUBLISH,--publish)
EXPORT_SCENE_FLAG = $(call boolean_flag,EXPORT_SCENE,--export-scene)
DEEP_FLAG = $(call boolean_flag,DEEP,--deep)
DRY_RUN_FLAG = $(call boolean_flag,DRY_RUN,--dry-run)
CONTRACT_ARGS = --recipe "$(RECIPE)"

EXPERIMENT_ARGS = \
	$(CONTRACT_ARGS) \
	--seed "$(SEED)" \
	--policy-gpu "$(POLICY_GPU)" \
	--env-gpu "$(ENV_GPU)"


SWEEP_ARGS = \
	$(foreach recipe,$(RECIPES),--recipe "$(recipe)") \
	--seed "$(SEED)" \
	--policy-gpu "$(POLICY_GPU)" \
	--env-gpu "$(ENV_GPU)"

.PHONY: help recipes setup preflight eval smoke benchmark tasks doctor results \
	assets-moonlake-office assets-moonlake-packing \
	lint lint-fix format format-check test pre-commit check _config-check _sweep-config-check _eval-config-check

##@ Workflow
help: ## Show the supported local workflow
	@printf 'RoboDojo local workflow\n\n  make recipes -> make eval RECIPE=<name>\n  optional machine defaults: .env (?= assignments)\n'
	@awk \
		'BEGIN {FS = ":.*## "} \
		/^##@ / {printf "\n%s\n", substr($$0, 5); next} \
		/^[a-zA-Z0-9_.-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' \
		"$(SELF)"
	@printf \
		'\nConfigured experiment\n  RECIPE=%s SEED=%s EVAL_NUM=%s\n  PUBLISH=%s EXPORT_SCENE=%s VERBOSITY=%s\n  POLICY_GPU=%s ENV_GPU=%s\n' \
		"$(RECIPE)" "$(SEED)" "$(EVAL_NUM)" \
		"$(PUBLISH)" "$(EXPORT_SCENE)" "$(VERBOSITY)" \
		"$(POLICY_GPU)" "$(ENV_GPU)"

recipes: ## List tracked evaluation recipes
	$(ROBODOJO_BASE) recipes $(ARGS)

_config-check:
	$(call require_experiment)
	@case "$(SEED)" in ''|*[!0-9]*) printf 'SEED must be a nonnegative integer.\n' >&2; exit 2;; esac
	@case "$(EVAL_NUM)" in native) ;; ''|*[!0-9]*) printf "EVAL_NUM must be 'native' or a positive integer.\n" >&2; exit 2;; 0) printf "EVAL_NUM must be 'native' or a positive integer.\n" >&2; exit 2;; esac
	@case "$(POLICY_GPU)" in auto) ;; ''|*[!0-9]*) printf "POLICY_GPU must be 'auto' or a nonnegative integer.\n" >&2; exit 2;; esac
	@case "$(ENV_GPU)" in auto) ;; ''|*[!0-9]*) printf "ENV_GPU must be 'auto' or a nonnegative integer.\n" >&2; exit 2;; esac
	@$(call boolean_flag,DEEP,true)
	@$(call boolean_flag,DRY_RUN,true)

_eval-config-check: _config-check
	@$(call boolean_flag,PUBLISH,true)
	@$(call boolean_flag,EXPORT_SCENE,true)

setup: _config-check ## Prepare submodules, locked env, assets, policy runtime, and checkpoint
	$(ROBODOJO_SETUP) setup $(CONTRACT_ARGS) --seed "$(SEED)" --policy-gpu "$(POLICY_GPU)" $(ARGS)

preflight: _config-check ## Validate the configured experiment; DEEP=true checks policy readiness
	$(ROBODOJO_SIM) preflight $(EXPERIMENT_ARGS) $(DEEP_FLAG) $(ARGS)

eval: _eval-config-check ## Prepare and evaluate locally with built-in fast preflight
ifneq ($(strip $(DRY_RUN)),true)
	$(ROBODOJO_SETUP) setup $(CONTRACT_ARGS) --seed "$(SEED)" --policy-gpu "$(POLICY_GPU)"
endif
	$(ROBODOJO_SIM) eval $(EXPERIMENT_ARGS) --eval-num "$(EVAL_NUM)" $(EXPORT_SCENE_FLAG) $(PUBLISH_FLAG) $(DRY_RUN_FLAG) $(ARGS)

_sweep-config-check:
	$(call require_recipes)
	@case "$(SEED)" in ''|*[!0-9]*) printf 'SEED must be a nonnegative integer.\n' >&2; exit 2;; esac
	@case "$(EVAL_NUM)" in native) ;; ''|*[!0-9]*) printf "EVAL_NUM must be 'native' or a positive integer.\n" >&2; exit 2;; 0) printf "EVAL_NUM must be 'native' or a positive integer.\n" >&2; exit 2;; esac
	@case "$(POLICY_GPU)" in auto) ;; ''|*[!0-9]*) printf "POLICY_GPU must be 'auto' or a nonnegative integer.\n" >&2; exit 2;; esac
	@case "$(ENV_GPU)" in auto) ;; ''|*[!0-9]*) printf "ENV_GPU must be 'auto' or a nonnegative integer.\n" >&2; exit 2;; esac
	@$(call boolean_flag,DRY_RUN,true)

smoke: _sweep-config-check ## Run one local episode for each selected recipe
	$(ROBODOJO_SIM) smoke $(SWEEP_ARGS) $(DRY_RUN_FLAG) $(ARGS)

benchmark: _sweep-config-check ## Run the configured local benchmark sweep
	$(ROBODOJO_SIM) benchmark $(SWEEP_ARGS) --eval-num "$(EVAL_NUM)" $(DRY_RUN_FLAG) $(ARGS)

##@ Inspection
tasks: ## List canonical tasks
	$(ROBODOJO_BASE) tasks $(ARGS)

doctor: _config-check ## Inspect the configured simulator and policy adapter
	$(ROBODOJO_SIM) doctor $(CONTRACT_ARGS) $(ARGS)

results: ## Summarize local evaluation results
	$(ROBODOJO_BASE) results summarize $(ARGS)

##@ Assets
assets-moonlake-office: ## Build the pinned internal Moonlake office fixture
	$(ROBODOJO_SIM) assets build-moonlake-office $(ARGS)

assets-moonlake-packing: assets-moonlake-office ## Build internal Moonlake packing task assets
	$(ROBODOJO_SIM) assets build-moonlake-packing $(ARGS)

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
	$(UV) run --locked robodojo --log-level "$(VERBOSITY)" tasks --format json --check >/dev/null
	$(UV) run --locked robodojo --log-level "$(VERBOSITY)" recipes --format json --check >/dev/null
