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
PRESET ?=
DATASET ?= RoboDojo
ACTION_TYPE ?= joint
SEED ?= 0
ENV_GPU ?= auto
POLICY_GPU ?= auto
EVAL_NUM ?= 1
VERBOSITY ?= INFO
PUBLISH ?= false
EXPORT_SCENE ?= false
DEEP ?= false
DRY_RUN ?= false
ONLY ?=
ARGS ?=

# Tracked experiment presets. Keep each registration on one line so the
# catalog remains easy to scan, diff, and render through `make presets`.
PRESETS :=

define register_preset
PRESETS += $(1)
PRESET.$(1).POLICY_DIR := $(2)
PRESET.$(1).POLICY_ENV := $(3)
PRESET.$(1).CKPT := $(4)
PRESET.$(1).ENV_CFG := $(5)
PRESET.$(1).SCENE := $(6)
PRESET.$(1).TASK := $(7)
endef

$(eval $(call register_preset,molmoact2-bimanual_yam-default-deposit_coin,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,deposit_coin))
$(eval $(call register_preset,molmoact2-bimanual_yam-default-fasten_screws,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,fasten_screws))
$(eval $(call register_preset,molmoact2-bimanual_yam-default-fold_clothes,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,fold_clothes))
$(eval $(call register_preset,molmoact2-bimanual_yam-default-general_pickup,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,general_pickup))
$(eval $(call register_preset,molmoact2-bimanual_yam-default-insert_key,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,insert_key))
$(eval $(call register_preset,molmoact2-bimanual_yam-default-play_Xylophone,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,play_Xylophone))
$(eval $(call register_preset,molmoact2-bimanual_yam-default-plug_in_charger,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,plug_in_charger))
$(eval $(call register_preset,molmoact2-bimanual_yam-default-push_T,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,push_T))
$(eval $(call register_preset,molmoact2-bimanual_yam-default-push_T_random,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,push_T_random))
$(eval $(call register_preset,molmoact2-bimanual_yam-default-swap_T,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,default,swap_T))
$(eval $(call register_preset,molmoact2-bimanual_yam-molmo_yam-fold_clothes,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,molmo_yam,fold_clothes))
$(eval $(call register_preset,molmoact2-bimanual_yam-molmo_yam-general_pickup,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_molmoact2,molmo_yam,general_pickup))
$(eval $(call register_preset,molmoact2-bimanual_yam-moonlake_office-pack_item_into_container,XPolicyLab/policy/MolmoACT2,molmoact2,molmoact2_bimanual_yam,bimanual_yam_moonlake_office,moonlake_office,pack_item_into_container))
$(eval $(call register_preset,pi05-arx_x5-default-fold_clothes,XPolicyLab/policy/Pi_05,uv,pi05_arx5_multitask_v1,arx_x5,default,fold_clothes))
$(eval $(call register_preset,pi05-bimanual_yam-molmo_yam-general_pickup,XPolicyLab/policy/Pi_05,uv,pi05_yam_molmoact2,bimanual_yam_molmoact2,molmo_yam,general_pickup))
$(eval $(call register_preset,pi05-bimanual_yam-moonlake_office-general_pickup,XPolicyLab/policy/Pi_05,uv,pi05_yam_molmoact2,bimanual_yam_moonlake_office,moonlake_office,general_pickup))
$(eval $(call register_preset,pi05-bimanual_yam-moonlake_office-pack_item_into_container,XPolicyLab/policy/Pi_05,uv,pi05_yam_molmoact2,bimanual_yam_moonlake_office,moonlake_office,pack_item_into_container))
$(eval $(call register_preset,pi05-bimanual_yam-moonlake_office-stack_blocks,XPolicyLab/policy/Pi_05,uv,pi05_yam_molmoact2,bimanual_yam_moonlake_office,moonlake_office,stack_blocks))
$(eval $(call register_preset,pi05-bimanual_yam-moonlake_office-stack_bowls,XPolicyLab/policy/Pi_05,uv,pi05_yam_molmoact2,bimanual_yam_moonlake_office,moonlake_office,stack_bowls))
$(eval $(call register_preset,lerobot_pi05_openarm-openarm_lerobot-default-fold_clothes,XPolicyLab/policy/LeRobot_Pi05_OpenArm,lerobot-pi05,folding_final,openarm_lerobot,default,fold_clothes))
$(eval $(call register_preset,smolvla-arx_x5-default-fold_clothes,XPolicyLab/policy/SmolVLA,smolvla,smolvla-aloha-bimanual,arx_x5,default,fold_clothes))

SELECTED_PRESET := $(strip $(PRESET))

ifneq ($(SELECTED_PRESET),)
ifeq ($(filter $(SELECTED_PRESET),$(PRESETS)),)
$(error unknown PRESET '$(PRESET)'; run 'make presets' to list valid names)
endif
POLICY_DIR := $(PRESET.$(SELECTED_PRESET).POLICY_DIR)
POLICY_ENV := $(PRESET.$(SELECTED_PRESET).POLICY_ENV)
CKPT := $(PRESET.$(SELECTED_PRESET).CKPT)
ENV_CFG := $(PRESET.$(SELECTED_PRESET).ENV_CFG)
SCENE := $(PRESET.$(SELECTED_PRESET).SCENE)
TASK := $(PRESET.$(SELECTED_PRESET).TASK)
endif

REQUIRED_EXPERIMENT_VARS := TASK ENV_CFG POLICY_DIR POLICY_ENV CKPT

define require_experiment
$(foreach name,$(REQUIRED_EXPERIMENT_VARS),$(if $(strip $($(name))),,$(error $(name) is required; select PRESET=..., pass $(name)=... to make, or export it; run 'make presets' to list presets)))
endef

define boolean_flag
$(if $(filter true,$(strip $($(1)))),$(2),$(if $(filter false,$(strip $($(1)))),,$(error $(1) must be true or false, got '$($(1))')))
endef

PUBLISH_FLAG = $(call boolean_flag,PUBLISH,--publish)
EXPORT_SCENE_FLAG = $(call boolean_flag,EXPORT_SCENE,--export-scene)
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

.PHONY: help presets setup preflight eval smoke benchmark tasks doctor results \
	assets-moonlake-office assets-moonlake-packing \
	lint lint-fix format format-check test pre-commit check _config-check _eval-config-check

##@ Workflow
help: ## Show the supported local workflow
	@printf 'RoboDojo local workflow\n\n  make presets -> make eval PRESET=<name>\n  optional machine defaults: .env (?= assignments)\n'
	@awk \
		'BEGIN {FS = ":.*## "} \
		/^##@ / {printf "\n%s\n", substr($$0, 5); next} \
		/^[a-zA-Z0-9_.-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' \
		"$(SELF)"
	@printf \
		'\nConfigured experiment\n  PRESET=%s\n  POLICY_DIR=%s POLICY_ENV=%s CKPT=%s\n  TASK=%s ENV_CFG=%s SCENE=%s SEED=%s EVAL_NUM=%s\n  PUBLISH=%s EXPORT_SCENE=%s VERBOSITY=%s\n  POLICY_GPU=%s ENV_GPU=%s\n' \
		"$(if $(SELECTED_PRESET),$(SELECTED_PRESET),custom)" "$(POLICY_DIR)" "$(POLICY_ENV)" "$(CKPT)" \
		"$(TASK)" "$(ENV_CFG)" "$(SCENE)" "$(SEED)" "$(EVAL_NUM)" \
		"$(PUBLISH)" "$(EXPORT_SCENE)" "$(VERBOSITY)" \
		"$(POLICY_GPU)" "$(ENV_GPU)"

presets: ## List tracked experiment presets
	@printf '%-68s %-39s %-14s %-24s %-18s %-17s %-24s\n' \
		'PRESET' 'POLICY_DIR' 'POLICY_ENV' 'CKPT' 'ENV_CFG' 'SCENE' 'TASK'
	@$(foreach preset,$(PRESETS),printf '%-68s %-39s %-14s %-24s %-18s %-17s %-24s\n' \
		'$(preset)' '$(PRESET.$(preset).POLICY_DIR)' '$(PRESET.$(preset).POLICY_ENV)' \
		'$(PRESET.$(preset).CKPT)' '$(PRESET.$(preset).ENV_CFG)' '$(PRESET.$(preset).SCENE)' \
		'$(PRESET.$(preset).TASK)';)

_config-check:
	$(call require_experiment)
	@case "$(SEED):$(EVAL_NUM)" in *[!0-9:]*|::*|:*:|*::*) printf 'SEED and EVAL_NUM must be nonnegative integers.\n' >&2; exit 2;; esac
	@case "$(POLICY_GPU)" in auto) ;; ''|*[!0-9]*) printf "POLICY_GPU must be 'auto' or a nonnegative integer.\n" >&2; exit 2;; esac
	@case "$(ENV_GPU)" in auto) ;; ''|*[!0-9]*) printf "ENV_GPU must be 'auto' or a nonnegative integer.\n" >&2; exit 2;; esac
	@$(call boolean_flag,DEEP,true)
	@$(call boolean_flag,DRY_RUN,true)

_eval-config-check: _config-check
	@$(call boolean_flag,PUBLISH,true)
	@$(call boolean_flag,EXPORT_SCENE,true)

setup: _config-check ## Prepare submodules, locked env, assets, policy runtime, and checkpoint
	$(ROBODOJO_SETUP) setup $(POLICY_ARGS) $(SCENE_FLAG) $(ARGS)

preflight: _config-check ## Validate the configured experiment; DEEP=true checks policy readiness
	$(ROBODOJO_SIM) preflight $(EXPERIMENT_ARGS) $(DEEP_FLAG) $(ARGS)

eval: _eval-config-check ## Prepare and evaluate locally with built-in fast preflight
ifneq ($(strip $(DRY_RUN)),true)
	$(ROBODOJO_SETUP) setup $(POLICY_ARGS) $(SCENE_FLAG)
endif
	$(ROBODOJO_SIM) eval $(EXPERIMENT_ARGS) --eval-num "$(EVAL_NUM)" $(EXPORT_SCENE_FLAG) $(PUBLISH_FLAG) $(DRY_RUN_FLAG) $(ARGS)

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
