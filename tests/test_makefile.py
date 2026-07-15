from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MAKE_VARIABLES = (
    "PRESET",
    "POLICY_DIR",
    "POLICY_ENV",
    "CKPT",
    "ENV_CFG",
    "SCENE",
    "TASK",
    "DATASET",
    "ACTION_TYPE",
    "SEED",
    "ENV_GPU",
    "POLICY_GPU",
    "EVAL_NUM",
    "VERBOSITY",
    "PUBLISH",
    "EXPORT_SCENE",
    "DEEP",
    "DRY_RUN",
    "ONLY",
    "ARGS",
)
MACHINE_VARIABLES = ("ROBODOJO_STORAGE_ROOT", "ROBODOJO_S3_URI", "AWS_PROFILE")

PRESETS = (
    (
        "molmoact2-bimanual_yam-default-deposit_coin",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "deposit_coin",
    ),
    (
        "molmoact2-bimanual_yam-default-fasten_screws",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "fasten_screws",
    ),
    (
        "molmoact2-bimanual_yam-default-fold_clothes",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "fold_clothes",
    ),
    (
        "molmoact2-bimanual_yam-default-general_pickup",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "general_pickup",
    ),
    (
        "molmoact2-bimanual_yam-default-insert_key",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "insert_key",
    ),
    (
        "molmoact2-bimanual_yam-default-play_Xylophone",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "play_Xylophone",
    ),
    (
        "molmoact2-bimanual_yam-default-plug_in_charger",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "plug_in_charger",
    ),
    (
        "molmoact2-bimanual_yam-default-push_T",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "push_T",
    ),
    (
        "molmoact2-bimanual_yam-default-push_T_random",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "push_T_random",
    ),
    (
        "molmoact2-bimanual_yam-default-swap_T",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "default",
        "swap_T",
    ),
    (
        "molmoact2-bimanual_yam-molmo_yam-fold_clothes",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "molmo_yam",
        "fold_clothes",
    ),
    (
        "molmoact2-bimanual_yam-molmo_yam-general_pickup",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "molmo_yam",
        "general_pickup",
    ),
    (
        "molmoact2-bimanual_yam-moonlake_office-pack_item_into_container",
        "XPolicyLab/policy/MolmoACT2",
        "molmoact2",
        "molmoact2_bimanual_yam",
        "bimanual_yam",
        "moonlake_office",
        "pack_item_into_container",
    ),
    (
        "pi05-arx_x5-default-fold_clothes",
        "XPolicyLab/policy/Pi_05",
        "uv",
        "pi05_arx5_multitask_v1",
        "arx_x5",
        "default",
        "fold_clothes",
    ),
    (
        "pi05-bimanual_yam-molmo_yam-general_pickup",
        "XPolicyLab/policy/Pi_05",
        "uv",
        "pi05_yam_molmoact2",
        "bimanual_yam",
        "molmo_yam",
        "general_pickup",
    ),
    (
        "pi05-bimanual_yam-moonlake_office-general_pickup",
        "XPolicyLab/policy/Pi_05",
        "uv",
        "pi05_yam_molmoact2",
        "bimanual_yam",
        "moonlake_office",
        "general_pickup",
    ),
    (
        "pi05-bimanual_yam-moonlake_office-pack_item_into_container",
        "XPolicyLab/policy/Pi_05",
        "uv",
        "pi05_yam_molmoact2",
        "bimanual_yam",
        "moonlake_office",
        "pack_item_into_container",
    ),
    (
        "pi05-bimanual_yam-moonlake_office-stack_blocks",
        "XPolicyLab/policy/Pi_05",
        "uv",
        "pi05_yam_molmoact2",
        "bimanual_yam",
        "moonlake_office",
        "stack_blocks",
    ),
    (
        "pi05-bimanual_yam-moonlake_office-stack_bowls",
        "XPolicyLab/policy/Pi_05",
        "uv",
        "pi05_yam_molmoact2",
        "bimanual_yam",
        "moonlake_office",
        "stack_bowls",
    ),
    (
        "lerobot_pi05_openarm-openarm_lerobot-default-fold_clothes",
        "XPolicyLab/policy/LeRobot_Pi05_OpenArm",
        "lerobot-pi05",
        "folding_final",
        "openarm_lerobot",
        "default",
        "fold_clothes",
    ),
    (
        "smolvla-arx_x5-default-fold_clothes",
        "XPolicyLab/policy/SmolVLA",
        "smolvla",
        "smolvla-aloha-bimanual",
        "arx_x5",
        "default",
        "fold_clothes",
    ),
)


def run_make(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy() if env is None else env.copy()
    if env is None:
        for name in MAKE_VARIABLES:
            environment.pop(name, None)
    return subprocess.run(
        ["make", *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def make_probe(tmp_path: Path) -> tuple[Path, Path]:
    makefile = tmp_path / "Makefile"
    probe = tmp_path / "probe.mk"
    shutil.copy2(ROOT / "Makefile", makefile)
    probe.write_text(
        ".PHONY: machine-env\n"
        "machine-env:\n"
        '\t@printf \'%s|%s|%s\\n\' "$$ROBODOJO_STORAGE_ROOT" "$$ROBODOJO_S3_URI" "$$AWS_PROFILE"\n',
        encoding="utf-8",
    )
    return makefile, probe


def run_make_probe(
    makefile: Path,
    probe: Path,
    *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy() if env is None else env.copy()
    if env is None:
        for name in MACHINE_VARIABLES:
            environment.pop(name, None)
    return subprocess.run(
        ["make", "-f", str(makefile), "-f", str(probe), "machine-env", *arguments],
        cwd=makefile.parent.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_make_loads_and_exports_machine_defaults_from_repo_dotenv(tmp_path: Path):
    makefile, probe = make_probe(tmp_path)
    (tmp_path / ".env").write_text(
        "ROBODOJO_STORAGE_ROOT ?= $(ROOT_DIR)/storage\n"
        "ROBODOJO_S3_URI ?= s3://dotenv/robodojo\n"
        "AWS_PROFILE ?= dotenv-profile\n",
        encoding="utf-8",
    )

    result = run_make_probe(makefile, probe)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{tmp_path}/storage|s3://dotenv/robodojo|dotenv-profile"


def test_shell_and_make_arguments_override_dotenv_defaults(tmp_path: Path):
    makefile, probe = make_probe(tmp_path)
    (tmp_path / ".env").write_text(
        "ROBODOJO_STORAGE_ROOT ?= /dotenv/storage\n"
        "ROBODOJO_S3_URI ?= s3://dotenv/robodojo\n"
        "AWS_PROFILE ?= dotenv-profile\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "ROBODOJO_STORAGE_ROOT": "/shell/storage",
            "ROBODOJO_S3_URI": "s3://shell/robodojo",
            "AWS_PROFILE": "shell-profile",
        }
    )

    shell = run_make_probe(makefile, probe, env=environment)
    command_line = run_make_probe(
        makefile,
        probe,
        "ROBODOJO_STORAGE_ROOT=/make/storage",
        "ROBODOJO_S3_URI=s3://make/robodojo",
        "AWS_PROFILE=make-profile",
        env=environment,
    )

    assert shell.returncode == 0, shell.stderr
    assert shell.stdout.strip() == "/shell/storage|s3://shell/robodojo|shell-profile"
    assert command_line.returncode == 0, command_line.stderr
    assert command_line.stdout.strip() == "/make/storage|s3://make/robodojo|make-profile"


def test_make_dotenv_is_optional(tmp_path: Path):
    makefile, probe = make_probe(tmp_path)

    environment = os.environ.copy()
    for name in MACHINE_VARIABLES:
        environment.pop(name, None)
    machine = run_make_probe(makefile, probe, env=environment)
    presets = subprocess.run(
        ["make", "-f", str(makefile), "presets"],
        cwd=makefile.parent.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert machine.returncode == 0, machine.stderr
    assert machine.stdout.strip() == "||"
    assert presets.returncode == 0, presets.stderr
    assert len(presets.stdout.splitlines()) == len(PRESETS) + 1


def test_make_workflow_dotenv_defaults_preserve_shell_and_argument_overrides(tmp_path: Path):
    makefile, _ = make_probe(tmp_path)
    (tmp_path / ".env").write_text(
        "PUBLISH ?= true\n"
        "EXPORT_SCENE ?= true\n"
        "VERBOSITY ?= DEBUG\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    for name in MAKE_VARIABLES:
        environment.pop(name, None)

    dotenv = subprocess.run(
        ["make", "-f", str(makefile), "help"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    shell_environment = environment | {
        "PUBLISH": "false",
        "EXPORT_SCENE": "false",
        "VERBOSITY": "WARNING",
    }
    shell = subprocess.run(
        ["make", "-f", str(makefile), "help"],
        cwd=tmp_path,
        env=shell_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = subprocess.run(
        [
            "make",
            "-f",
            str(makefile),
            "help",
            "PUBLISH=false",
            "EXPORT_SCENE=false",
            "VERBOSITY=INFO",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PUBLISH=true EXPORT_SCENE=true VERBOSITY=DEBUG" in dotenv.stdout
    assert "PUBLISH=false EXPORT_SCENE=false VERBOSITY=WARNING" in shell.stdout
    assert "PUBLISH=false EXPORT_SCENE=false VERBOSITY=INFO" in arguments.stdout


def test_make_requires_experiment_selection_without_a_preset():
    result = run_make("-n", "eval")

    assert result.returncode != 0
    assert "TASK is required" in result.stderr
    assert "select PRESET=..., pass TASK=... to make, or export it" in result.stderr


def test_make_lists_the_complete_aligned_preset_catalog():
    result = run_make("presets")

    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert lines[0].split() == ["PRESET", "POLICY_DIR", "POLICY_ENV", "CKPT", "ENV_CFG", "SCENE", "TASK"]
    assert tuple(tuple(line.split()) for line in lines[1:]) == PRESETS


def test_every_make_preset_resolves_its_experiment_contract():
    for preset, policy_dir, policy_env, checkpoint, env_config, scene, task in PRESETS:
        result = run_make("-n", "setup", f"PRESET={preset}")

        assert result.returncode == 0, result.stderr
        assert f'--policy-dir "{policy_dir}"' in result.stdout
        assert f'--policy-env "{policy_env}"' in result.stdout
        assert f'--ckpt "{checkpoint}"' in result.stdout
        assert f'--env-cfg "{env_config}"' in result.stdout
        assert f'--scene "{scene}"' in result.stdout
        assert f'--task "{task}"' in result.stdout


def test_preset_precedence_preserves_explicit_make_overrides():
    preset = "pi05-bimanual_yam-molmo_yam-general_pickup"
    environment = os.environ.copy()
    for name in MAKE_VARIABLES:
        environment.pop(name, None)
    environment.update(
        {
            "POLICY_DIR": "stale-policy",
            "POLICY_ENV": "stale-env",
            "CKPT": "stale-checkpoint",
            "ENV_CFG": "stale-environment",
            "SCENE": "stale-scene",
            "TASK": "stale-task",
        }
    )

    selected = run_make("-n", "setup", f"PRESET={preset}", env=environment)
    overridden = run_make("-n", "setup", f"PRESET={preset}", "TASK=fold_clothes", env=environment)
    custom = run_make("-n", "setup", env=environment)

    assert selected.returncode == 0
    assert '--policy-dir "XPolicyLab/policy/Pi_05"' in selected.stdout
    assert '--task "general_pickup"' in selected.stdout
    assert "stale-" not in selected.stdout
    assert overridden.returncode == 0
    assert '--task "fold_clothes"' in overridden.stdout
    assert custom.returncode == 0
    assert '--policy-dir "stale-policy"' in custom.stdout
    assert '--task "stale-task"' in custom.stdout


def test_make_rejects_unknown_presets_with_a_catalog_hint():
    result = run_make("-n", "eval", "PRESET=does-not-exist")

    assert result.returncode != 0
    assert "unknown PRESET 'does-not-exist'" in result.stderr
    assert "make presets" in result.stderr


def test_make_eval_sequences_setup_once_and_preserves_mutation_free_dry_runs():
    arguments = (
        "PRESET=pi05-bimanual_yam-molmo_yam-general_pickup",
        "POLICY_GPU=0",
        "ENV_GPU=1",
        "PUBLISH=false",
        "ARGS=--eval-only-marker",
    )
    normal = run_make("-n", "eval", *arguments)
    dry_run = run_make("-n", "eval", *arguments, "DRY_RUN=true")

    assert normal.returncode == 0
    assert normal.stdout.count(" setup --policy-dir") == 1
    assert normal.stdout.count(" preflight --policy-dir") == 0
    assert normal.stdout.count(" eval --policy-dir") == 1
    assert normal.stdout.index(" setup --policy-dir") < normal.stdout.index(" eval --policy-dir")
    assert normal.stdout.count("--eval-only-marker") == 1
    assert dry_run.returncode == 0
    assert " setup --policy-dir" not in dry_run.stdout
    assert " preflight --policy-dir" not in dry_run.stdout
    assert dry_run.stdout.count(" eval --policy-dir") == 1
    assert "--dry-run" in dry_run.stdout


def test_make_eval_stops_before_launch_when_setup_fails(tmp_path: Path):
    marker = tmp_path / "evaluation-started"
    result = run_make(
        "eval",
        "PRESET=pi05-bimanual_yam-molmo_yam-general_pickup",
        "POLICY_GPU=0",
        "ENV_GPU=1",
        "PUBLISH=false",
        "ROBODOJO_SETUP=false",
        f"ROBODOJO_SIM=touch {marker}",
    )

    assert result.returncode != 0
    assert not marker.exists()


def test_make_validates_eval_controls_before_setup(tmp_path: Path):
    marker = tmp_path / "setup-started"
    result = run_make(
        "eval",
        "PRESET=pi05-bimanual_yam-molmo_yam-general_pickup",
        "PUBLISH=maybe",
        f"ROBODOJO_SETUP=touch {marker}",
        "ROBODOJO_SIM=true",
    )

    assert result.returncode != 0
    assert "PUBLISH must be true or false" in result.stderr
    assert not marker.exists()
