from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
RECIPE = "pi05-bimanual_yam-molmo_yam-general_pickup"


def run_make(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-f", str(MAKEFILE), *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_make_requires_an_explicit_recipe():
    result = run_make("-n", "eval")
    assert result.returncode != 0
    assert "RECIPE is required" in result.stderr
    assert "make recipes" in result.stderr


def test_make_rejects_removed_preset_surface_with_migration_error():
    result = run_make("-n", "eval", f"PRESET={RECIPE}")
    assert result.returncode != 0
    assert "PRESET has been removed" in result.stderr
    assert "RECIPE=<name>" in result.stderr


def test_make_recipe_catalog_is_owned_by_typed_yaml():
    source = MAKEFILE.read_text(encoding="utf-8")
    recipes = yaml.safe_load((ROOT / "configs/recipes.yml").read_text(encoding="utf-8"))["recipes"]
    assert "register_preset" not in source
    assert "PRESET." not in source
    assert len(recipes) == 24
    assert "molmoact2-bimanual_yam-moonlake_office-moonlake_office_general_pickup" in recipes
    assert "pi05-bimanual_yam-moonlake_office-moonlake_office_general_pickup" in recipes


def test_make_lists_recipes_through_the_cli():
    result = run_make("-n", "recipes")
    assert result.returncode == 0
    assert "robodojo" in result.stdout
    assert " recipes --format table" in result.stdout
    source = MAKEFILE.read_text(encoding="utf-8")
    assert "\n\t@$(ROBODOJO_BASE) recipes --format table $(ARGS)" in source


def test_every_recipe_renders_as_one_opaque_selection():
    recipes = yaml.safe_load((ROOT / "configs/recipes.yml").read_text(encoding="utf-8"))["recipes"]
    for recipe in recipes:
        result = run_make("-n", "setup", f"RECIPE={recipe}")
        assert result.returncode == 0, result.stderr
        assert f'--recipe "{recipe}"' in result.stdout
        assert "--task" not in result.stdout
        assert "--scene" not in result.stdout
        assert "--env-cfg" not in result.stdout


def test_make_eval_defaults_to_protocol_native_count():
    result = run_make("-n", "eval", f"RECIPE={RECIPE}")
    assert result.returncode == 0, result.stderr
    assert '--eval-num "native"' in result.stdout
    assert "--publish" not in result.stdout
    assert "--export-scene" not in result.stdout


def test_make_eval_forwards_explicit_one_episode_publication():
    result = run_make(
        "-n",
        "eval",
        f"RECIPE={RECIPE}",
        "EVAL_NUM=1",
        "PUBLISH=true",
        "EXPORT_SCENE=true",
        "POLICY_GPU=2",
        "ENV_GPU=3",
    )
    assert result.returncode == 0, result.stderr
    assert '--eval-num "1"' in result.stdout
    assert "--publish" in result.stdout
    assert "--export-scene" in result.stdout
    assert '--policy-gpu "2"' in result.stdout
    assert '--env-gpu "3"' in result.stdout


def test_make_sweeps_forward_only_explicit_recipe_lists():
    second = "molmoact2-bimanual_yam-molmo_yam-general_pickup"
    result = run_make(
        "-n",
        "smoke",
        f"RECIPES={RECIPE} {second}",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--recipe") == 2
    assert f'--recipe "{RECIPE}"' in result.stdout
    assert f'--recipe "{second}"' in result.stdout
    assert "--task" not in result.stdout


def test_make_rejects_invalid_controls():
    cases = (
        ("SEED=-1", "SEED must be a nonnegative integer"),
        ("EVAL_NUM=0", "EVAL_NUM must be 'native' or a positive integer"),
        ("EVAL_NUM=all", "EVAL_NUM must be 'native' or a positive integer"),
        ("POLICY_GPU=AUTO", "POLICY_GPU must be 'auto' or a nonnegative integer"),
        ("ENV_GPU=-1", "ENV_GPU must be 'auto' or a nonnegative integer"),
        ("PUBLISH=maybe", "PUBLISH must be true or false"),
    )
    for argument, message in cases:
        result = run_make("eval", f"RECIPE={RECIPE}", argument)
        assert result.returncode != 0
        assert message in result.stderr


def test_make_check_validates_tasks_and_recipes():
    result = run_make("-n", "check")
    assert result.returncode == 0, result.stderr
    assert "tasks --format json --check" in result.stdout
    assert "recipes --format json --check" in result.stdout
