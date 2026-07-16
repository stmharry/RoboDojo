from pathlib import Path
import shutil

from pydantic import ValidationError
import pytest
import yaml

from robodojo.core.experiments.selection import resolve_recipe
from robodojo.core.models.environment import EnvironmentVariant
from robodojo.core.models.requests import PreflightRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.environment import load_environment_profile
from robodojo.workflows.preflight import _configuration_checks

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)
PICKUP = "pi05-bimanual_yam-moonlake_office-moonlake_office_general_pickup"
SHIFTED = "pi05-bimanual_yam-moonlake_office-general_pickup"


def test_policy_descriptor_owns_execution_and_reference_setup():
    pickup = resolve_recipe(PATHS, PICKUP)
    shifted = resolve_recipe(PATHS, SHIFTED)

    assert pickup.policy.model_dump() == {
        "policy_dir": Path("XPolicyLab/policy/Pi_05"),
        "runtime": "uv",
        "checkpoint": "pi05_yam_abc_pickplace",
    }
    assert pickup.policy_descriptor.execution.model_dump() == {
        "strategy": "adaptive",
        "prediction_horizon": 50,
        "nominal_execution_horizon": 8,
        "maximum_execution_horizon": 50,
    }
    assert pickup.policy_reference_match == "reference_match"
    assert shifted.policy_reference_match == "domain_shift"
    assert shifted.policy_descriptor.training.reference_environments == ["bimanual_yam_molmoact2"]


def test_domain_shift_warns_without_blocking_composition():
    experiment = resolve_recipe(PATHS, SHIFTED)
    request = PreflightRequest(experiment=experiment.spec(PATHS))

    checks, environment, scene = _configuration_checks(PATHS, request)

    assert environment is not None
    assert scene is not None
    descriptor = next(check for check in checks if check.name == "policy_descriptor")
    assert descriptor.status == "WARN"
    assert "outside its declared reference setup" in descriptor.detail


def test_environment_variants_are_advisory_but_tuned_variants_require_provenance():
    reference = EnvironmentVariant.model_validate({"kind": "reference"})
    assert reference.derived_for is None
    with pytest.raises(ValidationError, match="must declare derived_for"):
        EnvironmentVariant.model_validate({"kind": "policy_tuned"})
    tuned = EnvironmentVariant.model_validate(
        {
            "kind": "policy_tuned",
            "derived_for": {
                "policy": "pi05_bimanual_yam_pickup",
                "scene": "moonlake_office",
                "task_protocol": "moonlake_office_general_pickup",
            },
        }
    )
    assert tuned.derived_for.policy == "pi05_bimanual_yam_pickup"


def test_environment_profiles_explicitly_own_generated_embodiment_assets():
    yam = load_environment_profile(PATHS, "bimanual_yam_moonlake_office")
    openarm = load_environment_profile(PATHS, "openarm_lerobot")
    arx = load_environment_profile(PATHS, "arx_x5")

    assert yam.document.asset_builds == ["yam"]
    assert yam.document.variant.kind == "reference"
    assert openarm.document.asset_builds == ["openarm"]
    assert arx.document.asset_builds == []


def _copy_contract_tree(target: Path) -> RepositoryPaths:
    shutil.copytree(ROOT / "configs", target / "configs")
    task_source = ROOT / "src/robodojo/sim/tasks"
    shutil.copytree(task_source, target / "src/robodojo/sim/tasks")
    for policy in ("MolmoACT2", "Pi_05", "LeRobot_Pi05_OpenArm", "SmolVLA"):
        destination = target / "XPolicyLab/policy" / policy
        destination.mkdir(parents=True)
        shutil.copy2(ROOT / "XPolicyLab/policy" / policy / "eval_contracts.yml", destination)
    return RepositoryPaths(root=target)


def test_experiment_hash_ignores_unrelated_catalog_entries_but_tracks_selected_descriptor(tmp_path):
    paths = _copy_contract_tree(tmp_path)
    original = resolve_recipe(paths, PICKUP).identity_hash

    policy_catalog = yaml.safe_load(paths.policy_profiles.read_text(encoding="utf-8"))
    policy_catalog["policies"]["smolvla_arx_x5"]["runtime"] = "unrelated-runtime-change"
    paths.policy_profiles.write_text(yaml.safe_dump(policy_catalog, sort_keys=False), encoding="utf-8")
    assert resolve_recipe(paths, PICKUP).identity_hash == original

    descriptor_path = tmp_path / "XPolicyLab/policy/Pi_05/eval_contracts.yml"
    descriptors = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    descriptors["profiles"]["pi05_yam_abc_pickplace"]["adapter"]["image_transform"] = "candidate_v2"
    descriptor_path.write_text(yaml.safe_dump(descriptors, sort_keys=False), encoding="utf-8")
    assert resolve_recipe(paths, PICKUP).identity_hash != original
