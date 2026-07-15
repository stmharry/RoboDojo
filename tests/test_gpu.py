from pathlib import Path
import subprocess

from pydantic import ValidationError
import pytest

from robodojo.core import gpu
from robodojo.core.gpu import GpuSelectionError
from robodojo.core.models import EvaluationRequest, PolicyServerLaunchRequest, SimulatorLaunchRequest


def _inventory(monkeypatch, output: str, *, returncode: int = 0, stderr: str = "") -> None:
    monkeypatch.setattr(gpu.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, output, stderr),
    )


@pytest.mark.parametrize(("value", "expected"), [("auto", "auto"), ("0", 0), ("01", 1), (2, 2)])
def test_gpu_selector_parser_accepts_only_auto_or_nonnegative_indices(value, expected):
    assert gpu.parse_gpu_selector(value) == expected


@pytest.mark.parametrize("value", ["AUTO", " auto ", "-1", "1.5", -1, True])
def test_gpu_selector_parser_rejects_invalid_values(value):
    with pytest.raises(GpuSelectionError, match="expected 'auto' or a nonnegative integer"):
        gpu.parse_gpu_selector(value)


def test_explicit_pair_does_not_inspect_inventory(monkeypatch):
    monkeypatch.setattr(gpu, "discover_gpus", lambda: pytest.fail("explicit selectors queried inventory"))

    assignment = gpu.resolve_gpus(policy_gpu=3, env_gpu=1)

    assert assignment.policy_gpu == 3
    assert assignment.env_gpu == 1
    assert assignment.policy_source == assignment.env_source == "explicit"


def test_paired_auto_assignment_ranks_free_memory_then_index(monkeypatch):
    _inventory(monkeypatch, "3, 100\n1, 100\n2, 200\n")

    assignment = gpu.resolve_gpus(policy_gpu="auto", env_gpu="auto")

    assert assignment.env_gpu == 2
    assert assignment.policy_gpu == 1
    assert assignment.policy_source == assignment.env_source == "auto"


@pytest.mark.parametrize(
    ("policy_selector", "env_selector", "expected_policy", "expected_env"),
    [("auto", 1, 0, 1), (0, "auto", 0, 2)],
)
def test_auto_assignment_excludes_and_validates_explicit_peer(
    monkeypatch,
    policy_selector,
    env_selector,
    expected_policy,
    expected_env,
):
    _inventory(monkeypatch, "0, 300\n1, 100\n2, 200\n")

    assignment = gpu.resolve_gpus(policy_gpu=policy_selector, env_gpu=env_selector)

    assert assignment.policy_gpu == expected_policy
    assert assignment.env_gpu == expected_env


@pytest.mark.parametrize("role", ["policy", "env"])
def test_single_auto_role_supports_one_gpu(monkeypatch, role):
    _inventory(monkeypatch, "4, 12000\n")

    assignment = gpu.resolve_gpus(**{f"{role}_gpu": "auto"})

    assert getattr(assignment, f"{role}_gpu") == 4


def test_paired_auto_requires_two_distinct_gpus(monkeypatch):
    _inventory(monkeypatch, "4, 12000\n")

    with pytest.raises(GpuSelectionError, match="no distinct GPU is available for the policy"):
        gpu.resolve_gpus(policy_gpu="auto", env_gpu="auto")


def test_auto_rejects_an_unavailable_explicit_peer(monkeypatch):
    _inventory(monkeypatch, "0, 300\n1, 200\n")

    with pytest.raises(GpuSelectionError, match=r"\[7\].*unavailable"):
        gpu.resolve_gpus(policy_gpu="auto", env_gpu=7)


def test_concrete_pair_must_be_distinct_without_discovery(monkeypatch):
    monkeypatch.setattr(gpu, "discover_gpus", lambda: pytest.fail("collision queried inventory"))

    with pytest.raises(GpuSelectionError, match="must be distinct"):
        gpu.resolve_gpus(policy_gpu=1, env_gpu=1)


def test_concrete_preflight_validation_inspects_availability(monkeypatch):
    _inventory(monkeypatch, "0, 300\n2, 200\n")

    with pytest.raises(GpuSelectionError, match=r"\[1\].*unavailable"):
        gpu.validate_gpu_assignment(policy_gpu=1, env_gpu=2)


def test_missing_nvidia_smi_is_actionable(monkeypatch):
    monkeypatch.setattr(gpu.shutil, "which", lambda name: None)

    with pytest.raises(GpuSelectionError, match="nvidia-smi is unavailable"):
        gpu.discover_gpus()


@pytest.mark.parametrize(
    ("output", "returncode", "stderr", "message"),
    [
        ("", 1, "driver failed", "could not query GPU memory"),
        ("0, N/A\n", 0, "", "malformed nvidia-smi output"),
        ("0, 10\n0, 20\n", 0, "", "reported GPU index 0 more than once"),
        ("", 0, "", "reported no GPUs"),
    ],
)
def test_gpu_inventory_failures_are_actionable(monkeypatch, output, returncode, stderr, message):
    _inventory(monkeypatch, output, returncode=returncode, stderr=stderr)

    with pytest.raises(GpuSelectionError, match=message):
        gpu.discover_gpus()


def test_selection_models_default_to_auto_but_launch_models_require_indices(tmp_path: Path):
    selection = EvaluationRequest(
        policy_dir=tmp_path,
        task="stack_bowls",
        checkpoint="test",
        policy_env="test",
        env_config="arx_x5",
        policy_contract="arx_x5",
        protocol="stack_bowls",
        episode_horizon=800,
        native_eval_num=25,
        scene_config="default",
    )
    assert selection.policy_gpu == selection.env_gpu == "auto"

    with pytest.raises(ValidationError):
        PolicyServerLaunchRequest(
            policy_dir=tmp_path,
            task="stack_bowls",
            checkpoint="test",
            policy_env="test",
            env_config="arx_x5",
            policy_contract="arx_x5",
            policy_gpu="auto",
        )
    with pytest.raises(ValidationError):
        SimulatorLaunchRequest(
            task="stack_bowls",
            protocol_name="stack_bowls",
            episode_horizon=800,
            native_eval_num=25,
            policy_name="TestPolicy",
            port=19000,
            scene_config="default",
            env_gpu="auto",
            additional_info="test",
        )
