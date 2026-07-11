from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_uv_project_owns_the_simulator_environment():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    uv = config["tool"]["uv"]

    assert project["requires-python"] == ">=3.11,<3.12"
    assert any(requirement.startswith("isaacsim[all,extscache]==5.1.0") for requirement in project["dependencies"])
    assert {"torch", "torchvision", "torchaudio"} <= set(uv["sources"])
    assert uv["sources"]["nvidia-curobo"]["editable"] is True
    assert (ROOT / "uv.lock").is_file()
    assert not (ROOT / "scripts/requirements.txt").exists()


def test_native_launchers_do_not_select_a_simulator_conda_environment():
    launcher = (ROOT / "scripts/robodojo.sh").read_text(encoding="utf-8")
    sweep = (ROOT / "scripts/internal/smoke_all_tasks.sh").read_text(encoding="utf-8")
    orchestrator = (ROOT / "scripts/internal/run_policy_eval.sh").read_text(encoding="utf-8")

    assert "--eval-env" not in launcher
    assert "--eval-env" not in sweep
    assert "setup_eval_env_client.sh" not in orchestrator
    assert 'CLIENT_SCRIPT="${ROOT_DIR}/scripts/eval_policy.sh"' in orchestrator


def test_docker_uses_the_locked_uv_environment():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ghcr.io/astral-sh/uv:0.11.21" in dockerfile
    assert "uv sync --locked --no-dev --no-cache" in dockerfile
    assert "COPY pyproject.toml uv.lock" in dockerfile
    assert "miniconda" not in dockerfile.lower()
    assert "python -m pip" not in dockerfile
