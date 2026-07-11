from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_uv_project_packages_a_lightweight_core_and_sim_extra():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    uv = config["tool"]["uv"]

    assert project["requires-python"] == ">=3.11,<3.12"
    assert project["scripts"]["robodojo"] == "robodojo.cli:app"
    assert any(
        requirement.startswith("isaacsim[all,extscache]==5.1.0")
        for requirement in project["optional-dependencies"]["sim"]
    )
    assert not any(requirement.startswith("isaacsim") for requirement in project["dependencies"])
    assert uv["package"] is True
    assert {"torch", "torchvision", "torchaudio"} <= set(uv["sources"])
    assert (ROOT / "uv.lock").is_file()


def test_only_xpolicy_adapter_shell_remains():
    shells = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.glob("scripts/**/*.sh"))
    assert shells == ["scripts/eval_policy.sh"]
    shim = (ROOT / shells[0]).read_text(encoding="utf-8")
    assert "robodojo _adapter-client" in shim


def test_lightweight_imports_do_not_load_simulator_modules():
    code = (
        "import sys; import robodojo, robodojo.core, robodojo.server; "
        "assert not any(name.startswith(('isaacsim', 'isaaclab', 'torch')) for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_docker_uses_the_locked_sim_extra_and_cli_entrypoint():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ghcr.io/astral-sh/uv:0.11.21" in dockerfile
    assert "uv sync --extra sim --locked --no-dev --no-cache" in dockerfile
    assert 'ENTRYPOINT ["/workspace/RoboDojo/.venv/bin/robodojo"]' in dockerfile
    assert "miniconda" not in dockerfile.lower()
