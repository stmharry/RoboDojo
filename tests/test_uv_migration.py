import configparser
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_uv_project_packages_a_lightweight_core_and_sim_extra():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    uv = config["tool"]["uv"]
    sim = project["optional-dependencies"]["sim"]

    assert project["requires-python"] == ">=3.11,<3.12"
    assert project["scripts"]["robodojo"] == "robodojo.cli:app"
    assert "isaaclab==2.3.2.post1" in sim
    assert "isaacsim[all,extscache]==5.1.0" in sim
    assert "nvidia-curobo[cu12]" in sim
    assert not any(requirement.startswith("isaaclab-") for requirement in sim)
    assert not any(requirement.startswith("isaacsim") for requirement in project["dependencies"])
    assert uv["package"] is True
    assert {"torch", "torchvision", "torchaudio"} <= set(uv["sources"])
    assert not any(name.startswith("isaaclab") for name in uv["sources"])
    assert uv["sources"]["nvidia-curobo"] == {
        "path": "third_party/curobo",
        "editable": True,
    }
    assert (ROOT / "uv.lock").is_file()


def test_only_xpolicylab_and_curobo_remain_as_submodules():
    modules = configparser.ConfigParser()
    modules.read(ROOT / ".gitmodules")

    assert modules.sections() == ['submodule "third_party/curobo"', 'submodule "XPolicyLab"']
    assert modules['submodule "third_party/curobo"']["path"] == "third_party/curobo"
    assert modules['submodule "XPolicyLab"']["path"] == "XPolicyLab"
    assert not (ROOT / "third_party/IsaacLab").exists()


def test_only_xpolicy_adapter_shell_remains():
    shells = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.glob("scripts/**/*.sh"))
    assert shells == ["scripts/eval_policy.sh"]
    shim = (ROOT / shells[0]).read_text(encoding="utf-8")
    assert "robodojo _adapter-client" in shim


def test_lightweight_imports_do_not_load_simulator_modules():
    code = (
        "import sys; import robodojo, robodojo.core, robodojo.policy, robodojo.orchestration; "
        "assert 'robodojo.sim' not in sys.modules; "
        "assert not any(name.startswith(('isaacsim', 'isaaclab', 'torch')) for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_lightweight_launchers_do_not_initialize_simulator_runtime():
    code = (
        "import sys; import robodojo.policy.adapter, robodojo.orchestration.evaluation; "
        "assert not any(name.startswith(('isaacsim', 'isaaclab', 'torch')) for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_removed_transport_packages_have_no_compatibility_shims():
    assert not (ROOT / "src/robodojo/client").exists()
    assert not (ROOT / "src/robodojo/server").exists()


def test_docker_uses_the_locked_sim_extra_and_cli_entrypoint():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ghcr.io/astral-sh/uv:0.11.21" in dockerfile
    assert "uv sync --extra sim --locked --no-dev --no-cache" in dockerfile
    assert 'ENTRYPOINT ["/workspace/RoboDojo/.venv/bin/robodojo"]' in dockerfile
    assert "COPY third_party/IsaacLab" not in dockerfile
    assert "COPY third_party/curobo" in dockerfile
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO" in dockerfile
    assert "miniconda" not in dockerfile.lower()
