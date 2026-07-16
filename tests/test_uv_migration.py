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
    assert "warp-lang==1.11.0" in sim
    assert not any(requirement.startswith("isaaclab-") for requirement in sim)
    assert not any(requirement.startswith("isaacsim") for requirement in project["dependencies"])
    assert uv["package"] is True
    assert {"torch", "torchvision", "torchaudio"} <= set(uv["sources"])
    assert not any(name.startswith("isaaclab") for name in uv["sources"])
    assert uv["sources"]["nvidia-curobo"] == {
        "git": "https://github.com/NVlabs/curobo.git",
        "rev": "3fd54dc782a82e5500a771cfd47856ea499d5fef",
    }
    assert (ROOT / "uv.lock").is_file()


def test_lock_pins_the_resolved_official_curobo_commit():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    curobo = next(package for package in lock["package"] if package["name"] == "nvidia-curobo")

    assert curobo["version"] == "0.8.0.post1.dev41"
    assert curobo["source"]["git"] == (
        "https://github.com/NVlabs/curobo.git"
        "?rev=3fd54dc782a82e5500a771cfd47856ea499d5fef"
        "#3fd54dc782a82e5500a771cfd47856ea499d5fef"
    )


def test_only_xpolicylab_remains_as_a_submodule():
    modules = configparser.ConfigParser()
    modules.read(ROOT / ".gitmodules")

    assert modules.sections() == ['submodule "XPolicyLab"']
    assert modules['submodule "XPolicyLab"']["path"] == "XPolicyLab"
    assert not (ROOT / "third_party/IsaacLab").exists()
    assert not (ROOT / "third_party/curobo").exists()


def test_simulator_preloads_the_locked_warp_before_app_launcher():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sim = config["project"]["optional-dependencies"]["sim"]
    source = (ROOT / "src/robodojo/sim/evaluation/main.py").read_text(encoding="utf-8")

    assert "warp-lang==1.11.0" in sim
    assert 'EXPECTED_WARP_VERSION = "1.11.0"' in source
    warp_import = source.index("import warp as _project_warp")
    warp_guard = source.index("if _loaded_warp_version != EXPECTED_WARP_VERSION")
    app_launcher_import = source.index("from isaaclab.app import AppLauncher")
    assert warp_import < warp_guard < app_launcher_import


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
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "ghcr.io/astral-sh/uv:0.11.21" in dockerfile
    assert "uv sync --extra sim --locked --no-dev --no-cache" in dockerfile
    assert 'ENTRYPOINT ["/workspace/RoboDojo/.venv/bin/robodojo"]' in dockerfile
    assert "COPY third_party/IsaacLab" not in dockerfile
    assert "COPY third_party/curobo" not in dockerfile
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO" not in dockerfile
    assert "third_party/curobo" not in dockerignore
    assert "**/.venv/" in dockerignore
    assert "miniconda" not in dockerfile.lower()
