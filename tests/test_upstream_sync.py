import json
from pathlib import Path
import subprocess

from typer.testing import CliRunner
import yaml

from robodojo.cli import app
from robodojo.core.models import UpstreamProject
from robodojo.core.paths import RepositoryPaths
from robodojo.workflows.upstream import check_upstreams, json_upstream_report

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _run(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repository: Path, files: dict[str, str], message: str) -> str:
    for relative, content in files.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _run(repository, "add", "-A")
    _run(repository, "commit", "-m", message)
    return _run(repository, "rev-parse", "HEAD")


def _write_manifest(root: Path, upstream: Path, reviewed: str) -> None:
    manifest = {
        "schema_version": 1,
        "projects": {
            "robodojo": {
                "repository": str(upstream),
                "ref": "main",
                "reviewed_commit": reviewed,
                "mappings": [
                    {
                        "kind": "exact",
                        "upstream": "README.md",
                        "local": "README.md",
                        "disposition": "adapted",
                        "comparison": "manual",
                    },
                    {
                        "kind": "prefix",
                        "upstream": "task/RoboDojo/tasks/",
                        "local": "src/robodojo/sim/tasks/",
                        "disposition": "mirrored",
                        "comparison": "python-body",
                    },
                    {
                        "kind": "prefix",
                        "upstream": "task/RoboDojo/config/",
                        "local": "configs/task/",
                        "disposition": "mirrored",
                        "comparison": "yaml-semantic",
                    },
                    {
                        "kind": "prefix",
                        "upstream": "env_cfg/scene/",
                        "local": "configs/scene/",
                        "disposition": "mirrored",
                        "comparison": "yaml-semantic",
                    },
                    {
                        "kind": "prefix",
                        "upstream": "env/",
                        "local": "src/robodojo/sim/environment/",
                        "disposition": "mirrored",
                        "comparison": "python-api",
                    },
                ],
            }
        },
        "intentional_divergences": [
            {
                "id": "fixture-layout",
                "upstream": "fixture upstream layout",
                "local": "fixture local layout",
                "rationale": "Exercise explicit adapter documentation.",
                "tests": ["tests/guard.py"],
            }
        ],
    }
    (root / "upstream_sync.yml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _set_reviewed(root: Path, commit: str) -> None:
    path = root / "upstream_sync.yml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest["projects"]["robodojo"]["reviewed_commit"] = commit
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[RepositoryPaths, Path, str]:
    upstream = tmp_path / "official"
    upstream.mkdir()
    _run(upstream, "init", "-b", "main")
    _run(upstream, "config", "user.email", "test@example.com")
    _run(upstream, "config", "user.name", "Test")
    task_source = """from env.environment.task_env import TaskEnv

class Example(TaskEnv):
    def run(self, value=1):
        return value
"""
    api_source = """class PublicAPI:
    def __init__(self, value=1):
        self.value = value

    def execute(self, item, option=None):
        return item, option
"""
    reviewed = _commit(
        upstream,
        {
            "README.md": "fixture\n",
            "task/RoboDojo/tasks/example.py": task_source,
            "task/RoboDojo/config/example.yml": "name: example\nvalue: 1\n",
            "env_cfg/scene/default.yml": "name: default\nobjects: []\n",
            "env/api.py": api_source,
        },
        "reviewed",
    )

    local = tmp_path / "local"
    (local / "src/robodojo/sim/tasks").mkdir(parents=True)
    (local / "src/robodojo/sim/environment").mkdir(parents=True)
    (local / "configs/task").mkdir(parents=True)
    (local / "configs/scene").mkdir(parents=True)
    (local / "tests").mkdir(parents=True)
    (local / "pyproject.toml").write_text('[project]\nname = "robodojo"\n', encoding="utf-8")
    (local / "README.md").write_text("local fixture\n", encoding="utf-8")
    (local / "tests/guard.py").write_text("# manifest guard\n", encoding="utf-8")
    (local / "src/robodojo/sim/tasks/example.py").write_text(
        task_source.replace("from env.", "from robodojo.sim.environment."),
        encoding="utf-8",
    )
    (local / "src/robodojo/sim/environment/api.py").write_text(
        api_source.replace("def __init__(self, value=1):", "def __init__(self, value=1, local_option=None):"),
        encoding="utf-8",
    )
    (local / "configs/task/example.yml").write_text("value: 1\nname: example\n", encoding="utf-8")
    (local / "configs/scene/default.yml").write_text("objects: []\nname: default\n", encoding="utf-8")
    _write_manifest(local, upstream, reviewed)
    return RepositoryPaths.resolve(local), upstream, reviewed


def test_upstream_check_is_clean_with_import_rewrites_yaml_reordering_and_optional_api_extensions(tmp_path):
    paths, upstream, _ = _fixture(tmp_path)

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 0
    assert report["status"] == "clean"
    assert report["projects"][0]["alignment"] == {"status": "clean", "checked": 4, "failures": []}
    assert json.loads(json_upstream_report(report))["status"] == "clean"

    result = runner.invoke(
        app,
        [
            "upstream",
            "check",
            "--project",
            "robodojo",
            "--source",
            str(upstream),
            "--format",
            "json",
            "--root",
            str(paths.root),
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "clean"


def test_remote_fetch_accepts_a_commit_ref(tmp_path):
    paths, _, reviewed = _fixture(tmp_path)

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, ref=reviewed)

    assert code == 0
    assert report["projects"][0]["current_commit"] == reviewed
    assert report["projects"][0]["ref"] == reviewed


def test_new_task_and_scene_are_mapped_then_fail_parity_if_baseline_advances_without_adoption(tmp_path):
    paths, upstream, _ = _fixture(tmp_path)
    current = _commit(
        upstream,
        {
            "task/RoboDojo/tasks/new_task.py": "def run():\n    return True\n",
            "task/RoboDojo/config/new_task.yml": "name: new_task\n",
            "env_cfg/scene/new_scene.yml": "name: new_scene\nobjects: []\n",
        },
        "new task and scene",
    )

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 1
    mapped = {change["upstream_path"]: change["local_path"] for change in report["projects"][0]["changes"]}
    assert mapped == {
        "env_cfg/scene/new_scene.yml": "configs/scene/new_scene.yml",
        "task/RoboDojo/config/new_task.yml": "configs/task/new_task.yml",
        "task/RoboDojo/tasks/new_task.py": "src/robodojo/sim/tasks/new_task.py",
    }
    result = runner.invoke(
        app,
        [
            "upstream",
            "check",
            "--project",
            "robodojo",
            "--source",
            str(upstream),
            "--format",
            "json",
            "--root",
            str(paths.root),
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "pending"

    _set_reviewed(paths.root, current)
    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 1
    assert report["projects"][0]["changes"] == []
    assert {failure["local_path"] for failure in report["projects"][0]["alignment"]["failures"]} == {
        "configs/scene/new_scene.yml",
        "configs/task/new_task.yml",
        "src/robodojo/sim/tasks/new_task.py",
    }


def test_renames_and_deletions_preserve_git_status_and_old_path(tmp_path):
    paths, upstream, _ = _fixture(tmp_path)
    _run(upstream, "mv", "task/RoboDojo/tasks/example.py", "task/RoboDojo/tasks/renamed.py")
    _run(upstream, "rm", "task/RoboDojo/config/example.yml")
    _run(upstream, "commit", "-m", "rename and delete")

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 1
    changes = report["projects"][0]["changes"]
    rename = next(change for change in changes if change["status"].startswith("R"))
    deletion = next(change for change in changes if change["status"] == "D")
    assert rename["old_path"] == "task/RoboDojo/tasks/example.py"
    assert rename["old_local_path"] == "src/robodojo/sim/tasks/example.py"
    assert rename["upstream_path"] == "task/RoboDojo/tasks/renamed.py"
    assert rename["local_path"] == "src/robodojo/sim/tasks/renamed.py"
    assert deletion["local_path"] == "configs/task/example.yml"


def test_unmapped_upstream_path_is_an_error(tmp_path):
    paths, upstream, _ = _fixture(tmp_path)
    _commit(upstream, {"new_surface.py": "value = 1\n"}, "unmapped")

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 2
    assert report["status"] == "error"
    assert report["projects"][0]["unmapped_paths"] == ["new_surface.py"]
    result = runner.invoke(
        app,
        [
            "upstream",
            "check",
            "--project",
            "robodojo",
            "--source",
            str(upstream),
            "--format",
            "json",
            "--root",
            str(paths.root),
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "error"


def test_non_ancestor_and_unavailable_reviewed_commits_are_errors(tmp_path):
    paths, upstream, reviewed = _fixture(tmp_path)
    _run(upstream, "checkout", "-b", "side", reviewed)
    side = _commit(upstream, {"README.md": "side\n"}, "side")
    _run(upstream, "checkout", "main")
    _set_reviewed(paths.root, side)

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)
    assert code == 2
    assert "not an ancestor" in report["projects"][0]["error"]

    _set_reviewed(paths.root, "f" * 40)
    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)
    assert code == 2
    assert "unavailable" in report["projects"][0]["error"]


def test_malformed_manifest_is_an_error(tmp_path):
    paths, upstream, _ = _fixture(tmp_path)
    (paths.root / "upstream_sync.yml").write_text("schema_version: 99\n", encoding="utf-8")

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 2
    assert report["projects"] == []
    assert "schema_version" in report["error"]


def test_task_body_and_public_api_drift_are_reported(tmp_path):
    paths, upstream, _ = _fixture(tmp_path)
    task = paths.root / "src/robodojo/sim/tasks/example.py"
    task.write_text(task.read_text(encoding="utf-8").replace("return value", "return value + 1"), encoding="utf-8")
    api = paths.root / "src/robodojo/sim/environment/api.py"
    api.write_text(
        api.read_text(encoding="utf-8").replace(
            "def execute(self, item, option=None):",
            "def execute(self, renamed, option=None):",
        ),
        encoding="utf-8",
    )

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 1
    failures = {
        failure["upstream_path"]: failure["message"] for failure in report["projects"][0]["alignment"]["failures"]
    }
    assert failures["task/RoboDojo/tasks/example.py"] == "Python bodies differ"
    assert "incompatible signatures: PublicAPI.execute" in failures["env/api.py"]


def test_manifest_pins_xpolicylab_to_a_descendant_of_official_main():
    manifest = yaml.safe_load((ROOT / "upstream_sync.yml").read_text(encoding="utf-8"))
    config = manifest["projects"]["xpolicylab"]
    official = config["reviewed_commit"]
    fork = config["local_fork"]["reviewed_commit"]
    submodule = ROOT / config["local_fork"]["path"]

    assert _run(ROOT, "rev-parse", "HEAD:XPolicyLab") == fork
    assert _run(submodule, "rev-parse", "HEAD") == fork
    subprocess.run(
        ["git", "-C", str(submodule), "merge-base", "--is-ancestor", official, fork],
        check=True,
    )
