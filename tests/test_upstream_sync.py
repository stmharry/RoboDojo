import json
from pathlib import Path
import subprocess

from typer.testing import CliRunner
import yaml

from robodojo.cli import app
from robodojo.core.models import UpstreamProject
from robodojo.core.paths import RepositoryPaths
from robodojo.workflows.upstream import check_upstreams, json_upstream_report


def _run(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repository: Path, relative: str, content: str, message: str) -> str:
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _run(repository, "add", relative)
    _run(repository, "commit", "-m", message)
    return _run(repository, "rev-parse", "HEAD")


def _fixture(tmp_path: Path):
    upstream = tmp_path / "official"
    upstream.mkdir()
    _run(upstream, "init", "-b", "main")
    _run(upstream, "config", "user.email", "test@example.com")
    _run(upstream, "config", "user.name", "Test")
    reviewed = _commit(upstream, "env/base.py", "base\n", "base")

    root = tmp_path / "local"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "robodojo"\n', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "projects": {
            "robodojo": {
                "repository": str(upstream),
                "ref": "main",
                "reviewed_commit": reviewed,
                "mappings": [
                    {
                        "kind": "prefix",
                        "upstream": "env/",
                        "local": "src/robodojo/sim/environment/",
                        "disposition": "mapped",
                    }
                ],
            }
        },
    }
    (root / "upstream_sync.yml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return RepositoryPaths.resolve(root), upstream


def test_upstream_check_is_clean_at_reviewed_commit(tmp_path):
    paths, upstream = _fixture(tmp_path)

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 0
    assert report["status"] == "clean"
    assert report["projects"][0]["changes"] == []


def test_upstream_check_reports_mapped_pending_changes_and_json(tmp_path):
    paths, upstream = _fixture(tmp_path)
    _commit(upstream, "env/new.py", "new\n", "mapped")

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 1
    assert report["status"] == "pending"
    assert report["projects"][0]["changes"] == [
        {
            "upstream_path": "env/new.py",
            "local_path": "src/robodojo/sim/environment/new.py",
            "disposition": "mapped",
        }
    ]
    assert json.loads(json_upstream_report(report))["status"] == "pending"

    result = CliRunner().invoke(
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


def test_upstream_check_rejects_unmapped_changes(tmp_path):
    paths, upstream = _fixture(tmp_path)
    _commit(upstream, "new_surface.py", "new\n", "unmapped")

    report, code = check_upstreams(paths, project=UpstreamProject.ROBODOJO, source=upstream)

    assert code == 2
    assert report["status"] == "error"
    assert report["projects"][0]["changes"][0]["disposition"] == "unmapped"
