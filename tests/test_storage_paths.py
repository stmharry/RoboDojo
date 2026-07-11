import subprocess

import pytest

from robodojo.core import storage

STORAGE_ENV = [
    "ROBODOJO_STORAGE_ROOT",
    "ROBODOJO_S3_URI",
    "ROBODOJO_LOCAL_SCRATCH_ROOT",
    "ROBODOJO_ASSETS_ROOT",
    "ROBODOJO_DATA_ROOT",
    "ROBO_DOJO_DATA_ROOT",
    "ROBODOJO_MODEL_ROOT",
    "ROBODOJO_CHECKPOINT_ROOT",
    "ROBODOJO_EVAL_ROOT",
    "ROBODOJO_EVAL_WORK_ROOT",
    "ROBODOJO_RUN_ROOT",
    "ROBODOJO_RUN_WORK_ROOT",
    "ROBODOJO_SUMMARY_PATH",
]


@pytest.fixture(autouse=True)
def clear_storage_environment(monkeypatch):
    for name in STORAGE_ENV:
        monkeypatch.delenv(name, raising=False)


def test_unset_storage_preserves_repo_local_defaults():
    assert storage.assets_root() == storage.REPO_ROOT / "Assets"
    assert storage.data_root() == storage.REPO_ROOT / "data"
    assert storage.eval_root() == storage.REPO_ROOT / "eval_result" / "RoboDojo"
    assert storage.eval_work_root() == storage.eval_root()
    assert storage.run_root() == storage.REPO_ROOT / "smoke_results"
    assert storage.summary_path() == storage.REPO_ROOT / "eval_result" / "RoboDojo" / "_summary.md"


def test_canonical_storage_layout_and_local_work(monkeypatch, tmp_path):
    durable = tmp_path / "mount"
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(durable))
    monkeypatch.setenv("ROBODOJO_LOCAL_SCRATCH_ROOT", str(scratch))

    assert storage.assets_root() == durable / "assets"
    assert storage.data_root() == durable / "datasets"
    assert storage.model_root() == durable / "model_weights"
    assert storage.checkpoint_root() == durable / "model_weights"
    assert storage.eval_root() == durable / "runs" / "eval_result" / "RoboDojo"
    assert storage.eval_work_root() == scratch / "eval_result" / "RoboDojo"
    assert storage.run_root() == durable / "runs"
    assert storage.run_work_root() == scratch / "runs"
    assert storage.summary_path() == scratch / "runs" / "reports" / "_summary.md"


def test_exact_overrides_and_legacy_data_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(tmp_path / "mount"))
    monkeypatch.setenv("ROBO_DOJO_DATA_ROOT", str(tmp_path / "legacy"))
    assert storage.data_root() == tmp_path / "legacy"
    monkeypatch.setenv("ROBODOJO_DATA_ROOT", str(tmp_path / "current"))
    monkeypatch.setenv("ROBODOJO_EVAL_ROOT", str(tmp_path / "eval"))
    assert storage.data_root() == tmp_path / "current"
    assert storage.eval_root() == tmp_path / "eval"


def test_summary_path_override_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(tmp_path / "mount"))
    monkeypatch.setenv("ROBODOJO_LOCAL_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("ROBODOJO_SUMMARY_PATH", str(tmp_path / "environment.md"))
    assert storage.summary_path() == tmp_path / "environment.md"
    assert storage.summary_path(tmp_path / "cli.md") == tmp_path / "cli.md"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("folding_final", "folding_final"),
        ("/storage/robodojo/model_weights/Pi/checkpoint-10", "checkpoint-10"),
        ("relative/checkpoint-20", "checkpoint-20"),
    ],
)
def test_checkpoint_labels_are_safe(value, expected):
    assert storage.checkpoint_label(value) == expected


def test_explicit_checkpoint_label_is_validated():
    assert storage.checkpoint_label("/some/path", "release-1") == "release-1"
    with pytest.raises(ValueError):
        storage.checkpoint_label("/some/path", "../escape")


def test_docker_storage_supports_protected_env_file_and_named_profile():
    source = (storage.REPO_ROOT / "src/robodojo/workflows/docker.py").read_text(encoding="utf-8")
    assert "ROBODOJO_AWS_ENV_FILE" in source
    assert "AWS_PROFILE" in source
    assert '"--env-file"' in source


def test_documented_data_compatibility_link_is_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "data"],
        cwd=storage.REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0
    assert "data" in (storage.REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
