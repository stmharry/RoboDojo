import pytest

from robodojo.core import storage
from robodojo.core.paths import RepositoryPaths
from robodojo.core.settings import RuntimeSettings


@pytest.fixture(autouse=True)
def clear_storage_environment(monkeypatch):
    for name in {"ROBODOJO_STORAGE_ROOT", "ROBODOJO_S3_URI", *RuntimeSettings.REMOVED_STORAGE_VARIABLES}:
        monkeypatch.delenv(name, raising=False)


def test_unset_storage_uses_repository_local_root():
    root = storage.REPO_ROOT / ".robodojo"
    assert storage.storage_root() == root
    assert storage.assets_root() == root / "assets"
    assert storage.data_root() == root / "datasets"
    assert storage.model_root() == root / "model_weights"
    assert storage.checkpoint_root() == root / "model_weights"
    assert storage.eval_root() == root / "runs/eval_result/RoboDojo"
    assert storage.eval_work_root() == storage.eval_root()
    assert storage.run_root() == root / "runs"
    assert storage.run_work_root() == storage.run_root()
    assert storage.summary_path() == root / "runs/reports/_summary.md"


def test_explicit_storage_root_controls_every_payload(monkeypatch, tmp_path):
    root = tmp_path / "local"
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(root))
    assert storage.assets_root() == root / "assets"
    assert storage.data_root() == root / "datasets"
    assert storage.eval_root() == root / "runs/eval_result/RoboDojo"
    assert storage.summary_path(tmp_path / "cli.md") == tmp_path / "cli.md"


@pytest.mark.parametrize("name", sorted(RuntimeSettings.REMOVED_STORAGE_VARIABLES))
def test_removed_storage_variables_fail_fast(monkeypatch, tmp_path, name):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "robodojo"\n', encoding="utf-8")
    monkeypatch.setenv(name, "/legacy")
    with pytest.raises(RuntimeError, match="removed storage variable"):
        RuntimeSettings.load(RepositoryPaths.resolve(root))


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


def test_docker_storage_supports_credentials_and_single_root():
    source = (storage.REPO_ROOT / "src/robodojo/workflows/docker.py").read_text(encoding="utf-8")
    assert "ROBODOJO_AWS_ENV_FILE" in source
    assert "AWS_PROFILE" in source
    assert "ROBODOJO_STORAGE_ROOT" in source
    assert "ROBODOJO_LOCAL_SCRATCH_ROOT" not in source
