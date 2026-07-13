import os

import pytest

from robodojo.core import storage
from robodojo.core.paths import RepositoryPaths
from robodojo.core.settings import RuntimeSettings


@pytest.fixture(autouse=True)
def clear_storage_environment(monkeypatch):
    for name in {"ROBODOJO_STORAGE_ROOT", "ROBODOJO_S3_URI", *RuntimeSettings.REMOVED_STORAGE_VARIABLES}:
        monkeypatch.delenv(name, raising=False)


def test_unset_storage_uses_default_checkout_root():
    root = storage._default_storage_checkout(storage.REPO_ROOT) / ".robodojo"
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


def test_normal_checkout_keeps_repository_local_storage(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text('[project]\nname = "robodojo"\n', encoding="utf-8")

    assert storage._default_storage_checkout(checkout) == checkout


def test_linked_worktree_uses_primary_checkout_storage(tmp_path):
    primary = tmp_path / "primary"
    common_gitdir = primary / ".git"
    worktree_gitdir = common_gitdir / "worktrees" / "feature"
    worktree_gitdir.mkdir(parents=True)
    (primary / "pyproject.toml").write_text('[project]\nname = "robodojo"\n', encoding="utf-8")

    linked = tmp_path / "linked"
    linked.mkdir()
    relative_gitdir = os.path.relpath(worktree_gitdir, start=linked)
    (linked / ".git").write_text(f"gitdir: {relative_gitdir}\n", encoding="utf-8")
    (worktree_gitdir / "commondir").write_text("../..\n", encoding="utf-8")

    assert storage._linked_worktree_primary(linked) == primary
    assert storage._default_storage_checkout(linked) == primary


@pytest.mark.parametrize("marker", ["not a gitdir", "gitdir: ", "gitdir: missing"])
def test_invalid_linked_worktree_metadata_stays_local(tmp_path, marker):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text(marker, encoding="utf-8")

    assert storage._default_storage_checkout(checkout) == checkout


def test_explicit_storage_root_controls_every_payload(monkeypatch, tmp_path):
    root = tmp_path / "local"
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(root))
    assert storage.storage_root() == root
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
