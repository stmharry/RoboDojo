
import pytest

from utils import storage

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


def test_exact_overrides_and_legacy_data_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(tmp_path / "mount"))
    monkeypatch.setenv("ROBO_DOJO_DATA_ROOT", str(tmp_path / "legacy"))
    assert storage.data_root() == tmp_path / "legacy"
    monkeypatch.setenv("ROBODOJO_DATA_ROOT", str(tmp_path / "current"))
    monkeypatch.setenv("ROBODOJO_EVAL_ROOT", str(tmp_path / "eval"))
    assert storage.data_root() == tmp_path / "current"
    assert storage.eval_root() == tmp_path / "eval"


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
    smoke = (storage.REPO_ROOT / "docker/smoke_docker.sh").read_text(encoding="utf-8")
    assert '[[ -r "${ROBODOJO_AWS_ENV_FILE}" ]]' in smoke
    assert 'storage_mounts+=( --env-file "${ROBODOJO_AWS_ENV_FILE}" )' in smoke
    assert 'storage_mounts+=( -e "AWS_PROFILE=${AWS_PROFILE}" )' in smoke
