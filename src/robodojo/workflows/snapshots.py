"""Sequential first-frame snapshot batches and artifact reports."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from robodojo.core.artifacts.snapshots import (
    SNAPSHOT_SCHEMA_VERSION,
    normalize_recipe_metadata,
    normalize_snapshot_summary,
)
from robodojo.core.experiments.catalogs import load_recipe_catalog
from robodojo.core.experiments.selection import resolve_recipe
from robodojo.core.models.reports import (
    SnapshotRecord,
    SnapshotSummary,
)
from robodojo.core.models.requests import (
    SnapshotBatchRequest,
    SnapshotCaptureRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.core.storage import run_work_root, s3_uri
from robodojo.orchestration.snapshots import run_snapshot_capture
from robodojo.sim.scene_export.contracts import (
    ExportIdentity,
    completed_export_matches,
    normalize_export_manifest,
)
from robodojo.sim.scene_export.first_frame import FirstFrameIdentity, completed_first_frame_matches
from robodojo.workflows.snapshot_gallery import render_snapshot_gallery
from robodojo.workflows.storage import publish_snapshot_run

logger = logging.getLogger(__name__)


def _publication_prerequisites() -> int:
    remote = s3_uri()
    if remote is None or not remote.startswith("s3://"):
        logger.error("--publish requires ROBODOJO_S3_URI to name a dedicated s3:// prefix")
        return 2
    if shutil.which("aws") is None:
        logger.error("--publish requires the AWS CLI to be installed and available on PATH")
        return 2
    return 0


def _publish_snapshot(output: Path, run_id: str) -> int:
    from robodojo.workflows.errors import StorageError

    try:
        publish_snapshot_run(run_id, output)
    except StorageError as exc:
        logger.error("snapshot completed, but S3 publication failed: %s", exc)
        return 1
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        logger.error("snapshot completed, but S3 publication failed: %s", detail)
        return exc.returncode or 1
    except OSError as exc:
        logger.error("snapshot completed, but S3 publication failed: %s", exc)
        return 1
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _selected_recipes(paths: RepositoryPaths, request: SnapshotBatchRequest) -> list[str]:
    available = load_recipe_catalog(paths).recipes
    if not request.recipes:
        return sorted(available)
    unknown = sorted(set(request.recipes) - set(available))
    if unknown:
        raise ValueError(f"unknown recipe(s): {', '.join(unknown)}")
    return list(request.recipes)


def _output_root(request: SnapshotBatchRequest) -> Path:
    if request.output_dir is not None:
        return request.output_dir.expanduser().resolve()
    run_id = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return (run_work_root() / "snapshots" / run_id).resolve()


def _record(paths: RepositoryPaths, recipe_name: str, output: Path) -> SnapshotRecord:
    contract = resolve_recipe(paths, recipe_name)
    return SnapshotRecord(
        status="PENDING",
        recipe=recipe_name,
        policy=contract.policy_name,
        environment=contract.environment.name,
        scene=contract.scene.name,
        task_protocol=contract.task_protocol,
        task=contract.protocol.task,
        experiment_hash=contract.identity_hash,
        output_dir=str((output / recipe_name).resolve()),
    )


def _first_frame_identity(record: SnapshotRecord, summary: SnapshotSummary) -> FirstFrameIdentity:
    return FirstFrameIdentity(
        recipe=record.recipe,
        experiment_hash=record.experiment_hash,
        task=record.task,
        task_protocol=record.task_protocol,
        environment=record.environment,
        scene=record.scene,
        seed=summary.seed,
        layout_id=summary.layout_id,
    )


def _expected_recipe_identity(record: SnapshotRecord, summary: SnapshotSummary) -> dict[str, Any]:
    return {
        "recipe": record.recipe,
        "experiment_hash": record.experiment_hash,
        "policy": record.policy,
        "environment": record.environment,
        "scene": record.scene,
        "task_protocol": record.task_protocol,
        "task": record.task,
        "seed": summary.seed,
        "layout_id": summary.layout_id,
        "export_scene": summary.export_scene,
    }


def _validate_scene_snapshot(path: Path, record: SnapshotRecord, summary: SnapshotSummary) -> dict[str, Any]:
    manifest_path = path / "scene_manifest.json"
    try:
        manifest = normalize_export_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        identity = ExportIdentity(**manifest["identity"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"scene snapshot manifest is missing or invalid: {exc}") from exc
    expected = {
        "recipe": record.recipe,
        "experiment_hash": record.experiment_hash,
        "task": record.task,
        "task_protocol": record.task_protocol,
        "environment": record.environment,
        "scene": record.scene,
        "seed": summary.seed,
        "layout_id": summary.layout_id,
    }
    actual = {key: getattr(identity, key) for key in expected}
    if actual != expected or not completed_export_matches(path, identity):
        raise RuntimeError("scene snapshot does not match the requested recipe, seed, layout, and contract")
    return {
        "path": "scene_snapshot/scene_manifest.json",
        "sha256": _sha256_file(manifest_path),
        "format_version": manifest.get("format_version"),
    }


def _inspect_recipe_bundle(record: SnapshotRecord, summary: SnapshotSummary) -> dict[str, Any]:
    recipe_dir = Path(record.output_dir)
    first_frame_dir = recipe_dir / "first_frame"
    identity = _first_frame_identity(record, summary)
    if not completed_first_frame_matches(first_frame_dir, identity):
        raise RuntimeError("first-frame RGB bundle is incomplete or does not match the requested identity")
    first_metadata = first_frame_dir / "metadata.json"
    artifacts: dict[str, Any] = {
        "first_frame": {
            "path": "first_frame/metadata.json",
            "sha256": _sha256_file(first_metadata),
        }
    }
    if summary.export_scene:
        artifacts["scene_snapshot"] = _validate_scene_snapshot(recipe_dir / "scene_snapshot", record, summary)
    return artifacts


def _write_recipe_metadata(record: SnapshotRecord, summary: SnapshotSummary) -> None:
    recipe_dir = Path(record.output_dir)
    payload = {
        "format_version": SNAPSHOT_SCHEMA_VERSION,
        "complete": True,
        "created_at": datetime.now(UTC).isoformat(),
        "identity": _expected_recipe_identity(record, summary),
        "artifacts": _inspect_recipe_bundle(record, summary),
    }
    _atomic_text(recipe_dir / "metadata.json", json.dumps(payload, indent=2) + "\n")


def _completed_recipe_matches(record: SnapshotRecord, summary: SnapshotSummary) -> bool:
    try:
        metadata = json.loads((Path(record.output_dir) / "metadata.json").read_text(encoding="utf-8"))
        metadata = normalize_recipe_metadata(metadata)
        return bool(
            metadata.get("complete")
            and metadata.get("format_version") == SNAPSHOT_SCHEMA_VERSION
            and metadata.get("identity") == _expected_recipe_identity(record, summary)
            and metadata.get("artifacts") == _inspect_recipe_bundle(record, summary)
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _summary_markdown(summary: SnapshotSummary) -> str:
    counts = {status: sum(record.status == status for record in summary.results) for status in ("PASS", "SKIP", "FAIL")}
    lines = [
        f"# RoboDojo first-frame snapshots `{summary.run_id}`",
        "",
        f"- usable: `{counts['PASS'] + counts['SKIP']}`",
        f"- failed: `{counts['FAIL']}`",
        f"- seed/layout: `{summary.seed}/{summary.layout_id}`",
        f"- scene bundles: `{'enabled' if summary.export_scene else 'disabled'}`",
        "- gallery: [`index.html`](index.html)",
        "",
        "| Status | Recipe | Environment | Scene | Seconds | Output | Message |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    lines.extend(
        f"| {record.status} | `{record.recipe}` | `{record.environment}` | `{record.scene}` | "
        f"{record.elapsed_sec:.2f} | [`artifacts`]({record.recipe}/) | {record.message} |"
        for record in summary.results
    )
    return "\n".join(lines) + "\n"


def _write_reports(summary: SnapshotSummary, output: Path) -> None:
    _atomic_text(output / "summary.json", summary.model_dump_json(indent=2) + "\n")
    _atomic_text(output / "summary.md", _summary_markdown(summary))
    _atomic_text(output / "index.html", render_snapshot_gallery(summary))


def _load_resume(output: Path, expected: SnapshotSummary) -> SnapshotSummary:
    try:
        payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        previous = SnapshotSummary.model_validate(normalize_snapshot_summary(payload))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"--resume requires a valid snapshot summary in {output}: {exc}") from exc
    fields = ("output_dir", "seed", "layout_id", "export_scene", "recipes")
    mismatch = [field for field in fields if getattr(previous, field) != getattr(expected, field)]
    if mismatch:
        raise ValueError(f"snapshot resume identity differs in: {', '.join(mismatch)}")
    expected_by_recipe = {record.recipe: record for record in expected.results}
    if set(expected_by_recipe) != {record.recipe for record in previous.results}:
        raise ValueError("snapshot resume summary has a different recipe result set")
    for record in previous.results:
        expected_record = expected_by_recipe[record.recipe]
        normalized = record.model_copy(
            update={"status": expected_record.status, "exit_code": None, "elapsed_sec": 0.0, "message": ""}
        )
        if normalized != expected_record:
            raise ValueError(f"snapshot resume contract changed for recipe {record.recipe}")
    return previous


def run_snapshot_batch(paths: RepositoryPaths, request: SnapshotBatchRequest) -> int:
    """Capture selected recipes sequentially and maintain an offline gallery."""
    try:
        recipes = _selected_recipes(paths, request)
        output = _output_root(request)
        records = [_record(paths, recipe, output) for recipe in recipes]
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Snapshot selection failed: %s", exc)
        return 2

    run_id = output.name
    summary = SnapshotSummary(
        run_id=run_id,
        output_dir=str(output),
        seed=request.seed,
        layout_id=request.layout_id,
        export_scene=request.export_scene,
        recipes=tuple(recipes),
        results=records,
    )

    if request.publish and not request.dry_run:
        prerequisite_code = _publication_prerequisites()
        if prerequisite_code != 0:
            return prerequisite_code

    if request.dry_run:
        result = 0
        for record in summary.results:
            experiment = resolve_recipe(paths, record.recipe)
            capture = SnapshotCaptureRequest(
                experiment=experiment.spec(paths),
                environment_gpu=request.environment_gpu,
                output_dir=Path(record.output_dir),
                layout_id=request.layout_id,
                export_scene=request.export_scene,
                run_id=run_id,
                dry_run=True,
            )
            code = run_snapshot_capture(paths, capture)
            if code != 0:
                result = code
                if request.fail_fast:
                    return result
        return result

    try:
        if request.resume:
            summary = _load_resume(output, summary)
        elif output.exists():
            raise ValueError(f"snapshot output already exists; choose another --output-dir or use --resume: {output}")
        else:
            output.mkdir(parents=True)
        _write_reports(summary, output)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Snapshot output initialization failed: %s", exc)
        return 2

    for index, record in enumerate(summary.results):
        if record.status in {"PASS", "SKIP"}:
            if _completed_recipe_matches(record, summary):
                summary.results[index] = record.model_copy(
                    update={"status": "SKIP", "message": "reused exact completed bundle"}
                )
                _write_reports(summary, output)
                continue
            summary.results[index] = record.model_copy(
                update={"status": "FAIL", "exit_code": 2, "message": "previous completed bundle no longer matches"}
            )
            _write_reports(summary, output)
            if request.fail_fast:
                break
            continue

        experiment = resolve_recipe(paths, record.recipe)
        capture = SnapshotCaptureRequest(
            experiment=experiment.spec(paths),
            environment_gpu=request.environment_gpu,
            output_dir=Path(record.output_dir),
            layout_id=request.layout_id,
            export_scene=request.export_scene,
            run_id=run_id,
        )
        started = time.monotonic()
        message = ""
        try:
            code = run_snapshot_capture(paths, capture)
            if code == 0:
                _write_recipe_metadata(record, summary)
            else:
                message = f"snapshot process exited {code}"
        except Exception as exc:
            code = 1
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("Snapshot capture failed for %s", record.recipe)
        status = "PASS" if code == 0 else "FAIL"
        summary.results[index] = record.model_copy(
            update={
                "status": status,
                "exit_code": code,
                "elapsed_sec": time.monotonic() - started,
                "message": message,
            }
        )
        _write_reports(summary, output)
        if code != 0 and request.fail_fast:
            break

    summary.complete = not any(record.status == "PENDING" for record in summary.results)
    _write_reports(summary, output)
    logger.info("Snapshot gallery: %s", output / "index.html")
    if not summary.complete or any(record.status == "FAIL" for record in summary.results):
        return 1
    return _publish_snapshot(output, run_id) if request.publish else 0
