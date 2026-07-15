#!/usr/bin/env python3
"""Manage canonical local RoboDojo storage and immutable S3 payloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from robodojo.core.storage import (
    eval_work_root,
    s3_uri,
    storage_root,
)

INTERNAL_FILES = {"_MANIFEST.json", "_COMPLETE.json"}
EXCLUDED_DIRS = {".cache", ".git"}
EXCLUDED_SUFFIXES = (".lock", ".partial", ".part", ".tmp", ".incomplete")
CANONICAL_TOP_LEVEL = {"assets", "datasets", "model_weights", "runs"}


def _aws(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["aws", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def _base_uri() -> str:
    base = s3_uri()
    if not base or not base.startswith("s3://"):
        raise SystemExit("ROBODOJO_S3_URI must be set to the dedicated s3://.../robodojo prefix")
    return base.rstrip("/")


def _destination(relative: str) -> str:
    clean = relative.strip("/")
    if not clean or ".." in Path(clean).parts:
        raise SystemExit(f"invalid storage destination: {relative!r}")
    if Path(clean).parts[0] not in CANONICAL_TOP_LEVEL:
        raise SystemExit(f"destination must use a canonical storage root: {relative!r}")
    destination = f"{_base_uri()}/{clean}"
    if not destination.startswith(_base_uri() + "/"):
        raise SystemExit("refusing destination outside ROBODOJO_S3_URI")
    return destination


def _relative_path(relative: str) -> Path:
    clean = relative.strip("/")
    path = Path(clean)
    if not clean or ".." in path.parts:
        raise SystemExit(f"invalid storage destination: {relative!r}")
    if path.parts[0] not in CANONICAL_TOP_LEVEL:
        raise SystemExit(f"destination must use a canonical storage root: {relative!r}")
    return path


def _s3_location(uri: str) -> tuple[str, str]:
    bucket_and_key = uri.removeprefix("s3://")
    if "/" not in bucket_and_key:
        return bucket_and_key, ""
    bucket, key = bucket_and_key.split("/", 1)
    return bucket, key


def _payload_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        excluded_part = next(
            (
                part
                for part in relative.parts
                if part in EXCLUDED_DIRS or part.endswith(EXCLUDED_SUFFIXES) or ".partial." in part or ".part-" in part
            ),
            None,
        )
        if excluded_part is not None or path.name in INTERNAL_FILES:
            continue
        if path.is_symlink():
            raise SystemExit(f"payload contains unsupported symlink: {path}")
        if path.is_file():
            files.append(path)
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(source: Path, relative: str) -> tuple[dict, dict]:
    files = _payload_files(source)
    entries = [
        {
            "path": str(path.relative_to(source)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    created = datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- system Python 3.10 compatibility
    manifest = {
        "schema_version": 1,
        "destination": relative.strip("/"),
        "created_at": created,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "files": entries,
    }
    complete = {
        "schema_version": 1,
        "created_at": created,
    }
    return manifest, complete


def publish(source: Path, relative: str, *, replace: bool = False, dry_run: bool = False) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise SystemExit(f"publish source is not a directory: {source}")
    destination = _destination(relative)
    if dry_run:
        sys.stdout.write(f"publish {source} -> {destination}\n")
        return

    bucket, key = _s3_location(destination)
    completed = (
        _aws(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            f"{key}/_COMPLETE.json",
            check=False,
        ).returncode
        == 0
    )
    if completed and not replace:
        raise SystemExit(f"destination is already complete: {destination}")
    if completed and replace:
        _aws("s3", "rm", destination, "--recursive", "--only-show-errors")

    manifest, complete = _metadata(source, relative)
    staging_root = storage_root() / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_root) as temporary:
        temporary_path = Path(temporary)
        manifest_path = temporary_path / "_MANIFEST.json"
        complete_path = temporary_path / "_COMPLETE.json"
        manifest_text = json.dumps(manifest, indent=2) + "\n"
        complete["manifest_sha256"] = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        manifest_path.write_text(manifest_text, encoding="utf-8")
        complete_path.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")

        sync_args = [
            "s3",
            "sync",
            str(source),
            destination,
            "--only-show-errors",
            "--exclude",
            ".git/*",
            "--exclude",
            "*/.git/*",
            "--exclude",
            ".cache/*",
            "--exclude",
            "*/.cache/*",
            "--exclude",
            "*.lock",
            "--exclude",
            "*.lock/*",
            "--exclude",
            "*.partial",
            "--exclude",
            "*.partial/*",
            "--exclude",
            "*.partial.*",
            "--exclude",
            "*.partial.*/*",
            "--exclude",
            "*.part",
            "--exclude",
            "*.part/*",
            "--exclude",
            "*.part-*",
            "--exclude",
            "*.part-*/*",
            "--exclude",
            "*.tmp",
            "--exclude",
            "*.tmp/*",
            "--exclude",
            "*.incomplete",
            "--exclude",
            "*.incomplete/*",
            "--exclude",
            "_MANIFEST.json",
            "--exclude",
            "_COMPLETE.json",
        ]
        result_path = source / "_result.json"
        if result_path.is_file():
            sync_args.extend(["--exclude", "_result.json"])
        _aws(*sync_args)
        _aws("s3", "cp", str(manifest_path), f"{destination}/_MANIFEST.json", "--only-show-errors")
        if result_path.is_file():
            _aws("s3", "cp", str(result_path), f"{destination}/_result.json", "--only-show-errors")
        _aws("s3", "cp", str(complete_path), f"{destination}/_COMPLETE.json", "--only-show-errors")
    sys.stdout.write(f"published {source} -> {destination}\n")


def _find_eval_run(run_id: str) -> Path:
    matches = [path.parent for path in eval_work_root().rglob("_result.json") if path.parent.name == run_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one completed local eval run named {run_id!r}, found {len(matches)}")
    return matches[0]


def publish_evaluation_run(run_id: str, *, replace: bool = False, dry_run: bool = False) -> None:
    """Publish one completed evaluation selected by its runtime identifier."""

    source = _find_eval_run(run_id)
    relative = str(Path("runs/eval_result/RoboDojo") / source.resolve().relative_to(eval_work_root().resolve()))
    publish(source, relative, replace=replace, dry_run=dry_run)


def _verify_materialized(path: Path) -> None:
    manifest_path = path / "_MANIFEST.json"
    complete_path = path / "_COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise SystemExit(f"durable payload is incomplete: {path}")
    manifest_bytes = manifest_path.read_bytes()
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if hashlib.sha256(manifest_bytes).hexdigest() != complete.get("manifest_sha256"):
        raise SystemExit(f"durable payload manifest hash mismatch: {path}")
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("files", [])
    expected = {entry["path"] for entry in entries}
    actual: set[str] = set()
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise SystemExit(f"materialized payload contains unsupported symlink: {candidate}")
        if candidate.is_file() and candidate.name not in INTERNAL_FILES:
            relative = str(candidate.relative_to(path))
            actual.add(relative)
            if relative not in expected:
                raise SystemExit(f"materialized payload contains unmanifested file: {candidate}")
    if actual != expected:
        missing = sorted(expected - actual)
        raise SystemExit(f"materialized payload is missing manifest files: {', '.join(missing)}")
    if manifest.get("file_count") != len(entries):
        raise SystemExit(f"materialized payload manifest file count mismatch: {path}")
    if manifest.get("total_bytes") != sum(entry["size"] for entry in entries):
        raise SystemExit(f"materialized payload manifest byte count mismatch: {path}")
    for entry in entries:
        candidate = path / entry["path"]
        if not candidate.is_file() or candidate.stat().st_size != entry["size"]:
            raise SystemExit(f"materialized payload failed size verification: {candidate}")
        if _sha256(candidate) != entry["sha256"]:
            raise SystemExit(f"materialized payload failed hash verification: {candidate}")


def pull(relative: str, *, replace: bool = False, dry_run: bool = False) -> None:
    """Download one completed S3 payload into its canonical local path."""
    relative_path = _relative_path(relative)
    source = _destination(relative)
    destination = storage_root() / relative_path
    if dry_run:
        sys.stdout.write(f"pull {source} -> {destination}\n")
        return
    destination_exists = destination.exists() or destination.is_symlink()
    if destination_exists and not replace:
        raise SystemExit(f"local destination already exists: {destination}")

    bucket, key = _s3_location(source)
    completed = (
        _aws(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            f"{key}/_COMPLETE.json",
            check=False,
        ).returncode
        == 0
    )
    if not completed:
        raise SystemExit(f"remote payload is incomplete: {source}")

    staging_root = storage_root() / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_root) as temporary:
        temporary_path = Path(temporary)
        payload = temporary_path / "payload"
        payload.mkdir()
        _aws("s3", "sync", source, str(payload), "--only-show-errors")
        _verify_materialized(payload)
        manifest = json.loads((payload / "_MANIFEST.json").read_text(encoding="utf-8"))
        if manifest.get("destination") != str(relative_path):
            raise SystemExit(
                f"remote manifest destination mismatch: {manifest.get('destination')!r} != {str(relative_path)!r}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = temporary_path / "backup"
        if destination_exists:
            os.replace(destination, backup)
        try:
            os.replace(payload, destination)
        except Exception:
            if (backup.exists() or backup.is_symlink()) and not (destination.exists() or destination.is_symlink()):
                os.replace(backup, destination)
            raise
        if backup.is_symlink() or backup.is_file():
            backup.unlink()
        else:
            shutil.rmtree(backup, ignore_errors=True)
    sys.stdout.write(f"pulled {source} -> {destination}\n")


def doctor() -> None:
    root = storage_root()
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".storage-doctor"
    probe.write_text("probe\n", encoding="utf-8")
    replacement = root / ".storage-doctor.replaced"
    os.replace(probe, replacement)
    replacement.unlink()
    sys.stdout.write(f"local storage supports replace/delete: {root}\n")
    if s3_uri() is not None:
        if shutil.which("aws") is None:
            raise SystemExit("aws CLI is not installed")
        bucket, key = _s3_location(_base_uri())
        _aws("s3api", "list-objects-v2", "--bucket", bucket, "--prefix", key, "--max-keys", "1")
        sys.stdout.write(f"S3 prefix accessible: {_base_uri()}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    subparsers.add_parser("status")
    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument("relative")
    pull_parser.add_argument("--replace", action="store_true")
    pull_parser.add_argument("--dry-run", action="store_true")

    def add_publish(name: str, positional: tuple[str, ...]) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        for argument in positional:
            child.add_argument(argument)
        child.add_argument("--replace", action="store_true")
        child.add_argument("--dry-run", action="store_true")
        return child

    add_publish("publish-assets", ("source",))
    add_publish("publish-data", ("dataset", "source"))
    add_publish("publish-model", ("policy", "model", "source"))
    add_publish("publish-checkpoint", ("policy", "checkpoint", "source"))
    add_publish("publish-reference-cache", ("name", "revision", "source"))
    eval_parser = add_publish("publish-eval", ("source",))
    eval_parser.add_argument("--run-id")
    add_publish("publish-run", ("kind", "run_id", "source"))
    add_publish("publish", ("source", "relative"))

    args = parser.parse_args(argv)
    if args.command in {"doctor", "status"}:
        doctor()
        return 0
    if args.command == "pull":
        pull(args.relative, replace=args.replace, dry_run=args.dry_run)
        return 0

    source = Path(args.source)
    if args.command == "publish":
        relative = args.relative
    elif args.command == "publish-assets":
        relative = "assets"
    elif args.command == "publish-data":
        relative = f"datasets/{args.dataset}"
    elif args.command == "publish-model":
        relative = f"model_weights/{args.policy}/{args.model}"
    elif args.command == "publish-checkpoint":
        relative = f"model_weights/{args.policy}/{args.checkpoint}"
    elif args.command == "publish-reference-cache":
        relative = f"datasets/reference_cache/{args.name}/{args.revision}"
    elif args.command == "publish-eval":
        if args.run_id:
            publish_evaluation_run(args.run_id, replace=args.replace, dry_run=args.dry_run)
            return 0
        relative = str(Path("runs/eval_result/RoboDojo") / source.resolve().relative_to(eval_work_root().resolve()))
    else:
        relative = f"runs/{args.kind}/{args.run_id}"
    publish(source, relative, replace=args.replace, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
