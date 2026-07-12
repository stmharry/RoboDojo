"""Repository and runtime validation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import yaml

from robodojo.core.calibration import calibration_name, load_hardware_calibration
from robodojo.core.models import EnvironmentConfigDocument
from robodojo.core.paths import RepositoryPaths
from robodojo.core.storage import assets_root
from robodojo.workflows.task_inventory import build_inventory


def run_doctor(paths: RepositoryPaths, task: str, env_config: str, policy_dir: Path | None = None) -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    record("uv", shutil.which("uv") is not None, shutil.which("uv") or "not installed")
    record("git", shutil.which("git") is not None, shutil.which("git") or "not installed")
    record("git-lfs", subprocess.run(["git", "lfs", "version"], capture_output=True).returncode == 0, "git lfs")

    config_path = paths.environment_configs / f"{env_config}.yml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        document = EnvironmentConfigDocument.model_validate({"config": payload.get("config", {})})
        record("environment config", True, str(config_path))
        calibration = calibration_name(payload)
        if calibration:
            try:
                load_hardware_calibration(paths.environment_configs, calibration)
                record("hardware calibration", True, calibration)
            except ValueError as exc:
                record("hardware calibration", False, str(exc))
        for kind, name in document.config.model_dump().items():
            referenced = paths.environment_configs / kind / f"{name}.yml"
            record(f"{kind} config", referenced.is_file(), str(referenced))
    except Exception as exc:
        record("environment config", False, str(exc))

    task_path = paths.task_configs / f"{task}.yml"
    record("task config", task_path.is_file(), str(task_path))
    inventory = build_inventory()
    broken = [item["name"] for item in inventory["tasks"] if not item["runnable"]]
    record("task inventory", not broken, ", ".join(broken) if broken else f"{inventory['counts']['runnable']} runnable")

    required_assets = ["Robots", "Object", "Material", "Eval_Layout"]
    assets = assets_root()
    missing_assets = [name for name in required_assets if not (assets / name).is_dir()]
    record("assets", not missing_assets, ", ".join(missing_assets) if missing_assets else str(assets))

    if policy_dir is not None:
        adapter = policy_dir.resolve() / "setup_eval_policy_server.sh"
        record("policy adapter", adapter.is_file(), str(adapter))

    print(json.dumps({"passed": sum(ok for _, ok, _ in checks), "total": len(checks)}))
    return 0 if all(ok for _, ok, _ in checks) else 1
