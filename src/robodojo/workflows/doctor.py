"""Repository and runtime validation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from robodojo.core.calibration import load_hardware_calibration
from robodojo.core.models import SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile
from robodojo.core.storage import assets_root
from robodojo.sim.launcher import resolve_scene_profile
from robodojo.workflows.task_inventory import build_inventory


def run_doctor(
    paths: RepositoryPaths,
    task: str,
    env_config: str,
    policy_dir: Path | None = None,
    scene_config: str | None = None,
) -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    record("uv", shutil.which("uv") is not None, shutil.which("uv") or "not installed")
    record("git", shutil.which("git") is not None, shutil.which("git") or "not installed")
    record("git-lfs", subprocess.run(["git", "lfs", "version"], capture_output=True).returncode == 0, "git lfs")

    try:
        profile = load_environment_profile(paths, env_config, validate_calibration=False)
        record("environment config", True, str(profile.path))
        selected_scene = resolve_scene_profile(
            paths,
            SimulatorLaunchRequest(
                task=task,
                policy_name="doctor",
                port=1,
                env_config=env_config,
                scene_config=scene_config,
                additional_info="doctor",
            ),
        )
        calibration = profile.document.hardware_calibration
        if calibration:
            try:
                load_hardware_calibration(paths.environment_configs, calibration)
                record("hardware calibration", True, calibration)
            except ValueError as exc:
                record("hardware calibration", False, str(exc))
        record("scene profile", True, str(selected_scene.path))
        record("layout set", True, selected_scene.document.layout_set)
        component_paths = dict(profile.component_paths)
        component_paths["scene"] = selected_scene.component_path
        for kind, referenced in component_paths.items():
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
