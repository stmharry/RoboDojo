"""Sequential smoke and benchmark sweeps."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import time

from robodojo.core.gpu import GpuSelectionError, resolve_gpus
from robodojo.core.models import EvaluationRequest, SimulatorLaunchRequest, SmokeRecord, SmokeSummary, SweepRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.storage import run_work_root
from robodojo.orchestration.evaluation import run_evaluation
from robodojo.sim.launcher import resolve_scene_config
from robodojo.workflows.task_inventory import build_inventory

logger = logging.getLogger(__name__)


def _selected_tasks(request: SweepRequest) -> list[str]:
    runnable = [item["name"] for item in build_inventory()["tasks"] if item["runnable"]]
    selected = list(request.only)
    if request.tasks_file:
        selected.extend(
            line.strip()
            for line in request.tasks_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if selected:
        unknown = sorted(set(selected) - set(runnable))
        if unknown:
            raise ValueError(f"unknown task(s): {', '.join(unknown)}")
        wanted = set(selected)
        runnable = [name for name in runnable if name in wanted]
    if request.limit:
        runnable = runnable[: request.limit]
    return runnable


def _write_summary(summary: SmokeSummary, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    counts = {
        status: sum(row.status == status for row in summary.results) for status in ("PASS", "FAIL", "SKIP", "DRY_RUN")
    }
    lines = [
        f"# RoboDojo Sweep `{summary.run_id}`",
        "",
        f"- pass: `{counts['PASS']}`",
        f"- fail: `{counts['FAIL']}`",
        f"- skip: `{counts['SKIP']}`",
        f"- dry run: `{counts['DRY_RUN']}`",
        "",
        "| Status | Task | Exit | Seconds | Result | Log | Message |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row.status} | `{row.task}` | {row.exit_code} | {row.elapsed_sec:.2f} | "
        f"`{row.result_path}` | `{row.log_path}` | {row.message} |"
        for row in summary.results
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(paths: RepositoryPaths, request: SweepRequest) -> int:
    try:
        assignment = resolve_gpus(policy_gpu=request.policy_gpu, env_gpu=request.env_gpu)
    except GpuSelectionError as exc:
        logger.error("GPU selection failed: %s", exc)
        return 2
    request = request.model_copy(update={"policy_gpu": assignment.policy_gpu, "env_gpu": assignment.env_gpu})
    tasks = _selected_tasks(request)
    if tasks and not request.dry_run:
        from robodojo.workflows.preflight import emit_report, request_from_evaluation, run_sweep_preflight

        report = run_sweep_preflight(paths, request_from_evaluation(request, task=tasks[0]), tasks)
        emit_report(report)
        if report.status == "FAIL":
            return 2
    run_id = request.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_sweep")
    run_dir = run_work_root() / "smoke" / run_id
    summary_path = run_dir / "summary.json"
    markdown_path = run_dir / "summary.md"
    results: list[SmokeRecord] = []
    if request.resume and summary_path.is_file():
        prior = SmokeSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
        results.extend(prior.results)
    passed = {(row.task, row.scene_config) for row in results if row.status == "PASS"}

    for task in tasks:
        scene_config = resolve_scene_config(
            paths,
            SimulatorLaunchRequest(
                task=task,
                policy_name=request.policy_dir.name,
                port=1,
                env_config=request.env_config,
                scene_config=request.scene_config,
                additional_info="sweep",
            ),
        )
        if (task, scene_config) in passed:
            continue
        started = time.monotonic()
        evaluation = EvaluationRequest(
            **request.model_dump(
                exclude={"only", "tasks_file", "limit", "resume", "fail_fast", "run_id", "task", "scene_config"}
            ),
            task=task,
            scene_config=scene_config,
        )
        code = run_evaluation(paths, evaluation, preflight=False)
        status = "DRY_RUN" if request.dry_run else ("PASS" if code == 0 else "FAIL")
        record = SmokeRecord(
            status=status,
            task=task,
            scene_config=scene_config,
            exit_code=code,
            elapsed_sec=time.monotonic() - started,
            message="" if code == 0 else f"evaluation exited {code}",
        )
        results = [row for row in results if row.task != task]
        results.append(record)
        summary = SmokeSummary(run_id=run_id, eval_num=request.eval_num or 1, results=results)
        _write_summary(summary, summary_path, markdown_path)
        if code != 0 and request.fail_fast:
            return code
    return 1 if any(row.status == "FAIL" for row in results) else 0
