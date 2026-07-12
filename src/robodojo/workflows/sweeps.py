"""Sequential smoke and benchmark sweeps."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

from robodojo.core.models import EvaluationRequest, SmokeRecord, SmokeSummary, SweepRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.storage import run_work_root, storage_mode
from robodojo.orchestration.evaluation import run_evaluation
from robodojo.workflows.task_inventory import build_inventory


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
    run_id = request.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_sweep")
    run_dir = run_work_root() / "smoke" / run_id if storage_mode() else paths.root / "smoke_results" / run_id
    summary_path = run_dir / "summary.json"
    markdown_path = run_dir / "summary.md"
    results: list[SmokeRecord] = []
    if request.resume and summary_path.is_file():
        prior = SmokeSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
        results.extend(prior.results)
    passed = {row.task for row in results if row.status == "PASS"}

    for task in _selected_tasks(request):
        if task in passed:
            continue
        started = time.monotonic()
        evaluation = EvaluationRequest(
            **request.model_dump(exclude={"only", "tasks_file", "limit", "resume", "fail_fast", "run_id", "task"}),
            task=task,
        )
        code = run_evaluation(paths, evaluation)
        status = "DRY_RUN" if request.dry_run else ("PASS" if code == 0 else "FAIL")
        record = SmokeRecord(
            status=status,
            task=task,
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
