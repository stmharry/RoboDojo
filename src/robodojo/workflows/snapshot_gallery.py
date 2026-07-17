"""Render the offline first-frame snapshot inspection gallery."""

from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
from urllib.parse import quote

from robodojo.core.models.reports import (
    SnapshotRecord,
    SnapshotSummary,
)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _url(*parts: str) -> str:
    return "/".join(quote(part, safe="-_.") for part in parts)


def _options(records: list[SnapshotRecord], field: str, label: str) -> str:
    values = sorted({str(getattr(record, field)) for record in records})
    return f'<option value="">All {label}</option>' + "".join(
        f'<option value="{_escape(value)}">{_escape(value)}</option>' for value in values
    )


def _first_frame_artifacts(root: Path, record: SnapshotRecord) -> tuple[str | None, list[tuple[str, str]]]:
    metadata_path = root / record.recipe / "first_frame" / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None, []
    artifacts = metadata.get("artifacts", {})
    sheet = artifacts.get("contact_sheet", {}).get("path")
    cameras = artifacts.get("cameras", {})
    links = []
    if isinstance(cameras, dict):
        for camera_name in metadata.get("camera_order", []):
            artifact = cameras.get(camera_name, {})
            relative = artifact.get("path")
            if isinstance(relative, str):
                links.append((str(camera_name), _url(record.recipe, "first_frame", relative)))
    sheet_url = _url(record.recipe, "first_frame", sheet) if isinstance(sheet, str) else None
    return sheet_url, links


def _card(root: Path, record: SnapshotRecord, index: int, total: int, export_scene: bool) -> str:
    complete = record.status in {"PASS", "SKIP"}
    sheet_url, cameras = _first_frame_artifacts(root, record) if complete else (None, [])
    search = " ".join(
        (
            record.recipe,
            record.policy,
            record.environment,
            record.scene,
            record.task_protocol,
            record.task,
            record.status,
        )
    ).lower()
    status_label = "Reused" if record.status == "SKIP" else record.status.title()
    if sheet_url:
        visual = (
            f'<a class="frame-window" href="{sheet_url}" aria-label="Open {_escape(record.recipe)} contact sheet">'
            f'<span class="frame-boundary">POST-RESET&nbsp;&nbsp;/&nbsp;&nbsp;PRE-ACTION</span>'
            f'<img src="{sheet_url}" alt="First-frame camera contact sheet for '
            f'{_escape(record.recipe)}" loading="lazy">'
            "</a>"
        )
    else:
        detail = record.message or (
            "Capture has not started." if record.status == "PENDING" else "No RGB bundle found."
        )
        visual = (
            '<div class="frame-window frame-empty">'
            '<span class="frame-boundary">NO FRAME AVAILABLE</span>'
            f"<p>{_escape(detail)}</p>"
            "</div>"
        )

    camera_links = "".join(f'<a href="{href}">{_escape(camera_name)}</a>' for camera_name, href in cameras)
    if camera_links:
        camera_links += f'<a href="{_url(record.recipe, "first_frame", "metadata.json")}">RGB metadata</a>'

    usd_links = ""
    if export_scene and complete:
        usd_links = "".join(
            (
                f'<a href="{_url(record.recipe, "scene_snapshot", filename)}">{label}</a>'
                for label, filename in (
                    ("Referenced USDA", "scene_referenced.usda"),
                    ("Flattened USDC", "scene_flattened.usdc"),
                    ("Preview USDZ", "scene_preview.usdz"),
                    ("Scene manifest", "scene_manifest.json"),
                )
            )
        )

    return f"""
    <article class="capture-card status-{record.status.lower()}"
      data-search="{_escape(search)}" data-policy="{_escape(record.policy)}"
      data-environment="{_escape(record.environment)}" data-scene="{_escape(record.scene)}"
      data-task="{_escape(record.task)}" data-status="{_escape(record.status)}">
      <div class="specimen-rail" aria-hidden="true">
        <span>T+000</span><strong>{index:02d}</strong><small>/ {total:02d}</small>
      </div>
      <div class="card-body">
        <header class="card-header">
          <div>
            <p class="eyebrow">{_escape(record.task)} · {_escape(record.scene)}</p>
            <h2>{_escape(record.recipe)}</h2>
          </div>
          <span class="status-chip">{_escape(status_label)}</span>
        </header>
        {visual}
        <dl class="identity-grid">
          <div><dt>Policy</dt><dd>{_escape(record.policy)}</dd></div>
          <div><dt>Environment</dt><dd>{_escape(record.environment)}</dd></div>
          <div><dt>Task protocol</dt><dd>{_escape(record.task_protocol)}</dd></div>
          <div><dt>Elapsed</dt><dd>{record.elapsed_sec:.1f}s</dd></div>
        </dl>
        <nav class="artifact-links" aria-label="Artifacts for {_escape(record.recipe)}">
          {camera_links}{usd_links}
        </nav>
      </div>
    </article>
    """


def render_snapshot_gallery(summary: SnapshotSummary) -> str:
    """Return a self-contained gallery that works directly from ``file://``."""
    records = summary.results
    root = Path(summary.output_dir)
    counts = Counter(record.status for record in records)
    finished = sum(counts[status] for status in ("PASS", "SKIP", "FAIL", "DRY_RUN"))
    cards = "".join(
        _card(root, record, index, len(records), summary.export_scene) for index, record in enumerate(records, 1)
    )
    page = _PAGE_TEMPLATE
    replacements = {
        "__RUN_ID__": _escape(summary.run_id),
        "__OUTPUT_DIR__": _escape(summary.output_dir),
        "__SEED__": str(summary.seed),
        "__LAYOUT__": str(summary.layout_id),
        "__FINISHED__": str(finished),
        "__TOTAL__": str(len(records)),
        "__PASSED__": str(counts["PASS"] + counts["SKIP"]),
        "__FAILED__": str(counts["FAIL"]),
        "__BUNDLE_MODE__": "RGB + USD" if summary.export_scene else "RGB only",
        "__POLICY_OPTIONS__": _options(records, "policy", "policies"),
        "__ENVIRONMENT_OPTIONS__": _options(records, "environment", "environments"),
        "__SCENE_OPTIONS__": _options(records, "scene", "scenes"),
        "__TASK_OPTIONS__": _options(records, "task", "tasks"),
        "__STATUS_OPTIONS__": _options(records, "status", "statuses"),
        "__CARDS__": cards,
    }
    for marker, value in replacements.items():
        page = page.replace(marker, value)
    return page


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>RoboDojo first-frame inspection · __RUN_ID__</title>
  <style>
    :root {
      --bench: #dce5eb;
      --surface: #f7fafb;
      --ink: #14212b;
      --muted: #60727e;
      --line: #aab8c2;
      --cobalt: #2458d3;
      --optical: #5fb8c8;
      --amber: #d7812a;
      --fault: #b43e3e;
      --display: "Arial Narrow", "Aptos Narrow", "Roboto Condensed", sans-serif;
      --body: Inter, Aptos, system-ui, -apple-system, sans-serif;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }
    * { box-sizing: border-box; }
    html { background: var(--bench); color: var(--ink); font-family: var(--body); }
    body { margin: 0; min-width: 300px; }
    a { color: inherit; }
    a:focus-visible, input:focus-visible, select:focus-visible {
      outline: 3px solid var(--cobalt); outline-offset: 3px;
    }
    .masthead {
      display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(260px, .8fr);
      gap: 2rem; padding: clamp(2rem, 6vw, 5.5rem); border-bottom: 1px solid var(--line);
      background:
        linear-gradient(115deg, rgba(95, 184, 200, .19), transparent 38%),
        repeating-linear-gradient(90deg, transparent 0 79px, rgba(20, 33, 43, .045) 80px);
    }
    .kicker, .eyebrow, dt, .run-label {
      font-family: var(--mono); font-size: .68rem; letter-spacing: .11em; text-transform: uppercase;
    }
    .kicker { margin: 0 0 1.4rem; color: var(--cobalt); font-weight: 700; }
    h1 {
      max-width: 12ch; margin: 0; font-family: var(--display); font-size: clamp(3.4rem, 8vw, 7.5rem);
      font-stretch: condensed; font-weight: 800; letter-spacing: -.055em; line-height: .82; text-transform: uppercase;
    }
    .thesis { max-width: 58ch; margin: 1.8rem 0 0; color: var(--muted); font-size: 1rem; line-height: 1.65; }
    .run-plate {
      align-self: end; border-top: 7px solid var(--ink); background: rgba(247, 250, 251, .74);
      box-shadow: 8px 8px 0 rgba(20, 33, 43, .09); padding: 1.35rem;
    }
    .run-plate code {
      display: block; margin: .35rem 0 1.2rem; font: 700 1rem/1.4 var(--mono); overflow-wrap: anywhere;
    }
    .run-stats { display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid var(--line); }
    .run-stats div { padding: 1rem .5rem 0; border-right: 1px solid var(--line); }
    .run-stats div:last-child { border: 0; }
    .run-stats strong { display: block; font: 800 1.8rem/1 var(--display); }
    .run-stats span { color: var(--muted); font: .65rem var(--mono); text-transform: uppercase; }
    .controls {
      position: sticky; top: 0; z-index: 5; display: grid;
      grid-template-columns: minmax(220px, 2fr) repeat(5, minmax(125px, 1fr)); gap: .65rem;
      padding: 1rem clamp(1rem, 4vw, 3rem); background: rgba(220, 229, 235, .94);
      border-bottom: 1px solid var(--line); backdrop-filter: blur(14px);
    }
    .controls label { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); }
    .controls input, .controls select {
      width: 100%; min-height: 2.8rem; border: 1px solid var(--line); border-radius: 0;
      background: var(--surface); color: var(--ink); padding: .65rem .8rem; font: .75rem var(--mono);
    }
    .gallery-shell { padding: clamp(1rem, 4vw, 3rem); }
    .gallery-count { margin: 0 0 1rem; color: var(--muted); font: .72rem var(--mono); }
    .gallery { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.2rem; }
    .capture-card {
      display: grid; grid-template-columns: 54px minmax(0, 1fr); min-width: 0; background: var(--surface);
      border: 1px solid var(--line); box-shadow: 5px 5px 0 rgba(20, 33, 43, .08);
    }
    .capture-card[hidden] { display: none; }
    .specimen-rail {
      display: flex; align-items: center; gap: .45rem; padding: .9rem 0; writing-mode: vertical-rl;
      transform: rotate(180deg); background: var(--ink); color: var(--surface); font-family: var(--mono);
    }
    .specimen-rail span { color: var(--optical); font-size: .68rem; letter-spacing: .12em; }
    .specimen-rail strong { margin-top: auto; font-size: 1.25rem; }
    .specimen-rail small { color: var(--line); }
    .card-body { min-width: 0; padding: 1.1rem; }
    .card-header { display: flex; align-items: start; justify-content: space-between; gap: 1rem; margin-bottom: 1rem; }
    .eyebrow { margin: 0 0 .35rem; color: var(--muted); }
    h2 { margin: 0; font: 800 clamp(1.15rem, 2vw, 1.65rem)/1.05 var(--display); overflow-wrap: anywhere; }
    .status-chip {
      flex: 0 0 auto; padding: .42rem .55rem; border: 1px solid currentColor; color: var(--cobalt);
      font: 700 .62rem var(--mono); letter-spacing: .08em; text-transform: uppercase;
    }
    .status-fail .status-chip { color: var(--fault); }
    .status-pending .status-chip { color: var(--muted); }
    .status-skip .status-chip { color: #3d7770; }
    .frame-window { position: relative; display: block; min-height: 220px; overflow: hidden; background: #090e12; }
    .frame-window img {
      display: block; width: 100%; height: auto; min-height: 220px;
      object-fit: contain; transition: transform .35s ease;
    }
    .frame-window:hover img { transform: scale(1.012); }
    .frame-boundary {
      position: absolute; z-index: 1; top: 0; left: 0; padding: .45rem .6rem;
      background: var(--ink); color: var(--optical); font: .6rem var(--mono); letter-spacing: .09em;
    }
    .frame-empty { display: grid; place-items: center; padding: 3rem 1.5rem 1.5rem; color: #d6e0e6; }
    .frame-empty p { max-width: 46ch; font: .75rem/1.55 var(--mono); }
    .identity-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem; margin: 1rem 0; }
    .identity-grid div { min-width: 0; border-top: 1px solid var(--line); padding-top: .55rem; }
    dt { color: var(--muted); }
    dd { margin: .24rem 0 0; font: .72rem/1.35 var(--mono); overflow-wrap: anywhere; }
    .artifact-links { display: flex; flex-wrap: wrap; gap: .45rem; min-height: 1.8rem; }
    .artifact-links a {
      padding: .42rem .55rem; border-bottom: 2px solid var(--optical); background: #e7eef2;
      font: .65rem var(--mono); text-decoration: none;
    }
    .artifact-links a:hover { background: var(--ink); color: var(--surface); border-color: var(--amber); }
    footer { padding: 2rem clamp(1rem, 4vw, 3rem) 4rem; color: var(--muted); font: .7rem/1.5 var(--mono); }
    @media (max-width: 1050px) {
      .controls { grid-template-columns: repeat(3, 1fr); }
      .controls input { grid-column: 1 / -1; }
      .gallery { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      .masthead { grid-template-columns: 1fr; padding: 2.2rem 1rem; }
      .controls { position: static; grid-template-columns: 1fr 1fr; }
      .controls input { grid-column: 1 / -1; }
      .gallery-shell { padding: 1rem; }
      .capture-card { grid-template-columns: 38px minmax(0, 1fr); }
      .identity-grid { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
    }
  </style>
</head>
<body>
  <header class="masthead">
    <div>
      <p class="kicker">RoboDojo optical inspection · __BUNDLE_MODE__</p>
      <h1>First frame, every recipe.</h1>
      <p class="thesis">The exact post-reset observation before a policy acts. Compare camera framing,
        robot initialization, task layout, and optional scene bundles from one offline bench.</p>
    </div>
    <aside class="run-plate" aria-label="Run details">
      <span class="run-label">Run</span><code>__RUN_ID__</code>
      <span class="run-label">Output</span><code>__OUTPUT_DIR__</code>
      <span class="run-label">Seed / layout</span><code>__SEED__ / __LAYOUT__</code>
      <div class="run-stats">
        <div><strong>__FINISHED__</strong><span>finished</span></div>
        <div><strong>__PASSED__</strong><span>usable</span></div>
        <div><strong>__FAILED__</strong><span>failed</span></div>
      </div>
    </aside>
  </header>
  <section class="controls" aria-label="Filter snapshots">
    <label for="search">Search recipes</label>
    <input id="search" type="search" placeholder="Search recipe, task, policy…">
    <label for="policy">Filter policy</label><select id="policy">__POLICY_OPTIONS__</select>
    <label for="environment">Filter environment</label><select id="environment">__ENVIRONMENT_OPTIONS__</select>
    <label for="scene">Filter scene</label><select id="scene">__SCENE_OPTIONS__</select>
    <label for="task">Filter task</label><select id="task">__TASK_OPTIONS__</select>
    <label for="status">Filter status</label><select id="status">__STATUS_OPTIONS__</select>
  </section>
  <main class="gallery-shell">
    <p class="gallery-count" id="gallery-count" aria-live="polite">Showing __TOTAL__ of __TOTAL__ recipes</p>
    <div class="gallery" id="gallery">__CARDS__</div>
  </main>
  <footer>Generated by <code>robodojo eval snapshots</code>. All images and scene links are relative;
    this inspection page requires no server or network connection.</footer>
  <script>
    (() => {
      const cards = [...document.querySelectorAll('.capture-card')];
      const controls = ['search', 'policy', 'environment', 'scene', 'task', 'status']
        .map(id => document.getElementById(id));
      const count = document.getElementById('gallery-count');
      function filterCards() {
        const [search, ...selects] = controls;
        const query = search.value.trim().toLowerCase();
        let visible = 0;
        cards.forEach(card => {
          const selectMatch = selects.every(select => !select.value || card.dataset[select.id] === select.value);
          const matches = selectMatch && (!query || card.dataset.search.includes(query));
          card.hidden = !matches;
          visible += Number(matches);
        });
        count.textContent = `Showing ${visible} of ${cards.length} recipes`;
      }
      controls.forEach(control => control.addEventListener('input', filterCards));
    })();
  </script>
</body>
</html>
"""
