#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the validated DYNA OpenARM report and atomically repoint the stable site."""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
import shutil

MODEL_REVISION = "695abe40dbf3aac04efda59c1501d748681fa0fb"
LEROBOT_REVISION = "1396b9fab7aecddd10006c33c47a487ffdcb54b4"
DATASET_REVISION = "2e1b2e913cd367d74dc4481736954eed4a051ddc"
OPENARM_REVISION = "bad82e23716e6941c2de78ccb978f57c78b37734"
HARDWARE_REVISION = "ffe34b93c070343042eb9412fbfeffce16139947"
ARTICLE_REVISION = "170e1d479579e0b4be1afe0c99ebf868b24803db"
RIGS = {
    "dyna": {
        "title": "DYNA counterpart",
        "profile": "openarm_dyna",
        "base": "Waveshare OV2710 SKU 14121 · 145° · f 316.1146 px",
        "meaning": "Original rig · availability-driven base-camera substitution",
    },
}
CAMERAS = (
    ("Base / fixture", "cam_head", "base"),
    ("Left wrist / original target +90°", "cam_left_wrist", "left_wrist"),
    ("Right wrist / original target −90°", "cam_right_wrist", "right_wrist"),
)


def video_for(run_dir: Path, camera: str) -> Path:
    matches = sorted(run_dir.glob(f"episode_0000000_{camera}_*.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {camera} video in {run_dir}, found {len(matches)}")
    return matches[0]


def read_run(run_dir: Path, expected_profile: str) -> tuple[dict, dict, str]:
    result = json.loads((run_dir / "_result.json").read_text())
    validation = json.loads((run_dir / "visual_validation/visual_validation.json").read_text())
    if not validation["passed"]:
        raise RuntimeError(f"refusing to publish failed visual validation: {run_dir}")
    if validation["camera_profile"] != expected_profile:
        raise RuntimeError(f"wrong profile in {run_dir}: {validation['camera_profile']}")
    warnings_path = run_dir / "runtime_warnings.txt"
    warnings = warnings_path.read_text().strip() if warnings_path.exists() else "No retained runtime warnings."
    return result, validation, warnings


def camera_cards(slug: str, run_dir: Path, validation: dict) -> str:
    cards = []
    for title, sim_name, validation_name in CAMERAS:
        video = video_for(run_dir, sim_name)
        metrics = validation["cameras"][validation_name]
        info = metrics["video_info"]
        correction = ""
        if validation_name != "base":
            correction = f'''<img class="correction" src="{slug}/camera_calibration/{validation_name}_before_after_contact_sheet.png"
            alt="Previous and roll-only corrected {escape(title)} frames">'''
        cards.append(f"""
        <article class="camera">
          <div class="camera-label"><b>{escape(title)}</b><span>{info['width']}×{info['height']} · {info['frames']}f · {info['fps']:g} FPS</span></div>
          {correction}
          <img src="{slug}/camera_calibration/{validation_name}_blog_contact_sheet.png" alt="Pinned article and rendered {escape(title)} comparison">
          <img src="{slug}/camera_calibration/{validation_name}_matched_state_contact_sheet.png" alt="Dataset-state and rendered {escape(title)} comparison">
          <video controls preload="metadata" src="{slug}/{escape(video.name)}"></video>
        </article>""")
    return "".join(cards)


def build_html(runs: dict[str, Path]) -> str:
    blocks, warning_blocks = [], []
    for slug, rig in RIGS.items():
        result, validation, warnings = read_run(runs[slug], rig["profile"])
        success = bool(result.get("success_rate", 0))
        status = "fold complete" if success else "runtime complete · fold incomplete"
        blocks.append(f"""
        <section class="rig" id="{slug}">
          <header class="rig-head"><div><p class="eyebrow">{escape(rig['meaning'])}</p><h2>{escape(rig['title'])}</h2><p>{escape(rig['base'])}</p></div>
          <dl><div><dt>Status</dt><dd>{escape(status)}</dd></div><div><dt>Score</dt><dd>{result.get('score', 0)}</dd></div><div><dt>Episodes</dt><dd>{result.get('eval_time', 0)} / 1</dd></div></dl></header>
          <div class="cameras">{camera_cards(slug, runs[slug], validation)}</div>
        </section>""")
        warning_blocks.append(f"<h3>{escape(rig['title'])}</h3><pre>{escape(warnings)}</pre>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenARM camera-rig calibration</title><style>
:root{{--ink:#13222b;--paper:#e9eef0;--panel:#f7fafb;--line:#83949d;--cyan:#007b99;--amber:#e8a317;--red:#bd3b28}}
*{{box-sizing:border-box}}html{{background:var(--paper);color:var(--ink);font:16px/1.45 "IBM Plex Sans Condensed","Arial Narrow",sans-serif}}body{{margin:0}}main{{width:min(1460px,calc(100% - 32px));margin:auto;padding:34px 0 88px}}
.hero{{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(260px,.7fr);gap:28px;border:2px solid var(--ink);background:var(--panel);padding:28px;box-shadow:8px 8px 0 var(--cyan)}}
.eyebrow,dt{{margin:0;color:var(--cyan);font:700 11px/1.2 "IBM Plex Mono","Courier New",monospace;text-transform:uppercase;letter-spacing:.13em}}h1{{font:900 clamp(42px,7vw,92px)/.85 "Arial Narrow",sans-serif;letter-spacing:-.055em;margin:14px 0 20px;text-transform:uppercase}}.hero p{{max-width:72ch}}.axis{{align-self:stretch;display:grid;place-items:center;background:linear-gradient(90deg,transparent 49.5%,var(--red) 49.5%,var(--red) 50.5%,transparent 50.5%);min-height:180px}}.axis span{{background:var(--ink);color:white;padding:10px 14px;font:700 12px "Courier New",monospace;transform:rotate(-3deg)}}
.rig{{margin-top:70px;border-top:6px solid var(--ink)}}.rig-head{{display:grid;grid-template-columns:1fr auto;gap:24px;padding:18px 0}}h2{{font:900 clamp(34px,5vw,66px)/.9 "Arial Narrow",sans-serif;text-transform:uppercase;margin:8px 0}}.rig-head p{{margin:0}}dl{{display:grid;grid-template-columns:repeat(3,minmax(110px,1fr));margin:0;border:1px solid var(--line)}}dl div{{padding:12px;border-left:1px solid var(--line)}}dl div:first-child{{border:0}}dd{{margin:6px 0 0;font-weight:800}}
.cameras{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.camera{{background:var(--panel);border:1px solid var(--ink);padding:10px}}.camera-label{{display:flex;justify-content:space-between;gap:12px;margin-bottom:9px;font-family:"Courier New",monospace;font-size:12px}}.camera-label span{{color:#53656e}}img,video{{display:block;width:100%;border:1px solid var(--line);background:#000}}.camera img+img{{margin-top:9px}}.correction{{border:3px solid var(--amber)}}video{{margin-top:9px;max-height:56vh}}
.axes-proof{{display:grid;grid-template-columns:minmax(0,1fr) minmax(260px,.6fr);gap:18px;margin-top:18px;padding:14px;border:1px solid var(--ink);background:var(--panel)}}.axes-proof figcaption{{align-self:center;font:700 13px/1.55 "Courier New",monospace}}.axes-proof strong{{display:block;margin-bottom:10px;color:var(--cyan);font:900 25px/1 "Arial Narrow",sans-serif;text-transform:uppercase}}
.ledger,.warnings{{margin-top:70px}}h3{{font:800 20px "Arial Narrow",sans-serif;text-transform:uppercase}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}td{{padding:11px 13px;border:1px solid var(--line);vertical-align:top}}td:first-child{{width:230px;font-family:"Courier New",monospace;color:var(--cyan)}}pre{{white-space:pre-wrap;background:var(--ink);color:#eaf1f3;padding:16px;overflow:auto}}a{{color:#006e87}}a:focus-visible,video:focus-visible{{outline:4px solid var(--amber);outline-offset:3px}}
@media(max-width:950px){{.hero,.rig-head,.axes-proof{{grid-template-columns:1fr}}.cameras{{grid-template-columns:1fr}}dl{{grid-template-columns:1fr}}dl div{{border-left:0;border-top:1px solid var(--line)}}}}
</style></head><body><main><header class="hero"><div><p class="eyebrow">RoboDojo · original OpenARM rig · seed 0</p><h1>One rig.<br>One substituted sensor.</h1><p>DYNA preserves the original embodiment and wrist rig exactly. Only the unavailable policy-original base camera is replaced by the Waveshare OV2710; official episode frames validate coverage without fitting geometry.</p></div><div class="axis"><span>UPSTREAM CAMERA-STAND AXIS</span></div></header>
{''.join(blocks)}
<figure class="axes-proof"><img src="dyna/camera_calibration/cad_anchor_diagram.png" alt="CAD-derived mount and optical-frame axes"><figcaption><strong>Mount → optical frame</strong>The CAD frame determines the holder-to-sensor transform. The original LeRobot optical target determines where that frame lands. Instance-mask validation then checks the observable consequence: both wrist clips enter from the bottom.</figcaption></figure>
<section class="ledger"><p class="eyebrow">Pinned protocol</p><h2>Calibration ledger</h2><table>
<tr><td>Policy</td><td>folding_final@{MODEL_REVISION} · LeRobot@{LEROBOT_REVISION} · RTC 30/20/5.0 LINEAR · 30→90 Hz · 240 Hz 3/3/2</td></tr>
<tr><td>Embodiment</td><td>OpenARM@{OPENARM_REVISION} · hardware changes@{HARDWARE_REVISION} · 5 cm extension · enlarged jaws · right-first 16-D</td></tr>
<tr><td>Mounts</td><td>Named CAD <code>OpticalFrame</code> targets derive holder attachment transforms. Wrists retain the original link-7 optical centers and viewing axes; only image-up changes by left +90° / right −90°. Runtime optical roll is zero.</td></tr>
<tr><td>Profile delta</td><td>DYNA differs from the policy-original rig only in base-camera vendor/model, published FOV, and focal projection. Wrist and embodiment contracts are identical.</td></tr>
<tr><td>Validation</td><td>level2_final_quality3@{DATASET_REVISION} — orientation and coverage oracle only</td></tr>
<tr><td>Sources</td><td><a href="https://huggingface.co/spaces/lerobot/robot-folding/tree/{ARTICLE_REVISION}">Pinned LeRobot hardware article</a> · <a href="https://huggingface.co/datasets/lerobot/openarms-hardware-modifications/tree/{HARDWARE_REVISION}">Pinned camera-holder CAD</a> · <a href="https://moonlakeai.slack.com/archives/C0BCJPA3T9R/p1782508159343489">DYNA Slack note</a> · <a href="https://www.waveshare.com/wiki/OV2710_2MP_USB_Camera_%28A%29">Waveshare SKU 14121</a></td></tr>
</table></section><section class="warnings"><p class="eyebrow">Retained diagnostics</p><h2>Runtime warnings</h2>{''.join(warning_blocks)}</section></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dyna_run", type=Path)
    parser.add_argument("--stable-root", type=Path, default=Path(".cache/openarm_report_site"))
    args = parser.parse_args()
    runs = {"dyna": args.dyna_run.resolve()}
    for slug, rig in RIGS.items():
        read_run(runs[slug], rig["profile"])

    stable_root = args.stable_root.resolve()
    stable_root.mkdir(parents=True, exist_ok=True)
    site = stable_root / f"run-{os.getpid()}"
    site.mkdir()
    for slug, run_dir in runs.items():
        (site / slug).symlink_to(run_dir, target_is_directory=True)
    (site / "index.html").write_text(build_html(runs))
    temporary_link = stable_root / f".current.{os.getpid()}"
    current_link = stable_root / "current"
    temporary_link.symlink_to(site, target_is_directory=True)
    os.replace(temporary_link, current_link)
    for previous in stable_root.glob("run-*"):
        if previous != site and not previous.samefile(current_link):
            shutil.rmtree(previous)
    print(site / "index.html")
    print(current_link)


if __name__ == "__main__":
    main()
