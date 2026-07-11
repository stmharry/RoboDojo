#!/usr/bin/env python3
"""Build the OpenARM rollout report and atomically update the stable site."""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path

MODEL_REVISION = "695abe40dbf3aac04efda59c1501d748681fa0fb"
LEROBOT_REVISION = "1396b9fab7aecddd10006c33c47a487ffdcb54b4"
DATASET_REVISION = "2e1b2e913cd367d74dc4481736954eed4a051ddc"
OPENARM_REVISION = "bad82e23716e6941c2de78ccb978f57c78b37734"
HARDWARE_REVISION = "ffe34b93c070343042eb9412fbfeffce16139947"


def video_for(run_dir: Path, camera: str) -> Path:
    matches = sorted(run_dir.glob(f"episode_0000000_{camera}_*.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {camera} video, found {len(matches)}")
    return matches[0]


def build_html(run_dir: Path) -> str:
    result = json.loads((run_dir / "_result.json").read_text())
    validation = json.loads((run_dir / "visual_validation/visual_validation.json").read_text())
    warnings_path = run_dir / "runtime_warnings.txt"
    warnings = warnings_path.read_text().strip() if warnings_path.exists() else "No retained runtime warnings."
    success = bool(result.get("success_rate", 0))
    score = result.get("score", 0)
    cameras = [
        ("Base overview", "cam_head", "base"),
        ("Left wrist", "cam_left_wrist", "left_wrist"),
        ("Right wrist", "cam_right_wrist", "right_wrist"),
    ]
    camera_sections = []
    for title, sim_name, validation_name in cameras:
        video = video_for(run_dir, sim_name)
        metrics = validation["cameras"][validation_name]
        info = metrics["video_info"]
        distance = max(metrics["histogram_js_distance"])
        camera_sections.append(
            f"""
            <section class="camera-block">
              <div class="camera-copy"><p class="channel">{escape(title)}</p><p>{info['width']}×{info['height']} · {info['frames']} frames · {info['fps']:g} FPS<br><span>validation distance ≤ {distance:.3f}</span></p></div>
              <img src="visual_validation/{validation_name}_comparison.png" alt="Official and rendered {escape(title)} frame comparison">
              <video controls preload="metadata" src="{escape(video.name)}"></video>
            </section>
            """
        )

    verdict = "task completed" if success else "runtime passed · fold incomplete"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenARM folding protocol run</title>
<style>
:root{{--paper:#edf0ed;--ink:#101a1a;--muted:#586565;--cobalt:#1749d2;--orange:#ed5a36;--rule:#b9c3be;--panel:#f8faf7}}
*{{box-sizing:border-box}} html{{background:var(--paper)}} body{{margin:0;color:var(--ink);font:16px/1.5 "Arial",sans-serif;background-image:linear-gradient(rgba(16,26,26,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(16,26,26,.045) 1px,transparent 1px);background-size:24px 24px}}
main{{width:min(1220px,calc(100% - 32px));margin:auto;padding:42px 0 90px;position:relative}} main:before{{content:"";position:absolute;left:10px;top:0;bottom:0;border-left:3px dashed var(--cobalt);opacity:.72}}
.mast,.camera-block,.ledger,.warnings{{margin-left:42px}} .kicker,.channel{{font:700 12px/1.2 "Courier New",monospace;text-transform:uppercase;letter-spacing:.15em;color:var(--cobalt)}}
h1{{font:900 clamp(48px,9vw,118px)/.82 "Arial Narrow","Helvetica Neue",sans-serif;letter-spacing:-.075em;text-transform:uppercase;max-width:1000px;margin:22px 0 30px}} h1 em{{font-style:normal;color:var(--orange)}}
.summary{{display:grid;grid-template-columns:2fr repeat(3,1fr);border-top:3px solid var(--ink);border-bottom:1px solid var(--ink)}} .summary>div{{padding:18px;border-right:1px solid var(--rule)}} .summary>div:last-child{{border:0}} .summary b{{display:block;font:800 25px/1 "Arial Narrow",sans-serif}} .summary span,.camera-copy span{{color:var(--muted)}}
.camera-block{{display:grid;grid-template-columns:180px 1fr;gap:14px 22px;margin-top:64px;padding-top:14px;border-top:3px solid var(--ink)}} .camera-copy{{grid-row:1 / span 2}} .camera-copy p{{margin:0 0 10px}} img,video{{width:100%;display:block;border:1px solid var(--ink);background:#000}} video{{max-height:72vh}}
h2{{font:800 30px/1 "Arial Narrow",sans-serif;text-transform:uppercase;margin:64px 0 16px}} table{{width:100%;border-collapse:collapse;background:var(--panel)}} td{{padding:11px 13px;border-top:1px solid var(--rule);vertical-align:top}} td:first-child{{width:230px;color:var(--muted);font-family:"Courier New",monospace;font-size:13px}} code{{overflow-wrap:anywhere;color:var(--cobalt)}} pre{{white-space:pre-wrap;background:var(--ink);color:#e7eee9;padding:18px;overflow:auto}}
a{{color:var(--cobalt)}} a:focus-visible,video:focus-visible{{outline:3px solid var(--orange);outline-offset:3px}}
@media(max-width:760px){{main:before{{display:none}}.mast,.camera-block,.ledger,.warnings{{margin-left:0}}.summary{{grid-template-columns:1fr 1fr}}.camera-block{{grid-template-columns:1fr}}.camera-copy{{grid-row:auto}}}}
</style></head><body><main>
<header class="mast"><p class="kicker">RoboDojo / OpenARM / protocol-corrected seed 0</p><h1>The stand is gone.<br><em>The protocol remains.</em></h1>
<div class="summary"><div><b>{escape(verdict)}</b><span>official setup defines the twin; episode data validates its rendered views</span></div><div><b>{score}</b><span>score</span></div><div><b>{str(success).lower()}</b><span>task success</span></div><div><b>{result.get('eval_time', 0)} / 1</b><span>episodes</span></div></div></header>
{''.join(camera_sections)}
<section class="ledger"><h2>Protocol ledger</h2><table>
<tr><td>Policy</td><td><code>folding_final@{MODEL_REVISION}</code></td></tr><tr><td>LeRobot</td><td><code>{LEROBOT_REVISION}</code></td></tr>
<tr><td>Execution</td><td>RTC queue 30 · horizon 20 · guidance 5.0 · LINEAR prefix · 30→90 Hz interpolation · 240 Hz physics (3/3/2)</td></tr>
<tr><td>Embodiment</td><td><code>OpenARM Isaac {OPENARM_REVISION}</code> · 5 cm extension · enlarged jaws · right-first 16-D</td></tr>
<tr><td>USD inspection</td><td>1 articulation · 14 actuated arm joints · 4 gripper joints · 4 enlarged jaw meshes · both wrist mount links valid</td></tr>
<tr><td>Cameras</td><td>OV2710 base 640×480 on bimanual center extrusion · two IMX708 wrist cameras 1280×720 on link 7 · equidistant fisheye · 30 FPS</td></tr>
<tr><td>Hardware changes</td><td><code>{HARDWARE_REVISION}</code></td></tr><tr><td>Visual validation</td><td><code>level2_final_quality3@{DATASET_REVISION}</code> — validation only, not camera/geometry authority</td></tr>
<tr><td>References</td><td><a href="https://huggingface.co/spaces/lerobot/robot-folding">official folding setup</a> · <a href="https://github.com/huggingface/lerobot/blob/{LEROBOT_REVISION}/examples/rtc/eval_with_real_robot.py">pinned evaluation code</a> · <a href="https://huggingface.co/lerobot-data-collection/folding_final/tree/{MODEL_REVISION}">policy checkpoint</a> · <a href="https://huggingface.co/datasets/lerobot/openarms-hardware-modifications/tree/{HARDWARE_REVISION}">hardware modifications</a> · <a href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=lerobot-data-collection/level2_final_quality3">dataset viewer</a></td></tr>
</table></section><section class="warnings"><h2>Runtime warnings</h2><pre>{escape(warnings)}</pre></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stable-root", type=Path, default=Path(".cache/openarm_report_site"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not json.loads((run_dir / "visual_validation/visual_validation.json").read_text())["passed"]:
        raise SystemExit("refusing to publish a run that failed visual validation")
    index = run_dir / "index.html"
    temporary_index = run_dir / ".index.html.next"
    temporary_index.write_text(build_html(run_dir))
    os.replace(temporary_index, index)

    stable_root = args.stable_root.resolve()
    stable_root.mkdir(parents=True, exist_ok=True)
    temporary_link = stable_root / f".current.{os.getpid()}"
    current_link = stable_root / "current"
    temporary_link.symlink_to(run_dir, target_is_directory=True)
    os.replace(temporary_link, current_link)
    print(index)
    print(current_link)


if __name__ == "__main__":
    main()
