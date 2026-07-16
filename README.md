<div align="center">

<img src="https://media.luminis-sim.com/media/challenge/posters/robodojo_logo.png"></img>

<h2 align="center">RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies</h2>

<h2 align="center"><a href="https://robodojo-benchmark.com/">Webpage</a> | <a href="https://robodojo-benchmark.com/doc/">Document</a> | <a href="https://arxiv.org/abs/2607.04434">Paper</a> | <a href="https://robodojo-benchmark.com/community">Community</a> | <a href="https://robodojo-benchmark.com/leaderboard">Leaderboard</a></h2>

</div>

https://private-user-images.githubusercontent.com/88101805/619409345-cc074c5d-4567-4418-8a29-1385aaba9d5b.mp4

## ✨ Highlights

<p align="center">
  <img src="https://media.luminis-sim.com/media/home/teaser.png" width="70%"></img>
</p>

<p align="center"><em>Overview of RoboDojo. RoboDojo unifies efficient simulation evaluation and reproducible real-world testing for generalist robot manipulation, covering 42 simulation tasks, 18 real-world tasks, heterogeneous parallel simulation, RoboDojo-RealEval, XPolicyLab, and a continuously updated leaderboard.</em></p>

> RoboDojo is **eval-only** in this release: it provides the simulator client, benchmark tasks, asset/config validation, and result artifacts. Policy integration and policy servers are owned by [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md).

- 🌐 **Unified sim-and-real benchmark** — 42 simulation tasks and 18 real-world tasks across 3 robot embodiments for generalist robot manipulation.
- 🧭 **Five capability dimensions** — Generalization, Memory, Precision, Long-Horizon, and Open, designed to probe different skills rather than simple object or layout reskins.
- 🧗 **Challenging by design** — intentionally hard, diverse, long-horizon tasks that expose failures hidden by simpler benchmarks.
- ⚡ **Heterogeneous parallel simulation** — runs different tasks, scenes, and processes concurrently on Isaac Sim for fast, scalable feedback.
- 🧱 **Physically grounded assets** — rigid, articulated, and deformable objects in a single configuration-driven scene.
- 🤖 **Integrate once, evaluate everywhere** — [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md) unifies 40+ policies behind one interface for both simulation and real-world runs.
- 📊 **Reproducible & leaderboard-ready** — seed-controlled layouts and one-command `results summarize` aggregation into a leaderboard table.

## 📚 Documentation

The [RoboDojo documentation](https://robodojo-benchmark.com/doc/) is the canonical reference. Key sections:

| Section | Description |
| :-- | :-- |
| [Usage Overview](https://robodojo-benchmark.com/doc/usage/) | End-to-end walkthrough of the evaluation workflow. |
| [Installation & Downloading (Assets and Data)](https://robodojo-benchmark.com/doc/usage/install-and-download/) | Environment setup and downloading robot/object/layout assets/training data. |
| [Quick Evaluation](https://robodojo-benchmark.com/doc/usage/evaluation/) | Quickly dispatch XPolicyLab to run a policy for testing. |
| [XPolicyLab](https://robodojo-benchmark.com/doc/usage/XPolicyLab/) | Integrates a large collection of policies and defines how to integrate new ones. |
| [Simulation Tasks Details](https://robodojo-benchmark.com/doc/tasks/) | The 42 Isaac Sim tasks across five capability dimensions. |
| [Real Robot Tasks Details](https://robodojo-benchmark.com/doc/real-tasks/) | The 18 real-world tasks on Piper X, Piper, and ARX X5. |
| [Configurations](https://robodojo-benchmark.com/doc/usage/configurations/) | Simulator, scene, robot, and camera configuration options. |
| [Common Issues](https://robodojo-benchmark.com/doc/common-issue/) | Troubleshooting for installation, assets, GPU memory, and evaluation. |

## 🗂️ Repository Structure

```text
src/robodojo/core/          lightweight settings, paths, storage, and process contracts
src/robodojo/policy/        XPolicyLab adapter validation and launching
src/robodojo/sim/           simulator runtime, tasks, evaluation, and scene export
src/robodojo/orchestration/ coordinated policy/simulator evaluation lifecycle
src/robodojo/workflows/     setup, assets, storage, result, and Docker workflows
configs/                    environment, task, simulator, scene, robot, and camera YAML
XPolicyLab/                 policy implementations and server adapters (submodule)
scripts/eval_policy.sh      private compatibility shim for XPolicyLab
```

The official RoboDojo paths `env_cfg/<profile>.yml` and
`task/RoboDojo/config/<task>.yml` map to this fork's canonical
`configs/environment/<profile>.yml` and `configs/task/<task>.yml`. Profile and
task names, schemas, and XPolicyLab's `env_cfg_type` values remain unchanged.

## 🚀 Local Setup

RoboDojo uses a locked [uv](https://docs.astral.sh/uv/) environment with Python
3.11. IsaacLab and the official NVlabs cuRobo repository are pinned by uv;
XPolicyLab remains the only submodule. The simulator preloads the locked Warp
1.11 wheel before Isaac Sim starts. Install the machine prerequisites first:
Git, Git LFS, uv, compiler/runtime tools, and NVIDIA drivers. Then list the
tracked evaluation recipes and run one end-to-end:

```bash
make recipes
make eval RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup
```

`make recipes` renders a grouped terminal table for people. Automation can use
`robodojo recipes --format plain` for TSV or `--format json` for structured
output.

Each recipe explicitly selects a typed policy profile, environment profile,
scene profile, and task protocol. Those components cannot be overridden under a
recipe. The direct CLI also supports a strict manual mode, but all four profiles
must be named together. This keeps policy checkpoints and embodiment contracts,
scene composition, and task-protocol settings independently reviewable. Make
loads an optional ignored `.env` from the repository root for machine-local
controls such as GPU selection and storage. Use Make-compatible `?=` assignments
so explicit arguments and exported variables retain precedence; direct Python
CLI commands do not load this file. Make defaults to seed 0, automatic policy
and simulator GPU selection, the protocol's native evaluation count, and `INFO`
diagnostics. Scene export and publication are opt-in.

For paired workflows, Python assigns the most-free GPU to the simulator and the
next-most-free GPU to the policy, breaking ties by device index. Override either
selector with a Make argument or exported `POLICY_GPU`/`ENV_GPU`; direct CLI
flags take precedence over those variables. `make eval` first performs the
idempotent setup workflow, which initializes the pinned XPolicyLab submodule,
synchronizes the locked simulator environment, prepares inferred assets, and
invokes the optional policy preparation hook. The managed evaluation then runs
fast preflight before launching. Inspect the deliberately small Make surface with:

```bash
make help
```

Capture the exact first RGB observation for every tracked recipe without
starting a policy server:

```bash
make snapshots
make snapshots EXPORT_SCENE=true
make snapshots PUBLISH=true
```

The first command writes every configured camera PNG, a per-recipe contact
sheet, structured summaries, and an offline `index.html` gallery below
`.robodojo/runs/snapshots/<timestamp>/`. `EXPORT_SCENE=true` also reuses the
normal scene exporter to add referenced USDA, flattened USDC, and preview USDZ
bundles. Use `RECIPES="<name> ..."`, `LAYOUT_ID=<n>`, or `SNAPSHOT_DIR=<path>`
to narrow or relocate a batch. Completed explicit output directories can be
continued with `ARGS=--resume`. Publication remains off by default;
`PUBLISH=true` uploads a fully successful batch to the immutable
`runs/snapshots/<run-id>` location below `ROBODOJO_S3_URI`.

Make is the opinionated argument- and environment-driven interface. The CLI is
explicit and reads runtime settings from the process environment only; support
operations remain grouped under `assets`, `data`, `storage`, `results`, and
`docker`. After setup, native commands run through the lockfile without
synchronizing dependencies:

```bash
uv run --extra sim --locked --no-sync robodojo doctor --skip-policy
uv run --locked --no-sync robodojo tasks
```

Standalone preparation and deeper policy readiness checks remain available for
diagnosis:

```bash
make setup RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup
make preflight RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup
make preflight RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup DEEP=true
make eval RECIPE=pi05-bimanual_yam-molmo_yam-general_pickup
```

Setup remains the consolidated mutation boundary. The setup phase of `make eval`
may install, download, build, or derive missing prerequisites; its subsequent
preflight and managed launch are read-only. Fast preflight checks the locked
simulator and policy runtimes, task/scene/layout and generated robot assets,
GPUs, checkpoints, policy contracts, and optional publication settings. Deep
preflight additionally starts the normal policy server on a temporary port and
always stops it, without starting Isaac Sim or publishing. See
[Experiment setup and preflight](docs/PREFLIGHT.md).

The simulator environment is always the repository's `.venv`. Policy servers
remain independent and `--policy-env` may identify a uv project, environment
path, or policy-specific Conda environment.

Large assets, datasets, model weights, and runs live below one writable local
root, `.robodojo/` by default. S3 is an optional explicit publication and
restore target; it is never mounted by the application. Direct CLI evaluations
export only with `robodojo eval --export-scene` and publish only with
`robodojo eval --publish`. The Make workflow also stays local and skips scene
export by default; opt in with `PUBLISH=true` and `EXPORT_SCENE=true` explicitly
or as machine defaults in `.env`. `VERBOSITY` defaults to `INFO` and controls
the global RoboDojo log level for Make-launched commands. See [Local storage and
S3 publication](docs/STORAGE.md) for the contract.

## 🔌 Policy Integration

This fork uses the official [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
and [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) repositories as
design references rather than exact mirrors. Cross-repository policy launch,
transport, and observation/action boundaries remain interoperable where needed,
while local layout and APIs may evolve independently. See the concise
[upstream review notes](docs/UPSTREAM.md) for the latest LLM-assisted audit.

Policies live in [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md), which owns policy structure, dependencies, checkpoint layout, and server behavior. RoboDojo only assumes a policy directory provides:

```text
XPolicyLab/policy/<POLICY_NAME>/setup_eval_policy_server.sh
XPolicyLab/policy/<POLICY_NAME>/deploy.yml
```

Adapters may also provide standardized `prepare_eval_policy.sh` and
`check_eval_policy.sh` hooks. Both receive the same eight-argument experiment
prefix used by the server adapter. RoboDojo invokes preparation only through
`robodojo setup --only policy` or the full `make setup` workflow; preflight
invokes the check hook read-only. Legacy adapters
remain launchable through generic runtime/import/checkpoint checks and report a
warning when policy-specific validation is unavailable.

`robodojo eval` starts the server adapter and simulator as managed process groups.
The remaining `scripts/eval_policy.sh` only supports unchanged XPolicyLab legacy
callbacks and immediately delegates to the Python CLI.

## 🏆 Leaderboard

View live rankings on the [RoboDojo Leaderboard](https://robodojo-benchmark.com/leaderboard).

**Simulation.** The full evaluation stack is open source, so you can debug locally and iterate on scores. Official RoboDojo-endorsed listings are submitted through the cloud evaluation pipeline with anti-cheating verification.

**Real world.** Real-robot leaderboard entries are accepted through the same cloud evaluation process; see the public documentation for protocol, rules, and submission details.

## 📝 Citation

**RoboDojo**

```bibtex
@article{chen2026robodojo,
  title={RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies},
  author={Chen, Tianxing and Chen, Yue and Li, Zixuan and Tang, Junyuan and Su, Kailun and Wan, Weijie and Chen, Baijun and Lu, Haoran and Yan, Haowen and Su, Honghao and others},
  journal={arXiv preprint arXiv:2607.04434},
  year={2026}
}
```

**RoboTwin 2.0**

```bibtex
@article{chen2025robotwin,
  title={Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Li, Zixuan and Liang, Qiwei and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**MagicSim**

```bibtex
@misc{lu2026magicsimunifiedinfrastructureexecutable,
      title={MagicSim: A Unified Infrastructure for Executable Embodied Interaction}, 
      author={Haoran Lu and Songling Liu and Yue Chen and Guo Ye and Mutian Shen and Shuyang Yu and Yu Xiao and Jihai Zhao and Shang Wu and Jianshu Zhang and Xiangtian Gui and Chuye Hong and Yuran Wang and Maojiang Su and Jiayi Wang and Ruihai Wu and Zhaoran Wang and Han Liu},
      year={2026},
      eprint={2606.17511},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.17511}, 
}
```

## 🙏 Acknowledgements

RoboDojo builds on [Isaac Sim](https://developer.nvidia.com/isaac/sim), [IsaacLab](https://github.com/isaac-sim/IsaacLab), [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena), [RoboTwin 2.0](https://github.com/robotwin-Platform/robotwin), [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), and [MagicSim](https://arxiv.org/abs/2606.17511). We thank the authors and maintainers for their open-source contributions to the robotics community.

Contact [Tianxing Chen](https://tianxingchen.github.io/) or [Yue Chen](https://yuechen0614.github.io/) if you have questions or suggestions.

## 🏫 Affiliations

RoboDojo is operated by **AI MMLab Club**, a non-profit, vendor-neutral organization, and is jointly maintained and supported by a global consortium of academic institutional partners. To preserve the fairness, neutrality, and independence of the official evaluation, RoboDojo does not involve commercial companies in its governance, operation, funding, sponsorship, compute, hardware, or other forms of project support. For inquiries from academic or non-profit partners regarding project collaboration or resource support, please contact [RoboDojoCommittee@gmail.com](mailto:RoboDojoCommittee@gmail.com).

<img src="https://media.luminis-sim.com/media/home/partners/affiliations.png"></img>

## ⚖️ License

Released under the [RoboDojo Non-Commercial Research License](LICENSE). RoboDojo is available for non-commercial research, education, and evaluation only. Commercial use requires prior written permission from the maintainers.


<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-475569?style=flat-square&logo=python&logoColor=white&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Sim 5.1" src="https://img.shields.io/badge/Isaac_Sim-5.1-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Lab 2.3" src="https://img.shields.io/badge/Isaac_Lab-2.3-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="License Non-Commercial" src="https://img.shields.io/badge/License-Non--Commercial-475569?style=flat-square&labelColor=64748b" height="22"/>
</p>
