# Upstream review notes

This document is a historical review snapshot and a conceptual porting guide.
Upstream is a design and content source, not a machine-enforced semantic
baseline.

## Review snapshot — 2026-07-15

- Official RoboDojo `a38d84e` was reviewed against this fork. Its 54 task
  modules and 55 task configs were represented locally, and the shared task,
  config, and scene behavior was conceptually aligned.
- Recent upstream pose restoration, tic-tac-toe, process cleanup, simulator
  parallelism, and XPolicyLab pin changes were already represented by local
  implementations or adapters.
- Three corrected upstream task instructions were adopted. The upstream
  `fold_clothes` instruction was restored, with the existing two-arm wording
  retained only for the classic `bimanual_yam_molmoact2` setup.
- Official XPolicyLab `fe71eb5` was an ancestor of the configured local fork
  revision `20b0af3`, which contained 13 additional commits.

## Task, scene, and protocol ownership

Task discovery checks only the local runnable unit: a module, an exported class
matching its basename, and an aligned task YAML. Tasks may exist without task
protocols or recipes. There is no source-token scan, AST/YAML upstream hash,
one-to-one protocol coverage rule, or automatic comparison with an upstream
snapshot.

Following the task/benchmark-protocol separation described in the
[MagicSim paper](https://arxiv.org/html/2606.17511), task modules own the MDP
while `configs/protocols.yml` owns benchmark horizon, evaluation count, and
scene-compatibility choices. Scene components, profiles, layout sets, task
protocols, and recipes remain independently selectable. `configs/recipes.yml`
schema v3 selects one policy, environment, scene, and `task_protocol` without
implicit rebinding.

Keep upstream-derived task modules and YAML structurally recognizable so future
diff review and selective ports remain straightforward. Local experiment
identity hashes the selected task module and YAML alongside the selected
policy, environment, scene, and protocol inputs for reproducibility; that hash
does not assert upstream equivalence. Policy training affinity remains advisory
metadata.

## Porting checklist

1. Review upstream task and scene diffs with an LLM and identify material
   prompt, observation, reward, reset, object-role, layout, or asset changes.
2. Port a task as an aligned module/class/YAML unit and verify meaningful
   success, reset/soft-reset state, and shared logging.
3. Port scene components, profiles, layout sets, and typed asset builders in
   their own domains; do not add scene or policy branching to task code.
4. Add a task protocol only when selectable benchmark settings are needed, and
   add recipes only for supported compositions.
5. Validate task discovery, catalog compatibility, local identity changes, and
   the actual XPolicyLab launch/WebSocket/observation/action boundary.

## Future reviews

Use an LLM to inspect upstream commits and diffs, concentrating on material new
tasks, scenes, simulator behavior, evaluation flow, and cross-repository policy
integration. For each relevant change, decide whether it should be selectively
ported, is already represented locally, or is intentionally unnecessary.

Exact file parity, matching public APIs, identical configuration layouts, and
machine-enforced baselines are not goals. Preserve transport, policy-launch,
and observation/action interoperability only where the repositories actually
integrate, and adapt useful upstream ideas to the local architecture.
