# Upstream review notes

This document is a historical review snapshot. Canonical task semantics are
now enforced separately by `configs/reference/upstream_task_contracts.yml`.

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

## Task and protocol ownership

The semantic lock covers the canonical task Python AST and normalized task YAML
at the reviewed upstream revision. It deliberately ignores local import-path
mapping while detecting prompt, observation, reward, reset, object-role, and
horizon changes. Task code is also rejected when it branches on policy,
environment, scene, camera, robot, or protocol identity.

Following the task/benchmark-protocol separation described in the
[MagicSim paper](https://arxiv.org/html/2606.17511), canonical task modules own
the MDP while `configs/protocols.yml` owns explicit layout, horizon, evaluation
count, and scene-compatibility choices. `configs/recipes.yml` then selects one
policy, environment, scene, and protocol without implicit rebinding.

The upstream task inventory remains a semantically locked subset of this
fork's composition system; it is not a requirement that every policy, concrete
embodiment realization, scene, or generated asset form one standardized cross
product. New combinations use the same explicit contracts and allowlists while
policy training affinity remains advisory metadata.

## Future reviews

Use an LLM to inspect upstream commits and diffs, concentrating on material new
tasks, scenes, simulator behavior, evaluation flow, and cross-repository policy
integration. For each relevant change, decide whether it should be selectively
ported, is already represented locally, or is intentionally unnecessary.

Exact file parity, matching public APIs, identical configuration layouts, and
machine-enforced baselines are not goals. Preserve transport, policy-launch,
and observation/action interoperability only where the repositories actually
integrate, and adapt useful upstream ideas to the local architecture.
