# Upstream review notes

This is a historical snapshot from an LLM-assisted review, not a compatibility
manifest or an enforced baseline.

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

## Future reviews

Use an LLM to inspect upstream commits and diffs, concentrating on material new
tasks, scenes, simulator behavior, evaluation flow, and cross-repository policy
integration. For each relevant change, decide whether it should be selectively
ported, is already represented locally, or is intentionally unnecessary.

Exact file parity, matching public APIs, identical configuration layouts, and
machine-enforced baselines are not goals. Preserve transport, policy-launch,
and observation/action interoperability only where the repositories actually
integrate, and adapt useful upstream ideas to the local architecture.
