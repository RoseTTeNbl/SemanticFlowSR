# Findings & Decisions

## Requirements
- Work under `/home/ywj/wyh/SFSR/SemanticFlowSR`.
- Read and understand algorithm implementation, theory, and current experiment progress before cleanup.
- Complete cleanup of historical-version traces so results, logs, docs, and code entrypoints are clean.
- Do not launch long training during cleanup unless explicitly requested.

## Research Findings
- Repository root is `/home/ywj/wyh/SFSR/SemanticFlowSR`.
- Current README mainline is graph categorical-block target-conditioned Stage1 Flow.
- Current docs are narrowed to `docs/ALGORITHM_COMPLETE_EXPRESSION_SEMANTIC_FM.md`, `docs/ARCHITECTURE_COMPLETE_EXPRESSION_SEMANTIC_FM.md`, `docs/MATH.md`, `docs/STRUCTURAL_CLOSURE.md`, and `docs/DIAGNOSTIC_EXPERIMENTS_COMPLETE_EXPRESSION_FLOW.md`.
- `AGENTS.md` says token construction, semantic endpoint correction, and online semantic mass guidance are legacy/ablation by default; do not write fixed-pool/proxy/failure-taxonomy/matrix/distillation details into the mainline algorithm docs.
- Top-level `logs/` is already mostly clean: only `logs/README.md` plus 20260710 semantic latent/register logs remain.
- `results/clean_benchmark_20260701/ablations/complete_expression_semantic_fm_20260707` still contains 20260709 cleanup manifests and semantic-mass branch diagnostic files that look like historical cleanup residue.
- `scripts/archive_legacy_semantic_pushforward_20260709/` was a historical code archive and was removed from the active scripts tree during cleanup.
- `docs/ALGORITHM_COMPLETE_EXPRESSION_SEMANTIC_FM.md`, `docs/ARCHITECTURE_COMPLETE_EXPRESSION_SEMANTIC_FM.md`, and `docs/MATH.md` now define the latest mainline as One-Step Semantic Fisher Flow / Cycle:
  `theta0 -> G_phi(theta0,D) -> complete-expression semantic tilt -> source-preserving Fisher recoupling -> legal Fisher velocity matching -> same-coupling proposer update`.
- The new mainline uses `scripts/train_complete_expression_semantic_fm.py --training-flow one_step_semantic_fisher_cycle --construction-graph register_categorical_blocks`.
- `docs/STRUCTURAL_CLOSURE.md` records 2026-07-10 evidence that the older `target_conditioned_reference` Stage1 run failed at sample level because `theta0_endpoint_coupling=none` reintroduced the low-t route ambiguity problem.
- `docs/STRUCTURAL_CLOSURE.md` also records the latest semantic latent-endpoint small validation: train/oracle overfit passed on compiled tasks, but holdout GT-family mass/top-4 recall were `0.0` and raw R2 mean was strongly negative, so it is not a generalization pass.
- There is a current documentation conflict: README still says graph categorical-block target-conditioned Stage1 Flow is the current mainline, while algorithm/math/architecture docs say One-Step Semantic Fisher Flow is the current mainline and mark older routes as legacy/ablation.
- Code confirms the 20260710 mainline: `scripts/train_complete_expression_semantic_fm.py` currently defaults to `one_step_semantic_fisher_cycle`; `run()` rejects any other `training_flow`; `run_one_step_semantic_fisher_cycle()` writes `one_step_cycle_*` artifacts and `objective_version=one_step_semantic_fisher_cycle_v1`.
- `semflow_sr/one_step_fisher.py` implements semantic log-quality weights, systematic endpoint resampling, active-block Fisher distance, Hungarian source-preserving coupling, and proposer Fisher map loss.
- `semflow_sr/latent_endpoint.py` implements the older semantic latent complete-endpoint family model with K learned component queries, active-block sharp endpoints, closed-form Fisher transport, and Hungarian family matching. Docs classify it as 2026-07-10 failed historical definition, not current mainline.
- `scripts/run_one_step_semantic_fisher_cycle_gpu.sh` is the clean current runner for the new mainline; it uses `--training-flow one_step_semantic_fisher_cycle --construction-graph register_categorical_blocks`.
- Current retained result runs under `complete_expression_semantic_fm_20260707/runs` include:
  - `one_step_semantic_fisher_cycle_cpu_small_20260710`
  - `one_step_semantic_fisher_cycle_cpu_two_iter_20260710`
  - `semantic_latent_endpoint_l6_k4_overfit10_20260710_validated`
  - several `theta0_register_*_20260710_*` older target-conditioned/endpoint probes.
- Current top-level logs include semantic latent endpoint and theta0 register logs; no one-step CPU log exists in `logs/complete_expression_semantic_fm`.
- `theta0_register_endpoint_masked_l12_20260710_smoke.log` ends in a `RecursionError`, so it is a failed probe log and should not remain beside current validated logs.
- `theta0_register_flow_l12_20260710_smoke.log` and `theta0_register_endpoint_l12_20260710_smoke.log` are old `target_conditioned_reference` probe logs with weak endpoint/flow metrics, not current mainline evidence.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Supersede the older README graph Stage1 contract with the 20260710 One-Step Semantic Fisher Cycle contract | Algorithm/math/architecture docs and current training script default all point to `one_step_semantic_fisher_cycle`. |
| Treat 20260709 semantic-pushforward diagnostics and cleanup manifests as cleanup candidates, not current evidence | Current README says token/semantic endpoint/online semantic mass are legacy/ablation. |
| Keep semantic latent endpoint as historical/failure evidence only | It passed training memorization but failed holdout family mass/raw R2, and docs say not to enter semantic energy/KL refinement from it. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Worktree contains many pre-existing uncommitted changes | Preserve unrelated/user changes and make only tightly scoped cleanup edits. |

## Resources
-
