# Complete Expression Semantic FM Plan

## Objective
Rebuild the graph `graph_dag_edge_simplex` Stage1 path as the active mainline:
target-conditioned reference flow with `X,y` conditioning, GT trace endpoints,
`theta0` active-choice coupling, and active-block-only velocity matching.
Token construction remains archived. Stage2 work has reopened only for the graph
branch under the updated iterative semantic endpoint-corrected bridge FM theory:
`v^k -> theta1_ref -> semantic terminal tilt -> theta1_plus -> new bridge
(theta0, theta1_plus) -> v^(k+1)`.

## Status
- [complete] Inspect current Stage1 endpoint-attractor diagnostic.
- [complete] Decide whether Stage1 passes the structure/sharpness gate.
- [complete] Fix the reference flow/model conditioning before running formal Stage2.
- [complete] Implement endpoint semantic pushforward correction as the clean Stage2 training path.
- [complete] Implement target-distance semantic pushforward diagnostics and correction plumbing.
- [complete] Implement refined semantic-pushforward metric: semantic-space tilt, lifted expression posterior, endpoint projection, and top-neighborhood concentration diagnostics without ODE inside training gradient steps.
- [complete] Align token Stage2 with the latest theory using weighted complete-trace posterior FM (`weighted_trace_fm`) instead of single posterior-trace compression as the default.
- [complete] Verify latest endpoint semantic-pushforward implementation against the new theory note with static checks and tiny GPU graph/token smoke runs.
- [complete] Implement endpoint semantic-space KL tilt v3: `theta1 -> q_theta1(z|D) -> semantic pushforward -> target-neighborhood KL tilt -> lifted q_plus -> cached theta1_plus -> supervised FM correction`, with explicit target-near lift / target-far suppression diagnostics.
- [complete] Implement endpoint semantic-space KL tilt v4: keep decoupled endpoint collection/training and add signed target-near/far semantic contrast diagnostics so the acceptance target is explicitly "raise target-near top semantics and suppress target-far semantics".
- [complete] Implement endpoint semantic-signature KL tilt v5: replace output-only distance with grouped curve/shape/dependency semantic signature distance while keeping the endpoint pushforward chain and no training-step ODE simulation.
- [complete] Run a GPU token endpoint-correction validation for v5 signature tilt and compare whether target-near semantic mass and projected endpoint quality improve beyond v4.
- [complete] Refresh permanent semantic-mass branch diagnostics under `results/clean_benchmark_20260701/ablations/complete_expression_semantic_fm_20260707/`.
- [complete] Implement endpoint semantic-signature pushforward projection v6: keep the endpoint-level semantic KL chain and add rank-utility diagnostics/acceptance so the sampled semantic distribution is checked for smooth target-near mass lift and target-far suppression, not only hard near/far quantiles.
- [complete] Implement endpoint semantic-signature pushforward projection v7: keep endpoint collection/training decoupled and add continuous target-soft-utility / background-utility diagnostics so the sampled semantic distribution is checked for target-near lift and unrelated semantic suppression without relying only on hard top/bottom bins.
- [complete] Implement endpoint semantic-signature pushforward projection v8: keep endpoint collection/training decoupled and replace the ambiguous background diagnostic with an independent semantic-distance far/background utility, while retaining v7 fields as compatibility aliases.
- [complete] Implement endpoint semantic-signature pushforward projection v9: keep the endpoint-only collection/training split and add a signed target-vs-far semantic utility mean so the main statistic directly checks "target-top semantics up and far/unrelated semantics down".
- [complete] Implement endpoint semantic-signature pushforward projection v10: keep the endpoint-only collection/training split and add a top-near/far-tail signed utility so acceptance explicitly checks target-top mass rises while far-tail semantics weaken.
- [complete] Add a read-only v10 run auditor that separates reference-field, semantic posterior, posterior projection, endpoint FM, and rollout/readout bottlenecks for both graph/token branches.
- [complete] Stop the pending v10 waiting controller/audit watcher before applying the latest endpoint semantic-space theory update.
- [complete] Implement endpoint semantic-signature pushforward projection v11: keep endpoint-only collection/training and add the hard full-expression posterior `target_top_tail_mass_contrast` mean statistic as the primary "target-top up / far-tail down" check.
- [complete] Update graph/token endpoint correction, branch metrics, audit gate, docs, tests, and smoke runner naming to the v11 objective.
- [complete] Archive token construction, Stage2 semantic endpoint/pushforward runners, and old failed graph gate artifacts as legacy evidence.
- [complete] Add the current graph Stage1 runner: `scripts/run_graph_target_conditioned_stage1_gpu.sh`.
- [complete] Run a representative graph Stage1 smoke with multi-variable eval coverage and inspect `typed_op_node_flow_samples.jsonl`.
- [in_progress] Run medium/full graph Stage1 with broader multi-variable coverage; judge by convergence, R2, and sample expression structure rather than a single hard gate.
- [complete] Update graph Stage2 correction to iterative semantic endpoint-corrected bridge FM instead of residual-only velocity correction.
- [complete] Add graph Stage2 corrected-bridge runner and validate the new code path with tiny train-only/eval smoke runs.
- [in_progress] Rerun full-data 8-epoch Stage1 with light eval to obtain a reusable clean full checkpoint and sample-level R2/structure evidence.
- [pending] Run graph Stage2 corrected-bridge iterations from the clean full Stage1 checkpoint with bounded endpoint buffers and light eval.

## Constraints
- Main construction graph: `graph_dag_edge_simplex` (`fixed_symbol_node_edges` remains an alias).
- Preserve `register_categorical_blocks` only as an ablation switch.
- GT traces provide teacher endpoints/active masks and `theta0` route coupling for Stage1 construction.
- Keep the training objective clean: single velocity matching loss for Stage1.
- Stage2 correction must train on corrected bridge samples from `(theta0, theta1_plus)`,
  where `theta1_plus` is obtained by terminal semantic tilt and graph endpoint
  projection. Residual-only/base-frozen correction is legacy ablation only.
- Current runner uses GPU, but validation judgment must include samples/GT inspection, not only aggregate metrics.
