# SemanticFlowSR — First Milestone Implementation Plan

Goal: implement the **semantic-conditioned local velocity flow** for symbolic regression
exactly as specified in `docs/prompts/coding提示词.md` + `docs/prompts/主线理论.md`, plus the
first-stage data/benchmark deployment from `docs/prompts/数据和实验建议.md`.

Core principle (non-negotiable): training is **strict velocity matching** to the closed-form
`ṗ_λ` from the semantic-weighted Fisher slerp path — NOT endpoint KL / action classification.
No GP, no graph measure, no full expression-space flow, no STOP action.

## Environment (running in background now)
- New conda env `semflow` (Python 3.11).
- `torch` from cu126 wheel index (RTX 3090 = sm_86, works under driver CUDA 13.2). + numpy/scipy/sympy/pandas/sklearn/pyyaml/tqdm/einops/pytest.
- `git-lfs` install → `git lfs pull` real PMLB data (current .tsv.gz are 4KB LFS pointers).
- clone `dso` (deep-symbolic-optimization) into `external/`.
- Baselines (PySR, DSR) get **their own isolated conda envs** later (documented), not the core env.

## Package layout: `SemanticFlowSR/semflow_sr/`
Following the prescribed architecture exactly:
```
sr/         ast, ops, protected_ops, evaluator, parser, printer, simplify
registers/  state (RegisterState/SemanticState), executor (symbolic), trace, compiler
actions/    action_space (encode/decode/valid_mask), action_executor (semantic+symbolic),
            action_masks, action_features
semantics/  probe (ProbeBatch+samplers), semantic_matrix, projection (ridge+hard backend),
            energy (E_{B,y}), rank
geometry/   weights, semantic_chart (S & S^-1), slerp_path (closed-form p_λ, ṗ_λ),
            velocities, distances
endpoints/  base, prior_uniform, prior_grammar, target_gt, target_semantic_oracle
data/       synthetic_generator, trace_dataset, collate, benchmark_loader (SRTask + formula YAML + PMLB)
models/     row_encoder, register_encoder, action_encoder, velocity_model, semantic_transformer
train/      losses (velocity MSE + optional metric), trainer_velocity, train_velocity_gt,
            train_velocity_semantic_oracle
search/     rollout_velocity, beam
eval/       metrics, evaluator, baselines (adapters)
utils/      seed, logging, checkpoint, numerical
```
Plus: `pyproject.toml`, `README.md`, `configs/` (data/model/train/eval YAML),
`scripts/` (generate_trace_dataset, materialize_formula_benchmark, cache_pmlb_subset,
run_pysr_baseline, run_dsr_baseline, run_gplearn_baseline), `tests/`.

## Key math contracts (verified against 主线理论.md §1.5–1.10)
- Energy: `E = ½‖y−Π_a y‖² + λ_r r_eff(Bᵃ) + λ_m‖Π_a−Π_B‖²_F + λ_op C_op(a)`.
- Ridge projection: `Π_{B,ρ}=B(BᵀB+ρI)⁻¹Bᵀ`; r_eff=Tr(Π); never materialize m×m.
- Weights: `w(a)=exp(−η/2·(E−min E))`, clamp ≥ w_min.
- Chart: `z = w·√p / ‖w·√p‖₂`; inverse `p ∝ z²/w²`.
- Path: slerp on z; `p_λ=z_λ²/w² (norm.)`; closed-form `ṗ_λ=p_λ[2·ż/z − Σ p_λ·2·ż/z]`.
- Small-θ fallback → normalized linear interp; clamp z away from 0; smoothed endpoints.
- Velocity model output tangent-projected: `v=raw−mean(raw)` so Σv=0.

## Data (first stage)
1. Synthetic fixed-register **trace dataset** (the core training data) via `synthetic_generator`
   + `scripts/generate_trace_dataset.py` → saves (B,y,actions,energies,p0,p1,λ,p_λ,ṗ_λ,gt_action).
2. Formula benchmarks: `configs/data/formula_benchmarks/{nguyen,constant,livermore,jin}.yaml`
   + `scripts/materialize_formula_benchmark.py` → CSV per seed (SRTask format).
3. PMLB Feynman subset loader + `scripts/cache_pmlb_subset.py` (uses pulled LFS data).

## Tests (the 12 required)
action bijection, protected ops no-NaN, semantic==symbolic execution, ridge==lstsq,
residual energy, projection distance, chart inverse, slerp legality, analytic ṗ vs finite-diff,
zero-sum velocity, overfit-tiny-dataset velocity loss, rollout beats random-policy energy.

## Build order
1. utils + sr/ops/protected_ops + ast/evaluator.
2. registers + actions (space/executor) + tests.
3. semantics (probe/projection/energy/rank) + tests.
4. geometry (weights/chart/slerp/velocities/distances) + tests  ← theoretical core.
5. endpoints.
6. data (synthetic_generator/trace_dataset/collate) + generate script + tests (trace compile).
7. models (semantic transformer) + tests (shapes).
8. train (losses/trainer) + overfit test + train entry scripts + configs.
9. search/rollout + rollout-vs-random test.
10. eval (metrics/evaluator/baselines) + benchmark_loader + formula YAML + materialize/pmlb scripts.
11. baselines: run_*.py adapters + docs for isolated baseline envs.
12. Docs: README, docs/ALGORITHM.md, docs/DATA.md, docs/THEORY_MAPPING.md (theory→code), docs/BASELINES.md.

## Verification before declaring done
`pytest -q` green in `semflow` env; a tiny end-to-end: generate small trace set →
train few steps (overfit) → rollout on one synthetic task shows energy decrease > random.
GPU used (`cuda:0`) with CPU fallback.
