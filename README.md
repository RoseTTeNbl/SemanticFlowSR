# SemanticFlowSR

Semantic-conditioned **local velocity flow** for symbolic regression.

The method defines a velocity flow on the one-step discrete **action simplex** `Δ(A)`,
conditioned on the current semantic matrix `B`, probe target `y`, and action-induced
semantic energies. The core chart is the **semantic-conditioned Fisher chart**

```
S_{B,y}(p)(a) = w_{B,y}(a)·√p(a) / ( Σ_b w_{B,y}(b)²·p(b) )^{1/2},
w_{B,y}(a)    = exp(-η/2 · E_{B,y}(a)).
```

Training is **strict velocity matching** to the closed-form `ṗ_λ` of the semantic Fisher
slerp path — *not* endpoint KL / action classification. No GP, no graph measure, no
full-expression-space flow, no STOP action (stop by semantic-energy threshold).

文档（中文）：[docs/](docs/README.md) — 架构、算法、理论映射、数据集、外部基线。

## Environment

GPU: 3× RTX 3090 (sm_86), CUDA driver 13.2 / nvcc 12.6.

```bash
conda create -n semflow python=3.11
conda activate semflow
pip install --index-url https://download.pytorch.org/whl/cu126 torch
pip install numpy scipy sympy pyyaml pandas scikit-learn tqdm einops pytest
pip install -e .          # from SemanticFlowSR/
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Run workflow

Four stages: **(0)** test → **(1)** build data → **(2)** train → **(3)** eval. Run from
`SemanticFlowSR/` in the `semflow` env.

```bash
# 0. verify install (24 correctness tests)
pytest -q

# 1a. core velocity-flow trace dataset (the training data)
python scripts/generate_trace_dataset.py \
  --num_tasks 2000 --num_vars 1 --max_depth 4 --K 8 --probe_size 128 \
  --target gt --out data/local_flow_traces/v0

# 1b. standard SR formula benchmarks -> per-seed CSVs
python scripts/materialize_formula_benchmark.py \
  --suite nguyen constant livermore jin --seeds 0 1 2 3 4 --out data/materialized

# 1c. PMLB Feynman subset (data already in external/pmlb/datasets/)
python scripts/cache_pmlb_subset.py --pattern feynman --out data/pmlb/feynman

# 2. train the velocity model (strict velocity matching)
python -m semflow_sr.train.train_velocity_gt --config configs/train/velocity_gt.yaml
python -m semflow_sr.train.train_velocity_semantic_oracle \
  --config configs/train/velocity_semantic_oracle.yaml

# 4. baselines (each in its OWN conda env — see docs/baselines/)
python scripts/run_pysr_baseline.py --data data/materialized/nguyen --out results/pysr
```

**Stage 3 (eval)** loads a checkpoint, integrates `v_θ` from `p0` over a λ-grid, executes
actions until residual energy is small, and reports R²/expression. See the runnable
snippet in [docs/datasets/adaptation.md](docs/datasets/adaptation.md).

Details per stage: data → [docs/datasets/](docs/datasets/README.md); training/eval flags →
each `configs/{train,eval}/*.yaml`; baselines → [docs/baselines/](docs/baselines/README.md).

## Layout

```
configs/   实验配置        semflow_sr/  核心包        scripts/  命令行入口
tests/     24 个测试        docs/        文档          external/ 参考仓库 + PMLB 数据
```

Full directory & file descriptions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
The theoretical core is `semflow_sr/semantics/` (B, projection, energy) and
`semflow_sr/geometry/` (chart, slerp path, closed-form velocity).
