# SemanticFlowSR

SemanticFlowSR is currently centered on:

```text
Edge-Parameterized Semantic Flow Matching
```

The mainline no longer trains a local action policy. It builds a
register-operator circuit template, places a product-of-simplexes edge
distribution over complete expression DAGs, samples complete expressions,
projects reward elites to an edge target, and trains a flow on edge
probabilities.

For a new coding session, read [AGENTS.md](AGENTS.md), then
[docs/ALGORITHM.md](docs/ALGORITHM.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Current Algorithm

```text
data D=(X,y)
-> RegisterOperatorTemplate
-> EdgeDistribution Theta0
-> sample complete expressions from q_Theta
-> reward complete expressions
-> project elites to Theta*
-> build Fisher square-root teacher path
-> train EdgeFlowModel on dot z
-> infer by integrating Theta and sampling complete expressions
```

Old action-level SFFM/TSSF code remains for legacy diagnostics, but it is not
the current main algorithm.

## Environment

Run commands from `SemanticFlowSR/`.

```bash
conda create -n semflow python=3.11
conda activate semflow
pip install --index-url https://download.pytorch.org/whl/cu126 torch
pip install numpy scipy sympy pyyaml pandas scikit-learn tqdm einops pytest
pip install deap gplearn
pip install -e .
```

## Quick Commands

```bash
# Current Edge Flow unit tests.
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  pytest -q tests/test_edge_flow_core.py tests/test_edge_flow_training.py

# Smoke train. This is not full training.
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/edge_flow_smoke.yaml

# Smoke evaluate.
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python scripts/run_edge_flow.py \
  --ckpt checkpoints/edge_flow_smoke.pt \
  --out results/edge_flow_smoke \
  --tag edge_flow_smoke

# Validate the unified benchmark manifest before future full experiments.
conda run -n semflow python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

## Main Code Map

| Purpose | File |
|---|---|
| Template and edge groups | `semflow_sr/edge_flow/template.py` |
| Edge distribution | `semflow_sr/edge_flow/edge_distribution.py` |
| Complete DAG sampler | `semflow_sr/edge_flow/circuit_sampler.py` |
| Complete-expression reward | `semflow_sr/edge_flow/reward.py` |
| Elite projection | `semflow_sr/edge_flow/projection.py` |
| Fisher-slerp teacher | `semflow_sr/edge_flow/flow_teacher.py` |
| Training records | `semflow_sr/edge_flow/dataset.py` |
| Edge-flow model/loss | `semflow_sr/edge_flow/model.py` |
| Smoke trainer | `semflow_sr/edge_flow/train_edge_flow.py` |
| Smoke evaluator | `scripts/run_edge_flow.py` |

## Results

Current tracked result is smoke-only:

```text
results/edge_flow_smoke/
```

It verifies the new path can run end to end. It is not a benchmark result.

## Legacy Code

The old action-level and block-flow code paths remain available for regression
comparison:

```text
semflow_sr/path_posterior/
semflow_sr/train/train_path_posterior_flow.py
scripts/run_path_posterior_flow.py
semflow_sr/train/train_block_flow.py
scripts/run_block_risk_flow.py
```

Do not treat them as the current algorithm unless the user explicitly asks for
a legacy comparison.
