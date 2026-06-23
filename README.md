# SemanticFlowSR

SemanticFlowSR 当前聚焦于：

```text
Conditional Semantic Edge Flow, CSEF
```

CSEF 在表达式构造图的显式 categorical 概率形上做 flow matching。GT 先编译为 canonical 构造路径，GT-neighborhood 只提供 noisy context，目标概率形 `P_1` 直接流向 clean GT decision 的 smoothed one-hot。隐参数网络 `psi` 根据当前任务、寄存器语义、构造前缀、当前概率状态 `p_t` 和时间 `t` 预测局部速度。语义只作为网络输入和速度误差校准，不进入 Fisher-Rao teacher path。

建议新会话先读：

```text
AGENTS.md
docs/ALGORITHM.md
docs/ARCHITECTURE.md
docs/MATH.md
docs/RESULTS.md
```

## 当前流程

```text
D=(X,y, optional GT)
-> RegisterOperatorTemplate
-> ConditionalEdgeFlowModel
-> GT compiler builds canonical CSEF paths
-> GT-neighborhood samples noisy symbolic contexts
-> clean GT decisions define target probability shape P_1
-> sample P_0 and t
-> build Fisher or Euclidean p_0 -> p_t -> teacher velocity
-> recompute model velocity at the same p_t,t
-> train semantic-calibrated velocity matching
-> evaluate with sparse linear head fitting and structure-prior rerank
```

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

## Main Commands

Fisher probability-shape training:

```bash
conda run --no-capture-output -n semflow \
  python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.yaml
```

Euclidean probability-coordinate ablation:

```bash
conda run --no-capture-output -n semflow \
  python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.yaml
```

Fisher evaluation:

```bash
conda run --no-capture-output -n semflow python scripts/run_edge_flow.py \
  --ckpt checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.pt \
  --out results/teacher_path_geometry_fisher_gpu \
  --tag teacher_path_geometry_fisher_gpu \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --manifest_suite nguyen constant livermore jin \
  --eval_samples 64 \
  --flow_steps 1 \
  --sampler_method policy \
  --head_fit_mode linear \
  --device cuda:1
```

Euclidean evaluation:

```bash
conda run --no-capture-output -n semflow python scripts/run_edge_flow.py \
  --ckpt checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.pt \
  --out results/teacher_path_geometry_euclidean_gpu_20260623 \
  --tag teacher_path_geometry_euclidean_gpu_20260623 \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --manifest_suite nguyen constant livermore jin \
  --eval_samples 64 \
  --flow_steps 1 \
  --sampler_method policy \
  --head_fit_mode linear \
  --device cuda:1
```

Build comparison metrics and figures:

```bash
conda run --no-capture-output -n semflow python scripts/archive_paper_metrics.py \
  --out results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623 \
  --suite nguyen constant livermore jin \
  --bootstrap_samples 1000 \
  --method CSEF-Fisher SFSR sfsr_method samples_jsonl results/teacher_path_geometry_fisher_gpu/teacher_path_geometry_fisher_gpu_samples.jsonl \
  --method CSEF-Euclidean SFSR sfsr_method samples_jsonl results/teacher_path_geometry_euclidean_gpu_20260623/teacher_path_geometry_euclidean_gpu_20260623_samples.jsonl
```

Regression tests:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n semflow \
  pytest tests/test_edge_flow_core.py tests/test_edge_flow_training.py \
         tests/test_external_adapter_outputs.py tests/test_paper_metrics.py -q
```

## Code Map

| Purpose | File |
|---|---|
| Template and register metadata | `semflow_sr/edge_flow/template.py` |
| Conditional CSEF model and sampler | `semflow_sr/edge_flow/conditional.py` |
| Probability-shape teacher velocity and semantic calibration | `semflow_sr/edge_flow/semantic_teacher.py` |
| GT-neighborhood noisy-context sampler | `semflow_sr/edge_flow/gt_neighborhood.py` |
| Formula-to-CSEF compiler | `semflow_sr/edge_flow/path_compiler.py` |
| Reward and sparse head fitting | `semflow_sr/edge_flow/reward.py` |
| Structure-prior rerank score | `semflow_sr/edge_flow/selection.py` |
| Benchmark loading and result writing | `semflow_sr/edge_flow/benchmark.py` |
| Training CLI | `semflow_sr/edge_flow/train_edge_flow.py` |
| Evaluation CLI | `scripts/run_edge_flow.py` |
| Paper metrics bundle builder | `scripts/archive_paper_metrics.py` |

## Current Results

Current comparison is recorded in [docs/RESULTS.md](docs/RESULTS.md).

```text
Fisher:    R2 mean 0.937383, solution rate 0.323529, skeleton accuracy 0.029412
Euclidean: R2 mean 0.940868, solution rate 0.294118, skeleton accuracy 0.0
```
