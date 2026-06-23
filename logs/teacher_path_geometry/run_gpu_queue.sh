#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ywj/wyh/SFSR/SemanticFlowSR

LOG=logs/teacher_path_geometry/gpu_queue.log
exec >> "${LOG}" 2>&1

trap 'status=$?; echo "[$(date -Is)] teacher_path_geometry GPU queue failed at line ${LINENO} with status ${status}"' ERR

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "$$" > logs/teacher_path_geometry/gpu_queue.pid
echo "[$(date -Is)] teacher_path_geometry GPU queue start pid=$$"
conda run --no-capture-output -n semflow python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count()); print('cuda1', torch.cuda.get_device_name(1) if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'missing')"
nvidia-smi

echo "[$(date -Is)] semantic train start"
taskset -c 0-3 conda run --no-capture-output -n semflow \
  python -u -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.yaml

echo "[$(date -Is)] semantic eval start"
taskset -c 0-3 conda run --no-capture-output -n semflow \
  python -u scripts/run_edge_flow.py \
  --ckpt checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.pt \
  --out results/teacher_path_geometry_semantic_gpu \
  --tag teacher_path_geometry_semantic_gpu \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --legacy_87 \
  --feynman_root data/materialized/feynman \
  --eval_samples 16 \
  --flow_steps 1 \
  --sampler_method policy \
  --seed 0 \
  --decoder_budgets 8 16 \
  --postprocess_top_k 8 \
  --selection_validation_fraction 0.25 \
  --selection_eta_logprob 0.0 \
  --device cuda:1 \
  --head_fit_mode linear \
  --complexity_weight 0.0

echo "[$(date -Is)] euclidean train start"
taskset -c 0-3 conda run --no-capture-output -n semflow \
  python -u -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.yaml

echo "[$(date -Is)] euclidean eval start"
taskset -c 0-3 conda run --no-capture-output -n semflow \
  python -u scripts/run_edge_flow.py \
  --ckpt checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.pt \
  --out results/teacher_path_geometry_euclidean_gpu \
  --tag teacher_path_geometry_euclidean_gpu \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --legacy_87 \
  --feynman_root data/materialized/feynman \
  --eval_samples 16 \
  --flow_steps 1 \
  --sampler_method policy \
  --seed 0 \
  --decoder_budgets 8 16 \
  --postprocess_top_k 8 \
  --selection_validation_fraction 0.25 \
  --selection_eta_logprob 0.0 \
  --device cuda:1 \
  --head_fit_mode linear \
  --complexity_weight 0.0

echo "[$(date -Is)] teacher_path_geometry GPU queue complete"
