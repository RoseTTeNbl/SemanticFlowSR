#!/usr/bin/env bash
set -euo pipefail
cd /home/ywj/wyh/SFSR/SemanticFlowSR
PY="${PY:-/home/ywj/miniconda3/envs/semflow/bin/python}"
export CUDA_VISIBLE_DEVICES="${RUN_GPU:-1}"
CACHE="${TRACE_CACHE_ROOT:-data/cache/semantic_flow}"
BASE="${RESULT_BASE:-results/clean_benchmark/semantic_flow/bootstrap}"
mkdir -p "$BASE"
if [ ! -f "$CACHE/compiled_trace_families_v1.jsonl" ] || [ ! -f "$CACHE/compiled_trace_families_v1.manifest.json" ]; then
  echo "Missing semantic-flow trace cache under $CACHE" >&2
  exit 4
fi

COMMON=(
  --device cuda:0 --training-flow one_step_semantic_fisher_cycle
  --construction-graph register_categorical_blocks --velocity-parameterization direct_velocity
  --task-conditioning xy --task-encoder-mode hybrid_stats --global-state-mode full
  --num-vars 3 --num-layers 12 --num-registers 17
  --ops copy,add,sub,mul,protected_div,sin,cos,square,cube,protected_log,protected_sqrt,exp
  --output-terms 1 --gt-traces-per-task 8
  --trace-cache-root "$CACHE" --trace-cache-mode require
  --hidden 128 --metadata-embedding-dim 16 --max-train-points 64 --max-eval-points 64
  --epochs "${BOOTSTRAP_EPOCHS:-6}" --steps-per-epoch "${BOOTSTRAP_STEPS:-60}" --train-batch-size 8
  --cycle-iterations 0 --cycle-particles-per-task 8 --cycle-proposer-rollout-steps 8
  --cycle-collection-timeout-sec 300 --bootstrap-source-mass-schedule 0.30,0.20,0.10
  --bootstrap-inactive-weight "${BOOTSTRAP_INACTIVE_WEIGHT:-0.20}" --no-cycle-eval-each-iteration
  --eval-theta0-samples 8 --eval-samples 1 --seed "${SEED:-20260711}" --log-epochs
)

run_gate() {
  local gate="$1"; local tasks="$2"; local tag="$3"
  "$PY" scripts/train_complete_expression_semantic_fm.py \
    --out "$BASE/$tag" --bootstrap-gate "$gate" \
    --suites livermore nguyen --task-id-filter "$tasks" --eval-fraction 0 --allow-empty-eval \
    --train-task-limit 0 --eval-task-limit 0 \
    "${COMMON[@]}" --seed "${SEED:-20260711}"
}

for gate_seed in 20260711 20260712 20260713; do
  SEED="$gate_seed" run_gate A "livermore/Livermore-5" "gate_a_livermore5_seed${gate_seed}"
done
run_gate B "livermore/Livermore-3,livermore/Livermore-4,livermore/Livermore-5,nguyen/Nguyen-1,nguyen/Nguyen-3,nguyen/Nguyen-5,nguyen/Nguyen-6,nguyen/Nguyen-7" gate_b_8tasks
"$PY" scripts/train_complete_expression_semantic_fm.py \
  --out "$BASE/gate_c_24_8" --bootstrap-gate C --suites nguyen constant livermore jin \
  --symbolicgpt-root data/generated/symbolicgpt_large_2000_200_200 \
  --symbolicgpt-train-limit 20 --symbolicgpt-eval-limit 8 \
  --train-task-limit 4 --eval-task-limit 0 "${COMMON[@]}"
