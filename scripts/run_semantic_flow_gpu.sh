#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; echo "[semantic flow exit] $(date -Is) code=${code}"; exit ${code}' EXIT

cd /home/ywj/wyh/SFSR/SemanticFlowSR
PY="${PY:-/home/ywj/miniconda3/envs/semflow/bin/python}"
RUN_GPU="${RUN_GPU:-1}"
export CUDA_VISIBLE_DEVICES="$RUN_GPU"

SCALE="${SCALE:-smoke}"
TAG="${TAG:-semantic_flow_l12_${SCALE}}"
BASE="${RESULT_BASE:-results/clean_benchmark/semantic_flow}"
LOG_DIR="${LOG_DIR:-logs/complete_expression_semantic_fm}"
OUT="${OUT:-$BASE/$TAG}"
LOG="$LOG_DIR/$TAG.log"
mkdir -p "$LOG_DIR"
if [ -e "$OUT" ]; then echo "Refusing existing OUT=$OUT" >&2; exit 3; fi
TRACE_CACHE_ROOT="${TRACE_CACHE_ROOT:-data/cache/semantic_flow}"
if [ ! -f "$TRACE_CACHE_ROOT/compiled_trace_families_v1.jsonl" ] || [ ! -f "$TRACE_CACHE_ROOT/compiled_trace_families_v1.manifest.json" ]; then
  echo "Missing semantic-flow trace cache under $TRACE_CACHE_ROOT" >&2
  echo "Build it first with scripts/build_semantic_flow_trace_cache.py" >&2
  exit 4
fi

case "$SCALE" in
  smoke)
    TRAIN_LIMIT="${TRAIN_LIMIT:-4}"; EVAL_LIMIT="${EVAL_LIMIT:-2}"
    BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-4}"; BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-30}"
    PARTICLES="${PARTICLES:-4}"
    FLOW_EPOCHS="${FLOW_EPOCHS:-3}"; CYCLE_STEPS="${CYCLE_STEPS:-30}"
    HIDDEN="${HIDDEN:-96}"
    EVAL_THETA0_SAMPLES="${EVAL_THETA0_SAMPLES:-4}"
    ;;
  medium)
    TRAIN_LIMIT="${TRAIN_LIMIT:-26}"; EVAL_LIMIT="${EVAL_LIMIT:-8}"
    SYMBOLICGPT_TRAIN_LIMIT="${SYMBOLICGPT_TRAIN_LIMIT:-200}"; SYMBOLICGPT_EVAL_LIMIT="${SYMBOLICGPT_EVAL_LIMIT:-30}"
    BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-6}"; BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-60}"
    PARTICLES="${PARTICLES:-8}"
    FLOW_EPOCHS="${FLOW_EPOCHS:-6}"; CYCLE_STEPS="${CYCLE_STEPS:-60}"
    HIDDEN="${HIDDEN:-128}"
    EVAL_THETA0_SAMPLES="${EVAL_THETA0_SAMPLES:-8}"
    ;;
  overfit)
    TRAIN_LIMIT="${TRAIN_LIMIT:-8}"; EVAL_LIMIT="${EVAL_LIMIT:-4}"
    BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-12}"; BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-80}"
    PARTICLES="${PARTICLES:-8}"
    FLOW_EPOCHS="${FLOW_EPOCHS:-6}"; CYCLE_STEPS="${CYCLE_STEPS:-80}"
    HIDDEN="${HIDDEN:-128}"
    EVAL_THETA0_SAMPLES="${EVAL_THETA0_SAMPLES:-8}"
    ;;
  *) echo "SCALE must be smoke, medium, or overfit" >&2; exit 2;;
esac
SYMBOLICGPT_TRAIN_LIMIT="${SYMBOLICGPT_TRAIN_LIMIT:-0}"
SYMBOLICGPT_EVAL_LIMIT="${SYMBOLICGPT_EVAL_LIMIT:-0}"
OUTER_ITERATIONS="${CYCLE_ITERATIONS:-3}"
CYCLE_EVAL_EACH_ITERATION="${CYCLE_EVAL_EACH_ITERATION:-0}"
EVAL_SELECTION_MODE="${EVAL_SELECTION_MODE:-train_fit}"

ARGS=(
  --out "$OUT" --device cuda:0
  --training-flow one_step_semantic_fisher_cycle
  --construction-graph register_categorical_blocks
  --velocity-parameterization direct_velocity
  --task-conditioning xy --task-encoder-mode hybrid_stats --global-state-mode full
  --suites nguyen constant livermore jin
  --train-task-limit "$TRAIN_LIMIT" --eval-task-limit "$EVAL_LIMIT"
  --symbolicgpt-root "${SYMBOLICGPT_ROOT:-data/generated/symbolicgpt_large_2000_200_200}"
  --symbolicgpt-train-limit "$SYMBOLICGPT_TRAIN_LIMIT" --symbolicgpt-eval-limit "$SYMBOLICGPT_EVAL_LIMIT"
  --trace-cache-root "$TRACE_CACHE_ROOT" --trace-cache-mode require
  --num-vars 3 --num-layers 12 --num-registers 17
  --ops copy,add,sub,mul,protected_div,sin,cos,square,cube,protected_log,protected_sqrt,exp
  --output-terms 1 --gt-traces-per-task 8
  --hidden "$HIDDEN" --metadata-embedding-dim 16
  --max-train-points 64 --max-eval-points 64
  --epochs "$BOOTSTRAP_EPOCHS" --steps-per-epoch "$BOOTSTRAP_STEPS" --train-batch-size 4
  --lr "${BOOTSTRAP_LR:-5e-4}" --weight-decay 1e-5 --grad-clip 1.0
  --theta0-noise-scale 1.0 --theta0-endpoint-coupling none
  --inactive-block-target-mode start --inactive-block-loss-weight 0.0
  --cycle-iterations "$OUTER_ITERATIONS"
  --cycle-particles-per-task "$PARTICLES" --cycle-expression-samples 0
  --cycle-proposer-rollout-steps "${PROPOSER_ROLLOUT_STEPS:-8}"
  --cycle-poisson-steps "${POISSON_STEPS:-64}" --cycle-poisson-lr "${POISSON_LR:-1e-3}"
  --cycle-correction-step "${CORRECTION_STEP:-0.1}" --cycle-support-variance-eps "${SUPPORT_VARIANCE_EPS:-1e-6}"
  --cycle-collection-timeout-sec "${COLLECTION_TIMEOUT_SEC:-300}"
  --bootstrap-source-mass-schedule "${BOOTSTRAP_SOURCE_MASS_SCHEDULE:-0.30,0.20,0.10}" --bootstrap-inactive-weight "${BOOTSTRAP_INACTIVE_WEIGHT:-0.10}"
  --bootstrap-gate "${BOOTSTRAP_GATE:-C}"
  --cycle-landscape-sources "${LANDSCAPE_SOURCES:-4}"
  --cycle-landscape-task-limit "${LANDSCAPE_TASK_LIMIT:-1}"
  --cycle-landscape-time-points "${LANDSCAPE_TIME_POINTS:-5}"
  --cycle-flow-epochs "$FLOW_EPOCHS"
  --cycle-steps-per-epoch "$CYCLE_STEPS"
  --cycle-flow-lr "${FLOW_LR:-5e-4}"
  --cycle-time-sampling stratified_fisher
  --time-sampling low_t_mixture --low-t-sampling-prob 0.4 --low-t-max 0.1
  --ode-steps "${ODE_STEPS:-32}" --ode-sweep-steps "${ODE_SWEEP_STEPS:-}"
  --eval-theta0-samples "$EVAL_THETA0_SAMPLES" --eval-samples "${EVAL_SAMPLES:-2}"
  --eval-flow-gt-probe-samples "${EVAL_FLOW_GT_PROBE_SAMPLES:-4}"
  --eval-theta0-mode deterministic_random
  --eval-endpoint-decode-mode hard_argmax
  --eval-oracle-free-selection-mode "$EVAL_SELECTION_MODE"
  --no-eval-terminal-retraction
  --no-eval-theta0-use-gt-trace
  --temporal-visualization-steps 16
  --seed "${SEED:-20260711}" --log-epochs
)

case "$CYCLE_EVAL_EACH_ITERATION" in
  1|true|TRUE|yes|YES) ARGS+=(--cycle-eval-each-iteration) ;;
  *) ARGS+=(--no-cycle-eval-each-iteration) ;;
esac

echo "[semantic flow] $(date -Is) scale=$SCALE gpu=$RUN_GPU out=$OUT" | tee "$LOG"
echo "[budget] classic_tasks=${TRAIN_LIMIT}/${EVAL_LIMIT} symbolicgpt_tasks=${SYMBOLICGPT_TRAIN_LIMIT}/${SYMBOLICGPT_EVAL_LIMIT} particles=$PARTICLES outer_iterations=$OUTER_ITERATIONS poisson_steps=${POISSON_STEPS:-64} correction_step=${CORRECTION_STEP:-0.1} residual_epochs=$FLOW_EPOCHS eval_selection=$EVAL_SELECTION_MODE eval_each_iter=$CYCLE_EVAL_EACH_ITERATION" | tee -a "$LOG"
"$PY" scripts/train_complete_expression_semantic_fm.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
