#!/usr/bin/env bash
set -euo pipefail

CONDA_EXE="${CONDA_EXE:-/home/ywj/miniconda3/bin/conda}"
PY_ENV="${PY_ENV:-semflow}"
PYTHON=python
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT="${ROOT:-results/clean_benchmark/paper_complete}"
CLEAN_ROOT="${CLEAN_ROOT:-results/clean_benchmark}"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR" "$CLEAN_ROOT/external_baselines/formula_dev" "$CLEAN_ROOT/external_baselines/symbolicgpt_large"

FORMULA_MANIFEST="${FORMULA_MANIFEST:-data/benchmark_suites/benchmark_manifest.json}"
FORMULA_ROOT="${FORMULA_ROOT:-data/benchmark_suites}"
SYMGPT_MANIFEST="${SYMGPT_MANIFEST:-data/benchmark_suites/symbolicgpt_large_2000_200_200/symbolicgpt_large_test_compilable_manifest.json}"
SYMGPT_ROOT="${SYMGPT_ROOT:-data/benchmark_suites}"
TRAIN_ROOT="${TRAIN_ROOT:-data/generated/symbolicgpt_large_2000_200_200}"

MAX_TASKS_ARG=()
if [[ "${MAX_TASKS:-}" != "" ]]; then
  MAX_TASKS_ARG=(--max_tasks "$MAX_TASKS")
fi

run_semflow() {
  local name="$1"
  shift
  echo "[paper-complete] $name"
  PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$CONDA_EXE" run -n "$PY_ENV" "$PYTHON" "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

run_env() {
  local env_name="$1"
  local name="$2"
  shift 2
  echo "[paper-complete] $name"
  PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$CONDA_EXE" run -n "$env_name" "$PYTHON" "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

run_formula() {
  local tag="$1"
  shift
  run_semflow "${tag}_formula_dev" "$@" \
    --manifest "$FORMULA_MANIFEST" \
    --suite nguyen constant livermore jin \
    --root "$FORMULA_ROOT" \
    --out "$CLEAN_ROOT/external_baselines/formula_dev" \
    --tag "${tag}_formula_dev" \
    "${MAX_TASKS_ARG[@]}"
}

run_symbolicgpt() {
  local tag="$1"
  shift
  run_semflow "${tag}_symbolicgpt_large" "$@" \
    --manifest "$SYMGPT_MANIFEST" \
    --suite symbolicgpt_large \
    --root "$SYMGPT_ROOT" \
    --out "$CLEAN_ROOT/external_baselines/symbolicgpt_large" \
    --tag "${tag}_symbolicgpt_large" \
    "${MAX_TASKS_ARG[@]}"
}

run_small_pair() {
  local tag="$1"
  local script="$2"
  shift 2
  run_formula "$tag" "$script" --train_root "$TRAIN_ROOT" "$@"
  run_symbolicgpt "$tag" "$script" --train_root "$TRAIN_ROOT" "$@"
}

METHODS="${METHODS:-gp deap dso tpsr pysr e2e localdiffusion symgpt nesymres hvae nggp}"
for method in $METHODS; do
  case "$method" in
    gp)
      run_formula gp scripts/run_gplearn_baseline.py --generations "${GP_GENERATIONS:-12}" --population_size "${GP_POPULATION:-500}" --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-180}"
      run_symbolicgpt gp scripts/run_gplearn_baseline.py --generations "${GP_GENERATIONS:-12}" --population_size "${GP_POPULATION:-500}" --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-180}"
      ;;
    deap)
      run_formula deap scripts/run_deap_baseline.py --generations "${DEAP_GENERATIONS:-20}" --population_size "${DEAP_POPULATION:-500}" --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-180}"
      run_symbolicgpt deap scripts/run_deap_baseline.py --generations "${DEAP_GENERATIONS:-20}" --population_size "${DEAP_POPULATION:-500}" --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-180}"
      ;;
    pysr)
      run_env "${PYSR_ENV:-pysr}" pysr_formula_dev scripts/run_pysr_baseline.py \
        --manifest "$FORMULA_MANIFEST" --suite nguyen constant livermore jin --root "$FORMULA_ROOT" \
        --out "$CLEAN_ROOT/external_baselines/formula_dev" --tag pysr_formula_dev \
        --niterations "${PYSR_NITERATIONS:-100}" "${MAX_TASKS_ARG[@]}"
      run_env "${PYSR_ENV:-pysr}" pysr_symbolicgpt_large scripts/run_pysr_baseline.py \
        --manifest "$SYMGPT_MANIFEST" --suite symbolicgpt_large --root "$SYMGPT_ROOT" \
        --out "$CLEAN_ROOT/external_baselines/symbolicgpt_large" --tag pysr_symbolicgpt_large \
        --niterations "${PYSR_NITERATIONS:-100}" "${MAX_TASKS_ARG[@]}"
      ;;
    dso)
      run_env "${DSO_ENV:-dso37}" dso_formula_dev scripts/run_dsr_baseline.py \
        --manifest "$FORMULA_MANIFEST" --suite nguyen constant livermore jin --root "$FORMULA_ROOT" \
        --out "$CLEAN_ROOT/external_baselines/formula_dev" --tag dso_formula_dev \
        --n_samples "${DSO_SAMPLES:-20000}" --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-300}" "${MAX_TASKS_ARG[@]}"
      run_env "${DSO_ENV:-dso37}" dso_symbolicgpt_large scripts/run_dsr_baseline.py \
        --manifest "$SYMGPT_MANIFEST" --suite symbolicgpt_large --root "$SYMGPT_ROOT" \
        --out "$CLEAN_ROOT/external_baselines/symbolicgpt_large" --tag dso_symbolicgpt_large \
        --n_samples "${DSO_SAMPLES:-20000}" --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-300}" "${MAX_TASKS_ARG[@]}"
      ;;
    tpsr)
      run_env "${TPSR_ENV:-tpsr}" tpsr_formula_dev scripts/run_tpsr_manifest_baseline.py \
        --manifest "$FORMULA_MANIFEST" --suite nguyen constant livermore jin --root "$FORMULA_ROOT" \
        --out "$CLEAN_ROOT/external_baselines/formula_dev" --tag tpsr_mcts_formula_dev \
        --mode mcts --beam_size "${TPSR_BEAM_SIZE:-2}" --n_trees_to_refine "${TPSR_REFINE:-2}" \
        --max_input_points "${TPSR_MAX_INPUT_POINTS:-64}" --max_number_bags "${TPSR_BAGS:-1}" \
        --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-300}" "${MAX_TASKS_ARG[@]}"
      run_env "${TPSR_ENV:-tpsr}" tpsr_symbolicgpt_large scripts/run_tpsr_manifest_baseline.py \
        --manifest "$SYMGPT_MANIFEST" --suite symbolicgpt_large --root "$SYMGPT_ROOT" \
        --out "$CLEAN_ROOT/external_baselines/symbolicgpt_large" --tag tpsr_mcts_symbolicgpt_large \
        --mode mcts --beam_size "${TPSR_BEAM_SIZE:-2}" --n_trees_to_refine "${TPSR_REFINE:-2}" \
        --max_input_points "${TPSR_MAX_INPUT_POINTS:-64}" --max_number_bags "${TPSR_BAGS:-1}" \
        --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-300}" "${MAX_TASKS_ARG[@]}"
      ;;
    e2e)
      run_env "${TPSR_ENV:-tpsr}" e2e_formula_dev scripts/run_e2e_baseline.py \
        --manifest "$FORMULA_MANIFEST" --suite nguyen constant livermore jin --root "$FORMULA_ROOT" \
        --out "$CLEAN_ROOT/external_baselines/formula_dev" --tag e2e_formula_dev \
        --beam_size "${E2E_BEAM_SIZE:-1}" --n_trees_to_refine "${E2E_REFINE:-1}" \
        --max_input_points "${E2E_MAX_INPUT_POINTS:-64}" --max_number_bags "${E2E_BAGS:-1}" \
        "${MAX_TASKS_ARG[@]}"
      run_env "${TPSR_ENV:-tpsr}" e2e_symbolicgpt_large scripts/run_e2e_baseline.py \
        --manifest "$SYMGPT_MANIFEST" --suite symbolicgpt_large --root "$SYMGPT_ROOT" \
        --out "$CLEAN_ROOT/external_baselines/symbolicgpt_large" --tag e2e_symbolicgpt_large \
        --beam_size "${E2E_BEAM_SIZE:-1}" --n_trees_to_refine "${E2E_REFINE:-1}" \
        --max_input_points "${E2E_MAX_INPUT_POINTS:-64}" --max_number_bags "${E2E_BAGS:-1}" \
        "${MAX_TASKS_ARG[@]}"
      ;;
    localdiffusion)
      run_formula localdiffusion scripts/run_diffusion_sr_baseline.py --proposal_limit "${DIFFUSION_PROPOSAL_LIMIT:-2000}" --candidate_limit "${DIFFUSION_CANDIDATE_LIMIT:-512}" --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-180}"
      run_symbolicgpt localdiffusion scripts/run_diffusion_sr_baseline.py --proposal_limit "${DIFFUSION_PROPOSAL_LIMIT:-2000}" --candidate_limit "${DIFFUSION_CANDIDATE_LIMIT:-512}" --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-180}"
      ;;
    symgpt)
      run_small_pair symgpt scripts/run_symbolicgpt_baseline.py --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-120}"
      ;;
    nesymres)
      run_small_pair nesymres scripts/run_nesymres_baseline.py --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-120}"
      ;;
    hvae)
      run_small_pair hvae scripts/run_hvae_baseline.py --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-120}"
      ;;
    nggp)
      run_small_pair nggp scripts/run_nggp_baseline.py --per_task_timeout_sec "${PER_TASK_TIMEOUT_SEC:-120}"
      ;;
    *)
      echo "unknown method: $method" >&2
      exit 2
      ;;
  esac
done

run_semflow build_paper_complete scripts/build_paper_complete_results.py --root "$ROOT" --clean-root "$CLEAN_ROOT" --include-sfsr
echo "[paper-complete] done: $ROOT"
