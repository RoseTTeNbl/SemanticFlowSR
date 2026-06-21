# Edge-Parameterized Semantic Flow Plan

Status: current plan, replaces the action-level Semantic-Fisher Flow Matching
plan.

Goal:

```text
Move the main probability object from local action simplexes to a
low-dimensional edge distribution over complete expression DAGs.
```

---

## Phase 0: Smoke Mainline

Implemented:

- `RegisterOperatorTemplate`
- `EdgeDistribution`
- `CircuitSampler`
- complete-expression reward with affine calibration
- top-k elite projection to `Theta*`
- Fisher square-root teacher path
- lightweight `EdgeFlowModel`
- smoke train/eval CLIs

Acceptance:

```text
tests/test_edge_flow_core.py and tests/test_edge_flow_training.py pass;
edge_flow_smoke train/eval commands run under CPU cap.
```

---

## Phase 1: Benchmark Integration

Tasks:

- Add benchmark-loader path to `scripts/run_edge_flow.py`.
- Emit per-task expression and statistics reports for Edge Flow runs.
- Add validation split candidate selection.
- Preserve smoke outputs as tracked wiring checks only.

Acceptance:

```text
Run a small manifest slice without changing the core algorithm.
```

---

## Phase 2: Mixture Modes

Tasks:

- Add `H=4` config.
- Track per-mode elite counts, per-mode best reward, and mode entropy.
- Keep hard mode projection first.
- Add soft responsibility projection only after hard projection is stable.

Acceptance:

```text
Mode diagnostics show whether mode collapse occurs.
```

---

## Phase 3: Decoding

Tasks:

- Add sampling decode budget sweep.
- Add beam decoding over high-probability edge choices.
- Compare reward-only selection with `reward + eta log q_Theta`.

Acceptance:

```text
Decoding bottleneck is separated from flow-training bottleneck.
```

---

## Phase 4: Full Experiments

Tasks:

- Define dated result directory.
- Run small 87-task slice.
- Only then run full 87-task training/evaluation.
- Rerun external baselines into the new result matrix if needed.

Acceptance:

```text
New Edge Flow results are not mixed with old action-level SFFM/TSSF results.
```
