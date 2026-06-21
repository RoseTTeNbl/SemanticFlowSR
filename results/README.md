# Results Layout

The active algorithm is Edge-Parameterized Semantic Flow Matching.

Tracked result families:

```text
edge_flow_smoke/
edge_flow_87_basic/
edge_flow_87_h4_l3_k8/
```

`edge_flow_smoke/` contains only small synthetic wiring checks. A full 87-task
result directory must contain:

```text
*_summary.json                    aggregate metrics
*_samples.jsonl                    one JSON record per task
*_task_expressions.csv             GT expression vs generated expression
*_task_expressions.md              readable GT/generated expression table
*_statistics_by_group.csv          all/suite/num_vars/jin-vs-87 statistics
*_statistics_by_group.json         same grouped statistics in JSON
*_diagnostics.json                 decoder/prior/projection/template diagnostics
```

Full or dated benchmark runs should use:

```text
edge_flow_87_<tag>/
edge_flow_<YYYYMMDD>/
```

Those full outputs are ignored by default. Promote only compact summaries or
report tables when a run becomes part of the current comparison.

All commands for official runs should keep CPU bounded:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 ...
```
