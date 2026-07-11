# External baseline result snapshot

This directory is the canonical paper-comparison snapshot for local
reproduction.  It contains 22 raw JSON files: 11 methods evaluated on the
34-task formula development suite and the 178-task SymbolicGPT-large suite.

The JSON files are intentionally preserved byte-for-byte.  Runtime paths,
checkpoint names, environment fields, error messages, and tracebacks remain in
the records because they document the local execution and allow the same jobs
to resume.  Failed task records are part of the benchmark result rather than
missing data.

Run a local integrity check with:

```bash
conda run -n semflow python scripts/check_external_baseline_preflight.py
conda run -n semflow python scripts/check_external_baseline_results.py
```

Run one task per configured method before a full reproduction:

```bash
MAX_TASKS=1 bash scripts/run_paper_complete_benchmarks.sh
```

The full driver resumes successful records by default.  External repositories
under `external/`, conda environments, datasets, and model weights are local
dependencies and are not committed with this result snapshot.
