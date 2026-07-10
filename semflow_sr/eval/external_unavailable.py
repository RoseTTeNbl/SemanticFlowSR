"""Helpers for external baselines that cannot run without official artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .baseline_runner import collect_tasks


def write_unavailable_baseline(
    *,
    manifest: str,
    suites: list[str] | None,
    root: str,
    out: str,
    tag: str,
    seed: int,
    max_tasks: int | None,
    method: str,
    repo_root: str,
    required: list[str],
    reason: str,
    no_resume: bool = False,
) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_seed{seed}.json"
    if out_path.exists() and not no_resume:
        results: dict[str, dict[str, Any]] = json.loads(out_path.read_text())
    else:
        results = {}
    repo = Path(repo_root)
    missing = [item for item in required if not (repo / item).exists()]
    status_reason = reason if missing else "adapter_not_implemented_for_official_inference_entrypoint"
    tasks = collect_tasks(
        manifest=manifest,
        suites=suites,
        root=root,
        seed=int(seed),
        limit=max_tasks,
    )
    for task in tasks:
        if task.name in results and results[task.name].get("status") == "ok" and not no_resume:
            continue
        started = time.perf_counter()
        row = {
            "task_id": task.name,
            "suite": task.metadata.get("suite", task.name.split("/", 1)[0] if "/" in task.name else "unknown"),
            "method": method,
            "status": "failed",
            "error": status_reason,
            "error_type": "ExternalArtifactUnavailable" if missing else "ExternalAdapterUnavailable",
            "missing_artifacts": missing,
            "repo_root": str(repo),
            "r2": 0.0,
            "nmse": None,
            "expression": "",
            "ground_truth": task.expression,
            "runtime_sec": time.perf_counter() - started,
            "n_train": int(task.X_train.shape[0]),
            "n_test": int(task.X_test.shape[0]),
            "n_vars": int(task.X_train.shape[1]),
            "budget": {
                "official_repo_root": str(repo),
                "required_artifacts": list(required),
                "adapter_policy": "structured_failure_no_proxy_results",
            },
        }
        results[task.name] = row
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"{task.name:32s} status=failed reason={row['error_type']}")
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"saved {out_path} ({len(results)} tasks)")
    return out_path
