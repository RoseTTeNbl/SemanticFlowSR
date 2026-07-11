"""Persistent compiled-trace families for SemanticFlowSR bootstrap training."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TRACE_CACHE_SCHEMA = "semantic_flow_compiled_trace_families_v1"
TRACE_COMPILER_VERSION = "register_canonical_ssa_cse_v1"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def task_data_fingerprint(task: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(task.task_id).encode("utf-8"))
    digest.update(str(task.ground_truth).encode("utf-8"))
    for value in (task.x_train, task.y_train, task.x_test, task.y_test):
        array = np.ascontiguousarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_stable_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def graph_signature(template: Any) -> dict[str, Any]:
    return {
        "construction_graph": "register_categorical_blocks",
        "num_vars": int(template.num_vars),
        "num_layers": int(template.num_layers),
        "num_registers": int(template.num_registers),
        "source_count": int(template.source_count),
        "ops": [str(value) for value in template.ops],
        "output_terms": int(template.output_terms),
        "canonical_ssa": True,
        "cse": True,
    }


def graph_signature_hash(template: Any) -> str:
    return _sha256(_stable_json(graph_signature(template)).encode("utf-8"))


@dataclass(frozen=True)
class TraceCachePaths:
    records: Path
    manifest: Path


def cache_paths(root: str | Path) -> TraceCachePaths:
    path = Path(root)
    return TraceCachePaths(
        records=path / "compiled_trace_families_v1.jsonl",
        manifest=path / "compiled_trace_families_v1.manifest.json",
    )


def trace_record(task: Any, template: Any, traces: Iterable[dict[str, Any]], failures: Iterable[str]) -> dict[str, Any]:
    serialized = []
    for trace in traces:
        serialized.append({
            "choices": [int(value) for value in trace["choices"]],
            "active_block_indices": [int(value) for value in trace["active_block_indices"]],
            "expression_string": str(trace["expression_string"]),
            "canonical_expression_key": str(trace["expression_string"]),
            "semantic_oracle_raw_r2": float(trace["semantic_oracle_raw_r2"]),
            "canonical_ssa": bool(trace.get("canonical_ssa", False)),
            "ssa_operation_count": int(trace.get("ssa_operation_count", 0)),
            "cse_reuse_count": int(trace.get("cse_reuse_count", 0)),
        })
    return {
        "schema": TRACE_CACHE_SCHEMA,
        "task_id": str(task.task_id),
        "suite": str(task.suite),
        "split": str(task.split),
        "ground_truth": str(task.ground_truth),
        "data_fingerprint": task_data_fingerprint(task),
        "graph_signature_hash": graph_signature_hash(template),
        "compiler_version": TRACE_COMPILER_VERSION,
        "traces": serialized,
        "compile_failures": [str(value) for value in failures],
    }


def write_trace_cache(root: str | Path, template: Any, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    paths = cache_paths(root)
    paths.records.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(list(records), key=lambda row: (str(row["split"]), str(row["task_id"])))
    payload = "".join(_stable_json(row) + "\n" for row in ordered)
    temporary = paths.records.with_suffix(paths.records.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(paths.records)
    manifest = {
        "schema": TRACE_CACHE_SCHEMA,
        "compiler_version": TRACE_COMPILER_VERSION,
        "graph_signature": graph_signature(template),
        "graph_signature_hash": graph_signature_hash(template),
        "record_count": len(ordered),
        "records_sha256": _sha256(payload.encode("utf-8")),
        "task_ids": [str(row["task_id"]) for row in ordered],
    }
    manifest_tmp = paths.manifest.with_suffix(paths.manifest.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_tmp.replace(paths.manifest)
    return manifest


def load_trace_cache(root: str | Path, template: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    paths = cache_paths(root)
    if not paths.records.is_file() or not paths.manifest.is_file():
        raise FileNotFoundError(f"compiled trace cache is incomplete under {Path(root)}")
    manifest = json.loads(paths.manifest.read_text())
    if str(manifest.get("schema")) != TRACE_CACHE_SCHEMA:
        raise ValueError("compiled trace cache schema mismatch")
    if str(manifest.get("compiler_version")) != TRACE_COMPILER_VERSION:
        raise ValueError("compiled trace cache compiler version mismatch")
    expected_graph = graph_signature_hash(template)
    if str(manifest.get("graph_signature_hash")) != expected_graph:
        raise ValueError("compiled trace cache graph signature mismatch")
    payload = paths.records.read_text()
    if str(manifest.get("records_sha256")) != _sha256(payload.encode("utf-8")):
        raise ValueError("compiled trace cache checksum mismatch")
    records: dict[str, dict[str, Any]] = {}
    for line in payload.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("schema")) != TRACE_CACHE_SCHEMA:
            raise ValueError("compiled trace record schema mismatch")
        task_id = str(row["task_id"])
        if task_id in records:
            raise ValueError(f"duplicate compiled trace cache task: {task_id}")
        traces = list(row.get("traces", []))
        if not traces:
            raise ValueError(f"compiled trace cache task has no valid trace: {task_id}")
        for trace in traces:
            if float(trace.get("semantic_oracle_raw_r2", -np.inf)) < 0.999999:
                raise ValueError(f"compiled trace cache contains a failed semantic oracle: {task_id}")
        records[task_id] = row
    if int(manifest.get("record_count", -1)) != len(records):
        raise ValueError("compiled trace cache record count mismatch")
    return records, manifest


def validate_task_record(task: Any, template: Any, record: dict[str, Any]) -> None:
    if str(record.get("ground_truth")) != str(task.ground_truth):
        raise ValueError(f"compiled trace cache GT mismatch for {task.task_id}")
    if str(record.get("data_fingerprint")) != task_data_fingerprint(task):
        raise ValueError(f"compiled trace cache data mismatch for {task.task_id}")
    if str(record.get("graph_signature_hash")) != graph_signature_hash(template):
        raise ValueError(f"compiled trace cache graph mismatch for {task.task_id}")
