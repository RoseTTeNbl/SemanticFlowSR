#!/usr/bin/env python3
"""Validate or snapshot the immutable external-baseline result bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/clean_benchmark/external_baselines")
DEFAULT_LOCK = DEFAULT_ROOT / "baseline_results.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not all(isinstance(key, str) and isinstance(value, dict) for key, value in payload.items()):
        raise ValueError("result must be a JSON object mapping task_id to records")
    return payload


def _label(records: dict[str, dict[str, Any]], key: str) -> str:
    values = {str(row.get(key, "")) for row in records.values() if str(row.get(key, ""))}
    return sorted(values)[0] if len(values) == 1 else "mixed"


def build_lock(root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(root.glob("*/*.json")):
        if path.name == "baseline_results.lock.json":
            continue
        records = _load_records(path)
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "records": len(records),
            "suite": _label(records, "suite"),
            "method": _label(records, "method"),
        }
    return {
        "schema_version": 1,
        "created_at": "2026-07-11",
        "raw_files_immutable": True,
        "source_revision": None,
        "expected_files": len(files),
        "files": files,
    }


def validate(root: Path, lock_path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        lock = json.loads(lock_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {}, [f"cannot read lock {lock_path}: {exc}"]
    expected = lock.get("files", {})
    if not isinstance(expected, dict):
        return {}, ["lock files field must be an object"]
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.glob("*/*.json"))
        if path.name != lock_path.name
    }
    missing = sorted(set(expected) - set(actual_paths))
    unexpected = sorted(set(actual_paths) - set(expected))
    errors.extend(f"missing result: {name}" for name in missing)
    errors.extend(f"unexpected result: {name}" for name in unexpected)
    total_records = 0
    status_counts: dict[str, int] = {}
    for relative in sorted(set(expected) & set(actual_paths)):
        path = actual_paths[relative]
        spec = expected[relative]
        if int(path.stat().st_size) != int(spec.get("bytes", -1)):
            errors.append(f"size mismatch: {relative}")
        if _sha256(path) != str(spec.get("sha256", "")):
            errors.append(f"sha256 mismatch: {relative}")
        try:
            records = _load_records(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid JSON schema {relative}: {exc}")
            continue
        total_records += len(records)
        if len(records) != int(spec.get("records", -1)):
            errors.append(f"record count mismatch: {relative}")
        for task_id, row in records.items():
            if str(row.get("task_id", task_id)) != task_id:
                errors.append(f"task_id mismatch: {relative}:{task_id}")
            status = str(row.get("status", "ok"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status not in {"ok", "failed", "error"}:
                errors.append(f"invalid status {status}: {relative}:{task_id}")
            if status == "ok":
                for metric in ("r2", "nmse"):
                    value = row.get(metric)
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        errors.append(f"non-finite {metric}: {relative}:{task_id}")
                if "expression" not in row:
                    errors.append(f"missing expression: {relative}:{task_id}")
    summary = {
        "root": str(root),
        "files": len(actual_paths),
        "records": total_records,
        "status_counts": status_counts,
        "valid": not errors,
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--write-lock", action="store_true", help="write a new immutable snapshot lock")
    args = parser.parse_args()
    if args.write_lock:
        lock = build_lock(args.root)
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    summary, errors = validate(args.root, args.lock)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
