from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_external_baseline_results import build_lock, validate
from scripts.check_external_baseline_preflight import validate_preflight


def _write_result(root: Path, *, name: str = "formula_dev/fake_seed0.json") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "suite/task": {
            "task_id": "suite/task",
            "suite": "suite",
            "method": "fake",
            "status": "ok",
            "r2": 1.0,
            "nmse": 0.0,
            "expression": "x0",
        }
    }, indent=2))
    return path


def test_result_lock_validates_and_detects_tampering(tmp_path):
    root = tmp_path / "external_baselines"
    path = _write_result(root)
    lock = build_lock(root)
    lock_path = root / "baseline_results.lock.json"
    lock_path.write_text(json.dumps(lock))

    summary, errors = validate(root, lock_path)
    assert not errors
    assert summary["files"] == 1
    assert summary["records"] == 1

    path.write_text(path.read_text() + "\n")
    _, errors = validate(root, lock_path)
    assert any("mismatch" in error for error in errors)


def test_result_lock_detects_missing_and_unexpected_files(tmp_path):
    root = tmp_path / "external_baselines"
    path = _write_result(root)
    lock = build_lock(root)
    lock_path = root / "baseline_results.lock.json"
    lock_path.write_text(json.dumps(lock))
    path.unlink()
    _write_result(root, name="symbolicgpt_large/other.json")

    _, errors = validate(root, lock_path)
    assert any("missing result" in error for error in errors)
    assert any("unexpected result" in error for error in errors)


def test_canonical_external_baseline_snapshot_is_complete():
    root = Path("results/clean_benchmark/external_baselines")
    lock_path = root / "baseline_results.lock.json"
    summary, errors = validate(root, lock_path)
    assert not errors
    assert summary["files"] == 22
    assert summary["records"] == 11 * (34 + 178)


def test_lock_hash_is_sha256(tmp_path):
    root = tmp_path / "external_baselines"
    path = _write_result(root)
    lock = build_lock(root)
    entry = next(iter(lock["files"].values()))
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_local_external_baseline_preflight_is_satisfied():
    summary, errors = validate_preflight(Path("configs/eval/external_baselines.local.lock.json"))
    assert not errors
    assert summary["valid"] is True
