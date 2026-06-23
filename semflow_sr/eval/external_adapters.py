"""Shared helpers for external baseline adapters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .metrics import nmse, r2_score


@dataclass(frozen=True)
class LocalDiffusionRunStatus:
    root: str
    approach1_metrics_available: bool
    approach2_direct_runnable: bool
    approach2_missing: list[str]
    approach3_direct_runnable: bool
    approach3_missing: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def normalize_tpsr_result(
    *,
    task_id: str,
    suite: str,
    expression: str,
    ground_truth: str,
    y_train: Iterable[float],
    train_pred: Iterable[float],
    y_test: Iterable[float],
    test_pred: Iterable[float],
    runtime_sec: float,
    mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y_train_arr = np.asarray(list(y_train), dtype=float).reshape(-1)
    train_pred_arr = np.asarray(list(train_pred), dtype=float).reshape(-1)
    y_test_arr = np.asarray(list(y_test), dtype=float).reshape(-1)
    test_pred_arr = np.asarray(list(test_pred), dtype=float).reshape(-1)
    test_pred_arr = np.nan_to_num(test_pred_arr, nan=0.0, posinf=0.0, neginf=0.0)
    train_pred_arr = np.nan_to_num(train_pred_arr, nan=0.0, posinf=0.0, neginf=0.0)
    row = {
        "task_id": task_id,
        "suite": suite,
        "method": "TPSR",
        "status": "ok",
        "error": "",
        "error_type": "",
        "r2": r2_score(y_test_arr, test_pred_arr),
        "nmse": nmse(y_test_arr, test_pred_arr),
        "expression": expression,
        "ground_truth": ground_truth,
        "runtime_sec": float(runtime_sec),
        "tpsr_mode": mode,
        "n_train": int(y_train_arr.shape[0]),
        "n_test": int(y_test_arr.shape[0]),
    }
    if extra:
        row.update(extra)
    return row


def build_local_diffusion_reference_records(root: str | Path) -> tuple[list[dict[str, Any]], LocalDiffusionRunStatus]:
    root = Path(root)
    data_dir = root / "Approach1" / "Data"
    diffusion_metrics_path = data_dir / "diffusion_performance_metrics_DICT.json"
    embeddings_metrics_path = data_dir / "embeddings_performance_metrics_DICT.json"
    diffusion_metrics = _read_json_if_exists(diffusion_metrics_path)
    embeddings_metrics = _read_json_if_exists(embeddings_metrics_path)
    approach2_required = [
        root / "Approach2" / "approach2.py",
        root / "Approach2" / "data_symbolic_regression" / "train",
        root / "Approach2" / "data_symbolic_regression" / "val",
        root / "Approach2" / "data_symbolic_regression" / "test",
        root / "Approach2" / "diffusion_model_final.pth",
    ]
    approach3_required = [
        root / "Approach3" / "approach3.ipynb",
        root / "Approach3" / "data_symbolic_regression",
    ]
    status = LocalDiffusionRunStatus(
        root=str(root),
        approach1_metrics_available=bool(diffusion_metrics),
        approach2_direct_runnable=all(path.exists() for path in approach2_required),
        approach2_missing=[str(path.relative_to(root)) for path in approach2_required if not path.exists()],
        approach3_direct_runnable=all(path.exists() for path in approach3_required),
        approach3_missing=[str(path.relative_to(root)) for path in approach3_required if not path.exists()],
    )
    approach1_loss = _last(diffusion_metrics.get("val_loss_list")) or _last(diffusion_metrics.get("train_loss_list"))
    records = [
        {
            "task_id": "local_diffusion/approach1_embedding_diffusion",
            "suite": "local_diffusion_native",
            "bleu": 0.023,
            "token_similarity": 0.030,
            "edit_distance": 12.42,
            "native_loss": approach1_loss,
            "embedding_native_loss": _last(embeddings_metrics.get("train_loss_list")),
            "status": "ok",
            "direct_run_status": "artifact_metrics_available" if diffusion_metrics else "missing_metrics_artifact",
        },
        {
            "task_id": "local_diffusion/approach2_self_attention",
            "suite": "local_diffusion_native",
            "bleu": 0.011,
            "edit_distance": 19.09,
            "native_loss": 0.21,
            "status": "ok",
            "direct_run_status": "direct_assets_available" if status.approach2_direct_runnable else "missing_direct_assets",
            "missing_assets": status.approach2_missing,
        },
        {
            "task_id": "local_diffusion/approach3_text_diffusion",
            "suite": "local_diffusion_native",
            "bleu": 0.25,
            "token_similarity": 0.61,
            "edit_distance": 7.22,
            "native_loss": 0.0091,
            "status": "ok",
            "direct_run_status": "direct_assets_available" if status.approach3_direct_runnable else "missing_direct_assets",
            "missing_assets": status.approach3_missing,
        },
    ]
    return records, status


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _last(values: Any) -> float | None:
    if not values:
        return None
    return float(values[-1])
