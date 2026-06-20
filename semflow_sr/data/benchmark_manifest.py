"""Benchmark manifest schema for materialized symbolic-regression tasks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


DEFAULT_METRICS = ["r2", "nmse", "complexity"]


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    task_id: str
    suite: str
    num_vars: int
    variable_names: list[str]
    train_path: str
    test_path: str
    val_path: str | None = None
    target_column: str = "target"
    ground_truth: str | None = None
    domain: str = "unknown"
    has_dummy_vars: bool = False
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    split: str = "main"
    tags: list[str] = field(default_factory=list)
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkTaskSpec":
        data = dict(raw)
        data.setdefault("metrics", list(DEFAULT_METRICS))
        data.setdefault("tags", [])
        data.setdefault("metadata", {})
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSuiteSpec:
    version: str
    suites: dict[str, list[BenchmarkTaskSpec]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkSuiteSpec":
        suites = {
            str(name): [BenchmarkTaskSpec.from_dict(item) for item in items]
            for name, items in raw.get("suites", {}).items()
        }
        return cls(version=str(raw.get("version", "1.0")), suites=suites, metadata=dict(raw.get("metadata", {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metadata": self.metadata,
            "suites": {
                name: [task.to_dict() for task in tasks]
                for name, tasks in self.suites.items()
            },
        }


def write_benchmark_manifest(manifest: BenchmarkSuiteSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


def load_benchmark_manifest(path: str | Path) -> BenchmarkSuiteSpec:
    return BenchmarkSuiteSpec.from_dict(json.loads(Path(path).read_text()))


def build_benchmark_index(manifest: BenchmarkSuiteSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite, tasks in manifest.suites.items():
        if not tasks:
            rows.append({
                "suite": suite,
                "n_tasks": 0,
                "split": "",
                "domains": [],
                "has_dummy_vars": False,
                "metrics": [],
                "num_vars_min": 0,
                "num_vars_max": 0,
            })
            continue
        metrics = list(dict.fromkeys(m for task in tasks for m in task.metrics))
        domains = sorted({task.domain for task in tasks})
        splits = sorted({task.split for task in tasks})
        rows.append({
            "suite": suite,
            "n_tasks": len(tasks),
            "split": splits[0] if len(splits) == 1 else splits,
            "domains": domains,
            "has_dummy_vars": any(task.has_dummy_vars for task in tasks),
            "metrics": metrics,
            "num_vars_min": min(task.num_vars for task in tasks),
            "num_vars_max": max(task.num_vars for task in tasks),
        })
    return rows
