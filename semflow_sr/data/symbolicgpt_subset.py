"""SymbolicGPT-style synthetic subset generation and loading.

The external diffusion README describes a SymbolicGPT subset as JSON formulas
with sampled points, a formula string, and properties. The original dataset is
not bundled with this repository, so this module provides a compatible local
format and a generator for training-scale synthetic subsets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..sr.ast import eval_expr
from ..sr.printer import to_string
from .benchmark_loader import SRTask
from .synthetic_generator import GenConfig, generate_expression, sample_probe_xy


@dataclass(frozen=True)
class SymbolicGPTSubsetConfig:
    root: str | Path
    train_count: int = 747
    val_count: int = 160
    test_count: int = 161
    num_vars: int = 3
    num_points: int = 100
    max_depth: int = 4
    seed: int = 0
    x_range: tuple[float, float] = (-5.0, 5.0)
    ops: tuple[str, ...] = (
        "add",
        "sub",
        "mul",
        "protected_div",
        "sin",
        "cos",
        "square",
        "cube",
        "exp",
        "protected_log",
        "protected_sqrt",
    )
    train_fraction: float = 0.8


def generate_symbolicgpt_subset(cfg: SymbolicGPTSubsetConfig) -> dict:
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(cfg.seed))
    manifest = {
        "format": "symbolicgpt_subset",
        "properties": asdict(cfg) | {"root": str(root)},
        "splits": {
            "train": int(cfg.train_count),
            "val": int(cfg.val_count),
            "test": int(cfg.test_count),
        },
        "files": {},
    }
    for split, count in manifest["splits"].items():
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for idx in range(int(count)):
            item = _generate_item(cfg, rng, split=split, index=idx)
            path = split_dir / f"task_{idx:06d}.json"
            path.write_text(json.dumps(item, indent=2))
            files.append(str(path.relative_to(root)))
        manifest["files"][split] = files
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_symbolicgpt_subset_tasks(
    root: str | Path,
    *,
    splits: list[str] | tuple[str, ...] = ("train",),
    limit: int | None = None,
    rng: random.Random | None = None,
    train_fraction: float = 0.8,
) -> list[SRTask]:
    root = Path(root)
    rng = rng or random.Random(0)
    paths: list[Path] = []
    for split in splits:
        paths.extend(sorted((root / str(split)).glob("*.json")))
    if limit is not None:
        paths = paths[: max(int(limit), 0)]
    tasks = []
    for path in paths:
        task = _load_task(path, root=root, rng=rng, train_fraction=train_fraction)
        if _is_valid_formula(task.expression):
            tasks.append(task)
    return tasks


def _generate_item(cfg: SymbolicGPTSubsetConfig, rng: random.Random, *, split: str, index: int) -> dict:
    gen = GenConfig(
        num_vars=int(cfg.num_vars),
        max_depth=int(cfg.max_depth),
        K=max(int(cfg.num_vars) + 2, 8),
        probe_size=int(cfg.num_points),
        x_range=tuple(cfg.x_range),
        ops=tuple(cfg.ops),
    )
    for _ in range(100):
        expr = generate_expression(gen, rng)
        x, y = sample_probe_xy(expr, gen, rng)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(torch.isfinite(y).all()) and float(y.std(unbiased=False).item()) > 1e-6 and float(y.abs().max().item()) < 1e6:
            formula = to_string(expr, int(cfg.num_vars), simplify=True)
            return {
                "formula": formula,
                "points": [
                    {"x": [float(v) for v in x[row].tolist()], "y": float(y[row].item())}
                    for row in range(int(x.shape[0]))
                ],
                "properties": {
                    "split": str(split),
                    "index": int(index),
                    "num_vars": int(cfg.num_vars),
                    "num_points": int(cfg.num_points),
                    "max_depth": int(cfg.max_depth),
                    "complexity": int(expr.complexity),
                    "depth": int(expr.depth),
                    "seed": int(cfg.seed),
                },
            }
    raise RuntimeError("failed to generate a finite SymbolicGPT-style formula")


def _load_task(path: Path, *, root: Path, rng: random.Random, train_fraction: float) -> SRTask:
    raw = json.loads(path.read_text())
    points = list(raw.get("points", []))
    if not points:
        raise ValueError(f"SymbolicGPT subset file has no points: {path}")
    x = np.asarray([item["x"] for item in points], dtype=float)
    y = np.asarray([item["y"] for item in points], dtype=float)
    n = int(x.shape[0])
    order = list(range(n))
    rng.shuffle(order)
    cut = min(max(int(round(n * float(train_fraction))), 1), n - 1)
    tr = order[:cut]
    te = order[cut:]
    props = dict(raw.get("properties", {}))
    num_vars = int(props.get("num_vars", x.shape[1]))
    variable_names = [f"x{i}" for i in range(num_vars)]
    split = path.parent.name
    rel = path.relative_to(root)
    suite = "symbolicgpt_large" if "symbolicgpt_large" in str(root.name) else "symbolicgpt_subset"
    return SRTask(
        f"{suite}/{split}/{path.stem}",
        x[tr],
        y[tr],
        x[te],
        y[te],
        str(raw.get("formula", "")),
        variable_names,
        {
            "suite": suite,
            "split": split,
            "source": "local_symbolicgpt_subset",
            "path": str(rel),
            **props,
        },
    )


def _is_valid_formula(formula: str | None) -> bool:
    text = str(formula or "")
    bad_tokens = ("zoo", "nan", "inf", "oo")
    return not any(token in text for token in bad_tokens)
