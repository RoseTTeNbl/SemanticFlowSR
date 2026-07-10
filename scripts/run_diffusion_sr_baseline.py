#!/usr/bin/env python
"""Run a local diffusion-repo proposal baseline on materialized SR tasks.

The external diffusion repository mainly ships notebooks/checkpoints and a
generated formula dataset, not a manifest-aware benchmark runner. This adapter
uses its generated formulas as a diffusion proposal library: each benchmark task
selects the proposal with the best affine-refit train R2, then reports the
held-out test metrics in the same JSON format as other external baselines.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semflow_sr.eval.baseline_runner import collect_tasks
from semflow_sr.eval.metrics import nmse, r2_score
from semflow_sr.sr.ast import eval_expr
from semflow_sr.sr.parser import parse_formula


@dataclass(frozen=True)
class Proposal:
    formula: str
    source: str
    used_vars: int
    skeleton: str = ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    ap.add_argument("--suite", nargs="+", default=None)
    ap.add_argument("--root", default="data/benchmark_suites")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tasks", "--max-tasks", type=int, default=None)
    ap.add_argument("--diffusion_root", default="external/Symbolic_Regression_With_Diffusion_Models")
    ap.add_argument("--proposal_limit", type=int, default=2000)
    ap.add_argument("--candidate_limit", type=int, default=256)
    ap.add_argument("--ridge", type=float, default=1.0e-8)
    ap.add_argument("--per_task_timeout_sec", type=float, default=180.0)
    ap.add_argument("--out", default="results/external_baselines")
    ap.add_argument("--tag", default="local_diffusion_proposal")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_seed{args.seed}.json"
    if out_path.exists() and not args.no_resume:
        results: dict[str, dict[str, Any]] = json.loads(out_path.read_text())
    else:
        results = {}

    proposals = _load_external_diffusion_proposals(
        Path(args.diffusion_root),
        limit=None if int(args.proposal_limit) <= 0 else int(args.proposal_limit),
    )
    if not proposals:
        raise RuntimeError(f"no diffusion proposals found under {args.diffusion_root}")
    tasks = collect_tasks(
        manifest=args.manifest,
        suites=args.suite,
        root=args.root,
        seed=int(args.seed),
        limit=args.max_tasks,
    )
    for task in tasks:
        if task.name in results and results[task.name].get("status") == "ok" and not args.no_resume:
            continue
        started = time.perf_counter()
        try:
            row = _with_timeout(
                lambda: _run_task(
                    task,
                    proposals,
                    candidate_limit=int(args.candidate_limit),
                    ridge=float(args.ridge),
                ),
                timeout_sec=float(args.per_task_timeout_sec),
            )
            row["runtime_sec"] = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 - archive task-level failures.
            row = {
                "task_id": task.name,
                "suite": task.metadata.get("suite", _infer_suite(task.name)),
                "method": "LocalDiffusionProposal",
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "r2": 0.0,
                "nmse": None,
                "expression": "",
                "ground_truth": task.expression,
                "runtime_sec": time.perf_counter() - started,
            }
        row.update({
            "task_id": task.name,
            "suite": task.metadata.get("suite", _infer_suite(task.name)),
            "domain": task.metadata.get("domain", "unknown"),
            "split": task.metadata.get("split", ""),
            "ground_truth": task.expression,
            "n_train": int(task.X_train.shape[0]),
            "n_test": int(task.X_test.shape[0]),
            "n_vars": int(task.X_train.shape[1]),
            "budget": {
                "proposal_limit": int(args.proposal_limit),
                "candidate_limit": int(args.candidate_limit),
                "ridge": float(args.ridge),
                "proposal_source": str(Path(args.diffusion_root) / "Approach1" / "Data"),
            },
        })
        results[task.name] = row
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"{task.name:32s} status={row['status']} r2={row.get('r2')}")
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"saved {out_path} ({len(results)} tasks)")


def _load_external_diffusion_proposals(root: Path, *, limit: int | None) -> list[Proposal]:
    candidates: list[Proposal] = []
    data_dir = root / "Approach1" / "Data"
    for path in [
        data_dir / "combined_dataset_5_variables_dynamic_seed940.json",
        data_dir / "preprocessed_data_with_embeddings.json",
    ]:
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                formula = str(
                    row.get("function")
                    or row.get("formula")
                    or row.get("formula_human_readable")
                    or ""
                ).strip()
                if not formula:
                    continue
                candidates.append(Proposal(
                    formula=_normalize_formula_text(formula),
                    source=path.name,
                    used_vars=_max_var_index(formula) + 1,
                    skeleton=str(row.get("skeleton", "")),
                ))
                if limit is not None and len(candidates) >= int(limit):
                    return _dedupe(candidates)
    return _dedupe(candidates)


def _run_task(task, proposals: list[Proposal], *, candidate_limit: int, ridge: float) -> dict[str, Any]:
    n_vars = int(task.X_train.shape[1])
    selected = [p for p in proposals if p.used_vars <= n_vars]
    selected = selected[: max(int(candidate_limit), 1)]
    if not selected:
        raise RuntimeError(f"no proposals compatible with {n_vars} variables")
    x_train = torch.tensor(task.X_train, dtype=torch.float32)
    x_test = torch.tensor(task.X_test, dtype=torch.float32)
    y_train = np.asarray(task.y_train, dtype=float).reshape(-1)
    y_test = np.asarray(task.y_test, dtype=float).reshape(-1)
    variables = [f"x{i}" for i in range(n_vars)]

    best: dict[str, Any] | None = None
    for proposal in selected:
        try:
            expr = parse_formula(proposal.formula, variables)
            train_raw = _finite(eval_expr(expr, x_train).detach().cpu().numpy())
            test_raw = _finite(eval_expr(expr, x_test).detach().cpu().numpy())
        except Exception:
            continue
        train_pred, test_pred, coef = _affine_refit(y_train, train_raw, test_raw, ridge=float(ridge))
        train_r2 = r2_score(y_train, train_pred)
        score = (train_r2, -float(expr.complexity), proposal.formula)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "proposal": proposal,
                "expr": expr,
                "train_pred": train_pred,
                "test_pred": test_pred,
                "train_raw": train_raw,
                "test_raw": test_raw,
                "coef": coef,
            }
    if best is None:
        raise RuntimeError("no proposal could be parsed/evaluated")

    proposal = best["proposal"]
    test_pred = best["test_pred"]
    train_pred = best["train_pred"]
    raw_test = best["test_raw"]
    coef = best["coef"]
    expression = _format_affine_expression(proposal.formula, coef)
    return {
        "method": "LocalDiffusionProposal",
        "status": "ok",
        "error": "",
        "error_type": "",
        "r2": r2_score(y_test, test_pred),
        "r2_raw": r2_score(y_test, raw_test),
        "r2_affine_refit": r2_score(y_test, test_pred),
        "train_r2_affine_refit": r2_score(y_train, train_pred),
        "nmse": nmse(y_test, test_pred),
        "nmse_affine_refit": nmse(y_test, test_pred),
        "expression": expression,
        "raw_expression": proposal.formula,
        "proposal_source": proposal.source,
        "proposal_skeleton": proposal.skeleton,
        "proposal_candidates_considered": int(len(selected)),
        "affine_scale": float(coef[0]),
        "affine_intercept": float(coef[1]),
        "complexity": float(best["expr"].complexity),
    }


def _affine_refit(y_train: np.ndarray, train_raw: np.ndarray, test_raw: np.ndarray, *, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_raw = _finite(train_raw).reshape(-1)
    test_raw = _finite(test_raw).reshape(-1)
    design = np.stack([train_raw, np.ones_like(train_raw)], axis=1)
    gram = design.T @ design
    rhs = design.T @ y_train
    reg = float(ridge) * np.eye(2)
    reg[-1, -1] = 0.0
    try:
        coef = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(design, y_train, rcond=None)[0]
    return design @ coef, np.stack([test_raw, np.ones_like(test_raw)], axis=1) @ coef, coef


def _finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _format_affine_expression(formula: str, coef: np.ndarray) -> str:
    scale = float(coef[0])
    bias = float(coef[1])
    if abs(scale - 1.0) < 1.0e-8 and abs(bias) < 1.0e-8:
        return formula
    if abs(bias) < 1.0e-8:
        return f"({scale:.12g})*({formula})"
    return f"({scale:.12g})*({formula}) + ({bias:.12g})"


def _normalize_formula_text(text: str) -> str:
    text = str(text).replace("^", "**")
    text = re.sub(r"\bx_(\d+)\b", r"x\1", text)
    text = re.sub(r"\bvar_(\d+)\b", r"x\1", text)
    return text


def _max_var_index(text: str) -> int:
    found = [int(item) for item in re.findall(r"\bx_?(\d+)\b|\bvar_(\d+)\b", str(text)) for item in item if item != ""]
    return max(found) if found else 0


def _dedupe(proposals: list[Proposal]) -> list[Proposal]:
    out = []
    seen = set()
    for proposal in proposals:
        key = proposal.formula
        if key in seen:
            continue
        seen.add(key)
        out.append(proposal)
    return out


def _infer_suite(task_id: str) -> str:
    return task_id.split("/", 1)[0] if "/" in task_id else "unknown"


class _TaskTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame) -> None:
    raise _TaskTimeout("per-task diffusion proposal timeout")


def _with_timeout(fn, *, timeout_sec: float):
    if float(timeout_sec) <= 0:
        return fn()
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    main()
