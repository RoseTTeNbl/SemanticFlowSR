"""Small locally-trained SR baselines for checkpoint-free external methods.

These adapters are deliberately labelled as ``*-small`` in result files.  They
do not claim to reproduce official pretrained checkpoints.  They provide a
real, train/eval data path for external methods whose repositories are present
but whose paper checkpoints are not available locally.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Callable

import numpy as np
import torch

from ..data.symbolicgpt_subset import load_symbolicgpt_subset_tasks
from ..sr.ast import eval_expr
from ..sr.parser import parse_formula
from .metrics import nmse, r2_score


TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|[()+\-*/]|\d+\.?\d*")


@dataclass(frozen=True)
class SmallBaselineConfig:
    method: str
    train_root: str = "data/generated/symbolicgpt_large_2000_200_200"
    train_splits: tuple[str, ...] = ("train", "val")
    artifact: str = "results/clean_benchmark_20260701/paper_complete_20260702/trained_small_models/small_sr_library.json"
    train_limit: int | None = None
    candidate_limit: int = 384
    prefilter_limit: int = 1600
    eval_subset: int = 48
    ridge: float = 1.0e-8
    seed: int = 0
    force_retrain: bool = False


def make_small_baseline(cfg: SmallBaselineConfig) -> Callable[[Any, Any, Any, Any], dict[str, Any]]:
    artifact = train_or_load_small_library(cfg)

    def _run(x_train, y_train, x_test, y_test) -> dict[str, Any]:
        return run_small_library_task(
            x_train,
            y_train,
            x_test,
            y_test,
            artifact=artifact,
            cfg=cfg,
        )

    return _run


def train_or_load_small_library(cfg: SmallBaselineConfig) -> dict[str, Any]:
    path = Path(cfg.artifact)
    if path.exists() and not bool(cfg.force_retrain):
        return json.loads(path.read_text())
    started = time.perf_counter()
    rng = random.Random(int(cfg.seed))
    tasks = load_symbolicgpt_subset_tasks(
        cfg.train_root,
        splits=tuple(cfg.train_splits),
        limit=cfg.train_limit,
        rng=rng,
    )
    candidates: list[dict[str, Any]] = []
    token_counts: dict[str, int] = {}
    for task in tasks:
        formula = _normalize_formula(str(task.expression))
        if not formula or _bad_formula(formula):
            continue
        used_vars = _max_var_index(formula) + 1
        if used_vars <= 0:
            used_vars = int(task.X_train.shape[1])
        try:
            expr = parse_formula(formula, [f"x{i}" for i in range(max(used_vars, 1))])
        except Exception:
            continue
        tokens = TOKEN_RE.findall(formula)
        for tok in tokens:
            token_counts[tok] = token_counts.get(tok, 0) + 1
        candidates.append({
            "formula": formula,
            "used_vars": int(used_vars),
            "complexity": int(getattr(expr, "complexity", max(len(tokens), 1))),
            "depth": int(getattr(expr, "depth", 1)),
            "tokens": tokens,
            "source_task": str(task.name),
            "train_y_stats": _feature_stats(task.X_train, task.y_train).tolist(),
        })
    if not candidates:
        raise RuntimeError(f"no trainable formulas found in {cfg.train_root} splits={cfg.train_splits}")
    total = max(sum(token_counts.values()), 1)
    token_logprob = {tok: math.log((count + 1.0) / (total + len(token_counts))) for tok, count in token_counts.items()}
    payload = {
        "format": "small_learned_sr_library_v1",
        "trained_small": True,
        "source": str(cfg.train_root),
        "train_splits": list(cfg.train_splits),
        "train_limit": cfg.train_limit,
        "seed": int(cfg.seed),
        "training_runtime_sec": time.perf_counter() - started,
        "candidate_count": len(candidates),
        "token_logprob": token_logprob,
        "candidates": candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def run_small_library_task(
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    artifact: dict[str, Any],
    cfg: SmallBaselineConfig,
) -> dict[str, Any]:
    x_train_arr = np.asarray(x_train, dtype=float)
    y_train_arr = np.asarray(y_train, dtype=float).reshape(-1)
    x_test_arr = np.asarray(x_test, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float).reshape(-1)
    n_vars = int(x_train_arr.shape[1])
    candidates = [
        item for item in artifact.get("candidates", [])
        if int(item.get("used_vars", 999)) <= n_vars
    ]
    if not candidates:
        raise RuntimeError(f"no small-baseline candidates compatible with {n_vars} variables")
    ordered = _order_candidates(candidates, artifact=artifact, cfg=cfg, x_train=x_train_arr, y_train=y_train_arr)
    prefilter = ordered[: max(int(cfg.prefilter_limit), int(cfg.candidate_limit), 1)]
    scored = _score_candidates(
        prefilter,
        x_train_arr,
        y_train_arr,
        n_vars=n_vars,
        eval_subset=max(int(cfg.eval_subset), 1),
        ridge=float(cfg.ridge),
    )
    if not scored:
        raise RuntimeError("no small-baseline candidate could be evaluated")
    shortlist = sorted(scored, key=lambda row: row["quick_score"], reverse=True)[: max(int(cfg.candidate_limit), 1)]
    best: dict[str, Any] | None = None
    for row in shortlist:
        item = row["candidate"]
        try:
            expr = parse_formula(str(item["formula"]), [f"x{i}" for i in range(n_vars)])
            train_raw = _eval_expr(expr, x_train_arr)
            test_raw = _eval_expr(expr, x_test_arr)
        except Exception:
            continue
        train_pred, test_pred, coef = _affine_refit(y_train_arr, train_raw, test_raw, ridge=float(cfg.ridge))
        train_r2 = r2_score(y_train_arr, train_pred)
        key = (float(train_r2), -float(item.get("complexity", 0)), str(item.get("formula", "")))
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "candidate": item,
                "train_pred": train_pred,
                "test_pred": test_pred,
                "test_raw": test_raw,
                "coef": coef,
                "train_r2": train_r2,
            }
    if best is None:
        raise RuntimeError("all shortlisted small-baseline candidates failed final evaluation")
    item = best["candidate"]
    expression = _format_affine_expression(str(item["formula"]), best["coef"])
    return {
        "method": cfg.method,
        "status": "ok",
        "error": "",
        "error_type": "",
        "r2": r2_score(y_test_arr, best["test_pred"]),
        "r2_raw": r2_score(y_test_arr, best["test_raw"]),
        "r2_affine_refit": r2_score(y_test_arr, best["test_pred"]),
        "train_r2_affine_refit": float(best["train_r2"]),
        "nmse": nmse(y_test_arr, best["test_pred"]),
        "nmse_affine_refit": nmse(y_test_arr, best["test_pred"]),
        "expression": expression,
        "raw_expression": str(item["formula"]),
        "complexity": float(item.get("complexity", 0.0)),
        "trained_small": True,
        "small_training_source": str(artifact.get("source", "")),
        "small_training_splits": list(artifact.get("train_splits", [])),
        "small_training_candidate_count": int(artifact.get("candidate_count", len(artifact.get("candidates", [])))),
        "small_baseline_mode": _mode_for_method(cfg.method),
        "candidate_limit": int(cfg.candidate_limit),
        "prefilter_limit": int(cfg.prefilter_limit),
        "affine_scale": float(best["coef"][0]),
        "affine_intercept": float(best["coef"][1]),
    }


def _order_candidates(
    candidates: list[dict[str, Any]],
    *,
    artifact: dict[str, Any],
    cfg: SmallBaselineConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> list[dict[str, Any]]:
    mode = _mode_for_method(cfg.method)
    task_feat = _feature_stats(x_train, y_train)
    token_logprob = artifact.get("token_logprob", {})
    rng = random.Random(int(cfg.seed) + len(candidates) * 17 + int(x_train.shape[1]))

    def token_score(item: dict[str, Any]) -> float:
        toks = list(item.get("tokens", []))
        if not toks:
            return -999.0
        return float(sum(float(token_logprob.get(tok, -12.0)) for tok in toks) / max(len(toks), 1))

    def feat_dist(item: dict[str, Any]) -> float:
        feat = np.asarray(item.get("train_y_stats", []), dtype=float)
        if feat.shape != task_feat.shape:
            return 1e6
        return float(np.linalg.norm(feat - task_feat))

    if mode == "symbolicgpt_token_lm":
        key = lambda item: (token_score(item), -abs(int(item.get("used_vars", 0)) - int(x_train.shape[1])), -float(item.get("complexity", 0.0)))
        return sorted(candidates, key=key, reverse=True)
    if mode == "nesymres_semantic_retrieval":
        return sorted(candidates, key=lambda item: (feat_dist(item), float(item.get("complexity", 0.0))))
    if mode == "hvae_latent_diverse":
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return sorted(shuffled, key=lambda item: (abs(float(item.get("depth", 0.0)) - 3.0), rng.random()))
    if mode == "nggp_seeded_search":
        return sorted(candidates, key=lambda item: (0.5 * feat_dist(item) + 0.05 * float(item.get("complexity", 0.0))))
    return list(candidates)


def _score_candidates(
    candidates: list[dict[str, Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_vars: int,
    eval_subset: int,
    ridge: float,
) -> list[dict[str, Any]]:
    if x_train.shape[0] > int(eval_subset):
        idx = np.linspace(0, x_train.shape[0] - 1, int(eval_subset)).astype(int)
        x_eval = x_train[idx]
        y_eval = y_train[idx]
    else:
        x_eval = x_train
        y_eval = y_train
    out = []
    variables = [f"x{i}" for i in range(n_vars)]
    for item in candidates:
        try:
            expr = parse_formula(str(item["formula"]), variables)
            raw = _eval_expr(expr, x_eval)
            pred, _, _ = _affine_refit(y_eval, raw, raw, ridge=float(ridge))
            quick = r2_score(y_eval, pred)
        except Exception:
            continue
        out.append({"candidate": item, "quick_score": float(quick)})
    return out


def _mode_for_method(method: str) -> str:
    low = str(method).lower()
    if "nesym" in low:
        return "nesymres_semantic_retrieval"
    if "hvae" in low:
        return "hvae_latent_diverse"
    if "nggp" in low:
        return "nggp_seeded_search"
    return "symbolicgpt_token_lm"


def _eval_expr(expr, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        pred = eval_expr(expr, torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    return _finite(pred).reshape(-1)


def _affine_refit(y_train: np.ndarray, train_raw: np.ndarray, test_raw: np.ndarray, *, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_raw = _finite(train_raw).reshape(-1)
    test_raw = _finite(test_raw).reshape(-1)
    design = np.stack([train_raw, np.ones_like(train_raw)], axis=1)
    gram = design.T @ design
    rhs = design.T @ y_train.reshape(-1)
    reg = float(ridge) * np.eye(2)
    reg[-1, -1] = 0.0
    try:
        coef = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(design, y_train.reshape(-1), rcond=None)[0]
    test_design = np.stack([test_raw, np.ones_like(test_raw)], axis=1)
    return design @ coef, test_design @ coef, coef


def _format_affine_expression(formula: str, coef: np.ndarray) -> str:
    scale = _snap(float(coef[0]))
    bias = _snap(float(coef[1]))
    if abs(scale - 1.0) < 1.0e-8 and abs(bias) < 1.0e-8:
        return formula
    if abs(scale) < 1.0e-10:
        return f"{bias:.12g}"
    if abs(bias) < 1.0e-8:
        return f"({scale:.12g})*({formula})"
    return f"({scale:.12g})*({formula}) + ({bias:.12g})"


def _snap(value: float) -> float:
    if abs(value) < 1.0e-10:
        return 0.0
    nearest = round(value)
    if abs(value - nearest) < 1.0e-8:
        return float(nearest)
    if abs(value - 1.0) < 1.0e-8:
        return 1.0
    if abs(value + 1.0) < 1.0e-8:
        return -1.0
    return float(value)


def _feature_stats(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = _finite(np.asarray(y, dtype=float).reshape(-1))
    vals = [
        float(np.mean(y)),
        float(np.std(y)),
        float(np.min(y)),
        float(np.max(y)),
        float(np.mean(np.abs(y))),
    ]
    for col in range(min(int(x.shape[1]), 3)):
        xc = _finite(x[:, col])
        if float(np.std(xc)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(xc, y)[0, 1])
        vals.append(0.0 if not math.isfinite(corr) else corr)
    while len(vals) < 8:
        vals.append(0.0)
    return np.asarray(vals, dtype=float)


def _finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_formula(text: str) -> str:
    text = str(text or "").strip().replace("^", "**")
    text = re.sub(r"\bx_(\d+)\b", r"x\1", text)
    text = re.sub(r"\bvar_(\d+)\b", r"x\1", text)
    return text


def _max_var_index(text: str) -> int:
    found = []
    for match in re.finditer(r"\bx_?(\d+)\b|\bvar_(\d+)\b", str(text)):
        value = match.group(1) or match.group(2)
        if value != "":
            found.append(int(value))
    return max(found) if found else 0


def _bad_formula(text: str) -> bool:
    low = str(text or "").lower()
    return any(tok in low for tok in ("nan", "inf", "zoo", "oo"))
