#!/usr/bin/env python
"""Run TPSR/E2E on SemanticFlowSR materialized benchmark tasks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semflow_sr.eval.baseline_runner import collect_tasks
from semflow_sr.eval.external_adapters import normalize_tpsr_result


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    ap.add_argument("--suite", nargs="+", default=None)
    ap.add_argument("--root", default="data/benchmark_suites")
    ap.add_argument("--out", default="results/external_baselines")
    ap.add_argument("--tag", default="tpsr_formula_dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tasks", "--max-tasks", type=int, default=None)
    ap.add_argument("--tpsr_root", default="external/TPSR")
    ap.add_argument("--model_path", default="symbolicregression/weights/model1.pt")
    ap.add_argument("--mode", choices=["e2e", "mcts"], default="mcts")
    ap.add_argument("--beam_size", type=int, default=2)
    ap.add_argument("--n_trees_to_refine", type=int, default=2)
    ap.add_argument("--max_input_points", type=int, default=200)
    ap.add_argument("--max_number_bags", type=int, default=1)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--rollout", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--per_task_timeout_sec", type=float, default=300.0)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args(argv)

    repo_root = Path.cwd()
    tpsr_root = (repo_root / args.tpsr_root).resolve()
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_seed{args.seed}.json"
    tasks = collect_tasks(
        manifest=args.manifest,
        suites=args.suite,
        root=args.root,
        seed=args.seed,
        limit=args.max_tasks,
    )
    if out_path.exists() and not args.no_resume:
        results: dict[str, dict[str, Any]] = json.loads(out_path.read_text())
    else:
        results = {}

    sys.path.insert(0, str(tpsr_root))
    with _pushd(tpsr_root):
        runner = _TPSRRunner(args)
        for task in tasks:
            if not args.no_resume and task.name in results and results[task.name].get("status") == "ok":
                continue
            started = time.perf_counter()
            try:
                row = _with_timeout(lambda: runner.run_task(task), timeout_sec=float(args.per_task_timeout_sec))
                row["runtime_sec"] = float(row.get("runtime_sec", 0.0)) + (time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001 - long baseline runs should archive per-task failures.
                row = {
                    "task_id": task.name,
                    "suite": task.metadata.get("suite", _infer_suite(task.name)),
                    "method": "TPSR",
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                    "r2": 0.0,
                    "nmse": None,
                    "expression": "",
                    "ground_truth": task.expression,
                    "runtime_sec": time.perf_counter() - started,
                    "tpsr_mode": args.mode,
                }
            row.update({
                "task_id": task.name,
                "suite": task.metadata.get("suite", _infer_suite(task.name)),
                "domain": task.metadata.get("domain", "unknown"),
                "split": task.metadata.get("split", ""),
                "ground_truth": task.expression,
                "n_vars": int(task.X_train.shape[1]),
                "budget": {
                    "mode": args.mode,
                    "beam_size": args.beam_size,
                    "n_trees_to_refine": args.n_trees_to_refine,
                    "max_input_points": args.max_input_points,
                    "max_number_bags": args.max_number_bags,
                    "width": args.width,
                    "rollout": args.rollout,
                    "horizon": args.horizon,
                },
            })
            results[task.name] = row
            out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
            print(f"{task.name:32s} status={row['status']} r2={row.get('r2')}")
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"saved {out_path} ({len(results)} tasks)")


class _TPSRRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        self.args = args
        self.torch = torch
        try:
            self.model = torch.load(args.model_path, map_location=torch.device("cpu"), weights_only=False)
        except TypeError:
            self.model = torch.load(args.model_path, map_location=torch.device("cpu"))
        self.model.embedder.eval()
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.env = self.model.env
        self.params = self.env.params
        self._configure_params()
        self._patch_torch_load_for_cpu()

    def _configure_params(self) -> None:
        self.params.device = self.torch.device("cpu")
        self.params.cpu = True
        self.params.beam_size = int(self.args.beam_size)
        self.params.n_trees_to_refine = int(self.args.n_trees_to_refine)
        self.params.max_input_points = int(self.args.max_input_points)
        self.params.max_number_bags = int(self.args.max_number_bags)
        self.params.width = int(self.args.width)
        self.params.num_beams = int(self.args.beam_size)
        self.params.rollout = int(self.args.rollout)
        self.params.horizon = int(self.args.horizon)
        self.params.ucb_constant = getattr(self.params, "ucb_constant", 1.0)
        self.params.ucb_base = getattr(self.params, "ucb_base", 10)
        self.params.uct_alg = getattr(self.params, "uct_alg", "uct")
        self.params.train_value = False
        self.params.debug = False
        self.params.no_seq_cache = True
        self.params.no_prefix_cache = True
        self.params.sample_only = False
        self.params.beam_type = getattr(self.params, "beam_type", "sampling")
        self.params.backbone_model = getattr(self.params, "backbone_model", "e2e")
        self.params.beam_temperature = getattr(self.params, "beam_temperature", 1.0)
        self.params.beam_length_penalty = getattr(self.params, "beam_length_penalty", 1.0)
        self.params.beam_early_stopping = getattr(self.params, "beam_early_stopping", True)
        self.params.max_generated_output_len = getattr(self.params, "max_generated_output_len", 200)
        self.params.rescale = getattr(self.params, "rescale", True)
        self.params.lam = getattr(self.params, "lam", 0.1)

    def _patch_torch_load_for_cpu(self) -> None:
        original = self.torch.load

        def load_cpu(*load_args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = self.torch.device("cpu")
            return original(*load_args, **kwargs)

        self.torch.load = load_cpu

    def run_task(self, task) -> dict[str, Any]:
        if self.args.mode == "e2e":
            return self._run_e2e(task)
        return self._run_mcts(task)

    def _run_e2e(self, task) -> dict[str, Any]:
        from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor

        started = time.perf_counter()
        y_train = np.asarray(task.y_train, dtype=float).reshape(-1, 1)
        dstr = SymbolicTransformerRegressor(
            model=self.model,
            max_input_points=self.args.max_input_points,
            n_trees_to_refine=self.args.n_trees_to_refine,
            max_number_bags=self.args.max_number_bags,
            rescale=True,
        )
        dstr.fit(np.asarray(task.X_train, dtype=float), y_train, verbose=False)
        train_pred = dstr.predict(np.asarray(task.X_train, dtype=float), refinement_type="BFGS")
        test_pred = dstr.predict(np.asarray(task.X_test, dtype=float), refinement_type="BFGS")
        best = dstr.retrieve_tree(refinement_type="BFGS", with_infos=True)
        expression = _tree_expression(best)
        return normalize_tpsr_result(
            task_id=task.name,
            suite=task.metadata.get("suite", _infer_suite(task.name)),
            expression=expression,
            ground_truth=task.expression,
            y_train=task.y_train,
            train_pred=train_pred,
            y_test=task.y_test,
            test_pred=test_pred,
            runtime_sec=time.perf_counter() - started,
            mode="e2e",
            extra={"refinement_type": best.get("refinement_type")},
        )

    def _run_mcts(self, task) -> dict[str, Any]:
        from tpsr import tpsr_fit
        from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor, get_top_k_features
        import symbolicregression.model.utils_wrapper as utils_wrapper

        started = time.perf_counter()
        x_train = np.asarray(task.X_train, dtype=float)
        x_test = np.asarray(task.X_test, dtype=float)
        y_train = np.asarray(task.y_train, dtype=float).reshape(-1, 1)
        top_k = get_top_k_features(x_train, y_train, k=self.env.params.max_input_dimension)
        x_selected = x_train[:, top_k]
        scaler = utils_wrapper.StandardScaler()
        scaled_x = scaler.fit_transform(x_selected)
        scale_params = scaler.get_params()
        state, planning_time, sample_times = tpsr_fit([scaled_x], [y_train], self.params, self.env, bag_number=1)
        generated_tree = list(filter(lambda item: item is not None, [self.env.idx_to_infix(state[1:], is_float=False, str_array=False)]))
        if not generated_tree:
            raise RuntimeError("TPSR generated no valid tree")
        dstr = SymbolicTransformerRegressor(
            model=self.model,
            max_input_points=self.args.max_input_points,
            n_trees_to_refine=self.args.n_trees_to_refine,
            max_number_bags=self.args.max_number_bags,
            rescale=True,
        )
        dstr.start_fit = time.time()
        dstr.top_k_features = [top_k]
        refined = dstr.refine(scaled_x, y_train, generated_tree, verbose=False)
        if not refined:
            raise RuntimeError("TPSR generated tree could not be refined")
        for idx, candidate in enumerate(refined):
            candidate["predicted_tree"] = scaler.rescale_function(self.env, candidate["predicted_tree"], *scale_params)
            refined[idx] = candidate
        dstr.tree = {0: refined}
        train_pred = dstr.predict(x_train, refinement_type="BFGS")
        test_pred = dstr.predict(x_test, refinement_type="BFGS")
        best = dstr.retrieve_tree(refinement_type="BFGS", with_infos=True)
        expression = _tree_expression(best)
        return normalize_tpsr_result(
            task_id=task.name,
            suite=task.metadata.get("suite", _infer_suite(task.name)),
            expression=expression,
            ground_truth=task.expression,
            y_train=task.y_train,
            train_pred=train_pred,
            y_test=task.y_test,
            test_pred=test_pred,
            runtime_sec=time.perf_counter() - started,
            mode="mcts",
            extra={
                "planning_time_sec": planning_time,
                "sample_times": sample_times,
                "refinement_type": best.get("refinement_type"),
            },
        )


def _tree_expression(best: dict[str, Any]) -> str:
    tree = best.get("relabed_predicted_tree") or best.get("predicted_tree")
    return "" if tree is None else tree.infix()


def _infer_suite(task_id: str) -> str:
    return task_id.split("/", 1)[0] if "/" in task_id else "unknown"


class _TaskTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame) -> None:
    raise _TaskTimeout("per-task TPSR timeout")


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
