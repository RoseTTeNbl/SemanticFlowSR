#!/usr/bin/env python
"""Run velocity-rollout evaluation over a benchmark and record sample/stat metrics.

Loads a trained checkpoint, evaluates each task it can fit (num_vars must match the
checkpoint's gen config), and writes <tag>_samples.jsonl + <tag>_summary.json.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import yaml
import torch

from semflow_sr.models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from semflow_sr.data.benchmark_loader import materialize_formula, PMLBLoader, FeynmanCSVLoader, load_materialized_task
from semflow_sr.data.benchmark_manifest import load_benchmark_manifest
from semflow_sr.eval.evaluator import evaluate_task
from semflow_sr.eval.results import save_results
from semflow_sr.gp_distill.gp_policy import GPPolicyDistillationPrior
from semflow_sr.semantics.energy import ActionEnergyConfig
from semflow_sr.sr.ops import NAME_TO_ID

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "data" / "formula_benchmarks"


@dataclass
class ModelRunner:
    ckpt: str
    model: SemanticTransformer
    gen_cfg: dict
    energy_cfg: dict
    ops_ids: list[int]
    cfg: dict
    beta: float


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["meta"]["cfg"]
    g = cfg["gen"]
    m = cfg["model"]
    model = SemanticTransformer(SemanticTransformerConfig(
        d=g["num_vars"], K=g["K"], hidden=m["hidden"],
        row_layers=m["row_layers"], heads=m["heads"],
        output_mode=m.get("output_mode", "semantic_fisher_lograte")))
    model.load_state_dict(ck["model"]); model.eval().to(device)
    return model, g, cfg.get("energy", {}), cfg


def load_runner(ckpt_path: str, device) -> ModelRunner:
    model, g, ecfg, cfg = load_model(ckpt_path, device)
    return ModelRunner(ckpt=str(ckpt_path), model=model, gen_cfg=g, energy_cfg=ecfg,
                       ops_ids=[NAME_TO_ID[o] for o in g["ops"]],
                       cfg=cfg, beta=_checkpoint_beta(cfg))


def _checkpoint_beta(cfg: dict) -> float:
    update = cfg.get("update", {})
    return float(update.get("beta", cfg.get("beta", cfg.get("eta", 1.0))))


def _checkpoint_gamma(cfg: dict) -> float:
    return float(cfg.get("path", {}).get("gamma", 0.1))


def parse_ckpt_by_vars(specs: list[str] | None) -> dict[int, str]:
    out: dict[int, str] = {}
    for spec in specs or []:
        if ":" not in spec:
            raise ValueError(f"bad --ckpt_by_vars entry {spec!r}; expected N:path")
        k, v = spec.split(":", 1)
        if not k.isdigit() or not v:
            raise ValueError(f"bad --ckpt_by_vars entry {spec!r}; expected N:path")
        out[int(k)] = v
    return out


def parse_target_kwargs(raw: str | None) -> dict:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--target_kwargs must be a JSON object")
    return parsed


def load_gp_distilled_scores(path: str | None) -> tuple[dict[int, float], dict[int | str, float]]:
    if not path:
        return {}, {}
    events = json.loads(Path(path).read_text())
    if isinstance(events, dict):
        events = events.get("events", [])
    if not isinstance(events, list):
        raise ValueError("--gp_distill_events must contain a list or {'events': list}")
    prior = GPPolicyDistillationPrior(events=events)
    return prior.merged_scores()


def merge_score_maps(base: dict | None, extra: dict | None) -> dict:
    out = dict(base or {})
    for key, value in (extra or {}).items():
        out[key] = float(value)
    return out


def resolve_gp_policy_weight(
    raw_weight: float | None,
    target_kwargs: dict,
    gp_action_scores: dict | None,
    gp_operator_scores: dict | None,
) -> float:
    """Resolve online GP bias weight.

    GP scores are now treated primarily as candidate priors. Online additive bias is
    an explicit low-weight ablation, so it is disabled unless the CLI passes a value.
    """
    if raw_weight is None:
        return 0.0
    return float(raw_weight)


def select_runner_for_task(num_vars: int, runners: dict[int, ModelRunner]):
    return runners.get(int(num_vars))


def missing_checkpoint_dims(tasks, runners: dict[int, ModelRunner]) -> list[int]:
    dims = {int(t.X_train.shape[1]) for t in tasks}
    return sorted(d for d in dims if d not in runners)


def task_passes_dim_filter(task, min_vars: int | None = None, max_vars: int | None = None) -> bool:
    n_vars = int(task.X_train.shape[1])
    if min_vars is not None and n_vars < int(min_vars):
        return False
    if max_vars is not None and n_vars > int(max_vars):
        return False
    return True


def gather_manifest_tasks(
    manifest: str | Path,
    *,
    suites: list[str] | None = None,
    root: str | Path = ".",
    min_vars: int | None = None,
    max_vars: int | None = None,
    limit: int | None = None,
):
    manifest_obj = load_benchmark_manifest(manifest)
    selected = set(suites or manifest_obj.suites.keys())
    tasks = []
    for suite, specs in manifest_obj.suites.items():
        if suite not in selected:
            continue
        for spec in specs:
            task = load_materialized_task(spec, root=root)
            if task_passes_dim_filter(task, min_vars=min_vars, max_vars=max_vars):
                tasks.append(task)
                if limit is not None and len(tasks) >= int(limit):
                    return tasks
    return tasks


def gather_tasks(args, num_vars=None):
    if getattr(args, "manifest", None):
        tasks = gather_manifest_tasks(
            args.manifest,
            suites=getattr(args, "manifest_suite", None),
            root=getattr(args, "manifest_root", "."),
            min_vars=getattr(args, "min_vars", None),
            max_vars=getattr(args, "max_vars", None),
            limit=getattr(args, "limit_tasks", None),
        )
        if num_vars is not None:
            tasks = [t for t in tasks if int(t.X_train.shape[1]) == int(num_vars)]
        return tasks
    tasks = []
    for suite in args.suite:
        for entry in yaml.safe_load((CFG_DIR / f"{suite}.yaml").read_text()):
            if num_vars is not None and len(entry["variables"]) != num_vars:
                continue
            if getattr(args, "min_vars", None) is not None and len(entry["variables"]) < int(args.min_vars):
                continue
            if getattr(args, "max_vars", None) is not None and len(entry["variables"]) > int(args.max_vars):
                continue
            entry.setdefault("suite", suite)
            tasks.append(materialize_formula(entry, args.seed))
    if args.pmlb_root and args.pmlb:
        loader = PMLBLoader(args.pmlb_root)
        for name in args.pmlb:
            try:
                t = loader.load(name, seed=args.seed)
            except Exception as e:
                print(f"skip {name}: {e}"); continue
            if (num_vars is None or t.X_train.shape[1] == num_vars) and task_passes_dim_filter(
                t, getattr(args, "min_vars", None), getattr(args, "max_vars", None)
            ):
                tasks.append(t)
    if args.feynman:
        floader = FeynmanCSVLoader()
        for name in floader.names(n_vars=num_vars):
            try:
                t = floader.load(name, seed=args.seed)
                if task_passes_dim_filter(t, getattr(args, "min_vars", None), getattr(args, "max_vars", None)):
                    tasks.append(t)
            except Exception as e:
                print(f"skip {name}: {e}")
    return tasks


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="single checkpoint; tasks are filtered to its num_vars")
    ap.add_argument("--ckpt_by_vars", nargs="*", default=None,
                    help="dimension-to-checkpoint mapping, e.g. 1:ckpt_1.pt 2:ckpt_2.pt")
    ap.add_argument("--suite", nargs="+", default=["nguyen"])
    ap.add_argument("--manifest", default=None,
                    help="unified benchmark manifest JSON; when set, formula --suite is ignored")
    ap.add_argument("--manifest_suite", nargs="+", default=None,
                    help="suite names inside --manifest; defaults to all manifest suites")
    ap.add_argument("--manifest_root", default=".",
                    help="root directory for relative paths stored in --manifest")
    ap.add_argument("--limit_tasks", type=int, default=None,
                    help="optional cap for manifest-loaded tasks, useful for smoke runs")
    ap.add_argument("--min_vars", type=int, default=None,
                    help="only evaluate tasks with at least this many variables")
    ap.add_argument("--max_vars", type=int, default=None,
                    help="only evaluate tasks with at most this many variables")
    ap.add_argument("--pmlb", nargs="+", default=[])
    ap.add_argument("--pmlb_root", default="external/pmlb")
    ap.add_argument("--feynman", action="store_true", help="纳入物化的 Feynman 任务(按 num_vars 过滤)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/semflow")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--max_support", type=int, default=128)
    ap.add_argument("--support_mode", default="mixed_topk_random",
                    choices=["full", "adaptive_full", "topk_reward", "mixed_topk_random", "proposal_importance"])
    ap.add_argument("--support_topk", type=int, default=64)
    ap.add_argument("--support_full_threshold", type=int, default=None,
                    help="for adaptive_full: use complete legal action simplex up to this full-action count")
    ap.add_argument("--target", default="one_step_advantage",
                    choices=["one_step_advantage", "group_advantage", "semantic_advantage_flow", "rollout_fitness_advantage",
                             "rollout_fitness", "global_trajectory", "global_trajectory_marginal",
                             "trajectory_marginal", "semantic_fisher_risk_flow", "risk_flow", "energy"])
    ap.add_argument("--target_kwargs", default="{}",
                    help="JSON object forwarded to the rollout target builder")
    ap.add_argument("--num_policy_updates", type=int, default=1,
                    help="number of repeated local model-flow policy updates per SR step")
    ap.add_argument("--integration_method", default="semantic_fisher_sphere",
                    choices=["semantic_fisher_sphere", "semantic_fisher_ode", "closed_form"],
                    help="mainline is semantic_fisher_sphere/ode; closed_form is the plain-Fisher potential ablation")
    ap.add_argument("--ode_steps", type=int, default=1,
                    help="internal ODE substeps for --integration_method semantic_fisher_ode")
    ap.add_argument("--beta", type=float, default=None,
                    help="fixed natural-flow update strength; defaults to checkpoint update.beta or 1.0")
    ap.add_argument("--gamma", type=float, default=None,
                    help="semantic-Fisher pullback weight; defaults to checkpoint path.gamma or 0.1")
    ap.add_argument("--gram_rank", type=int, default=None,
                    help="optional low-rank Gram rank for exact semantic-Fisher oracle computations")
    ap.add_argument("--gp_policy_weight", type=float, default=None,
                    help="weight for online GP action/operator prior added to model scores")
    ap.add_argument("--gp_action_scores", default=None,
                    help="JSON action-id score map for online GP policy guidance")
    ap.add_argument("--gp_operator_scores", default=None,
                    help="JSON operator-name/id score map for online GP policy guidance")
    ap.add_argument("--gp_distill_events", default=None,
                    help="JSON GP event list used to distill action/operator success-likelihood priors")
    ap.add_argument("--update_mode", default="fixed_beta", choices=["fixed_beta", "target_kl"])
    ap.add_argument("--target_kl", type=float, default=0.05)
    ap.add_argument("--beta_max", type=float, default=10.0)
    ap.add_argument("--bisection_steps", type=int, default=20)
    ap.add_argument("--record_diagnostics", action="store_true",
                    help="write per-rollout-step support/reward/probability diagnostics into samples")
    ap.add_argument("--record_path", action="store_true",
                    help="with --record_diagnostics, also record per-lambda velocity/path summaries")
    ap.add_argument("--execution_mode", default="action",
                    choices=["action", "global_block_commit", "block_commit", "full_selector"],
                    help="online execution mode: stepwise action, commit a selected block, or select a full trajectory")
    ap.add_argument("--block_size", type=int, default=3)
    ap.add_argument("--block_candidate_budget", type=int, default=64)
    ap.add_argument("--block_aggregation", default="mean",
                    choices=["mean", "topk_mean", "sum", "max", "softmax_weighted_reward"])
    ap.add_argument("--block_p0_mode", default="uniform", choices=["uniform", "frequency"])
    ap.add_argument("--global_block_selector", default="exact", choices=["oracle", "exact", "learned"])
    ap.add_argument("--risk_mode", default="top_alpha", choices=["top_alpha", "rank", "z_score"])
    ap.add_argument("--risk_alpha", type=float, default=0.1)
    ap.add_argument("--risk_normalize", default="rank", choices=["rank", "zscore", "none"])
    ap.add_argument("--trajectory_num_samples", type=int, default=64)
    ap.add_argument("--trajectory_max_len", type=int, default=None)
    ap.add_argument("--trajectory_temperature", type=float, default=1.0)
    ap.add_argument("--trajectory_exploration", type=float, default=0.0)
    ap.add_argument("--gp_population_path", default=None,
                    help="JSON/JSONL action trajectory population for GP-CandidatePool/full selector")
    ap.add_argument("--gp_sample_mode", default="base_plus_gp", choices=["base_plus_gp", "gp_only"])
    ap.add_argument("--plot_per_task", action="store_true",
                    help="also write per-run task-order R2 and energy-trace plots")
    ap.add_argument("--require_all_ckpts", action="store_true",
                    help="fail if --ckpt_by_vars does not cover every loaded task dimension")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main():
    ap = build_arg_parser()
    a = ap.parse_args()
    target_kwargs = parse_target_kwargs(a.target_kwargs)
    gp_action_scores = parse_target_kwargs(a.gp_action_scores) if a.gp_action_scores else target_kwargs.get("gp_action_scores")
    gp_operator_scores = (
        parse_target_kwargs(a.gp_operator_scores) if a.gp_operator_scores else target_kwargs.get("gp_operator_scores")
    )
    distilled_action_scores, distilled_operator_scores = load_gp_distilled_scores(a.gp_distill_events)
    gp_action_scores = merge_score_maps(gp_action_scores, distilled_action_scores)
    gp_operator_scores = merge_score_maps(gp_operator_scores, distilled_operator_scores)
    gp_policy_weight = resolve_gp_policy_weight(
        a.gp_policy_weight, target_kwargs, gp_action_scores, gp_operator_scores
    )

    device = torch.device(a.device)
    ckpt_by_vars = parse_ckpt_by_vars(a.ckpt_by_vars)
    if ckpt_by_vars:
        runners = {dim: load_runner(path, device) for dim, path in sorted(ckpt_by_vars.items())}
        tasks = gather_tasks(a, num_vars=None)
        missing = missing_checkpoint_dims(tasks, runners)
        if a.require_all_ckpts and missing:
            raise SystemExit(f"missing checkpoints for num_vars: {missing}")
        print(f"evaluating up to {len(tasks)} tasks with checkpoints for dims {sorted(runners)}")
    else:
        if not a.ckpt:
            raise SystemExit("provide either --ckpt or --ckpt_by_vars")
        runner = load_runner(a.ckpt, device)
        runners = {int(runner.gen_cfg["num_vars"]): runner}
        tasks = gather_tasks(a, runner.gen_cfg["num_vars"])
        print(f"evaluating {len(tasks)} tasks (num_vars={runner.gen_cfg['num_vars']})")

    reports = []
    skipped = []
    for t in tasks:
        n_vars = int(t.X_train.shape[1])
        runner = select_runner_for_task(n_vars, runners)
        if runner is None:
            skipped.append({"name": t.name, "n_vars": n_vars,
                            "reason": "missing checkpoint for num_vars"})
            print(f"  skip {t.name:16s} n_vars={n_vars}: missing checkpoint")
            continue
        g = runner.gen_cfg
        rep = evaluate_task(runner.model, t, K=g["K"], ops_ids=runner.ops_ids, device=device,
                            energy_cfg=ActionEnergyConfig(**runner.energy_cfg), max_steps=a.max_steps,
                            grid=a.grid, step_size=a.step_size, greedy=True, max_support=a.max_support,
                            support_mode=a.support_mode, support_topk=a.support_topk,
                            support_full_threshold=a.support_full_threshold,
                            target=a.target, target_kwargs=target_kwargs,
                            beta=(a.beta if a.beta is not None else runner.beta),
                            gamma=(a.gamma if a.gamma is not None else _checkpoint_gamma(runner.cfg)),
                            gram_rank=a.gram_rank,
                            gp_policy_weight=gp_policy_weight,
                            gp_action_scores=gp_action_scores,
                            gp_operator_scores=gp_operator_scores,
                            num_policy_updates=a.num_policy_updates,
                            integration_method=a.integration_method,
                            ode_steps=a.ode_steps,
                            update_mode=a.update_mode, target_kl=a.target_kl,
                            beta_max=a.beta_max, bisection_steps=a.bisection_steps,
                            record_diagnostics=a.record_diagnostics,
                            record_path=a.record_path,
                            execution_mode=a.execution_mode,
                            block_size=a.block_size,
                            block_candidate_budget=a.block_candidate_budget,
                            block_aggregation=a.block_aggregation,
                            block_p0_mode=a.block_p0_mode,
                            global_block_selector=a.global_block_selector,
                            risk_mode=a.risk_mode,
                            risk_alpha=a.risk_alpha,
                            risk_normalize=a.risk_normalize,
                            trajectory_num_samples=a.trajectory_num_samples,
                            trajectory_max_len=a.trajectory_max_len,
                            trajectory_temperature=a.trajectory_temperature,
                            trajectory_exploration=a.trajectory_exploration,
                            gp_population_path=a.gp_population_path,
                            gp_sample_mode=a.gp_sample_mode)
        rep.task_metadata = dict(rep.task_metadata or {})
        rep.task_metadata.update({"checkpoint": runner.ckpt, "checkpoint_num_vars": int(g["num_vars"])})
        reports.append(rep)
        print(f"  {rep.name:16s} r2={rep.r2:.4f} acc={rep.acc_tau:.0f} "
              f"steps={rep.steps} cplx={rep.complexity}")
    summary = save_results(reports, a.out, a.tag, make_plots=a.plot_per_task)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}_skipped.json").write_text(json.dumps(skipped, indent=2))
    summary["skipped"] = len(skipped)
    (out / f"{a.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", summary)


if __name__ == "__main__":
    main()
