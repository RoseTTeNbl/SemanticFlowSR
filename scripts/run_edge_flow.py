#!/usr/bin/env python
"""Smoke/evaluation runner for Edge-Parameterized Semantic Flow."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from semflow_sr.data.synthetic_generator import GenConfig, generate_expression, sample_probe_xy
from semflow_sr.edge_flow.benchmark import (
    load_edge_flow_benchmark_tasks,
    skeleton_match,
    summarize_benchmark_records,
    task_tensors,
    write_benchmark_result_files,
)
from semflow_sr.edge_flow.circuit_sampler import CircuitSampler
from semflow_sr.edge_flow.dataset import EdgeFlowRecord
from semflow_sr.edge_flow.edge_distribution import EdgeDistribution
from semflow_sr.edge_flow.model import EdgeFlowModel, EdgeFlowModelConfig
from semflow_sr.edge_flow.projection import project_elites_to_edge_target
from semflow_sr.edge_flow.reward import RewardConfig, evaluate_expression_rewards
from semflow_sr.edge_flow.template import RegisterOperatorTemplate
from semflow_sr.eval.metrics import accuracy_rate, nmse, r2_score
from semflow_sr.sr.ast import Expr, eval_expr
from semflow_sr.sr.ops import get_op
from semflow_sr.sr.printer import to_string


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="results/edge_flow_smoke")
    parser.add_argument("--tag", default="edge_flow_smoke")
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--flow_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--manifest_root", default="data/benchmark_suites")
    parser.add_argument("--manifest_suite", nargs="+", default=["nguyen", "constant", "livermore", "jin"])
    parser.add_argument("--legacy_87", action="store_true")
    parser.add_argument("--feynman_root", default="data/materialized/feynman")
    parser.add_argument("--limit_tasks", type=int, default=None)
    parser.add_argument("--complexity_weight", type=float, default=0.001)
    parser.add_argument("--decoder_budgets", nargs="+", type=int, default=None)
    parser.add_argument("--oracle_samples", type=int, default=0)
    parser.add_argument("--oracle_decode_samples", type=int, default=0)
    parser.add_argument("--elite_k", type=int, default=16)
    parser.add_argument("--target_smoothing", type=float, default=0.01)
    parser.add_argument("--projection_mode", default="global_topk")
    parser.add_argument("--selection_eta_logprob", type=float, default=0.0)
    args = parser.parse_args()

    result = run(args)
    out = Path(args.out)
    summary = write_benchmark_result_files(result["records"], out, args.tag)
    print(json.dumps(summary, indent=2))


def run(args) -> dict:
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    template = RegisterOperatorTemplate(**{
        **ckpt["template"],
        "primitives": tuple(ckpt["template"]["primitives"]),
    })
    model = EdgeFlowModel(EdgeFlowModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    rng = random.Random(int(args.seed))
    records = []
    if args.manifest:
        tasks = load_edge_flow_benchmark_tasks(
            manifest=args.manifest,
            suites=list(args.manifest_suite or []),
            root=args.manifest_root,
            seed=int(args.seed),
            legacy_87=bool(args.legacy_87),
            feynman_root=args.feynman_root,
            limit=args.limit_tasks,
        )
        for task in tasks:
            x_train, y_train, x_test, y_test = task_tensors(task, template_num_vars=template.num_vars)
            records.append(_evaluate_dataset_task(
                model,
                template,
                task_id=task.name,
                suite=str(task.metadata.get("suite", "unknown")),
                num_vars=int(task.X_train.shape[1]),
                ground_truth=task.expression or task.metadata.get("ground_truth", ""),
                variable_names=list(task.variable_names),
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                rng=rng,
                eval_samples=int(args.eval_samples),
                flow_steps=int(args.flow_steps),
                complexity_weight=float(args.complexity_weight),
                decoder_budgets=list(args.decoder_budgets or []),
                oracle_samples=int(args.oracle_samples),
                oracle_decode_samples=int(args.oracle_decode_samples),
                elite_k=int(args.elite_k),
                target_smoothing=float(args.target_smoothing),
                projection_mode=str(args.projection_mode),
                selection_eta_logprob=float(args.selection_eta_logprob),
            ))
    else:
        for idx in range(int(args.num_tasks)):
            expr, x, y = _task(template, rng)
            records.append(_evaluate_dataset_task(
                model,
                template,
                task_id=f"synthetic_{idx}",
                suite="synthetic",
                num_vars=template.num_vars,
                ground_truth=to_string(expr, template.num_vars, simplify=True),
                variable_names=[f"x{i}" for i in range(template.num_vars)],
                x_train=x,
                y_train=y,
                x_test=x,
                y_test=y,
                rng=rng,
                eval_samples=int(args.eval_samples),
                flow_steps=int(args.flow_steps),
                complexity_weight=float(args.complexity_weight),
                decoder_budgets=list(args.decoder_budgets or []),
                oracle_samples=int(args.oracle_samples),
                oracle_decode_samples=int(args.oracle_decode_samples),
                elite_k=int(args.elite_k),
                target_smoothing=float(args.target_smoothing),
                projection_mode=str(args.projection_mode),
                selection_eta_logprob=float(args.selection_eta_logprob),
            ))
    summary = summarize_benchmark_records(records)
    return {"summary": summary, "records": records}


def _evaluate_dataset_task(
    model: EdgeFlowModel,
    template: RegisterOperatorTemplate,
    *,
    task_id: str,
    suite: str,
    num_vars: int,
    ground_truth: str | None,
    variable_names: list[str],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    rng: random.Random,
    eval_samples: int,
    flow_steps: int,
    complexity_weight: float,
    decoder_budgets: list[int],
    oracle_samples: int,
    oracle_decode_samples: int,
    elite_k: int,
    target_smoothing: float,
    projection_mode: str,
    selection_eta_logprob: float,
) -> dict:
    theta0 = EdgeDistribution.uniform(template)
    theta = _integrate_model(model, theta0, x_train, y_train, steps=flow_steps)
    samples, train_rewards, best = _sample_and_select(
        template,
        theta,
        x_train,
        y_train,
        rng=rng,
        samples=eval_samples,
        complexity_weight=complexity_weight,
        eta_logprob=selection_eta_logprob,
    )
    coef = train_rewards.affine_coef[best].tolist()
    expr_text = to_string(samples[best].expression, template.num_vars, simplify=True)
    generated = f"{coef[0]:.6g}*({expr_text}) + {coef[1]:.6g}"
    r2, nmse_value, raw_r2 = _test_metrics(samples[best].expression, x_test, y_test, coef)
    reward = float(r2 - float(complexity_weight) * int(samples[best].complexity))
    decoder_curve = _decoder_budget_curve(
        template,
        theta,
        ground_truth=ground_truth or "",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        rng=rng,
        budgets=decoder_budgets,
        complexity_weight=complexity_weight,
        eta_logprob=selection_eta_logprob,
    )
    oracle_diag = _oracle_diagnostics(
        template,
        theta0,
        ground_truth=ground_truth or "",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        rng=rng,
        oracle_samples=oracle_samples,
        oracle_decode_samples=oracle_decode_samples,
        elite_k=elite_k,
        target_smoothing=target_smoothing,
        projection_mode=projection_mode,
        complexity_weight=complexity_weight,
    )
    structure = _expression_structure(samples[best].expression, template.num_vars)
    template_diag = _template_diagnostics(template)
    return {
        "task_id": str(task_id),
        "suite": str(suite),
        "num_vars": int(num_vars),
        "variable_names": list(variable_names),
        "ground_truth": str(ground_truth or ""),
        "expression": generated,
        "raw_expression": expr_text,
        "affine_a": float(coef[0]),
        "affine_b": float(coef[1]),
        "r2": float(r2),
        "nmse": float(nmse_value),
        "reward": reward,
        "complexity": int(samples[best].complexity),
        "solved": bool(accuracy_rate(float(r2))),
        "train_r2": float(train_rewards.r2[best].item()),
        "train_nmse": float(train_rewards.nmse[best].item()),
        "train_reward": float(train_rewards.rewards[best].item()),
        "raw_test_r2_without_affine": float(raw_r2),
        "calibration_gain": float(r2 - raw_r2),
        "train_test_r2_gap": float(float(train_rewards.r2[best].item()) - r2),
        "valid_expression_fraction": float(train_rewards.valid_mask.float().mean().item()),
        "unique_expression_fraction": float(len({str(s.expression) for s in samples}) / max(len(samples), 1)),
        "best_log_prob": float(samples[best].log_prob),
        "decoder_budget_curve": decoder_curve,
        **template_diag,
        **structure,
        **oracle_diag,
    }


def _sample_and_select(
    template: RegisterOperatorTemplate,
    theta: EdgeDistribution,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    rng: random.Random,
    samples: int,
    complexity_weight: float,
    eta_logprob: float,
):
    sampled = CircuitSampler(template).sample(theta, batch_size=int(samples), rng=rng)
    rewards = evaluate_expression_rewards(sampled, x, y, RewardConfig(complexity_weight=complexity_weight))
    if rewards.rewards.numel() == 0:
        return sampled, rewards, 0
    scores = rewards.rewards.clone()
    if float(eta_logprob) != 0.0:
        scores = scores + float(eta_logprob) * torch.tensor([s.log_prob for s in sampled], dtype=scores.dtype)
    return sampled, rewards, int(torch.argmax(scores).item())


def _test_metrics(expr, x_test: torch.Tensor, y_test: torch.Tensor, coef: list[float]) -> tuple[float, float, float]:
    try:
        semantics = torch.nan_to_num(eval_expr(expr, x_test), nan=0.0, posinf=0.0, neginf=0.0)
        pred = float(coef[0]) * semantics + float(coef[1])
        raw_pred = semantics
        finite = torch.isfinite(pred).all() and pred.abs().max() < 1e8
    except Exception:
        pred = torch.zeros_like(y_test)
        raw_pred = torch.zeros_like(y_test)
        finite = torch.tensor(False)
    if not bool(finite):
        return 0.0, float("inf"), 0.0
    return r2_score(y_test.detach().cpu().numpy(), pred.detach().cpu().numpy()), nmse(
        y_test.detach().cpu().numpy(),
        pred.detach().cpu().numpy(),
    ), r2_score(y_test.detach().cpu().numpy(), raw_pred.detach().cpu().numpy())


def _decoder_budget_curve(
    template: RegisterOperatorTemplate,
    theta: EdgeDistribution,
    *,
    ground_truth: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    rng: random.Random,
    budgets: list[int],
    complexity_weight: float,
    eta_logprob: float,
) -> list[dict]:
    rows = []
    for budget in budgets:
        sampled, rewards, best = _sample_and_select(
            template,
            theta,
            x_train,
            y_train,
            rng=rng,
            samples=int(budget),
            complexity_weight=complexity_weight,
            eta_logprob=eta_logprob,
        )
        if not sampled:
            continue
        coef = rewards.affine_coef[best].tolist()
        r2, nmse_value, raw_r2 = _test_metrics(sampled[best].expression, x_test, y_test, coef)
        expr_text = to_string(sampled[best].expression, template.num_vars, simplify=True)
        rows.append({
            "budget": int(budget),
            "r2": float(r2),
            "nmse": float(nmse_value),
            "raw_r2_without_affine": float(raw_r2),
            "skeleton_match": bool(skeleton_match(ground_truth, expr_text)),
            "complexity": int(sampled[best].complexity),
            "expression": expr_text,
            "unique_fraction": float(len({str(s.expression) for s in sampled}) / max(len(sampled), 1)),
            "valid_fraction": float(rewards.valid_mask.float().mean().item()),
            "best_log_prob": float(sampled[best].log_prob),
        })
    return rows


def _oracle_diagnostics(
    template: RegisterOperatorTemplate,
    theta0: EdgeDistribution,
    *,
    ground_truth: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    rng: random.Random,
    oracle_samples: int,
    oracle_decode_samples: int,
    elite_k: int,
    target_smoothing: float,
    projection_mode: str,
    complexity_weight: float,
) -> dict:
    if int(oracle_samples) <= 0:
        return {}
    prior_samples, prior_rewards, prior_best = _sample_and_select(
        template,
        theta0,
        x_train,
        y_train,
        rng=rng,
        samples=int(oracle_samples),
        complexity_weight=complexity_weight,
        eta_logprob=0.0,
    )
    prior_coef = prior_rewards.affine_coef[prior_best].tolist()
    prior_r2, _, _ = _test_metrics(prior_samples[prior_best].expression, x_test, y_test, prior_coef)
    prior_expr = to_string(prior_samples[prior_best].expression, template.num_vars, simplify=True)
    theta_star, proj_diag = project_elites_to_edge_target(
        theta0,
        prior_samples,
        prior_rewards.rewards,
        prior_rewards.valid_mask,
        elite_k=int(elite_k),
        smoothing=float(target_smoothing),
        projection_mode=str(projection_mode),
    )
    out = {
        "prior_oracle_samples": int(oracle_samples),
        "prior_best_r2": float(prior_r2),
        "prior_best_skeleton_match": bool(skeleton_match(ground_truth, prior_expr)),
        "prior_best_complexity": int(prior_samples[prior_best].complexity),
        "prior_best_expression": prior_expr,
        "prior_unique_fraction": float(len({str(s.expression) for s in prior_samples}) / max(len(prior_samples), 1)),
        "prior_valid_fraction": float(prior_rewards.valid_mask.float().mean().item()),
        "projection_mode": str(projection_mode),
        "projection_target_edge_entropy_mean": float(proj_diag.get("target_edge_entropy_mean", 0.0)),
        "projection_target_ess": float(proj_diag.get("target_ess", 0.0)),
        "projection_per_mode_elite_count": proj_diag.get("per_mode_elite_count", []),
        "projection_per_mode_best_reward": proj_diag.get("per_mode_best_reward", []),
    }
    if int(oracle_decode_samples) > 0:
        star_samples, star_rewards, star_best = _sample_and_select(
            template,
            theta_star,
            x_train,
            y_train,
            rng=rng,
            samples=int(oracle_decode_samples),
            complexity_weight=complexity_weight,
            eta_logprob=0.0,
        )
        star_coef = star_rewards.affine_coef[star_best].tolist()
        star_r2, _, _ = _test_metrics(star_samples[star_best].expression, x_test, y_test, star_coef)
        star_expr = to_string(star_samples[star_best].expression, template.num_vars, simplify=True)
        out.update({
            "theta_star_decode_samples": int(oracle_decode_samples),
            "theta_star_best_r2": float(star_r2),
            "theta_star_best_skeleton_match": bool(skeleton_match(ground_truth, star_expr)),
            "theta_star_best_complexity": int(star_samples[star_best].complexity),
            "theta_star_best_expression": star_expr,
            "theta_star_projection_drop": float(prior_r2 - star_r2),
            "theta_star_unique_fraction": float(len({str(s.expression) for s in star_samples}) / max(len(star_samples), 1)),
            "theta_star_valid_fraction": float(star_rewards.valid_mask.float().mean().item()),
        })
    return out


def _template_diagnostics(template: RegisterOperatorTemplate) -> dict:
    counts = [group.num_candidates for group in template.groups]
    return {
        "template_num_layers": int(template.num_layers),
        "template_num_registers": int(template.num_registers),
        "template_mixture_modes": int(template.mixture_modes),
        "template_num_edge_groups": int(len(template.groups)),
        "template_candidate_count_mean": float(sum(counts) / max(len(counts), 1)),
        "template_candidate_count_max": int(max(counts) if counts else 0),
        "template_primitive_set": list(template.primitives),
    }


def _expression_structure(expr: Expr, num_vars: int) -> dict:
    ops: list[str] = []
    used_vars: set[int] = set()
    root = "var" if expr.kind == "var" else "const" if expr.kind == "const" else get_op(int(expr.op_id)).name

    def visit(node: Expr) -> None:
        if node.kind == "var" and node.var_index is not None:
            used_vars.add(int(node.var_index))
            return
        if node.kind != "op":
            return
        name = get_op(int(node.op_id)).name
        ops.append(name)
        for child in node.children:
            visit(child)

    visit(expr)
    return {
        "active_node_count": int(expr.complexity),
        "output_depth": int(expr.depth),
        "used_variable_set": [int(v) for v in sorted(v for v in used_vars if v < int(num_vars))],
        "used_variable_count": int(len([v for v in used_vars if v < int(num_vars)])),
        "root_operator": root,
        "num_binary_ops": int(sum(1 for op in ops if get_op_name_arity(op) == 2)),
        "num_unary_ops": int(sum(1 for op in ops if get_op_name_arity(op) == 1)),
        "num_plus_minus": int(sum(1 for op in ops if op in {"add", "sub"})),
        "num_mul_div": int(sum(1 for op in ops if op in {"mul", "protected_div"})),
        "num_trig": int(sum(1 for op in ops if op in {"sin", "cos"})),
        "num_exp_log_sqrt": int(sum(1 for op in ops if op in {"exp", "protected_log", "protected_sqrt"})),
        "operator_histogram": {op: int(ops.count(op)) for op in sorted(set(ops))},
    }


def get_op_name_arity(name: str) -> int:
    # Small helper avoids building a reverse registry in hot code.
    from semflow_sr.sr.ops import NAME_TO_ID

    return int(get_op(NAME_TO_ID[name]).arity)


def _task(template: RegisterOperatorTemplate, rng: random.Random):
    gen = GenConfig(
        num_vars=template.num_vars,
        max_depth=3,
        K=template.num_registers,
        probe_size=64,
        ops=tuple(template.primitives),
    )
    expr = generate_expression(gen, rng)
    x, y = sample_probe_xy(expr, gen, rng)
    return expr, x, y


def _integrate_model(
    model: EdgeFlowModel,
    theta: EdgeDistribution,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
) -> EdgeDistribution:
    cur = theta.clone()
    for step in range(max(int(steps), 1)):
        rec = _dummy_record(cur, x, y)
        pred = model(rec)
        dt = 1.0 / max(int(steps), 1)
        mix = _step_sqrt(cur.mixture_probs, pred.mixture_zdot, dt)
        groups = {
            key: _step_sqrt(value, pred.group_zdot[key], dt)
            for key, value in cur.group_probs.items()
        }
        cur = EdgeDistribution(cur.template, mix, groups)
    return cur


def _step_sqrt(p: torch.Tensor, zdot: torch.Tensor, dt: float) -> torch.Tensor:
    z = p.clamp_min(1e-12).sqrt() + float(dt) * zdot
    z = z.clamp_min(1e-6)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out = z * z
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _dummy_record(theta: EdgeDistribution, x: torch.Tensor, y: torch.Tensor) -> EdgeFlowRecord:
    return EdgeFlowRecord(
        task_id="infer",
        x=x.float(),
        y=y.float(),
        template=theta.template,
        theta0=theta,
        theta_star=theta,
        theta_lambda=theta,
        z_lambda_mixture=theta.sqrt_mixture,
        zdot_mixture=torch.zeros_like(theta.mixture_probs),
        z_lambda_groups=theta.sqrt_groups,
        zdot_groups={key: torch.zeros_like(value) for key, value in theta.group_probs.items()},
        sampled_expressions=[],
        rewards=torch.zeros(0),
        diagnostics={},
    )


if __name__ == "__main__":
    main()
