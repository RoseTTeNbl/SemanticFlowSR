"""External complete-expression proposal samplers for SPFF training."""
from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path

import torch

from ..sr.ast import Expr
from ..sr.ops import NAME_TO_ID, get_op
from ..sr.printer import to_string
from .circuit_sampler import CircuitSample
from .reward import RewardConfig, evaluate_expression_rewards


@dataclass(frozen=True)
class ExpressionProposal:
    formula: str
    source: str
    expression: Expr | None = None
    task_id: str = ""


def simple_gp_proposals(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_vars: int,
    primitives: tuple[str, ...],
    rng: random.Random,
    proposal_count: int = 8,
    population_size: int = 32,
    generations: int = 3,
    max_depth: int = 3,
) -> list[ExpressionProposal]:
    """Small tree-GP population proposal over the local Expr AST.

    This is intentionally lightweight: it supplies non-neural complete-expression
    candidates so SPFF top-k is not fed only by its own random policy samples.
    """

    pop = [_random_expr(num_vars, primitives, max_depth=max_depth, rng=rng) for _ in range(max(int(population_size), 1))]
    for _ in range(max(int(generations), 0)):
        ranked = _rank_exprs(pop, x, y)
        survivors = [expr for expr, _ in ranked[:max(2, len(ranked) // 4)]]
        children = list(survivors)
        while len(children) < len(pop):
            parent = rng.choice(survivors)
            children.append(_mutate_expr(parent, num_vars, primitives, max_depth=max_depth, rng=rng))
        pop = children
    ranked = _rank_exprs(pop, x, y)
    out: list[ExpressionProposal] = []
    seen: set[str] = set()
    for expr, _ in ranked:
        formula = to_string(expr, num_vars, simplify=True)
        if formula in seen:
            continue
        seen.add(formula)
        out.append(ExpressionProposal(formula=formula, source="gp", expression=expr))
        if len(out) >= int(proposal_count):
            break
    return out


def load_diffusion_formula_proposals(
    path: str | Path,
    *,
    task_id: str | None = None,
    limit: int = 0,
) -> list[ExpressionProposal]:
    """Load formula strings produced by an external diffusion sampler."""

    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict | str]
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        raw = json.loads(p.read_text())
        rows = raw if isinstance(raw, list) else raw.get("proposals", [])
    out: list[ExpressionProposal] = []
    for row in rows:
        if isinstance(row, str):
            formula = row
            row_task_id = ""
            source = "diffusion"
        else:
            row_task_id = str(row.get("task_id") or "")
            if task_id is not None and row_task_id != str(task_id):
                continue
            formula = str(row.get("formula") or row.get("expression") or row.get("generated") or "")
            source = str(row.get("source") or "diffusion")
        if not formula:
            continue
        out.append(ExpressionProposal(formula=formula, source=source, task_id=row_task_id))
        if int(limit) > 0 and len(out) >= int(limit):
            break
    return out


def _rank_exprs(exprs: list[Expr], x: torch.Tensor, y: torch.Tensor) -> list[tuple[Expr, float]]:
    samples = [
        CircuitSample(
            sample_id=idx,
            mode=0,
            edge_choices={},
            expression=expr,
            log_prob=0.0,
            complexity=int(expr.complexity),
            head_terms=(expr,),
        )
        for idx, expr in enumerate(exprs)
    ]
    rewards = evaluate_expression_rewards(samples, x, y, RewardConfig(complexity_weight=0.0, head_fit_mode="selector"))
    scored = list(zip(exprs, rewards.r2.detach().cpu().tolist()))
    scored.sort(key=lambda item: float(item[1]), reverse=True)
    return [(expr, float(score)) for expr, score in scored]


def _random_expr(num_vars: int, primitives: tuple[str, ...], *, max_depth: int, rng: random.Random) -> Expr:
    if int(max_depth) <= 0 or rng.random() < 0.35:
        if rng.random() < 0.8:
            return Expr.var(rng.randrange(max(int(num_vars), 1)))
        return Expr.const(1.0 if rng.random() < 0.5 else 0.0)
    primitive = rng.choice(list(primitives))
    op_id = NAME_TO_ID[primitive]
    op = get_op(op_id)
    children = tuple(
        _random_expr(num_vars, primitives, max_depth=int(max_depth) - 1, rng=rng)
        for _ in range(op.arity)
    )
    return Expr.op(op_id, children)


def _mutate_expr(expr: Expr, num_vars: int, primitives: tuple[str, ...], *, max_depth: int, rng: random.Random) -> Expr:
    if rng.random() < 0.45 or expr.kind != "op":
        return _random_expr(num_vars, primitives, max_depth=max_depth, rng=rng)
    children = list(expr.children)
    if not children:
        return _random_expr(num_vars, primitives, max_depth=max_depth, rng=rng)
    idx = rng.randrange(len(children))
    children[idx] = _mutate_expr(children[idx], num_vars, primitives, max_depth=max(int(max_depth) - 1, 0), rng=rng)
    return Expr.op(int(expr.op_id), tuple(children))
