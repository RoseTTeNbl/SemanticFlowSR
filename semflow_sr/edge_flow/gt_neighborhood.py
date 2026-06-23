"""GT-neighborhood sampler for CSEF target probability shapes."""
from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from ..sr.ast import Expr
from ..sr.ops import NAME_TO_ID, get_op
from ..sr.parser import parse_formula
from .circuit_sampler import CircuitSample
from .conditional import ConditionalEdgeFlowModel
from .path_compiler import compile_expr_to_csef_sample
from .template import RegisterOperatorTemplate


@dataclass(frozen=True)
class GTNeighborhoodResult:
    samples: list[CircuitSample]
    diagnostics: dict


def build_gt_neighborhood_samples(
    formula: str,
    *,
    variable_count: int,
    template: RegisterOperatorTemplate,
    model: ConditionalEdgeFlowModel,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    flow_steps: int,
    flow_time: float | None,
    rng: random.Random,
    size: int,
    op_replace_prob: float = 0.3,
    source_replace_prob: float = 0.3,
) -> GTNeighborhoodResult:
    """Compile canonical GT plus small symbolic perturbations.

    The perturbation kernel is intentionally conservative: every generated
    neighbor must compile through the same CSEF compiler before it can affect
    the target probability shape.
    """

    text = str(formula or "").strip()
    if not text:
        return GTNeighborhoodResult([], _diag(0, 0, 0, 0))
    try:
        expr = parse_formula(text, [f"x{i}" for i in range(int(variable_count))])
    except Exception:
        return GTNeighborhoodResult([], _diag(0, 0, 0, 0))

    requested = max(int(size), 1)
    samples: list[CircuitSample] = []
    seen: set[str] = set()

    def add_formula(expr_value: Expr, *, canonical: bool = False) -> None:
        formula_text = _structural_key(expr_value)
        if formula_text in seen:
            return
        seen.add(formula_text)
        compiled = compile_expr_to_csef_sample(
            expr_value,
            variable_count=int(variable_count),
            template=template,
            model=model,
            x=x,
            y=y,
            method=str(method),
            flow_steps=int(flow_steps),
            flow_time=flow_time,
        )
        if compiled is None:
            return
        diag = dict(compiled.diagnostics or {})
        diag.update({
            "is_gt_neighborhood": True,
            "gt_neighborhood_canonical": bool(canonical),
            "gt_neighborhood_formula": formula_text,
        })
        if canonical:
            diag["is_gt_elite"] = True
        else:
            diag.setdefault("is_gt_elite", False)
        compiled.diagnostics = diag
        samples.append(compiled)

    add_formula(expr, canonical=True)
    candidates = _perturbation_candidates(
        expr,
        variable_count=int(variable_count),
        primitive_names=tuple(template.primitives),
        rng=rng,
        op_replace_prob=float(op_replace_prob),
        source_replace_prob=float(source_replace_prob),
    )
    attempted = 0
    for candidate in candidates:
        if len(samples) >= requested:
            break
        attempted += 1
        add_formula(candidate, canonical=False)
    return GTNeighborhoodResult(
        samples=samples,
        diagnostics=_diag(
            requested,
            attempted + 1,
            len(samples),
            1 if samples and bool((samples[0].diagnostics or {}).get("gt_neighborhood_canonical")) else 0,
        ),
    )


def _perturbation_candidates(
    expr: Expr,
    *,
    variable_count: int,
    primitive_names: tuple[str, ...],
    rng: random.Random,
    op_replace_prob: float,
    source_replace_prob: float,
) -> list[Expr]:
    primitive_ids = [NAME_TO_ID[name] for name in primitive_names if name in NAME_TO_ID]
    paths = _node_paths(expr)
    out: list[Expr] = []
    for path in paths:
        node = _get_at_path(expr, path)
        if node.kind == "op" and rng.random() <= max(0.0, min(1.0, float(op_replace_prob))):
            old_op = get_op(int(node.op_id))
            replacements = [
                op_id for op_id in primitive_ids
                if int(op_id) != int(node.op_id) and get_op(int(op_id)).arity == old_op.arity
            ]
            rng.shuffle(replacements)
            for op_id in replacements[:3]:
                out.append(_replace_at_path(expr, path, Expr.op(int(op_id), node.children)))
        if node.kind in {"var", "const"} and rng.random() <= max(0.0, min(1.0, float(source_replace_prob))):
            leaves = [Expr.var(idx) for idx in range(max(int(variable_count), 1))]
            leaves.extend([Expr.const(0.0), Expr.const(1.0)])
            rng.shuffle(leaves)
            for leaf in leaves[:3]:
                if leaf != node:
                    out.append(_replace_at_path(expr, path, leaf))
    rng.shuffle(out)
    return out


def _node_paths(expr: Expr) -> list[tuple[int, ...]]:
    out = [()]
    if expr.kind == "op":
        for idx, child in enumerate(expr.children):
            out.extend((idx, *path) for path in _node_paths(child))
    return out


def _get_at_path(expr: Expr, path: tuple[int, ...]) -> Expr:
    node = expr
    for idx in path:
        node = node.children[int(idx)]
    return node


def _replace_at_path(expr: Expr, path: tuple[int, ...], replacement: Expr) -> Expr:
    if not path:
        return replacement
    idx = int(path[0])
    children = list(expr.children)
    children[idx] = _replace_at_path(children[idx], path[1:], replacement)
    return Expr.op(int(expr.op_id), tuple(children))


def _structural_key(expr: Expr) -> str:
    if expr.kind == "var":
        return f"x{expr.var_index}"
    if expr.kind == "const":
        return f"{float(expr.value):.6g}"
    op = get_op(int(expr.op_id)).name
    return op + "(" + ",".join(_structural_key(child) for child in expr.children) + ")"


def _diag(requested: int, attempted: int, compiled: int, canonical: int) -> dict:
    return {
        "gt_neighborhood_requested": int(requested),
        "gt_neighborhood_attempted": int(attempted),
        "gt_neighborhood_compiled": int(compiled),
        "gt_neighborhood_compile_success_rate": float(compiled / max(attempted, 1)),
        "gt_neighborhood_canonical_count": int(canonical),
    }
