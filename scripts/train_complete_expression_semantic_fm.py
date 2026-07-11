#!/usr/bin/env python3
"""One-step semantic endpoint proposals and legal register-simplex flow matching."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

import numpy as np
import sympy as sp
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semflow_sr.data.benchmark_loader import SRTask, load_materialized_task
from semflow_sr.data.benchmark_manifest import load_benchmark_manifest
from semflow_sr.data.symbolicgpt_subset import load_symbolicgpt_subset_tasks
from semflow_sr.eval.metrics import accuracy_rate, nmse, r2_score
from semflow_sr.flow.semantic_poisson import (
    exponential_fisher_correction,
    poisson_summary,
    weighted_poisson_loss,
)
from semflow_sr.flow.trace_cache import load_trace_cache, validate_task_record
from semflow_sr.one_step_fisher import (
    FISHER_TIME_BINS,
    block_fisher_squared_distance,
    fisher_rao_probability_path_and_logit_velocity,
    source_conditioned_trace_target_probabilities,
)
from semflow_sr.semantic_mass import semantic_signature_distance, semantic_signature_vector
from semflow_sr.sr.ast import Expr, eval_expr
from semflow_sr.sr.ops import NAME_TO_ID, get_op
from semflow_sr.sr.parser import parse_formula
from semflow_sr.sr.printer import to_string

from scripts.train_fixed_symbol_node_stage1 import (
    Block,
    FixedSymbolTemplate,
    active_block_indices_for_choices as fixed_symbol_active_block_indices_for_choices,
    apply_op,
    block_index,
    center_theta,
    execute_choices as fixed_symbol_execute_choices,
    integrate,
    op_arity,
    pack_blocks,
    readout_block_index,
    random_trace as random_fixed_symbol_trace,
    sanitize_values,
    simplex_path,
    split_blocks,
    terminal_summary,
    theta_dim,
    velocity_loss,
)


DEFAULT_OPS = (
    "copy",
    "add",
    "sub",
    "mul",
    "protected_div",
    "sin",
    "cos",
    "square",
    "cube",
    "protected_log",
    "protected_sqrt",
    "exp",
)

OBJECTIVE_NAME = "semantic_poisson_residual_fisher_bootstrap"


@dataclass(frozen=True)
class RegisterOperatorSimplexTemplate:
    """Register-operator simplex chart.

    Each layer has one operator-choice block and two register-argument blocks.
    The chosen operator writes to a fixed scratch register for that layer; the
    operator simplex includes a KEEP action that leaves the destination register
    unchanged.  Readout selects a final register.
    """

    num_vars: int
    num_layers: int
    ops: tuple[str, ...]
    output_terms: int = 1
    num_registers: int = 0

    @property
    def base_count(self) -> int:
        return int(self.num_vars) + 2

    @property
    def zero_source_index(self) -> int:
        return int(self.num_vars)

    @property
    def one_source_index(self) -> int:
        return int(self.num_vars) + 1

    @property
    def register_count(self) -> int:
        return int(self.num_registers) if int(self.num_registers) > 0 else int(self.base_count) + int(self.num_layers)

    @property
    def keep_action_index(self) -> int:
        return len(self.ops)

    @property
    def node_count(self) -> int:
        return int(self.num_layers)

    @property
    def source_count(self) -> int:
        return max(int(self.register_count), int(len(self.ops)) + 1)

    def write_register_for_layer(self, layer: int) -> int:
        scratch = int(self.base_count) + int(layer)
        return min(max(scratch, 0), int(self.register_count) - 1)

    @property
    def blocks(self) -> tuple[Block, ...]:
        rows: list[Block] = []
        for layer in range(int(self.num_layers)):
            write_reg = int(self.write_register_for_layer(layer))
            rows.append(Block("reg_op", layer=layer, node=write_reg, slot=-1, size=int(self.source_count)))
            rows.append(Block("reg_arg", layer=layer, node=write_reg, slot=0, size=int(self.source_count)))
            rows.append(Block("reg_arg", layer=layer, node=write_reg, slot=1, size=int(self.source_count)))
        for term in range(int(self.output_terms)):
            rows.append(Block("readout", layer=int(self.num_layers), term=term, size=int(self.source_count)))
        return tuple(rows)


def _is_register_template(template: Any) -> bool:
    return isinstance(template, RegisterOperatorSimplexTemplate)


def _is_fixed_symbol_template(template: Any) -> bool:
    return isinstance(template, FixedSymbolTemplate)


def _expr_is_zero(expr: Expr) -> bool:
    if expr.kind == "const":
        return bool(abs(float(expr.value or 0.0)) < 1.0e-12)
    try:
        return bool(sp.simplify(_sympify(_expr_simplified_key(expr))) == 0)
    except Exception:
        return False


def _expr_num_vars(expr: Expr) -> int:
    if expr.kind == "var":
        return int(expr.var_index) + 1
    if expr.kind != "op":
        return 0
    return max((_expr_num_vars(child) for child in expr.children), default=0)


def _expr_simplified_key(expr: Expr) -> str:
    return to_string(expr, max(_expr_num_vars(expr), 1), simplify=True)


def _unique_nonzero_terms(terms: list[Expr]) -> tuple[list[Expr], int]:
    """Drop zero readout terms and collapse duplicate expression fibers.

    The register graph can route the same sub-expression through multiple
    readout slots.  For the population-flow algorithm that should not create a
    fake multi-term individual, so expression-level readout semantics dedupe
    repeated terms before forming the summed expression.
    """
    kept: list[Expr] = []
    seen: set[str] = set()
    duplicates = 0
    for term in terms:
        if _expr_is_zero(term):
            continue
        key = _expr_simplified_key(term)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(term)
    return kept, int(duplicates)


def _sum_exprs(terms: list[Expr]) -> Expr:
    kept, _duplicates = _unique_nonzero_terms(terms)
    if not kept:
        return Expr.const(0.0)
    out = kept[0]
    for term in kept[1:]:
        out = Expr.op(NAME_TO_ID["add"], (out, term))
    return out


def register_op_block_index(template: RegisterOperatorSimplexTemplate, layer: int) -> int:
    return int(layer) * 3


def register_arg_block_index(template: RegisterOperatorSimplexTemplate, layer: int, slot: int) -> int:
    return int(layer) * 3 + 1 + int(slot)


def register_readout_block_index(template: RegisterOperatorSimplexTemplate, term: int) -> int:
    return int(template.num_layers) * 3 + int(term)


def register_readable_count(template: RegisterOperatorSimplexTemplate, layer: int) -> int:
    return min(int(template.base_count) + max(int(layer), 0), int(template.register_count))


def register_graph_action_mask(
    template: RegisterOperatorSimplexTemplate,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for block in template.blocks:
        mask = torch.zeros((int(block.size),), dtype=torch.bool, device=device)
        if block.kind == "reg_op":
            valid = min(int(len(template.ops)) + 1, int(block.size))
            mask[:valid] = True
        elif block.kind == "reg_arg":
            mask[:register_readable_count(template, int(block.layer))] = True
        elif block.kind == "readout":
            mask[:int(template.register_count)] = True
        else:
            mask[:] = True
        rows.append(mask)
    return torch.stack(rows, dim=0)


def execute_register_choices(
    template: RegisterOperatorSimplexTemplate,
    choices: list[int],
) -> tuple[Expr, list[Expr], list[list[Expr]]]:
    regs = [Expr.var(i) for i in range(int(template.num_vars))]
    regs.append(Expr.const(0.0))
    regs.append(Expr.const(1.0))
    while len(regs) < int(template.register_count):
        regs.append(Expr.const(0.0))
    layers: list[list[Expr]] = []
    for layer in range(int(template.num_layers)):
        op_bidx = register_op_block_index(template, layer)
        arg0_bidx = register_arg_block_index(template, layer, 0)
        arg1_bidx = register_arg_block_index(template, layer, 1)
        op_choice = int(choices[op_bidx]) if op_bidx < len(choices) else int(template.keep_action_index)
        write_reg = int(template.write_register_for_layer(layer))
        if 0 <= op_choice < len(template.ops):
            op = str(template.ops[op_choice])
            arity = op_arity(op)
            r0 = int(choices[arg0_bidx]) if arg0_bidx < len(choices) else 0
            r1 = int(choices[arg1_bidx]) if arg1_bidx < len(choices) else 0
            readable = max(register_readable_count(template, layer), 1)
            r0 = max(0, min(r0, readable - 1))
            r1 = max(0, min(r1, readable - 1))
            children = (regs[r0],) if arity == 1 else (regs[r0], regs[r1])
            regs[write_reg] = apply_op(op, children)
        layers.append([regs[write_reg]])
    terms: list[Expr] = []
    for term in range(int(template.output_terms)):
        bidx = register_readout_block_index(template, term)
        src = int(choices[bidx]) if bidx < len(choices) else int(template.zero_source_index)
        src = max(0, min(src, int(template.register_count) - 1))
        terms.append(regs[src])
    return _sum_exprs(terms), terms, layers


def active_register_block_indices_for_choices(
    template: RegisterOperatorSimplexTemplate,
    choices: list[int],
) -> list[int]:
    active: set[int] = set()
    visiting_layers: set[int] = set()
    visited_layers: set[int] = set()

    def visit_register(src: int) -> None:
        src = int(src)
        if src < int(template.base_count):
            return
        layer = src - int(template.base_count)
        if 0 <= layer < int(template.num_layers):
            visit_layer(layer)

    def visit_layer(layer: int) -> None:
        if int(layer) in visited_layers or int(layer) in visiting_layers:
            return
        visiting_layers.add(int(layer))
        op_bidx = register_op_block_index(template, layer)
        if not (0 <= op_bidx < len(choices)):
            visiting_layers.discard(int(layer))
            return
        op_choice = int(choices[op_bidx])
        if op_choice == int(template.keep_action_index) or not (0 <= op_choice < len(template.ops)):
            visiting_layers.discard(int(layer))
            visited_layers.add(int(layer))
            return
        active.add(op_bidx)
        arity = op_arity(str(template.ops[op_choice]))
        for slot in range(arity):
            arg_bidx = register_arg_block_index(template, layer, slot)
            active.add(arg_bidx)
            if 0 <= arg_bidx < len(choices):
                visit_register(int(choices[arg_bidx]))
        visiting_layers.discard(int(layer))
        visited_layers.add(int(layer))

    for term in range(int(template.output_terms)):
        bidx = register_readout_block_index(template, term)
        active.add(bidx)
        if 0 <= bidx < len(choices):
            visit_register(int(choices[bidx]))
    return sorted(active)


def execute_choices(template: Any, choices: list[int]) -> tuple[Expr, list[Expr], list[list[Expr]]]:
    if _is_register_template(template):
        return execute_register_choices(template, choices)
    return fixed_symbol_execute_choices(template, choices)


def active_block_indices_for_choices(template: Any, choices: list[int]) -> list[int]:
    if _is_register_template(template):
        return active_register_block_indices_for_choices(template, choices)
    return fixed_symbol_active_block_indices_for_choices(template, choices)


def random_trace(template: Any, rng: random.Random, *, max_depth_bias: float = 0.7) -> dict[str, Any]:
    if not _is_register_template(template):
        return random_fixed_symbol_trace(template, rng, max_depth_bias=float(max_depth_bias))
    choices = [0 for _ in template.blocks]
    for layer in range(int(template.num_layers)):
        op_bidx = register_op_block_index(template, layer)
        arg0_bidx = register_arg_block_index(template, layer, 0)
        arg1_bidx = register_arg_block_index(template, layer, 1)
        choices[op_bidx] = int(template.keep_action_index) if rng.random() < 0.25 else rng.randrange(max(len(template.ops), 1))
        readable = max(register_readable_count(template, layer), 1)
        choices[arg0_bidx] = rng.randrange(readable)
        choices[arg1_bidx] = rng.randrange(readable)
    for term in range(int(template.output_terms)):
        bidx = register_readout_block_index(template, term)
        if term == 0:
            choices[bidx] = rng.randrange(max(int(template.register_count), 1))
        else:
            choices[bidx] = int(template.zero_source_index)
    active = active_block_indices_for_choices(template, choices)
    expr, terms, layers = execute_choices(template, choices)
    return {
        "choices": choices,
        "active_block_indices": active,
        "block_weights": [1.0 if idx in set(active) else 0.0 for idx in range(len(template.blocks))],
        "expression": expr,
        "expression_string": to_string(expr, int(template.num_vars), simplify=False),
        "term_count": int(len(terms)),
        "active_block_count": int(len(active)),
        "node_expressions_by_layer": layers,
    }

CONSTRUCTION_GRAPHS = ("register_categorical_blocks",)


def canonical_construction_graph(name: str) -> str:
    value = str(name)
    if value not in CONSTRUCTION_GRAPHS:
        raise ValueError(f"unsupported construction graph: {value}")
    return value


def parse_ops(ops_csv: str) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in str(ops_csv).split(",") if str(value).strip())


def make_construction_template(args: argparse.Namespace, graph_family: str) -> Any:
    if str(graph_family) != "register_categorical_blocks":
        raise ValueError("the one-step mainline only supports register_categorical_blocks")
    ops = parse_ops(str(args.ops))
    return RegisterOperatorSimplexTemplate(
        num_vars=int(args.num_vars),
        num_layers=int(args.num_layers),
        ops=ops,
        output_terms=int(args.output_terms),
        num_registers=int(getattr(args, "num_registers", 0)),
    )


@dataclass
class TaskBundle:
    task_id: str
    suite: str
    split: str
    num_vars: int
    variable_names: list[str]
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    ground_truth: str
    traces: list[dict[str, Any]]
    compile_failures: list[str]


@dataclass
class CycleCoupledExample:
    task: TaskBundle
    theta0: torch.Tensor
    theta1: torch.Tensor
    active_mask: torch.Tensor
    proposal_index: int
    diagnostics: dict[str, Any]
    sample_weight: float = 1.0
    target_choices: tuple[int, ...] | None = None
    is_gt_anchor: bool = False


@dataclass
class ResidualFlowExample:
    task: TaskBundle
    theta0: torch.Tensor
    reference_theta1: torch.Tensor
    corrected_theta1: torch.Tensor
    source_index: int
    energy: float
    diagnostics: dict[str, Any]


def _resolve_device(name: str) -> torch.device:
    key = str(name or "auto").lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(key)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _limit_points(x: torch.Tensor, y: torch.Tensor, max_points: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if int(max_points) <= 0 or int(x.shape[0]) <= int(max_points):
        return x.float(), y.float()
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    idx = torch.randperm(int(x.shape[0]), generator=gen, device=x.device)[: int(max_points)]
    return x[idx].float(), y[idx].float()


def _pad_x(x: torch.Tensor, num_vars: int) -> torch.Tensor:
    x = x.float()
    if int(x.shape[1]) > int(num_vars):
        raise ValueError(f"task has {int(x.shape[1])} vars but template supports {int(num_vars)}")
    if int(x.shape[1]) == int(num_vars):
        return x
    pad = torch.zeros((int(x.shape[0]), int(num_vars) - int(x.shape[1])), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def _normalize_vec(y: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    y = sanitize_values(y.float())
    return (y - y.mean()) / y.std().clamp_min(float(eps))


def _target_semantic_signature(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return semantic_signature_vector(y.float(), x.float())


def _semantic_features(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    v = _normalize_vec(values)
    y = _normalize_vec(target)
    residual = v - y
    mse = (residual * residual).mean()
    corr = (v * y).mean()
    proj = (values.float() * target.float()).mean() / (target.float().pow(2).mean().clamp_min(1.0e-6))
    return torch.stack([
        mse.clamp(0.0, 1.0e6).log1p() / 8.0,
        corr.clamp(-10.0, 10.0) / 4.0,
        proj.clamp(-10.0, 10.0) / 4.0,
        values.float().mean().clamp(-1.0e6, 1.0e6).sign() * values.float().mean().abs().clamp_min(0.0).log1p() / 8.0,
        values.float().std().clamp(0.0, 1.0e6).log1p() / 8.0,
        residual.float().std().clamp(0.0, 1.0e6).log1p() / 8.0,
        torch.isfinite(values).float().mean(),
        torch.tensor(float(values.numel()), device=values.device, dtype=values.dtype).log1p() / 8.0,
    ])


def _semantic_features_batch(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values = sanitize_values(values.float())
    if values.ndim == 1:
        values = values[None, :]
    target = sanitize_values(target.float()).to(values.device)
    y = _normalize_vec(target)[None, :]
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True).clamp_min(1.0e-6)
    v = (values - mean) / std
    residual = v - y
    mse = (residual * residual).mean(dim=1)
    corr = (v * y).mean(dim=1)
    proj = (values * target[None, :]).mean(dim=1) / target.pow(2).mean().clamp_min(1.0e-6)
    raw_mean = values.mean(dim=1)
    raw_std = values.std(dim=1)
    residual_std = residual.std(dim=1)
    finite_rate = torch.isfinite(values).float().mean(dim=1)
    size_feature = torch.full_like(mse, float(values.shape[1])).log1p() / 8.0
    return torch.stack([
        mse.clamp(0.0, 1.0e6).log1p() / 8.0,
        corr.clamp(-10.0, 10.0) / 4.0,
        proj.clamp(-10.0, 10.0) / 4.0,
        raw_mean.clamp(-1.0e6, 1.0e6).sign() * raw_mean.abs().clamp_min(0.0).log1p() / 8.0,
        raw_std.clamp(0.0, 1.0e6).log1p() / 8.0,
        residual_std.clamp(0.0, 1.0e6).log1p() / 8.0,
        finite_rate,
        size_feature,
    ], dim=1)


def random_theta(
    template: FixedSymbolTemplate,
    *,
    scale: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    rows = []
    for block in template.blocks:
        logits = scale * torch.randn(int(block.size), device=device, generator=generator)
        rows.append(logits - logits.mean())
    return pack_blocks(rows)


def _stable_task_seed(task_id: str) -> int:
    digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def theta0_diagnostics(theta0: torch.Tensor, template: Any) -> dict[str, Any]:
    choices = hard_decode_choices(theta0, template)
    key = ",".join(str(int(value)) for value in choices)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return {
        "theta0_argmax_key": key,
        "theta0_hash": digest,
        "theta0_terminal_entropy_mean": float(terminal_summary(theta0, template).get("terminal_entropy_mean", 0.0)),
        "theta0_terminal_max_prob_mean": float(terminal_summary(theta0, template).get("terminal_max_prob_mean", 0.0)),
    }


def sample_eval_theta0(
    template: FixedSymbolTemplate,
    task: TaskBundle,
    args: argparse.Namespace,
    rng: random.Random,
    device: torch.device,
    *,
    sample_index: int = 0,
) -> tuple[torch.Tensor, str]:
    del rng
    mode = str(getattr(args, "eval_theta0_mode", "deterministic_random"))
    if mode == "deterministic_random":
        seed = int(args.seed) + 910_003 + 10_007 * int(sample_index) + _stable_task_seed(task.task_id)
        gen = torch.Generator(device=device).manual_seed(seed)
        return random_theta(template, scale=float(args.theta0_noise_scale), device=device, generator=gen), mode
    if mode == "random":
        return random_theta(template, scale=float(args.theta0_noise_scale), device=device), mode
    raise ValueError(f"unknown eval theta0 mode: {mode}")


def _inactive_default_choice(template: FixedSymbolTemplate, bidx: int) -> int:
    # Default unused graph edges/readouts to ZERO. For the register chart,
    # unused operator blocks should KEEP rather than accidentally selecting an
    # arithmetic op whose index happens to match the zero register.
    if _is_register_template(template):
        block = template.blocks[int(bidx)]
        if block.kind == "reg_op":
            return int(template.keep_action_index)
        return int(template.zero_source_index)
    return int(template.zero_source_index)


def target_theta_with_inactive_defaults(
    template: FixedSymbolTemplate,
    start: torch.Tensor,
    trace: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    start_blocks = split_blocks(start, template)
    choices = list(trace["choices"])
    active = set(int(v) for v in trace["active_block_indices"])
    mode = str(getattr(args, "inactive_block_target_mode", "zero"))
    inactive_weight = float(getattr(args, "inactive_block_loss_weight", 0.02))
    out: list[torch.Tensor] = []
    weights: list[float] = []
    for bidx, block in enumerate(template.blocks):
        if bidx in active:
            action = int(choices[bidx])
            weight = 1.0
        elif mode == "zero":
            action = _inactive_default_choice(template, bidx)
            weight = max(float(inactive_weight), 0.0)
        elif mode == "start":
            out.append(start_blocks[bidx].clone())
            weights.append(0.0)
            continue
        else:
            raise ValueError(f"unknown inactive block target mode: {mode}")
        logits = torch.full((int(block.size),), float(args.target_low), device=start.device)
        logits[max(0, min(int(action), int(block.size) - 1))] = float(args.target_high)
        out.append(logits - logits.mean())
        weights.append(float(weight))
    return pack_blocks(out).detach(), torch.tensor(weights, dtype=torch.float32, device=start.device)


def make_stage1_target(
    template: FixedSymbolTemplate,
    start: torch.Tensor,
    task: TaskBundle,
    trace: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    return target_theta_with_inactive_defaults(
        template,
        start,
        trace,
        args,
    )


def trace_endpoint_fisher_distance(
    template: Any,
    theta0: torch.Tensor,
    trace: dict[str, Any],
    args: argparse.Namespace,
) -> float:
    endpoint, weights = target_theta_with_inactive_defaults(template, theta0, trace, args)
    p0 = masked_block_softmax(theta0.view(len(template.blocks), int(template.source_count)), template)
    p1 = masked_block_softmax(endpoint.view(len(template.blocks), int(template.source_count)), template)
    active = weights > 0
    if not bool(active.any()):
        return 0.0
    affinity = (p0.sqrt() * p1.sqrt()).sum(dim=-1).clamp(-1.0, 1.0)
    distance = torch.acos(affinity).square()
    return float(distance[active].mean().detach().cpu())


def select_trace_for_theta0(
    template: Any,
    theta0: torch.Tensor,
    task: TaskBundle,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assign one endpoint basin to one source state.

    Multiple optima are represented by different initial states selecting
    different endpoints.  A single trajectory never predicts a mixture of
    complete expressions.
    """
    if not task.traces:
        raise ValueError(f"task {task.task_id} has no semantically valid traces")
    return min(task.traces, key=lambda trace: trace_endpoint_fisher_distance(template, theta0, trace, args))


def apply_predicted_active_endpoint_mask(template: Any, theta0: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
    rows = endpoint.view(len(template.blocks), int(template.source_count)).clone()
    seed_rows = theta0.view(len(template.blocks), int(template.source_count))
    valid_mask = graph_action_mask(template, device=rows.device)
    rows = rows.masked_fill(~valid_mask, -1.0e9)
    choices = rows.argmax(dim=-1).tolist()
    active = set(active_block_indices_for_choices(template, choices))
    for block_index_value in range(len(template.blocks)):
        if block_index_value not in active:
            rows[block_index_value] = seed_rows[block_index_value]
    return rows.reshape(-1)


def make_stage1_state_and_velocity(
    template: FixedSymbolTemplate,
    theta0: torch.Tensor,
    p1: torch.Tensor,
    t: float,
    args: argparse.Namespace,
    *,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    del args, rng, device
    theta_t, bridge_velocity = stage1_simplex_path(theta0.detach(), p1, template, float(t))
    return theta_t.detach(), bridge_velocity.detach()


def _uniform_chart(template: FixedSymbolTemplate) -> bool:
    return all(int(block.size) == int(template.source_count) for block in template.blocks)


def graph_action_mask(template: Any, *, device: torch.device | None = None) -> torch.Tensor:
    """Layer-valid source support for the fixed-symbol graph chart.

    At layer 0 there is no previous layer of computed symbol nodes.  The
    implementation historically represented those unavailable sources by zero
    placeholders, which bloats the simplex with degenerate actions and makes
    low-t velocity matching unnecessarily noisy.  The mask keeps the same theta
    chart while restricting q_theta to valid construction sources.
    """
    if _is_register_template(template):
        return register_graph_action_mask(template, device=device)
    rows = []
    for block in template.blocks:
        mask = torch.ones((int(block.size),), dtype=torch.bool, device=device)
        if block.kind == "edge" and int(block.layer) == 0:
            mask[int(template.base_count):] = False
        rows.append(mask)
    return torch.stack(rows, dim=0)


def graph_block_mask(template: Any, bidx: int, *, device: torch.device | None = None) -> torch.Tensor:
    return graph_action_mask(template, device=device)[int(bidx)]


def masked_single_block_softmax(logits: torch.Tensor, template: Any, bidx: int, eps: float = 1.0e-8) -> torch.Tensor:
    mask = graph_block_mask(template, int(bidx), device=logits.device)
    masked = logits.float().masked_fill(~mask, -1.0e9)
    p = torch.softmax(masked, dim=-1)
    p = torch.where(mask, p, torch.zeros_like(p))
    p = p.clamp_min(float(eps))
    p = torch.where(mask, p, torch.zeros_like(p))
    return p / p.sum().clamp_min(float(eps))


def masked_block_softmax(logits: torch.Tensor, template: Any, eps: float = 1.0e-8) -> torch.Tensor:
    if not _uniform_chart(template):
        raise ValueError("masked_block_softmax expects uniform fixed-symbol chart")
    block_count = len(template.blocks)
    source_count = int(template.source_count)
    rows = logits.float().flatten().view(block_count, source_count)
    mask = graph_action_mask(template, device=rows.device)
    masked = rows.masked_fill(~mask, -1.0e9)
    p = torch.softmax(masked, dim=-1)
    p = torch.where(mask, p, torch.zeros_like(p))
    p = p.clamp_min(float(eps))
    p = torch.where(mask, p, torch.zeros_like(p))
    return p / p.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def masked_block_softmax_batch(logits: torch.Tensor, template: Any, eps: float = 1.0e-8) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("batched logits must have shape [particle, block, action]")
    mask = graph_action_mask(template, device=logits.device).unsqueeze(0)
    masked = logits.float().masked_fill(~mask, -1.0e9)
    p = torch.softmax(masked, dim=-1)
    p = torch.where(mask, p, torch.zeros_like(p)).clamp_min(float(eps))
    p = torch.where(mask, p, torch.zeros_like(p))
    return p / p.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def terminal_summary(theta: torch.Tensor, template: Any, trace: dict[str, Any] | None = None) -> dict[str, float]:
    entropies: list[float] = []
    max_probs: list[float] = []
    argmax: list[int] = []
    active_probs: list[float] = []
    blocks = split_blocks(theta, template)
    for bidx, block in enumerate(blocks):
        p = masked_single_block_softmax(block.float(), template, int(bidx))
        support_count = max(int(graph_block_mask(template, int(bidx), device=p.device).sum().detach().cpu().item()), 2)
        entropies.append(float((-(p * p.clamp_min(1.0e-8).log()).sum() / math.log(support_count)).detach().cpu().item()))
        max_probs.append(float(p.max().detach().cpu().item()))
        argmax.append(int(torch.argmax(p).detach().cpu().item()))
    out = {
        "terminal_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "terminal_max_prob_mean": float(np.mean(max_probs)) if max_probs else 0.0,
    }
    if trace is not None:
        matches = []
        for idx in [int(v) for v in trace.get("active_block_indices", [])]:
            if idx < 0 or idx >= len(blocks) or idx >= len(trace.get("choices", [])):
                continue
            p = masked_single_block_softmax(blocks[idx].float(), template, idx)
            action = int(trace["choices"][idx])
            if 0 <= action < int(p.numel()):
                active_probs.append(float(p[action].detach().cpu().item()))
                matches.append(float(argmax[idx] == action))
        out.update({
            "active_target_prob_mean": float(np.mean(active_probs)) if active_probs else 0.0,
            "active_argmax_match_mean": float(np.mean(matches)) if matches else 0.0,
        })
    return out


def stage1_simplex_path(
    theta0: torch.Tensor,
    theta1: torch.Tensor,
    template: FixedSymbolTemplate,
    t: float,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _uniform_chart(template):
        return simplex_path(theta0, theta1, template, float(t), eps=float(eps))
    block_count = len(template.blocks)
    source_count = int(template.source_count)
    start = theta0.float().flatten().view(block_count, source_count)
    end = theta1.float().flatten().view(block_count, source_count)
    p0 = masked_block_softmax(start, template, eps=float(eps))
    p1 = masked_block_softmax(end, template, eps=float(eps))
    mask = graph_action_mask(template, device=start.device)
    p, velocity, _diagnostics = fisher_rao_probability_path_and_logit_velocity(
        p0,
        p1,
        float(t),
        eps=float(eps),
        support_mask=mask,
    )
    theta_t = p.clamp_min(float(eps)).log()
    valid_mean = (theta_t * mask.float()).sum(dim=-1, keepdim=True) / mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    theta_t = torch.where(mask, theta_t - valid_mean, torch.full_like(theta_t, -20.0))
    return theta_t.reshape(-1).detach(), velocity.reshape(-1).detach()


def stage1_velocity_loss(
    theta_t: torch.Tensor,
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    template: FixedSymbolTemplate,
    weights: torch.Tensor,
    eps: float = 1.0e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not _uniform_chart(template):
        return velocity_loss(theta_t, pred_v, target_v, template, weights, eps=float(eps))
    block_count = len(template.blocks)
    source_count = int(template.source_count)
    logits = theta_t.float().flatten().view(block_count, source_count)
    pred = pred_v.float().flatten().view(block_count, source_count)
    target = target_v.float().flatten().view(block_count, source_count)
    w = weights.to(logits.device).float().flatten()
    p = masked_block_softmax(logits, template, eps=float(eps))
    pred_dot = p * (pred - (p * pred).sum(dim=-1, keepdim=True))
    target_dot = p * (target.detach() - (p * target.detach()).sum(dim=-1, keepdim=True))
    diff = pred_dot - target_dot
    block_loss = ((diff * diff) / p.clamp_min(float(eps))).sum(dim=-1)
    loss = (w * block_loss).sum() / w.sum().clamp_min(1.0)
    active = w > 0
    active_max = p.max(dim=-1).values[active]
    return loss, {
        "active_block_count": float(w.sum().detach().cpu().item()),
        "active_max_prob_mean": float(active_max.mean().detach().cpu().item()) if int(active_max.numel()) else 0.0,
    }


def stage1_velocity_block_losses(
    theta_t: torch.Tensor,
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    template: Any,
    *,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    """Return un-reduced per-block Fisher probability-tangent errors."""
    if not _uniform_chart(template):
        raise ValueError("block diagnostics require a uniform register chart")
    block_count = len(template.blocks)
    source_count = int(template.source_count)
    logits = theta_t.float().flatten().view(block_count, source_count)
    predicted = pred_v.float().flatten().view(block_count, source_count)
    target = target_v.float().flatten().view(block_count, source_count)
    probabilities = masked_block_softmax(logits, template, eps=float(eps))
    predicted_tangent = probabilities * (
        predicted - (probabilities * predicted).sum(dim=-1, keepdim=True)
    )
    target_tangent = probabilities * (
        target.detach() - (probabilities * target.detach()).sum(dim=-1, keepdim=True)
    )
    difference = predicted_tangent - target_tangent
    return ((difference * difference) / probabilities.clamp_min(float(eps))).sum(dim=-1)


def velocity_alignment_diagnostics(
    theta_t: torch.Tensor,
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    template: FixedSymbolTemplate,
    weights: torch.Tensor,
    eps: float = 1.0e-4,
) -> dict[str, float]:
    if not _uniform_chart(template):
        return {}
    block_count = len(template.blocks)
    source_count = int(template.source_count)
    logits = theta_t.float().flatten().view(block_count, source_count)
    pred = pred_v.float().flatten().view(block_count, source_count)
    target = target_v.float().flatten().view(block_count, source_count)
    w = weights.to(logits.device).float().flatten() > 0
    if not bool(w.any().detach().cpu().item()):
        return {
            "pred_fr_norm_mean": 0.0,
            "target_fr_norm_mean": 0.0,
            "pred_target_cosine_mean": 0.0,
            "pred_target_norm_ratio_mean": 0.0,
        }
    p = masked_block_softmax(logits, template, eps=float(eps))
    pred_dot = p * (pred - (p * pred).sum(dim=-1, keepdim=True))
    target_dot = p * (target - (p * target).sum(dim=-1, keepdim=True))
    inner = ((pred_dot * target_dot) / p.clamp_min(float(eps))).sum(dim=-1)
    pred_norm = torch.sqrt(((pred_dot * pred_dot) / p.clamp_min(float(eps))).sum(dim=-1).clamp_min(0.0))
    target_norm = torch.sqrt(((target_dot * target_dot) / p.clamp_min(float(eps))).sum(dim=-1).clamp_min(0.0))
    cosine = inner / (pred_norm * target_norm).clamp_min(1.0e-8)
    ratio = pred_norm / target_norm.clamp_min(1.0e-8)
    return {
        "pred_fr_norm_mean": float(pred_norm[w].mean().detach().cpu().item()),
        "target_fr_norm_mean": float(target_norm[w].mean().detach().cpu().item()),
        "pred_target_cosine_mean": float(cosine[w].mean().detach().cpu().item()),
        "pred_target_norm_ratio_mean": float(ratio[w].mean().detach().cpu().item()),
    }


def bridge_target_diagnostics(theta_t: torch.Tensor, target_v: torch.Tensor, template: FixedSymbolTemplate, weights: torch.Tensor, eps: float) -> dict[str, float]:
    if _uniform_chart(template):
        block_count = len(template.blocks)
        source_count = int(template.source_count)
        logits = theta_t.float().flatten().view(block_count, source_count)
        target = target_v.float().flatten().view(block_count, source_count)
        w = weights.to(logits.device).float().flatten() > 0
        if not bool(w.any().detach().cpu().item()):
            return {"bridge_target_fr_norm_mean": 0.0, "bridge_target_logit_norm_mean": 0.0}
        p = masked_block_softmax(logits, template, eps=float(eps))
        target_dot = p * (target - (p * target).sum(dim=-1, keepdim=True))
        fr = torch.sqrt(((target_dot * target_dot) / p.clamp_min(float(eps))).sum(dim=-1))
        logit = target.pow(2).mean(dim=-1).sqrt()
        return {
            "bridge_target_fr_norm_mean": float(fr[w].mean().detach().cpu().item()),
            "bridge_target_logit_norm_mean": float(logit[w].mean().detach().cpu().item()),
        }
    fr_norms: list[float] = []
    logit_norms: list[float] = []
    for logits, target, weight in zip(split_blocks(theta_t, template), split_blocks(target_v, template), weights):
        if float(weight.detach().cpu().item()) <= 0.0:
            continue
        p = torch.softmax(logits, dim=-1)
        target_dot = p * (target - (p * target).sum())
        fr = torch.sqrt(((target_dot * target_dot) / p.clamp_min(float(eps))).sum()).detach()
        fr_norms.append(float(fr.cpu().item()))
        logit_norms.append(float(target.detach().pow(2).mean().sqrt().cpu().item()))
    return {
        "bridge_target_fr_norm_mean": float(np.mean(fr_norms)) if fr_norms else 0.0,
        "bridge_target_logit_norm_mean": float(np.mean(logit_norms)) if logit_norms else 0.0,
    }


def differentiable_bridge_velocity(
    theta0: torch.Tensor,
    theta1: torch.Tensor,
    template: FixedSymbolTemplate,
    t: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if _uniform_chart(template):
        block_count = len(template.blocks)
        source_count = int(template.source_count)
        start = theta0.float().flatten().view(block_count, source_count)
        end = theta1.float().flatten().view(block_count, source_count)
        tt = torch.as_tensor(float(t), dtype=start.dtype, device=start.device)
        p0 = masked_block_softmax(start, template, eps=float(eps))
        p1 = masked_block_softmax(end, template, eps=float(eps))
        mask = graph_action_mask(template, device=start.device)
        _probability, velocity, _diagnostics = fisher_rao_probability_path_and_logit_velocity(
            p0,
            p1,
            float(tt.detach().cpu().item()),
            eps=float(eps),
            support_mask=mask,
        )
        return velocity.reshape(-1)
    rows: list[torch.Tensor] = []
    tt = torch.as_tensor(float(t), dtype=theta0.dtype, device=theta0.device)
    for start, end in zip(split_blocks(theta0, template), split_blocks(theta1, template)):
        p0 = torch.softmax(start, dim=-1).clamp_min(float(eps))
        p0 = p0 / p0.sum()
        p1 = torch.softmax(end, dim=-1).clamp_min(float(eps))
        p1 = p1 / p1.sum()
        r0 = p0.sqrt()
        r1 = p1.sqrt()
        dot = (r0 * r1).sum().clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega).clamp_min(1.0e-6)
        a = torch.sin((1.0 - tt) * omega) / sin_omega
        b = torch.sin(tt * omega) / sin_omega
        da = -omega * torch.cos((1.0 - tt) * omega) / sin_omega
        db = omega * torch.cos(tt * omega) / sin_omega
        r = a * r0 + b * r1
        dr = da * r0 + db * r1
        p = (r * r).clamp_min(float(eps))
        p = p / p.sum()
        dp = 2.0 * r * dr
        dp = dp - p * dp.sum()
        velocity = dp / p.clamp_min(float(eps))
        rows.append(velocity - velocity.mean())
    return center_theta(pack_blocks(rows), template)


def endpoint_attractor_velocity(
    theta: torch.Tensor,
    endpoint: torch.Tensor,
    template: FixedSymbolTemplate,
    t: float,
    *,
    min_remaining: float,
) -> torch.Tensor:
    remaining = max(1.0 - float(t), float(min_remaining), 1.0e-4)
    local_v = differentiable_bridge_velocity(theta, endpoint, template, 1.0e-6)
    return center_theta(local_v / float(remaining), template).detach()


def sample_time(rng: random.Random, mode: str, low_prob: float, low_max: float) -> float:
    if str(mode) == "low_t_mixture" and rng.random() < float(low_prob):
        return rng.random() * float(low_max)
    return rng.random()


def sample_cycle_time(
    rng: random.Random,
    sample_index: int,
    mode: str,
    *,
    inherited_mode: str,
    low_prob: float,
    low_max: float,
) -> float:
    if str(mode) == "stratified_fisher":
        low, high = FISHER_TIME_BINS[int(sample_index) % len(FISHER_TIME_BINS)]
        return float(low + (high - low) * rng.random())
    if str(mode) == "inherit":
        return sample_time(rng, str(inherited_mode), float(low_prob), float(low_max))
    raise ValueError(f"unknown cycle time sampling: {mode}")


def load_tasks(args: argparse.Namespace, template_num_vars: int, device: torch.device) -> tuple[list[SRTask], list[SRTask]]:
    manifest = load_benchmark_manifest(args.manifest)
    suites = list(args.suites or [])
    specs = []
    for suite in suites:
        specs.extend(manifest.suites.get(str(suite), []))
    if not specs:
        for items in manifest.suites.values():
            specs.extend(items)
    tasks: list[SRTask] = []
    for spec in specs:
        if spec.ground_truth is None:
            continue
        if int(spec.num_vars) > int(template_num_vars):
            continue
        try:
            task = load_materialized_task(spec, root=args.manifest_root)
        except Exception:
            continue
        x_train = _pad_x(torch.tensor(task.X_train, dtype=torch.float32, device=device), template_num_vars)
        x_test = _pad_x(torch.tensor(task.X_test, dtype=torch.float32, device=device), template_num_vars)
        y_train = torch.tensor(task.y_train, dtype=torch.float32, device=device)
        y_test = torch.tensor(task.y_test, dtype=torch.float32, device=device)
        x_train, y_train = _limit_points(x_train, y_train, int(args.max_train_points), int(args.seed))
        x_test, y_test = _limit_points(x_test, y_test, int(args.max_eval_points), int(args.seed) + 17)
        tasks.append(SRTask(
            task.name,
            x_train.detach().cpu().numpy(),
            y_train.detach().cpu().numpy(),
            x_test.detach().cpu().numpy(),
            y_test.detach().cpu().numpy(),
            task.expression,
            list(task.variable_names),
            dict(task.metadata),
        ))
    task_filter = {value.strip() for value in str(getattr(args, "task_id_filter", "")).split(",") if value.strip()}
    if task_filter:
        tasks = [task for task in tasks if str(task.name) in task_filter]
    train: list[SRTask] = []
    eval_: list[SRTask] = []
    for task in tasks:
        h = int(hashlib.sha1(str(task.name).encode("utf-8")).hexdigest()[:8], 16)
        if (h % 1000) < int(1000 * float(args.eval_fraction)):
            eval_.append(task)
        else:
            train.append(task)
    if int(args.train_task_limit) > 0:
        train = train[: int(args.train_task_limit)]
    if int(args.eval_task_limit) > 0:
        eval_ = eval_[: int(args.eval_task_limit)]
    if not eval_ and train and not bool(getattr(args, "allow_empty_eval", False)):
        eval_.append(train.pop())
    overlap = {t.name for t in train} & {t.name for t in eval_}
    if overlap:
        raise RuntimeError(f"train/eval leakage: {sorted(overlap)[:5]}")
    return train, eval_


def load_all_task_sources(args: argparse.Namespace, template_num_vars: int, device: torch.device) -> tuple[list[SRTask], list[SRTask], dict[str, int]]:
    train, eval_ = load_tasks(args, template_num_vars, device)
    counts = {
        "benchmark_train_task_count": int(len(train)),
        "benchmark_eval_task_count": int(len(eval_)),
        "symbolicgpt_train_task_count": 0,
        "symbolicgpt_eval_task_count": 0,
    }
    if str(args.symbolicgpt_root):
        rng = random.Random(int(args.seed) + 202)
        train_limit = None if int(args.symbolicgpt_train_limit) <= 0 else int(args.symbolicgpt_train_limit)
        sym_train = load_symbolicgpt_subset_tasks(
            args.symbolicgpt_root,
            splits=("train",),
            limit=train_limit,
            rng=rng,
            train_fraction=float(args.symbolicgpt_point_train_fraction),
        )
        sym_train = [task for task in sym_train if int(task.X_train.shape[1]) <= int(template_num_vars)]
        train.extend(sym_train)
        counts["symbolicgpt_train_task_count"] = int(len(sym_train))
        if int(args.symbolicgpt_eval_limit) > 0:
            eval_splits = tuple(str(v).strip() for v in str(args.symbolicgpt_eval_splits).split(",") if str(v).strip())
            sym_eval = load_symbolicgpt_subset_tasks(
                args.symbolicgpt_root,
                splits=eval_splits,
                limit=int(args.symbolicgpt_eval_limit),
                rng=random.Random(int(args.seed) + 303),
                train_fraction=float(args.symbolicgpt_point_train_fraction),
            )
            sym_eval = [task for task in sym_eval if int(task.X_train.shape[1]) <= int(template_num_vars)]
            eval_.extend(sym_eval)
            counts["symbolicgpt_eval_task_count"] = int(len(sym_eval))
    overlap = {task.name for task in train} & {task.name for task in eval_}
    if overlap:
        raise RuntimeError(f"train/eval leakage after adding task sources: {sorted(overlap)[:5]}")
    return train, eval_, counts


def _clone_variant(expr: Expr, rng: random.Random) -> Expr:
    """Clone an expression without changing its semantics.

    The previous implementation randomly removed additive/multiplicative
    constants and collapsed every non-zero constant to one.  That produced
    easy-to-fit labels which were not ground truth expressions.
    """
    if expr.kind == "var":
        return Expr.var(int(expr.var_index))
    if expr.kind == "const":
        return Expr.const(float(expr.value or 0.0))
    op = get_op(int(expr.op_id)).name
    children = [_clone_variant(child, rng) for child in expr.children]
    if op in {"add", "mul"} and len(children) == 2 and rng.random() < 0.5:
        children = [children[1], children[0]]
    if op not in NAME_TO_ID:
        raise ValueError(f"unsupported op {op}")
    return Expr.op(NAME_TO_ID[op], tuple(children))


def _expr_key(expr: Expr) -> str:
    if expr.kind == "var":
        return f"x{expr.var_index}"
    if expr.kind == "const":
        return f"const:{float(expr.value or 0.0):.17g}"
    return f"{get_op(int(expr.op_id)).name}(" + ",".join(_expr_key(child) for child in expr.children) + ")"


def _canonical_register_expr(expr: Expr) -> Expr:
    """Canonicalize commutative children before deterministic SSA emission."""
    if expr.kind == "var":
        return Expr.var(int(expr.var_index))
    if expr.kind == "const":
        return Expr.const(float(expr.value or 0.0))
    op = get_op(int(expr.op_id)).name
    if op not in NAME_TO_ID:
        raise ValueError(f"unsupported op {op}")
    children = [_canonical_register_expr(child) for child in expr.children]
    if op in {"add", "mul"}:
        children = sorted(children, key=_expr_key)
    return Expr.op(NAME_TO_ID[op], tuple(children))


def _depth(expr: Expr) -> int:
    if expr.kind != "op":
        return 0
    return 1 + max(_depth(child) for child in expr.children)


def _flatten_add_terms(expr: Expr) -> list[Expr]:
    if expr.kind == "op" and get_op(int(expr.op_id)).name == "add" and len(expr.children) == 2:
        terms: list[Expr] = []
        for child in expr.children:
            terms.extend(_flatten_add_terms(child))
        return terms
    return [expr]


def _base_source(expr: Expr, template: Any) -> int | None:
    if expr.kind == "var" and int(expr.var_index) < int(template.num_vars):
        return int(expr.var_index)
    if expr.kind == "const":
        value = float(expr.value or 0.0)
        if abs(value) < 1.0e-12:
            return int(template.zero_source_index)
        if abs(value - 1.0) < 1.0e-12:
            return int(template.one_source_index)
        raise ValueError(f"non-binary constant {value:g} is not representable in the discrete chart")
    return None


def _choose_copy(candidates: list[int], rng: random.Random, mode: str) -> int:
    if not candidates:
        raise ValueError("empty copy candidate set")
    if str(mode) == "random":
        return int(rng.choice(candidates))
    if str(mode) != "canonical":
        raise ValueError(f"unknown trace copy assignment: {mode}")
    return int(sorted(candidates)[0])


def compile_expr_to_trace(
    template: FixedSymbolTemplate,
    expr: Expr,
    rng: random.Random,
    *,
    copy_assignment: str = "canonical",
) -> dict[str, Any]:
    expr = _clone_variant(expr, rng)
    root_depth = _depth(expr)
    if root_depth <= 0:
        choices = [0 for _ in template.blocks]
        base_source = _base_source(expr, template)
        choices[readout_block_index(template, 0)] = int(template.zero_source_index if base_source is None else base_source)
        for term in range(1, int(template.output_terms)):
            choices[readout_block_index(template, term)] = int(template.zero_source_index)
        active = active_block_indices_for_choices(template, choices)
        decoded, _, layers = execute_choices(template, choices)
        return _trace_payload(template, choices, active, decoded, layers)
    if root_depth > int(template.num_layers):
        raise ValueError(f"expr depth {root_depth} exceeds layers {template.num_layers}")
    offset = int(template.num_layers) - int(root_depth)
    assignments: dict[str, tuple[int, int, Expr]] = {}
    used: set[tuple[int, int]] = set()

    def assign(node: Expr) -> None:
        if node.kind != "op":
            return
        key = _expr_key(node)
        if key in assignments:
            return
        op = get_op(int(node.op_id)).name
        if op not in template.ops:
            raise ValueError(f"op {op} not in fixed-symbol template")
        layer = offset + _depth(node) - 1
        candidate_nodes = [idx for idx, name in enumerate(template.ops) if name == op]
        free_nodes = [idx for idx in candidate_nodes if (int(layer), int(idx)) not in used]
        if not free_nodes:
            raise ValueError(f"node collision at layer={layer} op={op}")
        node_idx = _choose_copy(free_nodes, rng, str(copy_assignment))
        loc = (int(layer), int(node_idx))
        used.add(loc)
        assignments[key] = (int(layer), int(node_idx), node)
        for child in node.children:
            assign(child)

    assign(expr)
    choices = [0 for _ in template.blocks]
    copy_nodes = [idx for idx, name in enumerate(template.ops) if name == "copy"]
    carry: dict[tuple[str, int], int] = {}

    def source_for(node: Expr, target_layer: int) -> int:
        base = _base_source(node, template)
        if base is not None:
            return int(base)
        key = _expr_key(node)
        if key not in assignments:
            raise ValueError(f"missing assignment for {key}")
        layer, node_idx, _node = assignments[key]
        if layer == int(target_layer) - 1:
            return int(template.base_count + node_idx)
        if layer >= int(target_layer):
            raise ValueError(f"node {key} assigned too late for layer {target_layer}")
        if not copy_nodes:
            raise ValueError("copy op required for carry")
        last_source = int(template.base_count + node_idx)
        for layer_i in range(int(layer) + 1, int(target_layer)):
            ckey = (key, layer_i)
            if ckey in carry:
                last_source = int(template.base_count + carry[ckey])
                continue
            free_copy = [idx for idx in copy_nodes if (int(layer_i), int(idx)) not in used]
            if not free_copy:
                raise ValueError(f"copy carry collision at layer {layer_i}")
            copy_idx = _choose_copy(free_copy, rng, str(copy_assignment))
            bidx = block_index(template, layer=layer_i, node=copy_idx, slot=0)
            if choices[bidx] not in {0, last_source}:
                raise ValueError(f"copy carry conflict at layer {layer_i}")
            choices[bidx] = int(last_source)
            used.add((int(layer_i), int(copy_idx)))
            carry[ckey] = int(copy_idx)
            last_source = int(template.base_count + copy_idx)
        return last_source

    for key, (layer, node_idx, node) in sorted(assignments.items(), key=lambda item: item[1][0]):
        arity = op_arity(template.ops[node_idx])
        if arity != len(node.children):
            raise ValueError(f"arity mismatch for {key}")
        for slot, child in enumerate(node.children):
            choices[block_index(template, layer=layer, node=node_idx, slot=slot)] = source_for(child, layer)
    choices[readout_block_index(template, 0)] = source_for(expr, int(template.num_layers))
    for term in range(1, int(template.output_terms)):
        choices[readout_block_index(template, term)] = int(template.zero_source_index)
    active = active_block_indices_for_choices(template, choices)
    decoded, _, layers = execute_choices(template, choices)
    return _trace_payload(template, choices, active, decoded, layers)


def compile_expr_to_register_trace(
    template: RegisterOperatorSimplexTemplate,
    expr: Expr,
    rng: random.Random,
    *,
    copy_assignment: str = "canonical",
) -> dict[str, Any]:
    del copy_assignment
    expr = _canonical_register_expr(expr)
    flattened_terms = _flatten_add_terms(expr)
    term_plans: list[list[Expr]] = [[expr]]
    flattened_keys = [_expr_key(term) for term in flattened_terms]
    if (
        int(template.output_terms) > 1
        and 1 < len(flattened_terms) <= int(template.output_terms)
        and len(set(flattened_keys)) == len(flattened_keys)
    ):
        term_plans.insert(0, flattened_terms)

    first_error: Exception | None = None
    for term_plan in term_plans:
        choices = [0 for _ in template.blocks]
        next_layer = [0]
        expr_to_register: dict[str, int] = {}
        cse_reuse_count = [0]

        def emit(node: Expr) -> int:
            base = _base_source(node, template)
            if base is not None:
                return int(base)
            if node.kind != "op":
                raise ValueError(f"unsupported expression node kind: {node.kind}")
            key = _expr_key(node)
            if key in expr_to_register:
                cse_reuse_count[0] += 1
                return int(expr_to_register[key])
            if next_layer[0] >= int(template.num_layers):
                raise ValueError(f"expr operation count exceeds register layers {template.num_layers}")
            op = get_op(int(node.op_id)).name
            if op not in template.ops:
                raise ValueError(f"op {op} not in register template")
            child_regs = [emit(child) for child in node.children]
            layer = int(next_layer[0])
            next_layer[0] += 1
            op_bidx = register_op_block_index(template, layer)
            choices[op_bidx] = int(list(template.ops).index(op))
            for slot in range(2):
                arg_bidx = register_arg_block_index(template, layer, slot)
                if slot < len(child_regs):
                    choices[arg_bidx] = int(child_regs[slot])
                else:
                    choices[arg_bidx] = 0
            out_reg = int(template.write_register_for_layer(layer))
            expr_to_register[key] = out_reg
            return out_reg

        try:
            term_registers = [emit(term) for term in term_plan]
            for layer in range(next_layer[0], int(template.num_layers)):
                choices[register_op_block_index(template, layer)] = int(template.keep_action_index)
                choices[register_arg_block_index(template, layer, 0)] = 0
                choices[register_arg_block_index(template, layer, 1)] = 0
            for term_index in range(int(template.output_terms)):
                bidx = register_readout_block_index(template, term_index)
                if term_index < len(term_registers):
                    choices[bidx] = int(term_registers[term_index])
                else:
                    choices[bidx] = int(template.zero_source_index)
            # Equivalent traces preserve canonical SSA/CSE and vary only
            # commutative argument order.  This creates distinct legal graph
            # fibers without changing the decoded expression semantics.
            for layer in range(int(next_layer[0])):
                op_choice = int(choices[register_op_block_index(template, layer)])
                if 0 <= op_choice < len(template.ops) and str(template.ops[op_choice]) in {"add", "mul"}:
                    if rng.random() < 0.5:
                        left = register_arg_block_index(template, layer, 0)
                        right = register_arg_block_index(template, layer, 1)
                        choices[left], choices[right] = choices[right], choices[left]
            active = active_block_indices_for_choices(template, choices)
            decoded, terms, layers = execute_choices(template, choices)
            unique_terms, duplicate_terms = _unique_nonzero_terms(terms)
            payload = _trace_payload(template, choices, active, decoded, layers)
            payload["readout_slot_count"] = int(len(terms))
            payload["readout_term_count"] = int(sum(not _expr_is_zero(term) for term in terms))
            payload["unique_nonzero_term_count"] = int(len(unique_terms))
            payload["duplicate_term_count"] = int(duplicate_terms)
            payload["additive_decomposition_term_count"] = int(len(term_plan))
            payload["additive_decomposition_used"] = bool(len(term_plan) > 1)
            payload["canonical_ssa"] = True
            payload["ssa_operation_count"] = int(next_layer[0])
            payload["cse_reuse_count"] = int(cse_reuse_count[0])
            payload["term_expression_strings"] = [to_string(term, int(template.num_vars), simplify=False) for term in unique_terms]
            return payload
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
    raise first_error if first_error is not None else ValueError("register trace compilation failed")


def _trace_payload(
    template: FixedSymbolTemplate,
    choices: list[int],
    active: list[int],
    decoded: Expr,
    layers: list[list[Expr]],
) -> dict[str, Any]:
    return {
        "choices": list(map(int, choices)),
        "active_block_indices": list(map(int, active)),
        "block_weights": [1.0 if idx in set(active) else 0.0 for idx in range(len(template.blocks))],
        "expression": decoded,
        "expression_string": to_string(decoded, int(template.num_vars), simplify=False),
        "active_block_count": int(len(active)),
        "node_expressions_by_layer": layers,
    }


def compile_task_traces(
    task: SRTask,
    template: Any,
    *,
    k: int,
    seed: int,
    copy_assignment: str = "canonical",
) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    if not task.expression:
        return [], ["missing_ground_truth"]
    rng = random.Random(int(seed))
    try:
        expr = parse_formula(str(task.expression), list(task.variable_names))
    except Exception as exc:
        return [], [f"parse:{type(exc).__name__}:{exc}"]
    traces: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = max(int(k) * 8, 16)
    for _ in range(attempts):
        try:
            if _is_register_template(template):
                trace = compile_expr_to_register_trace(template, expr, rng, copy_assignment=str(copy_assignment))
            else:
                trace = compile_expr_to_trace(template, expr, rng, copy_assignment=str(copy_assignment))
        except Exception as exc:
            failures.append(f"compile:{type(exc).__name__}:{str(exc)[:120]}")
            continue
        key = tuple(trace["choices"])
        if key in seen:
            continue
        seen.add(key)
        decoded_values = sanitize_values(eval_expr(trace["expression"], torch.tensor(task.X_train, dtype=torch.float32)))
        target_values = sanitize_values(torch.tensor(task.y_train, dtype=torch.float32))
        raw_fit = float(r2_score(target_values.cpu().numpy(), decoded_values.cpu().numpy()))
        if not math.isfinite(raw_fit) or raw_fit < 0.999999:
            failures.append(f"semantic_oracle:raw_r2={raw_fit:.9g}:{trace['expression_string']}")
            continue
        trace["semantic_oracle_raw_r2"] = raw_fit
        traces.append(trace)
        if len(traces) >= int(k):
            break
    return traces, failures


def build_task_bundles(
    tasks: list[SRTask],
    template: FixedSymbolTemplate,
    *,
    traces_per_task: int,
    max_train_points: int,
    max_eval_points: int,
    device: torch.device,
    seed: int,
    split: str,
    copy_assignment: str,
    cached_records: dict[str, dict[str, Any]] | None = None,
    require_cache: bool = False,
) -> list[TaskBundle]:
    bundles: list[TaskBundle] = []
    for idx, task in enumerate(tasks):
        x_train = _pad_x(torch.tensor(task.X_train, dtype=torch.float32, device=device), template.num_vars)
        y_train = torch.tensor(task.y_train, dtype=torch.float32, device=device)
        x_test = _pad_x(torch.tensor(task.X_test, dtype=torch.float32, device=device), template.num_vars)
        y_test = torch.tensor(task.y_test, dtype=torch.float32, device=device)
        task_seed = int(seed) + _stable_task_seed(str(task.name))
        x_train, y_train = _limit_points(x_train, y_train, int(max_train_points), task_seed)
        x_test, y_test = _limit_points(x_test, y_test, int(max_eval_points), task_seed + 1000)
        bundle = TaskBundle(
            task_id=str(task.name),
            suite=str(task.metadata.get("suite", "unknown")),
            split=str(split),
            num_vars=int(task.X_train.shape[1]),
            variable_names=list(task.variable_names),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            ground_truth=str(task.expression or task.metadata.get("ground_truth", "")),
            traces=[],
            compile_failures=[],
        )
        record = None if cached_records is None else cached_records.get(bundle.task_id)
        if record is not None:
            validate_task_record(bundle, template, record)
            for cached in list(record.get("traces", []))[: int(traces_per_task)]:
                choices = [int(value) for value in cached["choices"]]
                decoded, _terms, layers = execute_choices(template, choices)
                trace = _trace_payload(
                    template,
                    choices,
                    [int(value) for value in cached["active_block_indices"]],
                    decoded,
                    layers,
                )
                trace.update({key: value for key, value in cached.items() if key not in {"choices", "active_block_indices"}})
                bundle.traces.append(trace)
            bundle.compile_failures = [str(value) for value in record.get("compile_failures", [])]
        elif require_cache:
            raise ValueError(f"compiled trace cache is missing task {bundle.task_id}")
        else:
            bundle.traces, bundle.compile_failures = compile_task_traces(
                task,
                template,
                k=int(traces_per_task),
                seed=task_seed + 7919,
                copy_assignment=str(copy_assignment),
            )
        if require_cache and not bundle.traces:
            raise ValueError(f"compiled trace cache has no usable trace for {bundle.task_id}")
        bundles.append(bundle)
    return bundles


def _task_stat_dim(num_vars: int) -> int:
    n = int(num_vars)
    return 6 + 8 * n + n * (n + 1) // 2


def _signed_log_value(value: torch.Tensor) -> torch.Tensor:
    value = sanitize_values(value.float())
    return value.sign() * value.abs().clamp_min(0.0).log1p() / 8.0


def _normalized_corr_feature(values: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    values = sanitize_values(values.float())
    v = _normalize_vec(values)
    return (v * target_norm).mean().clamp(-10.0, 10.0) / 4.0


def task_stat_features(x: torch.Tensor, y: torch.Tensor, num_vars: int) -> torch.Tensor:
    x = _pad_x(x.float(), int(num_vars))
    y = sanitize_values(y.float())
    y_norm = _normalize_vec(y)
    y_center = y - y.mean()
    y_std = y.std().clamp_min(1.0e-6)
    y_z = y_center / y_std
    rows: list[torch.Tensor] = [
        _signed_log_value(y.mean()),
        y.std().clamp(0.0, 1.0e6).log1p() / 8.0,
        _signed_log_value(y.min()),
        _signed_log_value(y.max()),
        (y_z.pow(3).mean()).clamp(-10.0, 10.0) / 10.0,
        (y_z.pow(4).mean()).clamp(0.0, 100.0).log1p() / 8.0,
    ]
    for idx in range(int(num_vars)):
        z = sanitize_values(x[:, idx])
        basis = [
            z,
            z * z,
            z * z * z,
            torch.sin(z),
            torch.cos(z),
            torch.log1p(z.abs()),
            torch.sqrt(z.abs().clamp_min(0.0) + 1.0e-6),
            1.0 / (1.0 + z.abs()),
        ]
        rows.extend(_normalized_corr_feature(item, y_norm) for item in basis)
    for i in range(int(num_vars)):
        for j in range(i, int(num_vars)):
            rows.append(_normalized_corr_feature(x[:, i] * x[:, j], y_norm))
    return torch.stack(rows, dim=0).float()


class TaskSemanticEncoder(nn.Module):
    def __init__(self, num_vars: int, hidden: int, mode: str = "point_mlp"):
        super().__init__()
        self.num_vars = int(num_vars)
        self.mode = str(mode)
        if self.mode not in {"point_mlp", "stats", "hybrid_stats"}:
            raise ValueError(f"unknown task encoder mode: {self.mode}")
        self.net = nn.Sequential(
            nn.Linear(int(num_vars) + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.stats_net = nn.Sequential(
            nn.Linear(_task_stat_dim(int(num_vars)), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.hybrid_net = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = _pad_x(x.float(), self.num_vars)
        y_norm = _normalize_vec(y.float())
        raw_mean = y.float().mean()
        raw_std = y.float().std(unbiased=False).clamp_min(1.0e-6)
        mean_feature = raw_mean.sign() * raw_mean.abs().log1p() / 8.0
        std_feature = raw_std.log1p() / 8.0
        point = self.net(torch.cat([
            x,
            y_norm[:, None],
            mean_feature.expand(y.shape[0], 1),
            std_feature.expand(y.shape[0], 1),
        ], dim=1)).mean(dim=0)
        if self.mode == "point_mlp":
            return point
        stats = self.stats_net(task_stat_features(x, y.float(), self.num_vars).to(x.device))
        if self.mode == "stats":
            return stats
        return self.hybrid_net(torch.cat([point, stats], dim=0))


class FixedSymbolConditionedVelocityNet(nn.Module):
    def __init__(
        self,
        template: Any,
        hidden: int,
        *,
        semantic_features: bool = True,
        active_node_semantic_features: bool = False,
        velocity_parameterization: str = "direct_velocity",
        global_state_mode: str = "summary",
        metadata_embedding_dim: int = 0,
        task_encoder_mode: str = "point_mlp",
        task_conditioning: str = "xy",
        semantic_feature_mode: str = "full",
    ):
        super().__init__()
        self.template = template
        self.theta_dim = theta_dim(template)
        self.hidden = int(hidden)
        self.block_count = len(template.blocks)
        self.source_count = int(template.source_count)
        if any(int(block.size) != self.source_count for block in template.blocks):
            raise ValueError("fixed-symbol velocity net expects uniform block simplex size")
        self.semantic_features = bool(semantic_features)
        self.active_node_semantic_features = bool(active_node_semantic_features)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.velocity_parameterization = str(velocity_parameterization)
        if self.velocity_parameterization not in {"direct_velocity", "endpoint_bridge"}:
            raise ValueError(f"unknown velocity parameterization: {self.velocity_parameterization}")
        self.global_state_mode = str(global_state_mode)
        if self.global_state_mode not in {"summary", "full"}:
            raise ValueError(f"unknown global state mode: {self.global_state_mode}")
        self.task_conditioning = str(task_conditioning)
        if self.task_conditioning not in {"xy", "xy_residual", "off"}:
            raise ValueError(f"unknown task conditioning mode: {self.task_conditioning}")
        self.task_encoder_mode = str(task_encoder_mode)
        self.task_encoder = TaskSemanticEncoder(template.num_vars, hidden, mode=self.task_encoder_mode)
        self.semantic_feature_mode = str(semantic_feature_mode)
        block_meta, action_meta = _meta_rows(template)
        block_ids, action_ids = _id_rows(template)
        self.register_buffer("block_meta", block_meta, persistent=False)
        self.register_buffer("action_meta", action_meta, persistent=False)
        self.register_buffer("block_ids", block_ids, persistent=False)
        self.register_buffer("action_ids", action_ids, persistent=False)
        if self.metadata_embedding_dim > 0:
            self.block_embedding = nn.Embedding(self.block_count, self.metadata_embedding_dim)
            self.action_embedding = nn.Embedding(self.source_count, self.metadata_embedding_dim)
        else:
            self.block_embedding = None
            self.action_embedding = None
        global_in = 17 + hidden
        if self.global_state_mode == "full":
            global_in += self.theta_dim * 2
        self.global_net = nn.Sequential(
            nn.Linear(global_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.semantic_feature_width = 10 if _is_register_template(template) else 8
        action_feature_dim = self.semantic_feature_width if self.semantic_features else 0
        active_node_feature_dim = self.semantic_feature_width if self.active_node_semantic_features else 0
        embedding_dim = 2 * self.metadata_embedding_dim if self.metadata_embedding_dim > 0 else 0
        local_dim = (
            hidden
            + int(block_meta.shape[1])
            + int(action_meta.shape[1])
            + embedding_dim
            + 5
            + 1
            + 5
            + 1
            + 2 * template.source_count
            + action_feature_dim
            + active_node_feature_dim
        )
        self.head = nn.Sequential(
            nn.Linear(local_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.task_residual = nn.Sequential(
            nn.Linear(local_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.task_residual[-1].weight)
        nn.init.zeros_(self.task_residual[-1].bias)

    def theta_summary(self, theta: torch.Tensor) -> torch.Tensor:
        logits = theta.float().flatten().view(self.block_count, self.source_count)
        p = masked_block_softmax(logits, self.template)
        ent = -(p * p.clamp_min(1.0e-8).log()).sum(dim=-1) / math.log(max(self.source_count, 2))
        mx = p.max(dim=-1).values
        non_readout = max(self.block_count - int(self.template.output_terms), 1)
        return torch.stack([
            ent.mean(),
            mx.mean(),
            ent[:non_readout].mean(),
            mx[:non_readout].mean(),
            ent[non_readout:].mean() if non_readout < self.block_count else ent.new_tensor(0.0),
            mx[non_readout:].mean() if non_readout < self.block_count else mx.new_tensor(0.0),
            torch.tensor(float(self.template.num_layers), device=theta.device) / 32.0,
            torch.tensor(float(self.template.node_count), device=theta.device) / 16.0,
        ])

    def probability_meta(self, theta: torch.Tensor) -> torch.Tensor:
        logits = theta.float().flatten().view(self.block_count, self.source_count)
        p = masked_block_softmax(logits, self.template)
        ent = (-(p * p.clamp_min(1.0e-8).log()).sum(dim=-1, keepdim=True) / math.log(max(self.source_count, 2))).float()
        mx = p.max(dim=-1, keepdim=True).values.float()
        rows = torch.stack([
            p,
            p.clamp_min(1.0e-8).log() / 8.0,
            ent.expand_as(p),
            mx.expand_as(p),
            mx.expand_as(p) - p,
        ], dim=-1)
        return rows.reshape(self.theta_dim, 5)

    def block_context_meta(self, theta: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
        logits = theta.float().flatten().view(self.block_count, self.source_count) / 8.0
        seed_logits = theta0.float().flatten().view(self.block_count, self.source_count) / 8.0
        ctx = torch.cat([logits, seed_logits], dim=-1)
        return ctx[:, None, :].expand(self.block_count, self.source_count, 2 * self.source_count).reshape(self.theta_dim, 2 * self.source_count)

    def forward(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, t: float, theta0: torch.Tensor) -> torch.Tensor:
        endpoint_logits = self.predict_endpoint(x, y, theta0) if str(self.velocity_parameterization) == "endpoint_bridge" else None
        if endpoint_logits is not None:
            return differentiable_bridge_velocity(theta0, endpoint_logits, self.template, float(t))
        return self._predict_field(x, y, theta, t, theta0)

    def predict_endpoint(self, x: torch.Tensor, y: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
        return self._predict_field(x, y, theta0, 0.0, theta0, endpoint_output=True)

    def _predict_field(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
        t: float,
        theta0: torch.Tensor,
        *,
        endpoint_output: bool = False,
        task_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        theta = theta.float().flatten()
        theta0 = theta0.to(theta.device).float().flatten()
        feature_theta = theta0 if endpoint_output else theta
        feature_t = 0.0 if endpoint_output else float(t)
        if task_embedding is not None:
            task = task_embedding.to(theta.device)
        elif self.task_conditioning in {"off", "xy_residual"}:
            task = theta.new_zeros(self.hidden)
        else:
            task = self.task_encoder(x.to(theta.device), y.to(theta.device))
        residual_task = (
            self.task_encoder(x.to(theta.device), y.to(theta.device))
            if self.task_conditioning == "xy_residual"
            else task
        )
        summary = torch.cat([
            self.theta_summary(feature_theta),
            self.theta_summary(theta0),
            torch.tensor([float(feature_t)], dtype=theta.dtype, device=theta.device),
        ])
        global_parts = [summary, task]
        if self.global_state_mode == "full":
            global_parts = [feature_theta, theta0, summary, task]
        g = self.global_net(torch.cat(global_parts, dim=0).unsqueeze(0)).squeeze(0)
        prob_meta = self.probability_meta(feature_theta)
        seed_prob_meta = self.probability_meta(theta0)
        block_ctx = self.block_context_meta(feature_theta, theta0)
        prefix_parts = [
            g.unsqueeze(0).expand(self.theta_dim, -1),
            self.block_meta.to(theta.device),
            self.action_meta.to(theta.device),
        ]
        if self.metadata_embedding_dim > 0 and self.block_embedding is not None and self.action_embedding is not None:
            prefix_parts.extend([
                self.block_embedding(self.block_ids.to(theta.device)),
                self.action_embedding(self.action_ids.to(theta.device)),
            ])
        prefix_parts.extend([
            prob_meta,
            feature_theta[:, None] / 8.0,
            seed_prob_meta,
            theta0[:, None] / 8.0,
            block_ctx,
        ])
        semantic_part = theta.new_zeros((self.theta_dim, self.semantic_feature_width))
        if self.semantic_features and self.task_conditioning in {"xy", "xy_residual"}:
            semantic_part = (
                register_low_cost_semantic_features(self.template, feature_theta, x.to(theta.device), y.to(theta.device))
                if self.semantic_feature_mode == "soft_register_low_cost" and _is_register_template(self.template)
                else action_consequence_features(self.template, feature_theta, x.to(theta.device), y.to(theta.device))
            )
        base_semantic_part = semantic_part if self.task_conditioning == "xy" else theta.new_zeros((self.theta_dim, self.semantic_feature_width))
        active_node_part = theta.new_zeros((self.theta_dim, self.semantic_feature_width))
        if self.active_node_semantic_features and self.task_conditioning in {"xy", "xy_residual"}:
            active_node_part = active_node_semantic_features(self.template, feature_theta, x.to(theta.device), y.to(theta.device))
        base_active_node_part = active_node_part if self.task_conditioning == "xy" else theta.new_zeros((self.theta_dim, self.semantic_feature_width))
        local = torch.cat(
            prefix_parts
            + ([base_semantic_part] if self.semantic_features else [])
            + ([base_active_node_part] if self.active_node_semantic_features else []),
            dim=-1,
        )
        out = self.head(local).squeeze(-1)
        if self.task_conditioning == "xy_residual":
            residual_local = torch.cat(
                prefix_parts
                + ([semantic_part] if self.semantic_features else [])
                + ([active_node_part] if self.active_node_semantic_features else []),
                dim=-1,
            )
            residual_in = torch.cat([residual_local, residual_task.unsqueeze(0).expand(self.theta_dim, -1)], dim=-1)
            out = out + self.task_residual(residual_in).squeeze(-1)
        if endpoint_output:
            return center_theta(8.0 * torch.tanh(out / 8.0), self.template)
        return center_theta(20.0 * torch.tanh(out / 20.0), self.template)


class TaskConditionedVelocityNet(FixedSymbolConditionedVelocityNet):
    """Route-free direct field used by the current Poisson residual cycle."""

    route_free = True

    def __init__(self, template: Any, hidden: int, **kwargs: Any):
        kwargs.pop("velocity_parameterization", None)
        kwargs.pop("active_node_semantic_features", None)
        # The old online register semantic path contains hard-prefix and signed
        # pair work.  Complete semantics are evaluated by the current flow at complete semantics once at the endpoint.
        kwargs["semantic_features"] = True
        kwargs["semantic_feature_mode"] = "soft_register_low_cost"
        super().__init__(
            template,
            hidden,
            velocity_parameterization="direct_velocity",
            active_node_semantic_features=False,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, t: float) -> torch.Tensor:
        # Reuse the established state/task trunk while making every former
        # seed channel a deterministic function of the current Eulerian state.
        velocity = self._predict_field(x, y, theta, float(t), theta)
        gates = register_soft_block_reachability(self.template, theta).repeat_interleave(int(self.template.source_count))
        return center_theta(velocity * gates, self.template)

    def encode_task(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.task_encoder(x, y)

    def forward_with_task_embedding(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
        t: float,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:
        velocity = self._predict_field(x, y, theta, float(t), theta, task_embedding=task_embedding)
        gates = register_soft_block_reachability(self.template, theta).repeat_interleave(int(self.template.source_count))
        return center_theta(velocity * gates, self.template)

    def forward_batch(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, t: float) -> torch.Tensor:
        if theta.ndim == 1:
            return self.forward(x, y, theta, t)
        task_embedding = self.encode_task(x, y)
        return self.forward_batch_with_task_embedding(x, y, theta, t, task_embedding)

    def forward_batch_with_task_embedding(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
        t: float,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return torch.vmap(
            lambda row: self.forward_with_task_embedding(x, y, row, t, task_embedding)
        )(theta)


class ResidualFeatureTrunk(nn.Module):
    """Shared task/state feature map evaluated once for all residual stages."""

    def __init__(self, template: Any, hidden: int):
        super().__init__()
        self.template = template
        self.theta_dim = theta_dim(template)
        stat_dim = _task_stat_dim(int(template.num_vars))
        self.width = max(int(hidden) // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(self.theta_dim + stat_dim + 1, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, t: float) -> torch.Tensor:
        state = theta.float().flatten()
        stats = task_stat_features(x.to(state.device), y.to(state.device), int(self.template.num_vars)).to(state.device)
        time_feature = state.new_tensor([float(t)])
        return self.net(torch.cat([state / 8.0, stats, time_feature], dim=0))


class ResidualVelocityHead(nn.Module):
    """One inexpensive output head on the shared residual feature trunk."""

    def __init__(self, template: Any, hidden: int):
        super().__init__()
        self.template = template
        self.theta_dim = theta_dim(template)
        self.feature_width = max(int(hidden) // 2, 64)
        self.output = nn.Linear(self.feature_width, self.theta_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.output(features)
        return center_theta(20.0 * torch.tanh(raw / 20.0), self.template)


class ConditionalSemanticPotential(nn.Module):
    """Scalar endpoint potential conditioned only on the task and endpoint."""

    def __init__(self, template: Any, hidden: int):
        super().__init__()
        self.template = template
        self.theta_dim = theta_dim(template)
        stat_dim = _task_stat_dim(int(template.num_vars))
        width = max(int(hidden) // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(self.theta_dim + stat_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, probabilities: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        particles = probabilities if probabilities.ndim == 3 else probabilities.unsqueeze(0)
        stats = task_stat_features(x.to(particles.device), y.to(particles.device), int(self.template.num_vars))
        stats = stats.unsqueeze(0).expand(int(particles.shape[0]), -1)
        features = torch.cat([particles.flatten(start_dim=1), stats], dim=1)
        return self.net(features).squeeze(-1)


class PoissonResidualVelocity(nn.Module):
    """One route-free base field plus inexpensive frozen residual stages."""

    route_free = True
    velocity_parameterization = "direct_velocity"

    def __init__(self, base: TaskConditionedVelocityNet):
        super().__init__()
        self.base = base
        self.template = base.template
        self.hidden = base.hidden
        self.semantic_features = False
        self.active_node_semantic_features = False
        self.residual_trunk = ResidualFeatureTrunk(self.template, int(self.hidden))
        self.residual_heads = nn.ModuleList()
        self.register_buffer("residual_step_sizes", torch.zeros(0, dtype=torch.float32))

    def add_residual(self, head: ResidualVelocityHead, step_size: float) -> None:
        self.residual_heads.append(head)
        steps = torch.cat([
            self.residual_step_sizes.detach().cpu(),
            torch.tensor([float(step_size)], dtype=torch.float32),
        ])
        self.residual_step_sizes = steps.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, t: float) -> torch.Tensor:
        velocity = self.base(x, y, theta, float(t))
        if self.residual_heads:
            features = self.residual_trunk(x, y, theta, float(t))
            for index, head in enumerate(self.residual_heads):
                velocity = velocity + self.residual_step_sizes[index].to(velocity) * head(features)
        return center_theta(velocity, self.template)

    def forward_batch(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, t: float) -> torch.Tensor:
        if theta.ndim == 1:
            return self.forward(x, y, theta, t)
        task_embedding = self.base.encode_task(x, y)
        velocity = self.base.forward_batch_with_task_embedding(x, y, theta, float(t), task_embedding)
        if self.residual_heads:
            features = torch.vmap(lambda row: self.residual_trunk(x, y, row, float(t)))(theta)
            for index, head in enumerate(self.residual_heads):
                velocity = velocity + self.residual_step_sizes[index].to(velocity) * torch.vmap(head)(features)
        return torch.vmap(lambda row: center_theta(row, self.template))(velocity)


def _call_velocity_field(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
    t: float,
) -> torch.Tensor:
    return model(x, y, theta, float(t))


def _meta_rows(template: FixedSymbolTemplate) -> tuple[torch.Tensor, torch.Tensor]:
    block_rows, action_rows = [], []
    max_layer = max(float(template.num_layers), 1.0)
    max_node = max(float(template.node_count - 1), 1.0)
    max_source = max(float(template.source_count - 1), 1.0)
    op_vocab = list(dict.fromkeys(str(op) for op in template.ops))
    op_to_idx = {op: idx for idx, op in enumerate(op_vocab)}
    for block in template.blocks:
        for action in range(int(block.size)):
            block_op = [0.0 for _ in op_vocab]
            block_arity = 0.0
            if block.kind == "edge" and int(block.node) >= 0:
                op_name = str(template.ops[int(block.node)])
                block_op[op_to_idx[op_name]] = 1.0
                block_arity = float(op_arity(op_name)) / 2.0
            elif block.kind == "reg_op":
                block_arity = 1.0
            block_rows.append([
                float(block.layer) / max_layer,
                1.0 if block.kind in {"edge", "reg_arg"} else 0.0,
                1.0 if block.kind == "readout" else 0.0,
                float(block.node) / max_node if block.node >= 0 else -1.0,
                float(block.slot) / 2.0 if block.slot >= 0 else -1.0,
                float(block.term) / max(float(template.output_terms - 1), 1.0) if block.term >= 0 else -1.0,
                block_arity,
            ] + block_op)
            src = int(action)
            source_op = [0.0 for _ in op_vocab]
            if src >= template.base_count:
                node = int(src) - int(template.base_count)
                if _is_fixed_symbol_template(template) and 0 <= node < int(template.node_count):
                    source_op[op_to_idx[str(template.ops[node])]] = 1.0
            action_rows.append([
                float(src) / max_source,
                1.0 if src < int(template.num_vars) else 0.0,
                1.0 if src == int(template.zero_source_index) else 0.0,
                1.0 if src == int(template.one_source_index) else 0.0,
                1.0 if src >= template.base_count else 0.0,
                float(src - template.base_count) / max_node if src >= template.base_count else -1.0,
            ] + source_op)
    return torch.tensor(block_rows, dtype=torch.float32), torch.tensor(action_rows, dtype=torch.float32)


def _id_rows(template: FixedSymbolTemplate) -> tuple[torch.Tensor, torch.Tensor]:
    block_ids: list[int] = []
    action_ids: list[int] = []
    for block_idx, block in enumerate(template.blocks):
        for action in range(int(block.size)):
            block_ids.append(int(block_idx))
            action_ids.append(int(action))
    return torch.tensor(block_ids, dtype=torch.long), torch.tensor(action_ids, dtype=torch.long)


def _probability_meta(theta: torch.Tensor, template: FixedSymbolTemplate) -> torch.Tensor:
    block_count = len(template.blocks)
    source_count = int(template.source_count)
    if any(int(block.size) != source_count for block in template.blocks):
        rows = []
        for logits in split_blocks(theta, template):
            p = torch.softmax(logits, dim=-1)
            ent = (-(p * p.clamp_min(1.0e-8).log()).sum() / math.log(max(int(p.numel()), 2))).float()
            mx = p.max().float()
            for idx in range(int(logits.numel())):
                prob = p[idx]
                rows.append(torch.stack([prob, prob.clamp_min(1.0e-8).log() / 8.0, ent, mx, mx - prob]))
        return torch.stack(rows, dim=0)
    logits = theta.float().flatten().view(block_count, source_count)
    p = masked_block_softmax(logits, template)
    ent = (-(p * p.clamp_min(1.0e-8).log()).sum(dim=-1, keepdim=True) / math.log(max(source_count, 2))).float()
    mx = p.max(dim=-1, keepdim=True).values.float()
    return torch.stack([
        p,
        p.clamp_min(1.0e-8).log() / 8.0,
        ent.expand_as(p),
        mx.expand_as(p),
        mx.expand_as(p) - p,
    ], dim=-1).reshape(theta_dim(template), 5)


def _safe_apply_semantic(op: str, args: list[torch.Tensor]) -> torch.Tensor:
    if str(op) == "copy":
        return sanitize_values(args[0])
    try:
        return sanitize_values(get_op(NAME_TO_ID[op]).fn(*args))
    except Exception:
        return torch.zeros_like(args[0])


def soft_layer_semantics(template: FixedSymbolTemplate, theta: torch.Tensor, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    x = _pad_x(x.float(), template.num_vars)
    base = [x[:, idx] for idx in range(int(template.num_vars))]
    base.append(torch.zeros(int(x.shape[0]), dtype=x.dtype, device=x.device))
    base.append(torch.ones(int(x.shape[0]), dtype=x.dtype, device=x.device))
    prev = [torch.zeros(int(x.shape[0]), dtype=x.dtype, device=x.device) for _ in range(int(template.node_count))]
    banks: list[torch.Tensor] = []
    layers: list[torch.Tensor] = []
    blocks = split_blocks(theta, template)
    cursor = 0
    for layer in range(int(template.num_layers)):
        bank = torch.stack(base + prev, dim=1)
        banks.append(bank)
        current = []
        for node, op in enumerate(template.ops):
            args = []
            for _slot in range(op_arity(op)):
                p = masked_single_block_softmax(blocks[cursor], template, cursor).to(bank.device)
                args.append((bank * p[None, :]).sum(dim=1))
                cursor += 1
            current.append(_safe_apply_semantic(op, args))
        prev = current
        layers.append(torch.stack(current, dim=1))
    banks.append(torch.stack(base + prev, dim=1))
    return banks, layers


def action_consequence_features(template: FixedSymbolTemplate, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if _is_register_template(template):
        return register_active_node_semantic_features(template, theta, x, y)
    if _uniform_chart(template):
        return fixed_symbol_action_consequence_features(template, theta, x, y)
    return torch.stack([_semantic_features(values, y) for values in action_consequence_values(template, theta, x)], dim=0)


def active_node_semantic_features(template: Any, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if _is_register_template(template):
        return register_hard_prefix_semantic_features(template, theta, x, y)
    return fixed_symbol_active_node_semantic_features(template, theta, x, y)


def fixed_symbol_active_node_semantic_features(template: FixedSymbolTemplate, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    theta = theta.float().flatten()
    banks, layers = soft_layer_semantics(template, theta, x)
    rows: list[torch.Tensor] = []
    for block in template.blocks:
        if block.kind == "readout":
            rows.append(_semantic_features_batch(banks[-1].transpose(0, 1), y))
        else:
            source_values = banks[int(block.layer)].transpose(0, 1)
            rows.append(_semantic_features_batch(source_values, y))
    return torch.cat(rows, dim=0)


def register_soft_semantics(
    template: RegisterOperatorSimplexTemplate,
    theta: torch.Tensor,
    x: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    x = _pad_x(x.float(), template.num_vars)
    base = [x[:, idx] for idx in range(int(template.num_vars))]
    base.append(torch.zeros(int(x.shape[0]), dtype=x.dtype, device=x.device))
    base.append(torch.ones(int(x.shape[0]), dtype=x.dtype, device=x.device))
    while len(base) < int(template.register_count):
        base.append(torch.zeros(int(x.shape[0]), dtype=x.dtype, device=x.device))
    regs = list(base)
    banks: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    blocks = split_blocks(theta, template)

    def bank_tensor() -> torch.Tensor:
        cols = list(regs)
        while len(cols) < int(template.source_count):
            cols.append(torch.zeros(int(x.shape[0]), dtype=x.dtype, device=x.device))
        return torch.stack(cols[: int(template.source_count)], dim=1)

    for layer in range(int(template.num_layers)):
        bank = bank_tensor()
        banks.append(bank)
        op_probs = masked_single_block_softmax(blocks[register_op_block_index(template, layer)], template, register_op_block_index(template, layer)).to(bank.device)
        arg0_probs = masked_single_block_softmax(blocks[register_arg_block_index(template, layer, 0)], template, register_arg_block_index(template, layer, 0)).to(bank.device)
        arg1_probs = masked_single_block_softmax(blocks[register_arg_block_index(template, layer, 1)], template, register_arg_block_index(template, layer, 1)).to(bank.device)
        arg0 = (bank * arg0_probs[None, :]).sum(dim=1)
        arg1 = (bank * arg1_probs[None, :]).sum(dim=1)
        keep_value = regs[int(template.write_register_for_layer(layer))]
        mixed = torch.zeros_like(keep_value)
        for op_idx, op in enumerate(template.ops):
            if int(op_idx) >= int(op_probs.numel()):
                continue
            args = [arg0] if op_arity(op) == 1 else [arg0, arg1]
            mixed = mixed + op_probs[int(op_idx)] * _safe_apply_semantic(str(op), args)
        if int(template.keep_action_index) < int(op_probs.numel()):
            mixed = mixed + op_probs[int(template.keep_action_index)] * keep_value
        regs[int(template.write_register_for_layer(layer))] = sanitize_values(mixed)
        writes.append(regs[int(template.write_register_for_layer(layer))])
    banks.append(bank_tensor())
    return banks, writes


def register_low_cost_semantic_features(
    template: RegisterOperatorSimplexTemplate,
    theta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Smooth register/action summaries without hard prefixes or pair search."""
    theta = theta.float().flatten()
    banks, _writes = register_soft_semantics(template, theta, x)
    blocks = split_blocks(theta, template)
    rows: list[torch.Tensor] = []
    for block in template.blocks:
        bank = banks[-1] if block.kind == "readout" else banks[int(block.layer)]
        if block.kind in {"readout", "reg_arg"}:
            features = _semantic_features_batch(bank.transpose(0, 1), y)
        elif block.kind == "reg_op":
            layer = int(block.layer)
            arg0_p = masked_single_block_softmax(blocks[register_arg_block_index(template, layer, 0)], template, register_arg_block_index(template, layer, 0))
            arg1_p = masked_single_block_softmax(blocks[register_arg_block_index(template, layer, 1)], template, register_arg_block_index(template, layer, 1))
            arg0 = (bank * arg0_p[None, :]).sum(dim=1)
            arg1 = (bank * arg1_p[None, :]).sum(dim=1)
            values = []
            for action in range(int(block.size)):
                if action < len(template.ops):
                    op = str(template.ops[action])
                    values.append(_safe_apply_semantic(op, [arg0] if op_arity(op) == 1 else [arg0, arg1]))
                elif action == int(template.keep_action_index):
                    values.append(bank[:, int(template.write_register_for_layer(layer))])
                else:
                    values.append(torch.zeros_like(arg0))
            features = _semantic_features_batch(torch.stack(values), y)
        else:
            features = theta.new_zeros((int(block.size), 8))
        rows.append(torch.cat([features, features.new_zeros((int(features.shape[0]), 2))], dim=1))
    return torch.cat(rows, dim=0)


def register_soft_block_reachability(
    template: RegisterOperatorSimplexTemplate,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Differentiable probability that each block can affect the soft readout."""
    blocks = split_blocks(theta.float().flatten(), template)
    readout_index = register_readout_block_index(template, 0)
    reach = masked_single_block_softmax(blocks[readout_index], template, readout_index).clone()
    gates = theta.new_zeros((len(template.blocks),), dtype=torch.float32)
    gates[readout_index] = 1.0
    for layer in reversed(range(int(template.num_layers))):
        write_register = int(template.write_register_for_layer(layer))
        gate = reach[write_register].clamp(0.0, 1.0)
        op_index = register_op_block_index(template, layer)
        arg0_index = register_arg_block_index(template, layer, 0)
        arg1_index = register_arg_block_index(template, layer, 1)
        gates[op_index] = gate
        gates[arg0_index] = gate
        gates[arg1_index] = gate
        op_probability = masked_single_block_softmax(blocks[op_index], template, op_index)
        arg0_probability = masked_single_block_softmax(blocks[arg0_index], template, arg0_index)
        arg1_probability = masked_single_block_softmax(blocks[arg1_index], template, arg1_index)
        unary_mass = gate.new_zeros(())
        binary_mass = gate.new_zeros(())
        for action, op in enumerate(template.ops):
            if action >= int(op_probability.numel()):
                continue
            if op_arity(str(op)) == 1:
                unary_mass = unary_mass + op_probability[action]
            else:
                binary_mass = binary_mass + op_probability[action]
        keep_mass = op_probability[int(template.keep_action_index)] if int(template.keep_action_index) < int(op_probability.numel()) else gate.new_zeros(())
        propagated = reach.clone()
        propagated[write_register] = gate * keep_mass
        propagated = propagated + gate * (unary_mass + binary_mass) * arg0_probability
        propagated = propagated + gate * binary_mass * arg1_probability
        reach = propagated.clamp(0.0, 1.0)
    return gates.clamp(0.0, 1.0)


def register_active_node_semantic_features(
    template: RegisterOperatorSimplexTemplate,
    theta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    theta = theta.float().flatten()
    banks, writes = register_soft_semantics(template, theta, x)
    blocks = split_blocks(theta, template)
    rows: list[torch.Tensor] = []

    def signed_pair_features(values: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        values = sanitize_values(values.float())
        if values.ndim == 1:
            values = values[None, :]
        partners = sanitize_values(bank.float()).transpose(0, 1)
        target = sanitize_values(y.float()).to(values.device)
        variance = (target - target.mean()).square().mean().clamp_min(1.0e-8)
        direct = (values - target[None, :]).square().mean(dim=1) / variance
        plus = (values[:, None, :] + partners[None, :, :] - target[None, None, :]).square().mean(dim=-1) / variance
        minus = (values[:, None, :] - partners[None, :, :] - target[None, None, :]).square().mean(dim=-1) / variance
        reverse = (partners[None, :, :] - values[:, None, :] - target[None, None, :]).square().mean(dim=-1) / variance
        reachable = torch.minimum(direct, torch.minimum(plus.amin(dim=1), torch.minimum(minus.amin(dim=1), reverse.amin(dim=1))))
        direct_unit = direct / (1.0 + direct)
        reachable_unit = reachable / (1.0 + reachable)
        gain = (direct_unit - reachable_unit).clamp(0.0, 1.0)
        return torch.stack([reachable_unit, gain], dim=1)

    def padded_features(values: torch.Tensor, bank: torch.Tensor, block_size: int) -> torch.Tensor:
        feats = torch.cat([
            _semantic_features_batch(values, y),
            signed_pair_features(values, bank),
        ], dim=1)
        if int(feats.shape[0]) >= int(block_size):
            return feats[: int(block_size)]
        pad = torch.zeros((int(block_size) - int(feats.shape[0]), int(feats.shape[1])), dtype=feats.dtype, device=feats.device)
        return torch.cat([feats, pad], dim=0)

    for block in template.blocks:
        if block.kind == "readout":
            rows.append(padded_features(banks[-1].transpose(0, 1), banks[-1], int(block.size)))
            continue
        bank = banks[int(block.layer)]
        bank_t = bank.transpose(0, 1)
        if block.kind == "reg_arg":
            rows.append(padded_features(bank_t, bank, int(block.size)))
            continue
        if block.kind == "reg_op":
            layer = int(block.layer)
            arg0_probs = masked_single_block_softmax(blocks[register_arg_block_index(template, layer, 0)], template, register_arg_block_index(template, layer, 0)).to(bank.device)
            arg1_probs = masked_single_block_softmax(blocks[register_arg_block_index(template, layer, 1)], template, register_arg_block_index(template, layer, 1)).to(bank.device)
            arg0 = (bank * arg0_probs[None, :]).sum(dim=1)
            arg1 = (bank * arg1_probs[None, :]).sum(dim=1)
            values: list[torch.Tensor] = []
            for action in range(int(block.size)):
                if 0 <= action < len(template.ops):
                    op = str(template.ops[int(action)])
                    args = [arg0] if op_arity(op) == 1 else [arg0, arg1]
                    values.append(_safe_apply_semantic(op, args))
                elif int(action) == int(template.keep_action_index):
                    values.append(bank[:, int(template.write_register_for_layer(layer))])
                else:
                    values.append(torch.zeros_like(arg0))
            action_values = torch.stack(values, dim=0)
            rows.append(torch.cat([
                _semantic_features_batch(action_values, y),
                signed_pair_features(action_values, bank),
            ], dim=1))
            continue
        rows.append(torch.zeros((int(block.size), 10), dtype=theta.dtype, device=theta.device))
    return torch.cat(rows, dim=0)


def register_hard_prefix_semantic_features(
    template: RegisterOperatorSimplexTemplate,
    theta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Candidate semantics after projecting the current state to its hard prefix."""
    choices = hard_decode_choices(theta, template)
    rows: list[torch.Tensor] = []
    for block_index, block in enumerate(template.blocks):
        support = graph_block_mask(template, block_index, device=theta.device)
        logits = torch.full((int(block.size),), -16.0, dtype=theta.dtype, device=theta.device)
        action = int(choices[block_index])
        if 0 <= action < int(block.size) and bool(support[action].detach().cpu()):
            logits[action] = 16.0
        rows.append(logits)
    hard_theta = center_theta(pack_blocks(rows), template)
    return register_active_node_semantic_features(template, hard_theta, x, y)


def fixed_symbol_action_consequence_features(template: FixedSymbolTemplate, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    theta = theta.float().flatten()
    banks, _layers = soft_layer_semantics(template, theta, x)
    blocks = split_blocks(theta, template)
    rows: list[torch.Tensor] = []
    cursor = 0
    for bidx, block in enumerate(template.blocks):
        if block.kind == "readout":
            rows.append(_semantic_features_batch(banks[-1].transpose(0, 1), y))
            cursor += 1
            continue
        bank = banks[int(block.layer)]
        bank_t = bank.transpose(0, 1)
        op = template.ops[int(block.node)]
        arity = op_arity(op)
        base_args = []
        for slot in range(arity):
            slot_bidx = cursor + slot - int(block.slot)
            slot_block = blocks[slot_bidx]
            p = masked_single_block_softmax(slot_block, template, slot_bidx).to(bank.device)
            base_args.append((bank * p[None, :]).sum(dim=1))
        if str(op) == "copy":
            values = bank_t
        else:
            args: list[torch.Tensor] = []
            for slot in range(arity):
                if int(slot) == int(block.slot):
                    args.append(bank_t)
                else:
                    args.append(base_args[slot][None, :].expand_as(bank_t))
            values = _safe_apply_semantic(op, args)
        rows.append(_semantic_features_batch(values, y))
        cursor += 1
    return torch.cat(rows, dim=0)


def action_consequence_values(template: FixedSymbolTemplate, theta: torch.Tensor, x: torch.Tensor) -> list[torch.Tensor]:
    theta = theta.float().flatten()
    banks, _layers = soft_layer_semantics(template, theta, x)
    blocks = split_blocks(theta, template)
    rows: list[torch.Tensor] = []
    cursor = 0
    for bidx, block in enumerate(template.blocks):
        logits = blocks[bidx]
        if block.kind == "readout":
            bank = banks[-1]
            for action in range(int(block.size)):
                rows.append(bank[:, action])
            cursor += 1
            continue
        bank = banks[int(block.layer)]
        op = template.ops[int(block.node)]
        arity = op_arity(op)
        base_args = []
        for slot in range(arity):
            slot_bidx = cursor + slot - int(block.slot)
            slot_block = blocks[slot_bidx]
            p = masked_single_block_softmax(slot_block, template, slot_bidx).to(bank.device)
            base_args.append((bank * p[None, :]).sum(dim=1))
        for action in range(int(block.size)):
            args = list(base_args)
            args[int(block.slot)] = bank[:, action]
            rows.append(_safe_apply_semantic(op, args))
        cursor += 1
    return rows


def hard_node_semantics(template: FixedSymbolTemplate, choices: list[int], x: torch.Tensor) -> list[torch.Tensor]:
    base = [x[:, idx] for idx in range(int(template.num_vars))]
    base.append(torch.zeros(int(x.shape[0]), dtype=x.dtype, device=x.device))
    base.append(torch.ones(int(x.shape[0]), dtype=x.dtype, device=x.device))
    prev = [torch.zeros(int(x.shape[0]), dtype=x.dtype, device=x.device) for _ in range(int(template.node_count))]
    layers: list[torch.Tensor] = []
    cursor = 0
    for _layer in range(int(template.num_layers)):
        bank = base + prev
        current = []
        for op in template.ops:
            args = []
            for _slot in range(op_arity(op)):
                src = int(choices[cursor])
                args.append(bank[max(0, min(src, len(bank) - 1))])
                cursor += 1
            current.append(_safe_apply_semantic(op, args))
        prev = current
        layers.append(torch.stack(current, dim=1))
    return layers


def semantic_score_labels(template: FixedSymbolTemplate, theta: torch.Tensor, task: TaskBundle, trace: dict[str, Any], *, temperature: float) -> torch.Tensor:
    consequences = action_consequence_values(template, theta, task.x_train)
    y = _normalize_vec(task.y_train)
    gt_layers = hard_node_semantics(template, trace["choices"], task.x_train)
    labels = []
    cursor = 0
    for bidx, block in enumerate(template.blocks):
        block_labels = []
        if block.kind == "readout":
            candidates = [_normalize_vec(gt_layers[-1][:, n]) for n in range(int(template.node_count))]
            candidates.append(y)
        else:
            start = max(0, int(block.layer))
            candidates = []
            for layer in range(start, int(template.num_layers)):
                for node in range(int(template.node_count)):
                    candidates.append(_normalize_vec(gt_layers[layer][:, node]))
            candidates.append(y)
        for _action in range(int(block.size)):
            consequence = _normalize_vec(consequences[cursor])
            base_score = -float(((consequence - y).pow(2)).mean().detach().cpu().item())
            if candidates:
                best = max(
                    base_score,
                    max(-float(((consequence - cand).pow(2)).mean().detach().cpu().item()) for cand in candidates),
                )
            else:
                best = base_score
            block_labels.append(float(best))
            cursor += 1
        labels.extend(block_labels)
    out = torch.tensor(labels, dtype=torch.float32, device=theta.device)
    # Add a small direct GT-action shaping term only to labels, not inputs.
    for bidx in trace["active_block_indices"]:
        start = sum(int(block.size) for block in template.blocks[: int(bidx)])
        action = int(trace["choices"][int(bidx)])
        out[start + action] = out[start + action] + 1.0
    return out / max(float(temperature), 1.0e-6)


def train_stage1(
    model: FixedSymbolConditionedVelocityNet,
    train_tasks: list[TaskBundle],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = random.Random(int(args.seed))
    curve: list[dict[str, float]] = []
    compiled = [task for task in train_tasks if task.traces]
    if not compiled:
        raise RuntimeError("no train tasks have compilable GT traces")
    fixed_examples: list[dict[str, Any]] | None = None
    if bool(args.fixed_batch_overfit):
        fixed_examples = []
        for _ in range(int(args.fixed_batch_size)):
            task = rng.choice(compiled)
            theta0 = random_theta(model.template, scale=float(args.theta0_noise_scale), device=device)
            trace = select_trace_for_theta0(model.template, theta0, task, args)
            t = sample_time(rng, str(args.time_sampling), float(args.low_t_sampling_prob), float(args.low_t_max))
            p1, weights = make_stage1_target(model.template, theta0.detach(), task, trace, args)
            theta_t, target_v = make_stage1_state_and_velocity(
                model.template,
                theta0.detach(),
                p1,
                float(t),
                args,
                rng=rng,
                device=device,
            )
            fixed_examples.append({
                "task": task,
                "trace": trace,
                "theta0": theta0.detach(),
                "t": float(t),
                "theta_t": theta_t.detach(),
                "target_v": target_v.detach(),
                "weights": weights.detach(),
            })
    steps_per_epoch = max(int(args.steps_per_epoch), 1)
    best = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(int(args.epochs)):
        if str(args.lr_schedule) == "cosine":
            denom = max(int(args.epochs) - 1, 1)
            progress = float(epoch) / float(denom)
            factor = float(args.lr_min_factor) + (1.0 - float(args.lr_min_factor)) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in opt.param_groups:
                group["lr"] = float(args.lr) * float(factor)
        elif str(args.lr_schedule) != "constant":
            raise ValueError(f"unknown lr schedule: {args.lr_schedule}")
        losses: list[float] = []
        active_counts: list[float] = []
        target_fr_norms: list[float] = []
        target_logit_norms: list[float] = []
        zero_pred_losses: list[float] = []
        pred_fr_norms: list[float] = []
        align_target_fr_norms: list[float] = []
        pred_target_cosines: list[float] = []
        pred_target_norm_ratios: list[float] = []
        endpoint_active_argmax_matches: list[float] = []
        time_bin_losses: dict[str, list[float]] = {
            "loss_t_bin_0_001": [],
            "loss_t_bin_001_005": [],
            "loss_t_bin_005_020": [],
            "loss_t_bin_020_100": [],
            "loss_t_bin_100_1000": [],
        }
        finite_count = 0
        total_count = 0
        for _step in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            batch_losses = []
            for _ in range(int(args.train_batch_size)):
                if fixed_examples is not None:
                    example = rng.choice(fixed_examples)
                    task = example["task"]
                    trace = example["trace"]
                    theta0 = example["theta0"].to(device)
                    t = float(example["t"])
                    theta_t = example["theta_t"].to(device)
                    target_v = example["target_v"].to(device)
                    weights = example["weights"].to(device)
                else:
                    task = rng.choice(compiled)
                    theta0 = random_theta(model.template, scale=float(args.theta0_noise_scale), device=device)
                    trace = select_trace_for_theta0(model.template, theta0, task, args)
                    t = sample_time(rng, str(args.time_sampling), float(args.low_t_sampling_prob), float(args.low_t_max))
                    p1, weights = make_stage1_target(model.template, theta0.detach(), task, trace, args)
                    theta_t, target_v = make_stage1_state_and_velocity(
                        model.template,
                        theta0.detach(),
                        p1,
                        float(t),
                        args,
                        rng=rng,
                        device=device,
                    )
                if str(model.velocity_parameterization) == "endpoint_bridge":
                    endpoint_logits = model.predict_endpoint(task.x_train.to(device), task.y_train.to(device), theta0)
                    endpoint_rows = endpoint_logits.view(len(model.template.blocks), int(model.template.source_count))
                    choices = torch.tensor(trace["choices"], dtype=torch.long, device=device)
                    active = weights > 0
                    valid_mask = graph_action_mask(model.template, device=device)
                    log_probs = torch.log_softmax(endpoint_rows.masked_fill(~valid_mask, -1.0e9), dim=-1)
                    loss = -log_probs[active, choices[active]].mean()
                    predicted_choices = endpoint_rows.argmax(dim=-1)
                    metrics = {
                        "active_block_count": float(active.sum().detach().cpu()),
                        "active_argmax_match_mean": float((predicted_choices[active] == choices[active]).float().mean().detach().cpu()),
                    }
                    pred = differentiable_bridge_velocity(theta0, endpoint_logits, model.template, float(t))
                else:
                    pred = model(task.x_train.to(device), task.y_train.to(device), theta_t, float(t), theta0)
                    loss, metrics = stage1_velocity_loss(theta_t, pred, target_v, model.template, weights, eps=float(args.fisher_eps))
                batch_losses.append(loss)
                loss_value = float(loss.detach().cpu().item())
                if float(t) < 0.001:
                    time_bin_losses["loss_t_bin_0_001"].append(loss_value)
                elif float(t) < 0.005:
                    time_bin_losses["loss_t_bin_001_005"].append(loss_value)
                elif float(t) < 0.02:
                    time_bin_losses["loss_t_bin_005_020"].append(loss_value)
                elif float(t) < 0.10:
                    time_bin_losses["loss_t_bin_020_100"].append(loss_value)
                else:
                    time_bin_losses["loss_t_bin_100_1000"].append(loss_value)
                active_counts.append(float(metrics.get("active_block_count", 0.0)))
                endpoint_active_argmax_matches.append(float(metrics.get("active_argmax_match_mean", 0.0)))
                diag = bridge_target_diagnostics(theta_t.detach(), target_v.detach(), model.template, weights.detach(), eps=float(args.fisher_eps))
                target_fr_norms.append(float(diag["bridge_target_fr_norm_mean"]))
                target_logit_norms.append(float(diag["bridge_target_logit_norm_mean"]))
                align = velocity_alignment_diagnostics(theta_t.detach(), pred.detach(), target_v.detach(), model.template, weights.detach(), eps=float(args.fisher_eps))
                if align:
                    pred_fr_norms.append(float(align["pred_fr_norm_mean"]))
                    align_target_fr_norms.append(float(align["target_fr_norm_mean"]))
                    pred_target_cosines.append(float(align["pred_target_cosine_mean"]))
                    pred_target_norm_ratios.append(float(align["pred_target_norm_ratio_mean"]))
                zero_loss, _ = stage1_velocity_loss(theta_t, torch.zeros_like(target_v), target_v, model.template, weights, eps=float(args.fisher_eps))
                zero_pred_losses.append(float(zero_loss.detach().cpu().item()))
                finite_count += int(torch.isfinite(pred).all().detach().cpu().item())
                total_count += 1
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        if mean_loss < best:
            best = mean_loss
            best_epoch = int(epoch + 1)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": float(epoch + 1),
            "stage1_fm_loss": mean_loss,
            "stage1_objective": "active_endpoint_categorical_nll" if str(model.velocity_parameterization) == "endpoint_bridge" else "fisher_probability_velocity_loss",
            "endpoint_categorical_nll": mean_loss if str(model.velocity_parameterization) == "endpoint_bridge" else 0.0,
            "train_endpoint_active_argmax_match": float(np.mean(endpoint_active_argmax_matches)) if endpoint_active_argmax_matches else 0.0,
            "stage1_best_loss": float(best),
            "stage1_best_epoch": float(best_epoch),
            "stage1b_fm_loss": mean_loss,
            "stage1b_best_loss": float(best),
            "stage1b_best_epoch": float(best_epoch),
            "active_block_count_mean": float(np.mean(active_counts)) if active_counts else 0.0,
            "bridge_target_fr_norm_mean": float(np.mean(target_fr_norms)) if target_fr_norms else 0.0,
            "bridge_target_logit_norm_mean": float(np.mean(target_logit_norms)) if target_logit_norms else 0.0,
            "bridge_zero_pred_loss_mean": float(np.mean(zero_pred_losses)) if zero_pred_losses else 0.0,
            "pred_fr_norm_mean": float(np.mean(pred_fr_norms)) if pred_fr_norms else 0.0,
            "target_fr_norm_mean": float(np.mean(align_target_fr_norms)) if align_target_fr_norms else 0.0,
            "pred_target_cosine_mean": float(np.mean(pred_target_cosines)) if pred_target_cosines else 0.0,
            "pred_target_norm_ratio_mean": float(np.mean(pred_target_norm_ratios)) if pred_target_norm_ratios else 0.0,
            "task_encoder_finite_rate": float(finite_count / max(total_count, 1)),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        for key, values in time_bin_losses.items():
            row[key] = float(np.mean(values)) if values else 0.0
            row[f"{key}_count"] = float(len(values))
        curve.append(row)
        if bool(args.log_epochs):
            print(json.dumps(row), flush=True)
        if float(args.early_stop_loss) > 0.0 and best <= float(args.early_stop_loss):
            break
    final = dict(curve[-1])
    final["stage1_fm_loss"] = curve[-1]["stage1_fm_loss"]
    final["stage1_best_loss"] = best
    final["stage1_best_epoch"] = float(best_epoch)
    final["stage1b_fm_loss"] = final["stage1_fm_loss"]
    final["stage1b_best_loss"] = final["stage1_best_loss"]
    final["stage1b_best_epoch"] = final["stage1_best_epoch"]
    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    return curve, final


def expression_semantic_energy(
    expr: Expr,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    complexity_weight: float,
    invalid_penalty: float,
    collapse_penalty: float,
    return_semantic_vector: bool = False,
) -> tuple[float, dict[str, Any]]:
    try:
        raw_pred = sanitize_values(eval_expr(expr, x))
        design = torch.stack([raw_pred.float(), torch.ones_like(raw_pred.float())], dim=1)
        coefficients = torch.linalg.lstsq(design, y.float()).solution
        if not bool(torch.isfinite(coefficients).all()):
            coefficients = torch.tensor([1.0, 0.0], dtype=raw_pred.dtype, device=raw_pred.device)
        pred = sanitize_values(coefficients[0] * raw_pred.float() + coefficients[1])
        pred_norm = _normalize_vec(pred)
        target_norm = _normalize_vec(y)
        output_mse = float(((pred_norm - target_norm) ** 2).mean().detach().cpu().item())
        signature = semantic_signature_vector(pred, x)
        target_signature = semantic_signature_vector(y, x).to(signature.device)
        signature_mse = float(semantic_signature_distance(pred, y, x).detach().cpu().item())
        raw_mse = float(((sanitize_values(pred.float()) - sanitize_values(y.float())) ** 2).mean().detach().cpu().item())
        unfitted_mse = float(((raw_pred.float() - y.float()) ** 2).mean().detach().cpu().item())
        target_variance_sum = (y.float() - y.float().mean()).square().sum().clamp_min(1.0e-12)
        fitted_r2 = 1.0 - (pred.float() - y.float()).square().sum() / target_variance_sum
        unfitted_r2 = 1.0 - (raw_pred.float() - y.float()).square().sum() / target_variance_sum
        corr = float((pred_norm * target_norm).mean().detach().cpu().item())
        complexity = float(getattr(expr, "complexity", 1.0))
        raw = to_string(expr, int(x.shape[1]), simplify=False)
        collapsed = bool(expr.kind != "op" or str(raw).strip() in {"0", "1", "x0", "x1", "x2"})
        energy = (
            float(signature_mse)
            + float(complexity_weight) * float(complexity)
            + (float(collapse_penalty) if collapsed else 0.0)
        )
        diag: dict[str, Any] = {
            "semantic_energy": float(energy),
            "semantic_mse": float(signature_mse),
            "semantic_output_mse": float(output_mse),
            "semantic_raw_mse": float(raw_mse),
            "semantic_corr": float(corr),
            "semantic_complexity": float(complexity),
            "semantic_collapsed": float(collapsed),
            "semantic_invalid": 0.0,
            "semantic_signature_metric_code": 1.0,
            "semantic_coefficient_fit_mode": "global_affine_train_lstsq",
            "semantic_fitted_scale": float(coefficients[0].detach().cpu()),
            "semantic_fitted_intercept": float(coefficients[1].detach().cpu()),
            "semantic_fitted_train_r2": float(fitted_r2.detach().cpu()),
            "semantic_unfitted_train_r2": float(unfitted_r2.detach().cpu()),
            "semantic_unfitted_raw_mse": float(unfitted_mse),
        }
        if bool(return_semantic_vector):
            diag["semantic_vector"] = signature.detach()
        return float(energy), diag
    except Exception:
        target_signature = semantic_signature_vector(y, x)
        diag = {
            "semantic_energy": float(invalid_penalty),
            "semantic_mse": float(invalid_penalty),
            "semantic_output_mse": float(invalid_penalty),
            "semantic_raw_mse": float(invalid_penalty),
            "semantic_corr": 0.0,
            "semantic_complexity": 0.0,
            "semantic_collapsed": 1.0,
            "semantic_invalid": 1.0,
            "semantic_signature_metric_code": 1.0,
            "semantic_coefficient_fit_mode": "fit_failed",
            "semantic_fitted_scale": 0.0,
            "semantic_fitted_intercept": 0.0,
            "semantic_fitted_train_r2": -1.0e9,
            "semantic_unfitted_train_r2": -1.0e9,
            "semantic_unfitted_raw_mse": float(invalid_penalty),
        }
        if bool(return_semantic_vector):
            diag["semantic_vector"] = torch.zeros_like(target_signature).detach()
        return float(invalid_penalty), diag


def logits_from_block_probabilities(
    probs_by_block: list[torch.Tensor],
    template: FixedSymbolTemplate,
    *,
    eps: float,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for bidx, probs in enumerate(probs_by_block):
        p = probs.float()
        mask = graph_block_mask(template, int(bidx), device=p.device)
        p = torch.where(mask, p, torch.zeros_like(p))
        p = p / p.sum().clamp_min(float(eps))
        logits = torch.where(mask, p.clamp_min(float(eps)).log(), torch.full_like(p, -20.0))
        valid_mean = logits[mask].mean() if bool(mask.any().detach().cpu().item()) else logits.mean()
        rows.append(logits - valid_mean)
    return center_theta(pack_blocks(rows), template)


def rollout(
    model: FixedSymbolConditionedVelocityNet,
    _unused_score_model: None,
    task: TaskBundle,
    theta0: torch.Tensor,
    *,
    steps: int,
    mode: str,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    del generator
    if str(mode) != "off":
        raise ValueError("online rollout guidance was removed; semantic selection happens at endpoints")
    if str(model.velocity_parameterization) == "endpoint_bridge":
        with torch.no_grad():
            endpoint = model.predict_endpoint(task.x_train, task.y_train, theta0)
            theta, _ = stage1_simplex_path(theta0, endpoint, model.template, 1.0)
        return theta, {
            "closed_form_endpoint_transport": 1.0,
            "integration_step_count": 0.0,
        }
    theta = theta0.clone()
    step_count = max(int(steps), 1)
    max_step_distance = 0.0
    finite_steps = 0
    for step in range(step_count):
        t = float(step) / float(step_count)
        dt = 1.0 / float(step_count)
        before_probability = masked_block_softmax(
            theta.view(len(model.template.blocks), int(model.template.source_count)),
            model.template,
        )
        with torch.no_grad():
            velocity = _call_velocity_field(model, task.x_train, task.y_train, theta, min(t, 1.0))
            mid_theta = integrate(theta, velocity, model.template, dt=0.5 * dt)
            mid_velocity = _call_velocity_field(
                model,
                task.x_train,
                task.y_train,
                mid_theta,
                min(t + 0.5 * dt, 1.0),
            )
            theta = integrate(theta, mid_velocity, model.template, dt=dt)
            after_probability = masked_block_softmax(
                theta.view(len(model.template.blocks), int(model.template.source_count)),
                model.template,
            )
            block_steps = torch.sqrt(
                torch.stack([
                    block_fisher_squared_distance(before_probability[index:index + 1], after_probability[index:index + 1], torch.ones(1, dtype=torch.bool, device=theta.device))
                    for index in range(int(before_probability.shape[0]))
                ]).clamp_min(0.0)
            )
            max_step_distance = max(max_step_distance, float(block_steps.max().detach().cpu()))
            finite_steps += int(torch.isfinite(theta).all().detach().cpu().item() and torch.isfinite(mid_velocity).all().detach().cpu().item())
    return theta, {
        "closed_form_endpoint_transport": 0.0,
        "integration_step_count": float(step_count),
        "rollout_integrator": "rk2",
        "max_block_fisher_step": float(max_step_distance),
        "rollout_finite_rate": float(finite_steps / max(step_count, 1)),
    }


def integrate_batch(theta: torch.Tensor, velocity: torch.Tensor, template: FixedSymbolTemplate, dt: float) -> torch.Tensor:
    """Integrate a [particle, theta] batch while centering every block."""
    rows = []
    cursor = 0
    for block in template.blocks:
        size = int(block.size)
        logits = theta[:, cursor:cursor + size]
        vel = velocity[:, cursor:cursor + size]
        nxt = logits + float(dt) * vel
        rows.append(nxt - nxt.mean(dim=1, keepdim=True))
        cursor += size
    return torch.cat(rows, dim=1)


def _block_fisher_distances(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    root_source = source.float().clamp_min(1.0e-12).sqrt()
    root_target = target.float().to(source.device).clamp_min(1.0e-12).sqrt()
    half_angle = torch.atan2(
        (root_source - root_target).norm(dim=-1),
        (root_source + root_target).norm(dim=-1).clamp_min(1.0e-12),
    )
    return 4.0 * half_angle


@torch.no_grad()
def rollout_batch(
    flow: PoissonResidualVelocity,
    task: TaskBundle,
    theta0: torch.Tensor,
    *,
    steps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Batched RK2 rollout; no endpoint/expression work occurs inside the loop."""
    if theta0.ndim != 2:
        raise ValueError("theta0 batch must have shape [particle, theta]")
    theta = theta0.clone()
    step_count = max(int(steps), 1)
    max_step_distance = 0.0
    for step in range(step_count):
        t = float(step) / float(step_count)
        dt = 1.0 / float(step_count)
        velocity = flow.forward_batch(task.x_train, task.y_train, theta, t)
        midpoint = integrate_batch(theta, velocity, flow.template, 0.5 * dt)
        midpoint_velocity = flow.forward_batch(task.x_train, task.y_train, midpoint, min(t + 0.5 * dt, 1.0))
        theta_next = integrate_batch(theta, midpoint_velocity, flow.template, dt)
        before = masked_block_softmax_batch(theta.view(theta.shape[0], len(flow.template.blocks), int(flow.template.source_count)), flow.template)
        after = masked_block_softmax_batch(theta_next.view(theta.shape[0], len(flow.template.blocks), int(flow.template.source_count)), flow.template)
        max_step_distance = max(max_step_distance, float(torch.linalg.vector_norm(after - before).detach().cpu()))
        theta = theta_next
    return theta, {
        "integration_step_count": float(step_count),
        "rollout_integrator": "rk2_batched",
        "max_batch_probability_step": float(max_step_distance),
    }


def rollout_with_snapshots(
    model: FixedSymbolConditionedVelocityNet,
    task: TaskBundle,
    theta0: torch.Tensor,
    *,
    steps: int,
    snapshot_count: int,
) -> tuple[torch.Tensor, dict[str, float], list[tuple[float, torch.Tensor]]]:
    """Run the direct RK2 field and retain a small fixed ODE-time grid."""
    if str(model.velocity_parameterization) != "direct_velocity":
        raise ValueError("lineage landscape snapshots require direct_velocity")
    theta = theta0.clone()
    step_count = max(int(steps), 1)
    count = max(int(snapshot_count), 2)
    capture_steps = sorted(set(int(round(value)) for value in np.linspace(0, step_count, count)))
    snapshots: list[tuple[float, torch.Tensor]] = []
    if 0 in capture_steps:
        snapshots.append((0.0, theta.detach().clone()))
    max_step_distance = 0.0
    finite_steps = 0
    for step in range(step_count):
        t = float(step) / float(step_count)
        dt = 1.0 / float(step_count)
        before_probability = masked_block_softmax(
            theta.view(len(model.template.blocks), int(model.template.source_count)),
            model.template,
        )
        with torch.no_grad():
            velocity = _call_velocity_field(model, task.x_train, task.y_train, theta, t)
            midpoint = integrate(theta, velocity, model.template, dt=0.5 * dt)
            midpoint_velocity = _call_velocity_field(
                model,
                task.x_train,
                task.y_train,
                midpoint,
                min(t + 0.5 * dt, 1.0),
            )
            theta = integrate(theta, midpoint_velocity, model.template, dt=dt)
            after_probability = masked_block_softmax(
                theta.view(len(model.template.blocks), int(model.template.source_count)),
                model.template,
            )
            block_step = _block_fisher_distances(before_probability, after_probability)
            max_step_distance = max(max_step_distance, float(block_step.max().detach().cpu()))
            finite_steps += int(torch.isfinite(theta).all().detach().cpu())
        completed_step = step + 1
        if completed_step in capture_steps:
            snapshots.append((float(completed_step) / float(step_count), theta.detach().clone()))
    return theta, {
        "closed_form_endpoint_transport": 0.0,
        "integration_step_count": float(step_count),
        "rollout_integrator": "rk2",
        "max_block_fisher_step": float(max_step_distance),
        "rollout_finite_rate": float(finite_steps / max(step_count, 1)),
    }, snapshots


def _lineage_landscape_row(
    template: RegisterOperatorSimplexTemplate,
    task: TaskBundle,
    theta: torch.Tensor,
    *,
    iteration: int,
    source_index: int,
    t: float,
    point_kind: str,
) -> dict[str, Any]:
    probabilities = masked_block_softmax(
        theta.view(len(template.blocks), int(template.source_count)),
        template,
    )
    choices = hard_decode_choices(theta, template)
    expr, _terms, _layers = execute_choices(template, choices)
    expression = to_string(expr, int(template.num_vars), simplify=False)
    prediction = sanitize_values(eval_expr(expr, task.x_train))
    target = sanitize_values(task.y_train.float()).to(prediction.device)
    target_variance = (target - target.mean()).square().mean().clamp_min(1.0e-8)
    raw_nmse = (prediction - target).square().mean() / target_variance
    semantic = semantic_signature_vector(prediction, task.x_train)
    return {
        "iteration": int(iteration),
        "task_id": task.task_id,
        "source_index": int(source_index),
        "t": float(t),
        "point_kind": str(point_kind),
        "expression": expression,
        "gt_symbolic_hit": float(_symbolic_equiv(task.ground_truth, expression)),
        "raw_nmse": float(raw_nmse.detach().cpu()),
        "parameter_vector": (2.0 * probabilities.clamp_min(0.0).sqrt()).flatten().detach().cpu().tolist(),
        "semantic_vector": semantic.detach().cpu().tolist(),
    }


def temporal_action_weight_rows(
    model: FixedSymbolConditionedVelocityNet,
    task: TaskBundle,
    theta0: torch.Tensor,
    args: argparse.Namespace,
    *,
    steps: int,
) -> list[dict[str, Any]]:
    """Record the probability path used to explain register-flow failures."""
    theta = theta0.clone()
    endpoint = None
    if str(model.velocity_parameterization) == "endpoint_bridge":
        with torch.no_grad():
            raw_endpoint = model.predict_endpoint(task.x_train, task.y_train, theta0)
            endpoint = (
                raw_endpoint
                if bool(getattr(model, "preserve_soft_endpoint", False))
                else apply_predicted_active_endpoint_mask(model.template, theta0, raw_endpoint)
            )
    rows: list[dict[str, Any]] = []
    total_steps = max(int(steps), 1)
    for step in range(total_steps + 1):
        t = float(step) / float(total_steps)
        if endpoint is not None:
            theta, _ = stage1_simplex_path(theta0, endpoint, model.template, t)
        probabilities = masked_block_softmax(
            theta.view(len(model.template.blocks), int(model.template.source_count)),
            model.template,
        )
        for block_index_value, block in enumerate(model.template.blocks):
            probability = probabilities[block_index_value]
            top_probability, top_action = probability.max(dim=-1)
            row = {
                "task_id": task.task_id,
                "step": step,
                "t": t,
                "block": block_index_value,
                "kind": block.kind,
                "layer": int(block.layer),
                "node": int(block.node),
                "slot": int(block.slot),
                "term": int(block.term),
                "top_action": int(top_action.cpu()),
                "top_probability": float(top_probability.cpu()),
                "entropy": float((-(probability * probability.clamp_min(1.0e-8).log()).sum()).cpu()),
                "probabilities": probability.cpu().tolist(),
            }
            if _is_register_template(model.template) and block.kind == "reg_op":
                action = int(top_action.cpu())
                row["top_action_name"] = "KEEP" if action == int(model.template.keep_action_index) else str(model.template.ops[action])
                row["action_names"] = [
                    str(model.template.ops[index]) if index < len(model.template.ops)
                    else "KEEP" if index == int(model.template.keep_action_index)
                    else f"invalid_{index}"
                    for index in range(int(model.template.source_count))
                ]
            else:
                row["top_action_name"] = f"r{int(top_action.cpu())}"
                row["action_names"] = [f"r{index}" for index in range(int(model.template.source_count))]
            support = graph_block_mask(model.template, int(block_index_value), device=probability.device)
            valid_indices = support.nonzero(as_tuple=False).flatten()
            valid_probabilities = probability.index_select(0, valid_indices)
            top_count = min(3, int(valid_indices.numel()))
            top_values, top_positions = valid_probabilities.topk(top_count)
            row["top_actions"] = [
                {
                    "index": int(valid_indices[position].detach().cpu()),
                    "name": row["action_names"][int(valid_indices[position].detach().cpu())],
                    "probability": float(value.detach().cpu()),
                }
                for value, position in zip(top_values, top_positions)
            ]
            rows.append(row)
        if step < total_steps and endpoint is None:
            mid_t = (float(step) + 0.5) / float(total_steps)
            with torch.no_grad():
                velocity = _call_velocity_field(model, task.x_train, task.y_train, theta, mid_t)
                theta = integrate(theta, velocity, model.template, dt=1.0 / float(total_steps))
    return rows


def write_temporal_action_visualization(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    jsonl_path = out_dir / "temporal_action_weights.jsonl"
    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    payload = json.dumps(_jsonable(rows), ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Register transport trajectories</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#637083;--line:#d8dde5;--panel:#f7f8fa;--accent:#146c5c}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;color:var(--ink);background:white}
header{padding:20px 24px;border-bottom:1px solid var(--line)}h1{font-size:20px;margin:0}main{padding:20px 24px;max-width:1500px;margin:auto}
.task{margin:0 0 30px}.task h2{font-size:16px;margin:0 0 10px}.canvas-wrap{overflow-x:auto;border:1px solid var(--line);background:white}
canvas{display:block;min-width:940px}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);margin:8px 0 12px}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--accent);margin-right:5px}
.snapshots{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:8px;margin-top:10px}
details{border:1px solid var(--line);background:var(--panel)}summary{cursor:pointer;padding:9px 10px;font-weight:650}
.state{padding:8px 10px;border-top:1px solid var(--line)}.state-title{font-family:ui-monospace,monospace;font-size:12px;margin-bottom:6px;overflow-wrap:anywhere}
.weight{display:grid;grid-template-columns:36px minmax(70px,1fr) 44px;align-items:center;gap:6px;margin:3px 0;color:var(--muted);font-size:11px}
.bar{height:7px;background:#e3e7ec}.fill{height:100%;background:var(--accent)}.readout .fill{background:#b94b36}
</style></head><body><header><h1>Register transport trajectories</h1></header><main id="root"></main>
<script id="data" type="application/json">""" + payload + """</script>
<script>
const rows=JSON.parse(document.getElementById('data').textContent),root=document.getElementById('root');
const palette=['#146c5c','#b94b36','#3568a8','#8a5b22','#7b4b94','#397047','#a23b72','#5a6675'];
const color=name=>palette[Math.abs([...name].reduce((n,c)=>((n*33)^c.charCodeAt(0))|0,5381))%palette.length];
const grouped=rows.reduce((a,r)=>((a[r.task_id]??=[]).push(r),a),{});
function byTime(items){return Object.values(items.reduce((a,r)=>{const k=Number(r.t).toFixed(6);(a[k]??=[]).push(r);return a},{})).sort((a,b)=>a[0].t-b[0].t)}
function layerState(snapshot,layer){const op=snapshot.find(r=>r.kind==='reg_op'&&r.layer===layer),a=snapshot.find(r=>r.kind==='reg_arg'&&r.layer===layer&&r.slot===0),b=snapshot.find(r=>r.kind==='reg_arg'&&r.layer===layer&&r.slot===1);if(!op)return null;const call=op.top_action_name==='KEEP'?'keep':`${op.top_action_name}(${a?.top_action_name??'?'}, ${b?.top_action_name??'?'})`;return {op,a,b,label:`L${layer}  r${op.node} <- ${call}`}}
function meter(label,row,readout=false){if(!row)return'';return `<div class="weight ${readout?'readout':''}"><span>${label}</span><div class="bar"><div class="fill" style="width:${Math.max(0,Math.min(100,100*row.top_probability))}%"></div></div><span>${row.top_probability.toFixed(3)}</span></div>`}
for(const [task,items] of Object.entries(grouped)){
 const snapshots=byTime(items),layers=[...new Set(items.filter(r=>r.kind==='reg_op').map(r=>r.layer))].sort((a,b)=>a-b);
 const section=document.createElement('section');section.className='task';section.innerHTML=`<h2>${task}</h2><div class="legend"><span>circle size = top-action probability</span><span>line color = selected operator</span><span>faded point = high entropy</span></div><div class="canvas-wrap"><canvas></canvas></div><div class="snapshots"></div>`;root.append(section);
 const canvas=section.querySelector('canvas'),width=Math.max(940,120+snapshots.length*72),height=70+layers.length*38+36,dpr=Math.max(1,window.devicePixelRatio||1);canvas.style.width=width+'px';canvas.style.height=height+'px';canvas.width=width*dpr;canvas.height=height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.font='12px system-ui';
 const x=i=>110+(width-145)*(snapshots.length===1?0:i/(snapshots.length-1));
 snapshots.forEach((s,i)=>{ctx.strokeStyle='#e5e8ed';ctx.beginPath();ctx.moveTo(x(i),34);ctx.lineTo(x(i),height-26);ctx.stroke();ctx.fillStyle='#637083';ctx.textAlign='center';ctx.fillText(`t=${Number(s[0].t).toFixed(2)}`,x(i),height-9)});
 layers.forEach((layer,li)=>{const y=48+li*38;ctx.fillStyle='#17202a';ctx.textAlign='left';ctx.fillText(`L${layer} -> r${layerState(snapshots[0],layer)?.op.node??'?'}`,8,y+4);let previous=null;snapshots.forEach((snapshot,i)=>{const state=layerState(snapshot,layer);if(!state)return;const px=x(i),c=color(state.op.top_action_name);if(previous){ctx.strokeStyle=c;ctx.globalAlpha=.7;ctx.lineWidth=2+3*Math.min(state.a?.top_probability??0,state.b?.top_probability??1);ctx.beginPath();ctx.moveTo(previous.x,y);ctx.lineTo(px,y);ctx.stroke()}ctx.globalAlpha=Math.max(.25,1-Math.min(1,state.op.entropy/Math.log(Math.max(state.op.probabilities.length,2))));ctx.fillStyle=c;ctx.beginPath();ctx.arc(px,y,3+7*state.op.top_probability,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;if(!previous||previous.name!==state.op.top_action_name||i===snapshots.length-1){ctx.fillStyle='#17202a';ctx.font='10px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(state.op.top_action_name,px,y-10);ctx.font='12px system-ui'}previous={x:px,name:state.op.top_action_name}})});
 const picks=[0,Math.floor((snapshots.length-1)/4),Math.floor((snapshots.length-1)/2),Math.floor(3*(snapshots.length-1)/4),snapshots.length-1].filter((v,i,a)=>a.indexOf(v)===i);
 const snapRoot=section.querySelector('.snapshots');for(const index of picks){const snapshot=snapshots[index],details=document.createElement('details');if(index===picks.length-1)details.open=true;let body='';for(const layer of layers){const state=layerState(snapshot,layer);if(!state)continue;body+=`<div class="state"><div class="state-title">${state.label}</div>${meter('op',state.op)}${meter('arg0',state.a)}${meter('arg1',state.b)}</div>`}for(const readout of snapshot.filter(r=>r.kind==='readout'))body+=`<div class="state"><div class="state-title">output <- ${readout.top_action_name}</div>${meter('read',readout,true)}</div>`;details.innerHTML=`<summary>t=${Number(snapshot[0].t).toFixed(3)}</summary>${body}`;snapRoot.append(details)}
}
</script></body></html>"""
    (out_dir / "temporal_action_weights.html").write_text(html)


def fit_affine(train_pred: torch.Tensor, y_train: torch.Tensor, test_pred: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    if float(train_pred.float().std().detach().cpu().item()) < 1.0e-8:
        intercept = y_train.float().mean()
        return torch.full_like(test_pred.float(), float(intercept.detach().cpu().item())), [0.0, float(intercept.detach().cpu().item())]
    a = torch.stack([train_pred.float(), torch.ones_like(train_pred.float())], dim=1)
    try:
        sol = torch.linalg.lstsq(a, y_train.float()).solution
    except Exception:
        sol = torch.tensor([1.0, 0.0], dtype=train_pred.dtype, device=train_pred.device)
    if not bool(torch.isfinite(sol).all().detach().cpu().item()):
        sol = torch.tensor([0.0, float(y_train.float().mean().detach().cpu().item())], dtype=train_pred.dtype, device=train_pred.device)
    return sol[0] * test_pred.float() + sol[1], [float(sol[0].detach().cpu().item()), float(sol[1].detach().cpu().item())]


def fit_linear_terms(
    train_terms: list[torch.Tensor],
    y_train: torch.Tensor,
    test_terms: list[torch.Tensor],
    *,
    ridge: float = 1.0e-8,
    max_abs: float = 1.0e6,
) -> tuple[torch.Tensor, torch.Tensor, list[float], float]:
    if not train_terms or not test_terms:
        intercept = float(y_train.float().mean().detach().cpu().item())
        return (
            torch.full_like(y_train.float(), intercept),
            torch.full((int(test_terms[0].shape[0]),), intercept, dtype=y_train.dtype, device=y_train.device)
            if test_terms
            else torch.empty((0,), dtype=y_train.dtype, device=y_train.device),
            [],
            intercept,
        )
    train_cols = [sanitize_values(value, clip=float(max_abs)) for value in train_terms]
    test_cols = [sanitize_values(value, clip=float(max_abs)) for value in test_terms]
    solve_dtype = torch.float64
    x_train = torch.stack(train_cols, dim=1).to(dtype=solve_dtype)
    x_test = torch.stack(test_cols, dim=1).to(dtype=solve_dtype)
    ones_train = torch.ones((int(x_train.shape[0]), 1), dtype=x_train.dtype, device=x_train.device)
    ones_test = torch.ones((int(x_test.shape[0]), 1), dtype=x_test.dtype, device=x_test.device)
    design_train = torch.cat([x_train, ones_train], dim=1)
    design_test = torch.cat([x_test, ones_test], dim=1)
    target = y_train.to(dtype=solve_dtype)
    eye = torch.eye(int(design_train.shape[1]), dtype=design_train.dtype, device=design_train.device)
    eye[-1, -1] = 0.0
    try:
        sol = torch.linalg.solve(
            design_train.T @ design_train + max(float(ridge), 0.0) * eye,
            design_train.T @ target,
        )
    except Exception:
        try:
            sol = torch.linalg.lstsq(design_train, target).solution
        except Exception:
            sol = torch.zeros((int(design_train.shape[1]),), dtype=design_train.dtype, device=design_train.device)
            sol[-1] = target.mean()
    if not bool(torch.isfinite(sol).all().detach().cpu().item()):
        sol = torch.zeros((int(design_train.shape[1]),), dtype=design_train.dtype, device=design_train.device)
        sol[-1] = target.mean()
    fitted_train = sanitize_values((design_train @ sol).to(dtype=y_train.dtype), clip=float(max_abs))
    fitted_test = sanitize_values((design_test @ sol).to(dtype=y_train.dtype), clip=float(max_abs))
    coeffs = [float(value.detach().cpu().item()) for value in sol[:-1]]
    intercept = float(sol[-1].detach().cpu().item())
    return fitted_train, fitted_test, coeffs, intercept


def _term_fit_expression(term_strings: list[str], coeffs: list[float], intercept: float) -> str:
    pieces: list[str] = []
    for coeff, term in zip(coeffs, term_strings):
        if abs(float(coeff)) <= 1.0e-10:
            continue
        pieces.append(f"({float(coeff):.8g})*({term})")
    if abs(float(intercept)) > 1.0e-10 or not pieces:
        pieces.append(f"{float(intercept):.8g}")
    return " + ".join(pieces)


def evaluate_expression(
    expr: Expr,
    task: TaskBundle,
    *,
    terms: list[Expr] | None = None,
    term_fit_ridge: float = 1.0e-8,
    term_fit_max_abs: float = 1.0e6,
) -> dict[str, Any]:
    readout_terms = list(terms) if terms is not None else [expr]
    raw_terms = [term for term in readout_terms if not _expr_is_zero(term)]
    unique_terms, duplicate_terms = _unique_nonzero_terms(raw_terms)
    if not unique_terms:
        unique_terms = [Expr.const(0.0)]
    train_pred = sanitize_values(eval_expr(expr, task.x_train), clip=float(term_fit_max_abs))
    test_pred = sanitize_values(eval_expr(expr, task.x_test), clip=float(term_fit_max_abs))
    fitted_test, coeffs = fit_affine(train_pred, task.y_train, test_pred)
    fitted_train = coeffs[0] * train_pred.float() + coeffs[1]
    train_term_values = [sanitize_values(eval_expr(term, task.x_train), clip=float(term_fit_max_abs)) for term in unique_terms]
    test_term_values = [sanitize_values(eval_expr(term, task.x_test), clip=float(term_fit_max_abs)) for term in unique_terms]
    term_fit_train, term_fit_test, term_coeffs, term_intercept = fit_linear_terms(
        train_term_values,
        task.y_train,
        test_term_values,
        ridge=float(term_fit_ridge),
        max_abs=float(term_fit_max_abs),
    )
    raw_r2 = r2_score(task.y_test.detach().cpu().numpy(), test_pred.detach().cpu().numpy())
    raw_train_r2 = r2_score(task.y_train.detach().cpu().numpy(), train_pred.detach().cpu().numpy())
    raw_nmse_value = nmse(task.y_test.detach().cpu().numpy(), test_pred.detach().cpu().numpy())
    raw_train_nmse_value = nmse(task.y_train.detach().cpu().numpy(), train_pred.detach().cpu().numpy())
    r2 = r2_score(task.y_test.detach().cpu().numpy(), fitted_test.detach().cpu().numpy())
    term_strings = [to_string(term, int(task.num_vars), simplify=False) for term in unique_terms]
    term_fit_r2 = r2_score(task.y_test.detach().cpu().numpy(), term_fit_test.detach().cpu().numpy())
    term_fit_train_r2 = r2_score(task.y_train.detach().cpu().numpy(), term_fit_train.detach().cpu().numpy())
    return {
        "r2": float(r2),
        "nmse": float(nmse(task.y_test.detach().cpu().numpy(), fitted_test.detach().cpu().numpy())),
        "train_r2": float(r2_score(task.y_train.detach().cpu().numpy(), fitted_train.detach().cpu().numpy())),
        "train_nmse": float(nmse(task.y_train.detach().cpu().numpy(), fitted_train.detach().cpu().numpy())),
        "raw_test_r2_without_affine": float(raw_r2),
        "raw_train_r2_without_affine": float(raw_train_r2),
        "raw_nmse_without_affine": float(raw_nmse_value),
        "raw_train_nmse_without_affine": float(raw_train_nmse_value),
        "term_fit_coefficients": coeffs[:1],
        "term_fit_intercept": coeffs[1],
        "term_fit_nonzero_coefficient_count": int(abs(coeffs[0]) > 1.0e-8) + int(abs(coeffs[1]) > 1.0e-8),
        "global_affine_test_r2": float(r2),
        "global_affine_test_nmse": float(nmse(task.y_test.detach().cpu().numpy(), fitted_test.detach().cpu().numpy())),
        "global_affine_train_r2": float(r2_score(task.y_train.detach().cpu().numpy(), fitted_train.detach().cpu().numpy())),
        "global_affine_train_nmse": float(nmse(task.y_train.detach().cpu().numpy(), fitted_train.detach().cpu().numpy())),
        "term_linear_fit_r2": float(term_fit_r2),
        "term_linear_fit_nmse": float(nmse(task.y_test.detach().cpu().numpy(), term_fit_test.detach().cpu().numpy())),
        "term_linear_fit_train_r2": float(term_fit_train_r2),
        "term_linear_fit_train_nmse": float(nmse(task.y_train.detach().cpu().numpy(), term_fit_train.detach().cpu().numpy())),
        "term_linear_fit_coefficients": term_coeffs,
        "term_linear_fit_intercept": float(term_intercept),
        "term_linear_fit_nonzero_coefficient_count": int(sum(abs(float(value)) > 1.0e-8 for value in term_coeffs)) + int(abs(float(term_intercept)) > 1.0e-8),
        "term_linear_fit_expression": _term_fit_expression(term_strings, term_coeffs, float(term_intercept)),
        "readout_slot_count": int(len(readout_terms)),
        "term_count": int(len(raw_terms)),
        "unique_nonzero_term_count": int(0 if all(_expr_is_zero(term) for term in unique_terms) else len(unique_terms)),
        "duplicate_term_count": int(duplicate_terms),
        "term_complexities": [int(term.complexity) for term in unique_terms],
        "term_expressions": term_strings,
    }


def sample_choices(theta: torch.Tensor, template: FixedSymbolTemplate, rng: torch.Generator) -> list[int]:
    out = []
    for bidx, logits in enumerate(split_blocks(theta, template)):
        p = masked_single_block_softmax(logits.float(), template, bidx)
        out.append(int(torch.multinomial(p, 1, generator=rng).detach().cpu().item()))
    return out


def hard_decode_choices(theta: torch.Tensor, template: Any) -> list[int]:
    choices: list[int] = []
    for bidx, logits in enumerate(split_blocks(theta, template)):
        support = graph_block_mask(template, int(bidx), device=logits.device)
        masked = logits.float().masked_fill(~support, -1.0e9)
        choices.append(int(torch.argmax(masked).detach().cpu().item()))
    return choices


def _gt_rollout_diagnostics(
    template: RegisterOperatorSimplexTemplate,
    theta: torch.Tensor,
    probabilities: torch.Tensor,
    choices: list[int],
    task: TaskBundle,
    generator: torch.Generator,
    *,
    sample_count: int,
) -> dict[str, Any]:
    expr, _terms, _layers = execute_choices(template, choices)
    expression = to_string(expr, int(template.num_vars), simplify=False)
    sampled_hits = 0
    for _ in range(max(int(sample_count), 0)):
        sampled_choices = sample_choices(theta, template, generator)
        sampled_expr, _sampled_terms, _sampled_layers = execute_choices(template, sampled_choices)
        sampled_expression = to_string(sampled_expr, int(template.num_vars), simplify=False)
        sampled_hits += int(_symbolic_equiv(task.ground_truth, sampled_expression))
    trace_rows: list[tuple[float, float, float]] = []
    for trace in task.traces:
        trace_choices = [int(value) for value in trace.get("choices", [])]
        active_indices = [int(value) for value in trace.get("active_block_indices", [])]
        if not trace_choices or not active_indices:
            continue
        active = torch.tensor(active_indices, dtype=torch.long, device=probabilities.device)
        selected = torch.tensor([trace_choices[index] for index in active_indices], dtype=torch.long, device=probabilities.device)
        selected_probability = probabilities[active, selected].clamp_min(1.0e-12)
        trace_rows.append((
            float(selected_probability.log().mean().exp().cpu()),
            float((probabilities.argmax(dim=-1)[active] == selected).float().mean().cpu()),
            float(selected_probability.log().sum().cpu()),
        ))
    best = max(trace_rows, key=lambda row: (row[0], row[2])) if trace_rows else (0.0, 0.0, -math.inf)
    return {
        "flow_hard_expression": expression,
        "flow_hard_gt_symbolic_hit": float(_symbolic_equiv(task.ground_truth, expression)),
        "flow_sample_gt_hit_rate": float(sampled_hits / max(int(sample_count), 1)) if sample_count > 0 else 0.0,
        "flow_gt_trace_probability_geometric_mean_max": best[0],
        "flow_gt_trace_active_argmax_match_max": best[1],
        "flow_nearest_gt_cell_fr_rms": None,
        "flow_nearest_gt_cell_fr_mean": None,
    }


def terminal_single_expression_retraction(
    theta: torch.Tensor,
    template: RegisterOperatorSimplexTemplate,
    *,
    projection_eps: float,
) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    """Retract a soft endpoint to its decoded epsilon-sharp expression cell."""
    choices = hard_decode_choices(theta, template)
    active_indices = active_block_indices_for_choices(template, choices)
    active_mask = torch.zeros(len(template.blocks), dtype=torch.bool, device=theta.device)
    if active_indices:
        active_mask[torch.tensor(active_indices, dtype=torch.long, device=theta.device)] = True
    raw_probabilities = masked_block_softmax(
        theta.view(len(template.blocks), int(template.source_count)),
        template,
    )
    eps = float(projection_eps)
    selected = torch.tensor(choices, dtype=torch.long, device=theta.device)
    selected_probability = raw_probabilities[
        torch.arange(len(template.blocks), device=theta.device), selected
    ]
    already_sharp = bool(
        (selected_probability[active_mask] >= (1.0 - eps - 1.0e-7)).all().detach().cpu()
    ) if bool(active_mask.any().detach().cpu()) else True
    if already_sharp:
        retracted_probabilities = raw_probabilities
        retracted_theta = theta.detach().clone()
    else:
        retracted_probabilities = source_conditioned_trace_target_probabilities(
            raw_probabilities,
            selected,
            active_mask,
            projection_eps=eps,
            projection_sharpness=1.0,
        )
        retracted_theta = logits_from_block_probabilities(
            [retracted_probabilities[index] for index in range(int(retracted_probabilities.shape[0]))],
            template,
            eps=1.0e-8,
        )
    root_raw = raw_probabilities.clamp_min(1.0e-12).sqrt()
    root_retracted = retracted_probabilities.clamp_min(1.0e-12).sqrt()
    half_angle = torch.atan2(
        (root_raw - root_retracted).norm(dim=-1),
        (root_raw + root_retracted).norm(dim=-1).clamp_min(1.0e-12),
    )
    block_distance = 4.0 * half_angle
    inactive_mask = ~active_mask
    active_distance = block_distance[active_mask] if bool(active_mask.any().detach().cpu()) else block_distance.new_zeros(1)
    pre_expr, _pre_terms, _pre_layers = execute_choices(template, choices)
    post_choices = hard_decode_choices(retracted_theta, template)
    post_expr, _post_terms, _post_layers = execute_choices(template, post_choices)
    diagnostics = {
        "terminal_retraction_enabled": 1.0,
        "terminal_retraction_applied": float(not already_sharp),
        "terminal_retraction_pre_expression": to_string(pre_expr, int(template.num_vars), simplify=False),
        "terminal_retraction_post_expression": to_string(post_expr, int(template.num_vars), simplify=False),
        "terminal_retraction_expression_preserved": float(
            to_string(pre_expr, int(template.num_vars), simplify=True)
            == to_string(post_expr, int(template.num_vars), simplify=True)
        ),
        "terminal_retraction_fr_mean": float(block_distance.mean().detach().cpu()),
        "terminal_retraction_fr_p95": float(torch.quantile(block_distance, 0.95).detach().cpu()),
        "terminal_retraction_fr_max": float(block_distance.max().detach().cpu()),
        "terminal_retraction_active_fr_mean": float(active_distance.mean().detach().cpu()),
        "terminal_retraction_active_fr_rms": float(torch.sqrt(active_distance.square().mean()).detach().cpu()),
        "terminal_retraction_active_fr_p95": float(torch.quantile(active_distance, 0.95).detach().cpu()),
        "terminal_retraction_active_fr_max": float(active_distance.max().detach().cpu()),
        "terminal_retraction_inactive_fr_max": float(block_distance[inactive_mask].max().detach().cpu()) if bool(inactive_mask.any().detach().cpu()) else 0.0,
        "terminal_retraction_active_block_count": int(active_mask.sum().detach().cpu()),
    }
    return retracted_theta, post_choices, diagnostics


def _candidate_train_selection_score(metrics: dict[str, Any], expr: Expr) -> float:
    raw_train = float(metrics.get("raw_train_r2_without_affine", -1.0e9))
    term_train = float(metrics.get("term_linear_fit_train_r2", metrics.get("train_r2", -1.0e9)))
    global_train = float(metrics.get("train_r2", -1.0e9))
    fitted_gap = max(0.0, max(term_train, global_train) - raw_train - 0.25)
    raw_guard = max(raw_train, -1.0)
    return float(
        0.70 * term_train
        + 0.25 * global_train
        + 0.05 * raw_guard
        - 0.02 * fitted_gap
        - 1.0e-3 * float(expr.complexity)
    )


def endpoint_probability_diagnostics(
    template: FixedSymbolTemplate,
    theta: torch.Tensor,
    task: TaskBundle,
    choices: list[int],
) -> dict[str, Any]:
    blocks = split_blocks(theta, template)
    entropies: list[float] = []
    max_probs: list[float] = []
    readout_max_probs: list[float] = []
    nonreadout_max_probs: list[float] = []
    argmax: list[int] = []
    for bidx, logits in enumerate(blocks):
        p = masked_single_block_softmax(logits.float(), template, int(bidx))
        support = graph_block_mask(template, int(bidx), device=p.device)
        support_count = max(int(support.sum().detach().cpu().item()), 2)
        entropies.append(float((-(p * p.clamp_min(1.0e-8).log()).sum() / math.log(support_count)).detach().cpu().item()))
        top = float(p.max().detach().cpu().item())
        max_probs.append(top)
        argmax.append(int(torch.argmax(p).detach().cpu().item()))
        if template.blocks[int(bidx)].kind == "readout":
            readout_max_probs.append(top)
        else:
            nonreadout_max_probs.append(top)

    def choice_stats(active: list[int], trace_choices: list[int]) -> dict[str, float]:
        probs: list[float] = []
        matches: list[float] = []
        for bidx in active:
            idx = int(bidx)
            if idx < 0 or idx >= len(blocks) or idx >= len(trace_choices):
                continue
            action = int(trace_choices[idx])
            if action < 0 or action >= int(blocks[idx].numel()):
                continue
            p = masked_single_block_softmax(blocks[idx].float(), template, idx)
            prob = float(p[action].detach().cpu().item())
            probs.append(prob)
            matches.append(float(argmax[idx] == action))
        log_probs = [math.log(max(value, 1.0e-12)) for value in probs]
        return {
            "block_count": float(len(probs)),
            "prob_mean": float(np.mean(probs)) if probs else 0.0,
            "prob_min": float(np.min(probs)) if probs else 0.0,
            "logprob_mean": float(np.mean(log_probs)) if log_probs else 0.0,
            "logprob_sum": float(np.sum(log_probs)) if log_probs else 0.0,
            "argmax_match_mean": float(np.mean(matches)) if matches else 0.0,
        }

    chosen_active = active_block_indices_for_choices(template, choices) if choices else []
    chosen = choice_stats(chosen_active, choices) if choices else {
        "block_count": 0.0,
        "prob_mean": 0.0,
        "prob_min": 0.0,
        "logprob_mean": 0.0,
        "logprob_sum": 0.0,
        "argmax_match_mean": 0.0,
    }
    best_trace_idx = -1
    best_trace = {
        "block_count": 0.0,
        "prob_mean": 0.0,
        "prob_min": 0.0,
        "logprob_mean": 0.0,
        "logprob_sum": 0.0,
        "argmax_match_mean": 0.0,
    }
    for trace_idx, trace in enumerate(task.traces):
        stats = choice_stats(
            [int(v) for v in trace.get("active_block_indices", [])],
            [int(v) for v in trace.get("choices", [])],
        )
        if (
            float(stats["argmax_match_mean"]) > float(best_trace["argmax_match_mean"])
            or (
                float(stats["argmax_match_mean"]) == float(best_trace["argmax_match_mean"])
                and float(stats["prob_mean"]) > float(best_trace["prob_mean"])
            )
        ):
            best_trace_idx = int(trace_idx)
            best_trace = stats

    return {
        "endpoint_masked_terminal_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "endpoint_masked_terminal_max_prob_mean": float(np.mean(max_probs)) if max_probs else 0.0,
        "endpoint_masked_readout_max_prob_mean": float(np.mean(readout_max_probs)) if readout_max_probs else 0.0,
        "endpoint_masked_nonreadout_max_prob_mean": float(np.mean(nonreadout_max_probs)) if nonreadout_max_probs else 0.0,
        "endpoint_sample_active_block_count": int(chosen["block_count"]),
        "endpoint_sample_active_prob_mean": float(chosen["prob_mean"]),
        "endpoint_sample_active_prob_min": float(chosen["prob_min"]),
        "endpoint_sample_active_logprob_mean": float(chosen["logprob_mean"]),
        "endpoint_sample_active_logprob_sum": float(chosen["logprob_sum"]),
        "endpoint_sample_active_argmax_match_mean": float(chosen["argmax_match_mean"]),
        "endpoint_trace_family_best_index": int(best_trace_idx),
        "endpoint_trace_family_best_active_block_count": int(best_trace["block_count"]),
        "endpoint_trace_family_best_argmax_match": float(best_trace["argmax_match_mean"]),
        "endpoint_trace_family_best_active_mean_prob": float(best_trace["prob_mean"]),
        "endpoint_trace_family_best_active_min_prob": float(best_trace["prob_min"]),
        "endpoint_trace_family_best_active_logprob_mean": float(best_trace["logprob_mean"]),
        "endpoint_trace_family_best_active_logprob_sum": float(best_trace["logprob_sum"]),
    }


def select_best_rollout(
    model: FixedSymbolConditionedVelocityNet,
    score_model: None,
    task: TaskBundle,
    args: argparse.Namespace,
    generator: torch.Generator,
    *,
    mode: str,
    steps: int,
) -> tuple[dict[str, Any], int, int, int, float]:
    start_time = time.perf_counter()
    best: dict[str, Any] | None = None
    valid_exprs: list[str] = []
    endpoint_decode_mode = str(getattr(args, "eval_endpoint_decode_mode", "hard_argmax"))
    candidates_per_theta0 = 1 if endpoint_decode_mode == "hard_argmax" else max(int(args.eval_samples), 1)
    total_candidates = max(int(args.eval_theta0_samples) * candidates_per_theta0, 1)
    hard_mode_counts: dict[str, int] = {}
    population_rows: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    eval_seed_rng = random.Random(int(args.seed) + 88_019)
    for _theta_sample in range(int(args.eval_theta0_samples)):
        theta0, theta0_mode = sample_eval_theta0(
            model.template,
            task,
            args,
            eval_seed_rng,
            next(model.parameters()).device,
            sample_index=int(_theta_sample),
        )
        theta0_diag = theta0_diagnostics(theta0, model.template)
        theta, guide_diag = rollout(
            model,
            score_model,
            task,
            theta0,
            steps=int(steps),
            mode=mode,
            args=args,
            generator=generator,
        )
        raw_terminal_diag = terminal_summary(theta, model.template)
        raw_choices = hard_decode_choices(theta, model.template)
        raw_probabilities = masked_block_softmax(
            theta.view(len(model.template.blocks), int(model.template.source_count)),
            model.template,
        )
        gt_probe_generator = torch.Generator(device=theta.device).manual_seed(
            int(args.seed) + _stable_task_seed(task.task_id) + 104_729 * int(_theta_sample)
        )
        pre_retraction_gt_diag = _gt_rollout_diagnostics(
            model.template,
            theta,
            raw_probabilities,
            raw_choices,
            task,
            gt_probe_generator,
            sample_count=max(int(getattr(args, "eval_flow_gt_probe_samples", 4)), 0),
        )
        direct_gt_diag = {
            "flow_pre_retraction_hard_expression": str(pre_retraction_gt_diag["flow_hard_expression"]),
            "flow_pre_retraction_hard_gt_symbolic_hit": float(pre_retraction_gt_diag["flow_hard_gt_symbolic_hit"]),
            "flow_pre_retraction_sample_gt_hit_rate": float(pre_retraction_gt_diag["flow_sample_gt_hit_rate"]),
            "flow_pre_retraction_sample_probe_count": int(pre_retraction_gt_diag["flow_sample_probe_count"]),
            "flow_pre_retraction_gt_trace_probability_geometric_mean": float(
                pre_retraction_gt_diag["flow_gt_trace_probability_geometric_mean_max"]
            ),
            "flow_pre_retraction_gt_trace_active_argmax_match": float(
                pre_retraction_gt_diag["flow_gt_trace_active_argmax_match_max"]
            ),
            "flow_pre_retraction_nearest_gt_cell_fr_rms": float(
                pre_retraction_gt_diag["flow_nearest_gt_cell_fr_rms"]
            ),
            "flow_pre_retraction_nearest_gt_cell_fr_mean": float(
                pre_retraction_gt_diag["flow_nearest_gt_cell_fr_mean"]
            ),
        }
        retraction_enabled = bool(getattr(args, "eval_terminal_retraction", True))
        retraction_eps = float(getattr(args, "eval_terminal_retraction_eps", -1.0))
        if retraction_eps < 0.0:
            retraction_eps = float(getattr(args, "cycle_projection_eps", 0.02))
        if retraction_enabled:
            decoded_theta, retracted_choices, retraction_diag = terminal_single_expression_retraction(
                theta,
                model.template,
                projection_eps=retraction_eps,
            )
        else:
            decoded_theta = theta
            retracted_choices = hard_decode_choices(theta, model.template)
            raw_expr, _raw_terms, _raw_layers = execute_choices(model.template, retracted_choices)
            raw_expr_string = to_string(raw_expr, int(model.template.num_vars), simplify=False)
            retraction_diag = {
                "terminal_retraction_enabled": 0.0,
                "terminal_retraction_applied": 0.0,
                "terminal_retraction_pre_expression": raw_expr_string,
                "terminal_retraction_post_expression": raw_expr_string,
                "terminal_retraction_expression_preserved": 1.0,
                "terminal_retraction_fr_mean": 0.0,
                "terminal_retraction_fr_p95": 0.0,
                "terminal_retraction_fr_max": 0.0,
                "terminal_retraction_active_fr_mean": 0.0,
                "terminal_retraction_active_fr_rms": 0.0,
                "terminal_retraction_active_fr_p95": 0.0,
                "terminal_retraction_active_fr_max": 0.0,
                "terminal_retraction_inactive_fr_max": 0.0,
                "terminal_retraction_active_block_count": int(len(active_block_indices_for_choices(model.template, retracted_choices))),
            }
        if endpoint_decode_mode == "hard_argmax":
            candidate_choices = [retracted_choices]
        elif endpoint_decode_mode == "soft_sample":
            candidate_choices = [sample_choices(decoded_theta, model.template, generator) for _ in range(max(int(args.eval_samples), 1))]
        else:
            raise ValueError(f"unknown eval endpoint decode mode: {endpoint_decode_mode}")
        for candidate_index, choices in enumerate(candidate_choices):
            try:
                expr, terms, _ = execute_choices(model.template, choices)
                raw_expr = to_string(expr, int(model.template.num_vars), simplify=False)
                valid_exprs.append(raw_expr)
                hard_mode_counts[raw_expr] = hard_mode_counts.get(raw_expr, 0) + 1
                metrics = evaluate_expression(
                    expr,
                    task,
                    terms=terms,
                    term_fit_ridge=float(getattr(args, "term_fit_ridge", 1.0e-8)),
                    term_fit_max_abs=float(getattr(args, "term_fit_max_abs", 1.0e6)),
                )
                # Candidate selection must use inference-available train fit.
                # Affine/term fits remain diagnostic only.
                diagnostic_fit_score = _candidate_train_selection_score(metrics, expr)
                raw_train_prediction = sanitize_values(eval_expr(expr, task.x_train))
                semantic_vector = semantic_signature_vector(raw_train_prediction, task.x_train).detach().cpu()
                target_semantic_energy = float(
                    semantic_signature_distance(raw_train_prediction, task.y_train, task.x_train).detach().cpu()
                )
                endpoint_diag = endpoint_probability_diagnostics(model.template, decoded_theta, task, choices)
            except Exception:
                continue
            population_row = {
                "theta0_index": int(_theta_sample),
                "candidate_index": int(candidate_index),
                "theta0_mode": str(theta0_mode),
                **theta0_diag,
                "raw_expression": raw_expr,
                "term_expressions": metrics.get("term_expressions", []),
                "term_count": int(metrics.get("term_count", 0)),
                "unique_nonzero_term_count": int(metrics.get("unique_nonzero_term_count", 0)),
                "duplicate_term_count": int(metrics.get("duplicate_term_count", 0)),
                "diagnostic_affine_term_selection_score": float(diagnostic_fit_score),
                "oracle_free_target_semantic_energy": float(target_semantic_energy),
                "raw_train_r2_without_affine": float(metrics.get("raw_train_r2_without_affine", -1.0e9)),
                "raw_test_r2_without_affine": float(metrics.get("raw_test_r2_without_affine", -1.0e9)),
                "global_affine_test_r2": float(metrics.get("r2", -1.0e9)),
                "term_linear_fit_test_r2": float(metrics.get("term_linear_fit_r2", -1.0e9)),
                "terminal_entropy_mean": float(terminal_summary(decoded_theta, model.template).get("terminal_entropy_mean", 0.0)),
                "terminal_max_prob_mean": float(terminal_summary(decoded_theta, model.template).get("terminal_max_prob_mean", 0.0)),
                "terminal_pre_retraction_entropy_mean": float(raw_terminal_diag.get("terminal_entropy_mean", 0.0)),
                "terminal_pre_retraction_max_prob_mean": float(raw_terminal_diag.get("terminal_max_prob_mean", 0.0)),
                "endpoint_sample_active_prob_mean": float(endpoint_diag.get("endpoint_sample_active_prob_mean", 0.0)),
                "endpoint_trace_family_best_argmax_match": float(endpoint_diag.get("endpoint_trace_family_best_argmax_match", 0.0)),
                **direct_gt_diag,
                **retraction_diag,
            }
            population_rows.append(population_row)
            candidate_records.append({
                "expr": expr,
                "raw_expression": raw_expr,
                "raw_terms": [to_string(term, int(model.template.num_vars), simplify=False) for term in terms],
                "choices": choices,
                "theta": decoded_theta,
                "theta0": theta0,
                "eval_theta0_index": int(_theta_sample),
                "eval_theta0_mode": theta0_mode,
                **theta0_diag,
                "eval_endpoint_decode_mode": endpoint_decode_mode,
                "diagnostic_affine_term_selection_score": float(diagnostic_fit_score),
                "oracle_free_target_semantic_energy": float(target_semantic_energy),
                "_semantic_vector": semantic_vector,
                **metrics,
                **terminal_summary(decoded_theta, model.template),
                "terminal_pre_retraction_entropy_mean": float(raw_terminal_diag.get("terminal_entropy_mean", 0.0)),
                "terminal_pre_retraction_max_prob_mean": float(raw_terminal_diag.get("terminal_max_prob_mean", 0.0)),
                **endpoint_diag,
                **direct_gt_diag,
                **retraction_diag,
                **guide_diag,
            })
    if candidate_records:
        semantic_matrix = torch.stack([row["_semantic_vector"] for row in candidate_records], dim=0).float()
        pairwise = (semantic_matrix[:, None, :] - semantic_matrix[None, :, :]).square().mean(dim=-1)
        medoid_distance = pairwise.mean(dim=1)
        medoid_index = min(
            range(len(candidate_records)),
            key=lambda index: (
                float(medoid_distance[index]),
                float(candidate_records[index]["oracle_free_target_semantic_energy"]),
                int(candidate_records[index]["expr"].complexity),
            ),
        )
        fit_index = max(
            range(len(candidate_records)),
            key=lambda index: (
                float(candidate_records[index].get("diagnostic_affine_term_selection_score", -1.0e9)),
                float(candidate_records[index].get("term_linear_fit_train_r2", -1.0e9)),
                float(candidate_records[index].get("train_r2", -1.0e9)),
                -int(candidate_records[index]["expr"].complexity),
            ),
        )
        selection_mode = str(getattr(args, "eval_oracle_free_selection_mode", "train_fit"))
        selected_index = medoid_index if selection_mode == "semantic_medoid" else fit_index
        for index, population_row in enumerate(population_rows):
            population_row["oracle_free_population_medoid_distance"] = float(medoid_distance[index])
            population_row["oracle_free_population_medoid_selected"] = float(index == medoid_index)
            population_row["oracle_free_population_train_fit_selected"] = float(index == fit_index)
            population_row["oracle_free_population_selected"] = float(index == selected_index)
        best = dict(candidate_records[selected_index])
        best.pop("_semantic_vector", None)
        if selection_mode == "semantic_medoid":
            best["selection_score"] = -float(medoid_distance[selected_index])
            best["eval_oracle_free_selection"] = "raw_semantic_population_medoid"
        else:
            best["selection_score"] = float(best.get("diagnostic_affine_term_selection_score", -1.0e9))
            best["eval_oracle_free_selection"] = "train_fit_score"
        best["eval_oracle_free_medoid_distance"] = float(medoid_distance[selected_index])
        best["eval_oracle_free_selection_mode"] = selection_mode
        best["eval_oracle_free_medoid_candidate_expression"] = str(candidate_records[medoid_index]["raw_expression"])
        best["eval_oracle_free_train_fit_candidate_expression"] = str(candidate_records[fit_index]["raw_expression"])
        best["eval_oracle_free_train_fit_score"] = float(
            candidate_records[fit_index].get("diagnostic_affine_term_selection_score", -1.0e9)
        )
        best["eval_oracle_free_medoid_score"] = -float(medoid_distance[medoid_index])
        selected_expression = str(best["raw_expression"])
        best["eval_oracle_free_medoid_expression_share"] = float(
            hard_mode_counts.get(selected_expression, 0) / max(sum(hard_mode_counts.values()), 1)
        )
    hard_counts = list(hard_mode_counts.values())
    hard_total = int(sum(hard_counts))
    if hard_total > 0:
        probs = np.asarray(hard_counts, dtype=np.float64) / float(hard_total)
        entropy = float(-(probs * np.log(np.maximum(probs, 1.0e-12))).sum() / math.log(max(len(hard_counts), 2)))
        top_share = float(max(hard_counts) / float(hard_total))
    else:
        entropy = 0.0
        top_share = 0.0
    if best is None:
        device = next(model.parameters()).device
        best = {
            "expr": Expr.const(0.0),
            "raw_expression": "0",
            "raw_terms": [],
            "choices": [],
            "theta": torch.zeros(theta_dim(model.template), device=device),
            "theta0": torch.zeros(theta_dim(model.template), device=device),
            "selection_score": -1.0e9,
            "r2": 0.0,
            "nmse": 1.0e9,
            "train_r2": 0.0,
            "train_nmse": 1.0e9,
            "raw_test_r2_without_affine": 0.0,
            "raw_train_r2_without_affine": 0.0,
            "raw_nmse_without_affine": 1.0e9,
            "raw_train_nmse_without_affine": 1.0e9,
            "term_linear_fit_r2": 0.0,
            "term_linear_fit_nmse": 1.0e9,
            "term_linear_fit_train_r2": 0.0,
            "term_linear_fit_train_nmse": 1.0e9,
            "term_linear_fit_coefficients": [],
            "term_linear_fit_intercept": 0.0,
            "term_fit_coefficients": [],
            "term_fit_intercept": 0.0,
            "term_fit_nonzero_coefficient_count": 0,
            "term_count": 0,
            "unique_nonzero_term_count": 0,
            "duplicate_term_count": 0,
            "term_expressions": [],
            "eval_theta0_mode": "none",
            "eval_theta0_index": -1,
            "eval_endpoint_decode_mode": endpoint_decode_mode,
            "theta0_hash": "",
            "theta0_argmax_key": "",
            "terminal_entropy_mean": 0.0,
            "terminal_max_prob_mean": 0.0,
            "endpoint_masked_terminal_entropy_mean": 0.0,
            "endpoint_masked_terminal_max_prob_mean": 0.0,
            "endpoint_masked_readout_max_prob_mean": 0.0,
            "endpoint_masked_nonreadout_max_prob_mean": 0.0,
            "endpoint_sample_active_block_count": 0,
            "endpoint_sample_active_prob_mean": 0.0,
            "endpoint_sample_active_prob_min": 0.0,
            "endpoint_sample_active_logprob_mean": 0.0,
            "endpoint_sample_active_logprob_sum": 0.0,
            "endpoint_sample_active_argmax_match_mean": 0.0,
            "endpoint_trace_family_best_index": -1,
            "endpoint_trace_family_best_active_block_count": 0,
            "endpoint_trace_family_best_argmax_match": 0.0,
            "endpoint_trace_family_best_active_mean_prob": 0.0,
            "endpoint_trace_family_best_active_min_prob": 0.0,
            "endpoint_trace_family_best_active_logprob_mean": 0.0,
            "endpoint_trace_family_best_active_logprob_sum": 0.0,
            "guidance_fr_norm": 0.0,
            "guidance_cap_ratio": 0.0,
        }
    raw_population = [
        float(row["raw_test_r2_without_affine"])
        for row in population_rows
        if math.isfinite(float(row.get("raw_test_r2_without_affine", float("nan"))))
    ]
    term_population = [
        float(row["term_linear_fit_test_r2"])
        for row in population_rows
        if math.isfinite(float(row.get("term_linear_fit_test_r2", float("nan"))))
    ]
    retraction_population = [
        float(row["terminal_retraction_fr_mean"])
        for row in population_rows
        if math.isfinite(float(row.get("terminal_retraction_fr_mean", float("nan"))))
    ]
    active_retraction_population = [
        float(row["terminal_retraction_active_fr_mean"])
        for row in population_rows
        if math.isfinite(float(row.get("terminal_retraction_active_fr_mean", float("nan"))))
    ]
    active_retraction_p95_population = [
        float(row["terminal_retraction_active_fr_p95"])
        for row in population_rows
        if math.isfinite(float(row.get("terminal_retraction_active_fr_p95", float("nan"))))
    ]
    flow_hard_gt_hits = [
        float(row["flow_pre_retraction_hard_gt_symbolic_hit"])
        for row in population_rows
    ]
    flow_sample_gt_hits = [
        float(row["flow_pre_retraction_sample_gt_hit_rate"])
        for row in population_rows
    ]
    flow_gt_trace_mass = [
        float(row["flow_pre_retraction_gt_trace_probability_geometric_mean"])
        for row in population_rows
        if math.isfinite(float(row.get("flow_pre_retraction_gt_trace_probability_geometric_mean", float("nan"))))
    ]
    flow_gt_argmax_match = [
        float(row["flow_pre_retraction_gt_trace_active_argmax_match"])
        for row in population_rows
        if math.isfinite(float(row.get("flow_pre_retraction_gt_trace_active_argmax_match", float("nan"))))
    ]
    flow_nearest_gt_cell = [
        float(row["flow_pre_retraction_nearest_gt_cell_fr_rms"])
        for row in population_rows
        if math.isfinite(float(row.get("flow_pre_retraction_nearest_gt_cell_fr_rms", float("nan"))))
    ]
    best.update({
        "eval_hard_expression_mode_entropy": float(entropy),
        "eval_hard_top_expression_share": float(top_share),
        "eval_hard_unique_expression_count": int(len(hard_mode_counts)),
        "eval_hard_decoded_candidate_count": int(hard_total),
        "eval_theta0_population": population_rows,
        "eval_theta0_unique_hash_count": int(len({str(row.get("theta0_hash", "")) for row in population_rows})),
        "eval_theta0_unique_argmax_count": int(len({str(row.get("theta0_argmax_key", "")) for row in population_rows})),
        "eval_population_unique_expression_count": int(len({str(row.get("raw_expression", "")) for row in population_rows})),
        "eval_population_multi_term_rate": float(np.mean([float(row.get("unique_nonzero_term_count", 0)) > 1.0 for row in population_rows])) if population_rows else 0.0,
        "eval_population_zero_expression_rate": float(np.mean([float(row.get("unique_nonzero_term_count", 0)) <= 0.0 for row in population_rows])) if population_rows else 0.0,
        "eval_population_raw_r2_mean": float(np.mean(raw_population)) if raw_population else None,
        "eval_population_raw_r2_median": float(np.median(raw_population)) if raw_population else None,
        "eval_population_raw_r2_best": float(np.max(raw_population)) if raw_population else None,
        "eval_population_term_fit_r2_mean": float(np.mean(term_population)) if term_population else None,
        "eval_population_term_fit_r2_best": float(np.max(term_population)) if term_population else None,
        "eval_gt_oracle_best_of_n_raw_r2": float(np.max(raw_population)) if raw_population else None,
        "eval_gt_oracle_best_of_n_term_fit_r2": float(np.max(term_population)) if term_population else None,
        "eval_terminal_retraction_fr_mean": float(np.mean(retraction_population)) if retraction_population else 0.0,
        "eval_terminal_retraction_fr_p95": float(np.quantile(retraction_population, 0.95)) if retraction_population else 0.0,
        "eval_terminal_retraction_active_fr_mean": float(np.mean(active_retraction_population)) if active_retraction_population else 0.0,
        "eval_terminal_retraction_active_fr_p95": float(np.quantile(active_retraction_p95_population, 0.95)) if active_retraction_p95_population else 0.0,
        "eval_terminal_retraction_expression_preserved_rate": float(np.mean([
            float(row.get("terminal_retraction_expression_preserved", 0.0)) for row in population_rows
        ])) if population_rows else 0.0,
        "eval_population_flow_hard_gt_hit_rate": float(np.mean(flow_hard_gt_hits)) if flow_hard_gt_hits else 0.0,
        "eval_population_flow_sample_gt_hit_rate": float(np.mean(flow_sample_gt_hits)) if flow_sample_gt_hits else 0.0,
        "eval_population_flow_gt_trace_probability_geometric_mean": float(np.mean(flow_gt_trace_mass)) if flow_gt_trace_mass else 0.0,
        "eval_population_flow_gt_trace_active_argmax_match": float(np.mean(flow_gt_argmax_match)) if flow_gt_argmax_match else 0.0,
        "eval_population_flow_nearest_gt_cell_fr_rms": float(np.mean(flow_nearest_gt_cell)) if flow_nearest_gt_cell else None,
    })
    return best, len(valid_exprs), len(set(valid_exprs)), total_candidates, float(time.perf_counter() - start_time)


def _reference_field_endpoint(
    model: FixedSymbolConditionedVelocityNet,
    task: TaskBundle,
    theta0: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Use the analytic source-conditioned Fisher bridge as the endpoint generator.

    This is an oracle/reference diagnostic: it uses the compiled trace family to
    select a target atom for the tracked source particle, then follows the
    closed-form Fisher-Rao bridge solution to ``t=1``.  No learned proposer and
    no learned velocity model participate in the endpoint generation.
    """
    if not task.traces:
        raise ValueError(f"task {task.task_id} has no compiled reference trace")
    trace = select_trace_for_theta0(model.template, theta0, task, args)
    device = theta0.device
    source_probabilities = masked_block_softmax(
        theta0.view(len(model.template.blocks), int(model.template.source_count)),
        model.template,
    )
    active_mask = torch.zeros(len(model.template.blocks), dtype=torch.bool, device=device)
    active_indices = [int(value) for value in trace.get("active_block_indices", [])]
    if active_indices:
        active_mask[torch.tensor(active_indices, dtype=torch.long, device=device)] = True
    projection_sharpness = float(getattr(args, "eval_reference_projection_sharpness", -1.0))
    if projection_sharpness < 0.0:
        projection_sharpness = float(getattr(args, "cycle_projection_sharpness", 0.7))
    target_probabilities = source_conditioned_trace_target_probabilities(
        source_probabilities,
        torch.tensor([int(value) for value in trace["choices"]], dtype=torch.long, device=device),
        active_mask,
        projection_eps=float(getattr(args, "cycle_projection_eps", 0.02)),
        projection_sharpness=projection_sharpness,
    )
    theta1 = logits_from_block_probabilities(
        [target_probabilities[block_index] for block_index in range(int(target_probabilities.shape[0]))],
        model.template,
        eps=float(args.fisher_eps),
    )
    endpoint, endpoint_velocity = stage1_simplex_path(theta0, theta1, model.template, 1.0)
    active_cost = (
        block_fisher_squared_distance(source_probabilities, target_probabilities, active_mask)
        if bool(active_mask.any().detach().cpu())
        else torch.tensor(0.0, device=device)
    )
    return endpoint, {
        "reference_field_oracle": 1.0,
        "reference_field_source": "compiled_trace_fisher_bridge",
        "reference_field_closed_form_ode_solution": 1.0,
        "reference_field_projection_sharpness": float(projection_sharpness),
        "reference_field_projection_eps": float(getattr(args, "cycle_projection_eps", 0.02)),
        "reference_field_active_block_count": int(active_mask.sum().detach().cpu()),
        "reference_field_source_to_target_fr": float(active_cost.detach().cpu()),
        "reference_field_endpoint_velocity_norm": float(endpoint_velocity.norm().detach().cpu()),
        "reference_trace_expression": str(trace.get("expression_string", "")),
        "reference_trace_active_choice_key": ",".join(str(int(trace["choices"][idx])) for idx in active_indices),
    }


def select_best_reference_field_rollout(
    model: FixedSymbolConditionedVelocityNet,
    task: TaskBundle,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[dict[str, Any], int, int, int, float]:
    start_time = time.perf_counter()
    best: dict[str, Any] | None = None
    valid_exprs: list[str] = []
    endpoint_decode_mode = str(getattr(args, "eval_endpoint_decode_mode", "hard_argmax"))
    candidates_per_theta0 = 1 if endpoint_decode_mode == "hard_argmax" else max(int(args.eval_samples), 1)
    total_candidates = max(int(args.eval_theta0_samples) * candidates_per_theta0, 1)
    hard_mode_counts: dict[str, int] = {}
    eval_seed_rng = random.Random(int(args.seed) + 188_019)
    for _theta_sample in range(int(args.eval_theta0_samples)):
        theta0, theta0_mode = sample_eval_theta0(
            model.template,
            task,
            args,
            eval_seed_rng,
            next(model.parameters()).device,
            sample_index=int(_theta_sample),
        )
        try:
            theta, reference_diag = _reference_field_endpoint(model, task, theta0, args)
        except Exception:
            continue
        if endpoint_decode_mode == "hard_argmax":
            candidate_choices = [hard_decode_choices(theta, model.template)]
        elif endpoint_decode_mode == "soft_sample":
            candidate_choices = [sample_choices(theta, model.template, generator) for _ in range(max(int(args.eval_samples), 1))]
        else:
            raise ValueError(f"unknown eval endpoint decode mode: {endpoint_decode_mode}")
        for choices in candidate_choices:
            try:
                expr, terms, _ = execute_choices(model.template, choices)
                raw_expr = to_string(expr, int(model.template.num_vars), simplify=False)
                valid_exprs.append(raw_expr)
                hard_mode_counts[raw_expr] = hard_mode_counts.get(raw_expr, 0) + 1
                metrics = evaluate_expression(
                    expr,
                    task,
                    terms=terms,
                    term_fit_ridge=float(getattr(args, "term_fit_ridge", 1.0e-8)),
                    term_fit_max_abs=float(getattr(args, "term_fit_max_abs", 1.0e6)),
                )
                score = _candidate_train_selection_score(metrics, expr)
            except Exception:
                continue
            if best is None or float(score) > float(best["selection_score"]):
                best = {
                    "expr": expr,
                    "raw_expression": raw_expr,
                    "raw_terms": [to_string(term, int(model.template.num_vars), simplify=False) for term in terms],
                    "choices": choices,
                    "theta": theta,
                    "theta0": theta0,
                    "eval_theta0_mode": theta0_mode,
                    "eval_endpoint_decode_mode": endpoint_decode_mode,
                    "selection_score": float(score),
                    **metrics,
                    **terminal_summary(theta, model.template),
                    **endpoint_probability_diagnostics(model.template, theta, task, choices),
                    **reference_diag,
                }
    hard_counts = list(hard_mode_counts.values())
    hard_total = int(sum(hard_counts))
    if hard_total > 0:
        probs = np.asarray(hard_counts, dtype=np.float64) / float(hard_total)
        entropy = float(-(probs * np.log(np.maximum(probs, 1.0e-12))).sum() / math.log(max(len(hard_counts), 2)))
        top_share = float(max(hard_counts) / float(hard_total))
    else:
        entropy = 0.0
        top_share = 0.0
    if best is None:
        device = next(model.parameters()).device
        best = {
            "expr": Expr.const(0.0),
            "raw_expression": "0",
            "raw_terms": [],
            "choices": [],
            "theta": torch.zeros(theta_dim(model.template), device=device),
            "theta0": torch.zeros(theta_dim(model.template), device=device),
            "selection_score": -1.0e9,
            "r2": 0.0,
            "nmse": 1.0e9,
            "raw_test_r2_without_affine": 0.0,
            "raw_train_r2_without_affine": 0.0,
            "raw_nmse_without_affine": 1.0e9,
            "raw_train_nmse_without_affine": 1.0e9,
            "term_linear_fit_r2": 0.0,
            "term_linear_fit_nmse": 1.0e9,
            "term_linear_fit_train_r2": 0.0,
            "term_linear_fit_train_nmse": 1.0e9,
            "term_linear_fit_coefficients": [],
            "term_linear_fit_intercept": 0.0,
            "term_fit_coefficients": [],
            "term_fit_intercept": 0.0,
            "term_fit_nonzero_coefficient_count": 0,
            "term_count": 0,
            "unique_nonzero_term_count": 0,
            "duplicate_term_count": 0,
            "term_expressions": [],
            "eval_theta0_mode": "none",
            "eval_endpoint_decode_mode": endpoint_decode_mode,
            "terminal_entropy_mean": 0.0,
            "terminal_max_prob_mean": 0.0,
            "endpoint_sample_active_prob_mean": 0.0,
            "endpoint_sample_active_argmax_match_mean": 0.0,
            "endpoint_trace_family_best_argmax_match": 0.0,
            "endpoint_trace_family_best_active_mean_prob": 0.0,
            "reference_field_oracle": 1.0,
            "reference_field_source": "missing_compiled_trace",
        }
    best.update({
        "eval_hard_expression_mode_entropy": float(entropy),
        "eval_hard_top_expression_share": float(top_share),
        "eval_hard_unique_expression_count": int(len(hard_mode_counts)),
        "eval_hard_decoded_candidate_count": int(hard_total),
    })
    return best, len(valid_exprs), len(set(valid_exprs)), total_candidates, float(time.perf_counter() - start_time)


def evaluate_reference_field_oracle(
    model: FixedSymbolConditionedVelocityNet,
    eval_tasks: list[TaskBundle],
    args: argparse.Namespace,
    device: torch.device,
    *,
    progress_out_dir: Path | None = None,
    progress_prefix: str = "reference_field",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    gen = torch.Generator(device=device).manual_seed(int(args.seed) + 52_421)
    started = time.perf_counter()
    for task_idx, task in enumerate(eval_tasks):
        best, valid_count, unique_count, total_candidates, runtime_sec = select_best_reference_field_rollout(
            model,
            task,
            args,
            gen,
        )
        expression = _affine_expression(
            best["raw_expression"],
            best.get("term_fit_coefficients", [1.0]),
            best.get("term_fit_intercept", 0.0),
        )
        record = with_structural_metrics({
            "task_id": task.task_id,
            "suite": task.suite,
            "method": "ReferenceFisherBridgeOracle",
            "eval_status": "ok" if task.traces else "no_compiled_trace",
            "rollout_guidance_mode": "reference_field_oracle",
            "num_vars": int(task.num_vars),
            "ground_truth": task.ground_truth,
            "expression": expression,
            "raw_expression": best["raw_expression"],
            "global_affine_expression": expression,
            "term_fitted_expression": best.get("term_linear_fit_expression", ""),
            "structural_expression": best["raw_expression"],
            "head_fit_mode": "affine_raw_and_term_linear_diagnostics",
            "complexity": int(best["expr"].complexity),
            "raw_complexity": int(best["expr"].complexity),
            "solved": bool(accuracy_rate(float(best["r2"]))),
            "valid_expression_fraction": float(valid_count / max(total_candidates, 1)),
            "unique_expression_fraction": float(unique_count / max(valid_count, 1)),
            "eval_candidate_count": int(total_candidates),
            "eval_valid_expression_count": int(valid_count),
            "eval_unique_expression_count": int(unique_count),
            "eval_runtime_sec": float(runtime_sec),
            **{key: value for key, value in best.items() if key not in {"expr", "theta", "theta0", "choices"}},
        })
        records.append(record)
        endpoint_rows.append(endpoint_diagnostics(task, model.template, best["theta"]))
    if progress_out_dir is not None:
        progress_out_dir.mkdir(parents=True, exist_ok=True)
        _write_cycle_jsonl(progress_out_dir / f"{progress_prefix}_samples.partial.jsonl", records)
        _write_cycle_jsonl(progress_out_dir / f"{progress_prefix}_endpoint_rankings.partial.jsonl", endpoint_rows)
        progress = {
            "status": "complete",
            "completed_tasks": int(len(eval_tasks)),
            "total_tasks": int(len(eval_tasks)),
            "records": int(len(records)),
            "endpoint_rows": int(len(endpoint_rows)),
            "elapsed_sec": float(time.perf_counter() - started),
            "partial_samples": f"{progress_prefix}_samples.partial.jsonl",
            "partial_endpoint_rankings": f"{progress_prefix}_endpoint_rankings.partial.jsonl",
        }
        (progress_out_dir / f"{progress_prefix}_eval_progress.json").write_text(
            json.dumps(_jsonable(progress), indent=2, ensure_ascii=False) + "\n"
        )
    return records, endpoint_rows, []


def evaluate_model(
    model: FixedSymbolConditionedVelocityNet,
    score_model: None,
    eval_tasks: list[TaskBundle],
    args: argparse.Namespace,
    device: torch.device,
    *,
    progress_out_dir: Path | None = None,
    progress_prefix: str = "typed_op_node_flow",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    modes = ["off"]
    gen = torch.Generator(device=device).manual_seed(int(args.seed) + 4242)
    progress_interval = max(int(getattr(args, "eval_progress_interval", 0)), 0)
    progress_started = time.perf_counter()

    def flush_progress(status: str, task_idx: int, task_id: str) -> None:
        if progress_out_dir is None:
            return
        progress_out_dir.mkdir(parents=True, exist_ok=True)
        partial_path = progress_out_dir / f"{progress_prefix}_samples.partial.jsonl"
        endpoint_path = progress_out_dir / f"{progress_prefix}_endpoint_rankings.partial.jsonl"
        sweep_path = progress_out_dir / f"{progress_prefix}_ode_sweep.partial.jsonl"
        with partial_path.open("w") as f:
            for row in records:
                f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        with endpoint_path.open("w") as f:
            for row in endpoint_rows:
                f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        with sweep_path.open("w") as f:
            for row in sweep_rows:
                f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        progress = {
            "status": status,
            "completed_tasks": int(task_idx),
            "total_tasks": int(len(eval_tasks)),
            "records": int(len(records)),
            "endpoint_rows": int(len(endpoint_rows)),
            "sweep_rows": int(len(sweep_rows)),
            "last_task_id": str(task_id),
            "elapsed_sec": float(time.perf_counter() - progress_started),
            "partial_samples": partial_path.name,
            "partial_endpoint_rankings": endpoint_path.name,
            "partial_ode_sweep": sweep_path.name,
        }
        (progress_out_dir / f"{progress_prefix}_eval_progress.json").write_text(
            json.dumps(_jsonable(progress), indent=2, ensure_ascii=False) + "\n"
        )
        if bool(getattr(args, "log_epochs", False)):
            print(json.dumps({"eval_progress": progress}, ensure_ascii=False), flush=True)

    for task_idx, task in enumerate(eval_tasks):
        for mode in modes:
            best, valid_count, unique_count, total_candidates, runtime_sec = select_best_rollout(
                model,
                score_model,
                task,
                args,
                gen,
                mode=mode,
                steps=int(args.ode_steps),
            )
            expression = _affine_expression(best["raw_expression"], best.get("term_fit_coefficients", [1.0]), best.get("term_fit_intercept", 0.0))
            record = {
                "task_id": task.task_id,
                "suite": task.suite,
                "method": "CompleteExpressionSemanticFM",
                "eval_status": "ok",
                "rollout_guidance_mode": mode,
                "num_vars": int(task.num_vars),
                "ground_truth": task.ground_truth,
                "expression": expression,
                "raw_expression": best["raw_expression"],
                "global_affine_expression": expression,
                "term_fitted_expression": best.get("term_linear_fit_expression", ""),
                "structural_expression": best["raw_expression"],
                "head_fit_mode": "affine_raw_and_term_linear_diagnostics",
                "complexity": int(best["expr"].complexity),
                "raw_complexity": int(best["expr"].complexity),
                "solved": bool(accuracy_rate(float(best["r2"]))),
                "valid_expression_fraction": float(valid_count / max(total_candidates, 1)),
                "unique_expression_fraction": float(unique_count / max(valid_count, 1)),
                "eval_candidate_count": int(total_candidates),
                "eval_valid_expression_count": int(valid_count),
                "eval_unique_expression_count": int(unique_count),
                "eval_runtime_sec": float(runtime_sec),
                **{k: v for k, v in best.items() if k not in {"expr", "theta", "theta0", "choices"}},
            }
            record = with_structural_metrics(record)
            records.append(record)
            endpoint_rows.append(endpoint_diagnostics(task, model.template, best["theta"]))
            if mode == "off" and _is_register_template(model.template):
                temporal_rows.extend(temporal_action_weight_rows(
                    model,
                    task,
                    best["theta0"],
                    args,
                    steps=int(args.temporal_visualization_steps),
                ))
            for steps in [int(v) for v in str(args.ode_sweep_steps).split(",") if str(v).strip()]:
                sweep_best, sweep_valid, sweep_unique, sweep_total, sweep_runtime = select_best_rollout(
                    model,
                    score_model,
                    task,
                    args,
                    gen,
                    mode=mode,
                    steps=int(steps),
                )
                sweep_expression = _affine_expression(
                    sweep_best["raw_expression"],
                    sweep_best.get("term_fit_coefficients", [1.0]),
                    sweep_best.get("term_fit_intercept", 0.0),
                )
                sweep_record = with_structural_metrics({
                    "task_id": task.task_id,
                    "suite": task.suite,
                    "mode": mode,
                    "ode_steps": int(steps),
                    "ground_truth": task.ground_truth,
                    "expression": sweep_expression,
                    "raw_expression": sweep_best["raw_expression"],
                    "global_affine_expression": sweep_expression,
                    "term_fitted_expression": sweep_best.get("term_linear_fit_expression", ""),
                    "structural_expression": sweep_best["raw_expression"],
                    "r2": float(sweep_best.get("r2", 0.0)),
                    "nmse": float(sweep_best.get("nmse", 0.0)),
                    "valid_expression_fraction": float(sweep_valid / max(sweep_total, 1)),
                    "unique_expression_fraction": float(sweep_unique / max(sweep_valid, 1)),
                    "terminal_entropy_mean": float(sweep_best.get("terminal_entropy_mean", 0.0)),
                    "terminal_max_prob_mean": float(sweep_best.get("terminal_max_prob_mean", 0.0)),
                    "eval_runtime_sec": float(sweep_runtime),
                    "guidance_fr_norm": float(sweep_best.get("guidance_fr_norm", 0.0)),
                    "guidance_cap_ratio": float(sweep_best.get("guidance_cap_ratio", 0.0)),
                })
                sweep_rows.append(sweep_record)
        completed = int(task_idx) + 1
        if progress_interval > 0 and (completed % progress_interval == 0 or completed == len(eval_tasks)):
            flush_progress("running", completed, task.task_id)
    flush_progress("complete", len(eval_tasks), eval_tasks[-1].task_id if eval_tasks else "")
    if progress_out_dir is not None:
        write_temporal_action_visualization(progress_out_dir, temporal_rows)
    return records, endpoint_rows, sweep_rows


def _affine_expression(raw: str, coeffs: list[float], intercept: float) -> str:
    coef = float(coeffs[0]) if coeffs else 1.0
    return f"{coef:.6g}*({raw}) + {float(intercept):.6g}"


def endpoint_diagnostics(task: TaskBundle, template: FixedSymbolTemplate, theta: torch.Tensor) -> dict[str, Any]:
    best_match, best_prob = 0.0, 0.0
    blocks = split_blocks(theta, template)
    for idx, trace in enumerate(task.traces):
        active = list(trace["active_block_indices"])
        if not active:
            continue
        matches, probs = [], []
        for bidx in active:
            p = masked_single_block_softmax(blocks[int(bidx)], template, int(bidx))
            action = int(trace["choices"][int(bidx)])
            matches.append(float(torch.argmax(p).detach().cpu().item() == action))
            probs.append(float(p[action].detach().cpu().item()))
        match = float(np.mean(matches))
        prob = float(np.mean(probs))
        if match > best_match or (match == best_match and prob > best_prob):
            best_match, best_prob = match, prob
    return {
        "task_id": task.task_id,
        "suite": task.suite,
        "gt_trace_count": int(len(task.traces)),
        "endpoint_family_best_argmax_match": best_match,
        "endpoint_family_best_active_mean_prob": best_prob,
        "endpoint_trace_family_best_argmax_match": best_match,
        "endpoint_trace_family_best_active_mean_prob": best_prob,
        "compiler_semantic_oracle_pass": bool(task.traces),
        "compiler_semantic_oracle_raw_r2_min": min(
            [float(trace.get("semantic_oracle_raw_r2", 0.0)) for trace in task.traces],
            default=0.0,
        ),
        "terminal_gt_active_action_match": best_match,
        "terminal_gt_active_action_probability": best_prob,
    }


def with_structural_metrics(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    gt = str(out.get("ground_truth", "") or "")
    pred = str(out.get("structural_expression", out.get("raw_expression", out.get("expression", ""))) or "")
    out["gt_skeleton"] = _skeleton(gt)
    out["pred_skeleton"] = _skeleton(pred)
    out["skeleton_match"] = bool(out["gt_skeleton"] and out["pred_skeleton"] and out["gt_skeleton"] == out["pred_skeleton"])
    out["operator_dependency_gt"] = _operator_dependency(gt)
    out["operator_dependency_pred"] = _operator_dependency(pred)
    out["operator_dependency_match"] = bool(out["operator_dependency_gt"] and out["operator_dependency_gt"] == out["operator_dependency_pred"])
    try:
        gt_vars = {str(symbol) for symbol in _sympify(gt).free_symbols}
        pred_vars = {str(symbol) for symbol in _sympify(pred).free_symbols}
        out["ground_truth_variables"] = sorted(gt_vars)
        out["predicted_variables"] = sorted(pred_vars)
        out["variable_set_match"] = bool(gt_vars == pred_vars)
    except Exception:
        out["ground_truth_variables"] = []
        out["predicted_variables"] = []
        out["variable_set_match"] = False
    out["simplified_symbolic_equivalence"] = _symbolic_equiv(gt, pred)
    out.update(_token_metrics(gt, pred))
    return out


def _sympify(text: str):
    return sp.sympify(str(text or ""), locals={"Abs": sp.Abs})


def _skeleton(text: str) -> str:
    try:
        return _skel_node(_sympify(text))
    except Exception:
        return ""


def _skel_node(expr) -> str:
    expr = sp.sympify(expr)
    if expr.is_Number:
        return "C"
    if expr.is_Symbol:
        return str(expr)
    if expr.is_Add:
        parts = [_skel_node(arg) for arg in expr.args if getattr(arg, "free_symbols", set())]
        parts = [part for part in parts if part and part != "C"]
        return "add(" + ",".join(sorted(parts)) + ")" if len(parts) > 1 else (parts[0] if parts else "C")
    if expr.is_Mul:
        parts = [_skel_node(arg) for arg in expr.args if getattr(arg, "free_symbols", set())]
        parts = [part for part in parts if part and part != "C"]
        return "mul(" + ",".join(sorted(parts)) + ")" if len(parts) > 1 else (parts[0] if parts else "C")
    if expr.is_Pow:
        return f"pow({_skel_node(expr.args[0])},C)"
    return f"{expr.func.__name__}(" + ",".join(_skel_node(arg) for arg in expr.args) + ")"


def _operator_dependency(text: str) -> str:
    try:
        expr = _sympify(text)
    except Exception:
        return ""
    ops: dict[str, int] = {}
    vars_seen = sorted(str(v) for v in getattr(expr, "free_symbols", set()))

    def visit(node):
        if node.is_Atom:
            return
        name = node.func.__name__
        ops[name] = ops.get(name, 0) + 1
        for child in node.args:
            visit(child)

    visit(expr)
    return "ops[" + ",".join(f"{k}:{ops[k]}" for k in sorted(ops)) + "]|vars[" + ",".join(vars_seen) + "]"


def _symbolic_equiv(gt: str, pred: str) -> bool:
    if len(str(gt or "")) > 1000 or len(str(pred or "")) > 1000:
        return False
    try:
        diff = _sympify(gt) - _sympify(pred)
        if diff == 0:
            return True
        if sp.count_ops(diff) > 120:
            return False
        expanded = sp.expand(diff)
        return bool(expanded == 0)
    except Exception:
        return False


def _token_metrics(gt: str, pred: str) -> dict[str, float]:
    import re
    pattern = r"[a-zA-Z_][a-zA-Z0-9_]*|[()+\-*/]|\d+\.?\d*"
    a = re.findall(pattern, str(gt or ""))
    b = re.findall(pattern, str(pred or ""))
    if not a or not b:
        return {"formula_bleu": 0.0, "formula_token_accuracy": 0.0, "formula_edit_distance": float(len(a) or len(b))}
    acc = sum(1 for x, y in zip(a, b) if x == y) / max(len(a), 1)
    return {"formula_bleu": float(acc), "formula_token_accuracy": float(acc), "formula_edit_distance": float(abs(len(a) - len(b)))}


def make_task_split(train_tasks: list[TaskBundle], eval_tasks: list[TaskBundle]) -> dict[str, Any]:
    return {
        "mode": "fixed_hash_by_task_id",
        "train_task_ids": [task.task_id for task in train_tasks],
        "eval_task_ids": [task.task_id for task in eval_tasks],
        "compiled_train_task_ids": [task.task_id for task in train_tasks if task.traces],
        "compiled_eval_task_ids": [task.task_id for task in eval_tasks if task.traces],
        "train_trace_counts": {task.task_id: len(task.traces) for task in train_tasks},
        "eval_trace_counts": {task.task_id: len(task.traces) for task in eval_tasks},
        "train_compile_failures": {task.task_id: task.compile_failures[:5] for task in train_tasks if not task.traces},
        "eval_compile_failures": {task.task_id: task.compile_failures[:5] for task in eval_tasks if not task.traces},
        "train_semantic_oracle_raw_r2": {
            task.task_id: [float(trace.get("semantic_oracle_raw_r2", 0.0)) for trace in task.traces]
            for task in train_tasks
        },
        "eval_semantic_oracle_raw_r2": {
            task.task_id: [float(trace.get("semantic_oracle_raw_r2", 0.0)) for trace in task.traces]
            for task in eval_tasks
        },
    }


def _cycle_compiler_metrics(tasks: list[TaskBundle]) -> dict[str, Any]:
    traces = [trace for task in tasks for trace in task.traces]
    oracle_r2 = [float(trace.get("semantic_oracle_raw_r2", -math.inf)) for trace in traces]
    return {
        "task_count": int(len(tasks)),
        "compiled_task_count": int(sum(bool(task.traces) for task in tasks)),
        "compile_failure_count": int(sum(len(task.compile_failures) for task in tasks)),
        "accepted_trace_count": int(len(traces)),
        "semantic_oracle_pass_count": int(sum(value >= 0.999999 for value in oracle_r2)),
        "semantic_oracle_raw_r2_min": float(min(oracle_r2)) if oracle_r2 else None,
        "semantic_oracle_raw_r2_mean": float(np.mean(oracle_r2)) if oracle_r2 else None,
    }


def _write_cycle_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def write_outer_iteration_coupling_figure(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist the trajectory records consumed by the paper visualization."""
    if not rows:
        return {"status": "not_run", "row_count": 0}
    path = out_dir / "outer_iteration_flow_landscape.jsonl"
    _write_cycle_jsonl(path, rows)
    iterations = sorted({int(row.get("iteration", 0)) for row in rows})
    time_points = sorted({float(row.get("t", 0.0)) for row in rows})
    metadata = {
        "status": "complete",
        "row_count": int(len(rows)),
        "iteration_count": int(len(iterations)),
        "iterations": iterations,
        "time_points": time_points,
        "records": str(path),
    }
    (out_dir / "outer_iteration_flow_landscape.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return metadata


def _cycle_eval_record(record: dict[str, Any], *, model_role: str) -> dict[str, Any]:
    out = dict(record)
    out["model_role"] = str(model_role)
    return out


def _cycle_eval_summary(records: list[dict[str, Any]], *, model_role: str) -> dict[str, Any]:
    if not records:
        return {"model_role": str(model_role), "evaluation_status": "not_run", "n_tasks": 0}

    def values(key: str) -> list[float]:
        return [
            float(row[key])
            for row in records
            if key in row and isinstance(row[key], (int, float, bool)) and math.isfinite(float(row[key]))
        ]

    def mean(key: str) -> float | None:
        rows = values(key)
        return float(np.mean(rows)) if rows else None

    def median(key: str) -> float | None:
        rows = values(key)
        return float(np.median(rows)) if rows else None

    return {
        "model_role": str(model_role),
        "evaluation_status": "complete",
        "n_tasks": int(len(records)),
        "raw_test_r2_mean": mean("raw_test_r2_without_affine"),
        "raw_test_r2_median": median("raw_test_r2_without_affine"),
        "coefficient_fitted_test_r2_mean": mean("r2"),
        "coefficient_fitted_test_r2_median": median("r2"),
        "term_linear_fit_test_r2_mean": mean("term_linear_fit_r2"),
        "skeleton_accuracy": mean("skeleton_match"),
        "operator_dependency_accuracy": mean("operator_dependency_match"),
        "symbolic_equivalence_rate": mean("simplified_symbolic_equivalence"),
        "valid_expression_fraction_mean": mean("valid_expression_fraction"),
        "population_unique_expression_count_mean": mean("eval_population_unique_expression_count"),
        "population_raw_r2_mean": mean("eval_population_raw_r2_mean"),
        "population_raw_r2_median": mean("eval_population_raw_r2_median"),
        "population_term_fit_r2_mean": mean("eval_population_term_fit_r2_mean"),
        "terminal_retraction_fr_mean": mean("eval_terminal_retraction_fr_mean"),
        "terminal_retraction_fr_p95": mean("eval_terminal_retraction_fr_p95"),
        "terminal_retraction_expression_preserved_rate": mean("eval_terminal_retraction_expression_preserved_rate"),
        "evaluation_runtime_sec_mean": mean("eval_runtime_sec"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _inherit_graph_architecture_from_checkpoint(
    args: argparse.Namespace,
    loaded_ckpt: dict[str, Any] | None,
) -> dict[str, Any]:
    if loaded_ckpt is None:
        return {"checkpoint_architecture_inherited_count": 0, "checkpoint_architecture_inherited_fields": ""}
    objective_name = str(loaded_ckpt.get("objective_name", ""))
    if objective_name != OBJECTIVE_NAME:
        raise ValueError("checkpoint objective is incompatible with the current semantic-flow trainer")
    template_cfg = loaded_ckpt.get("template", {})
    model_cfg = loaded_ckpt.get("model_cfg", {})
    if not isinstance(template_cfg, dict) or not isinstance(model_cfg, dict):
        raise ValueError("checkpoint is missing template/model_cfg metadata")
    inherited: list[str] = []

    def inherit(name: str, value: Any) -> None:
        if getattr(args, name, None) != value:
            setattr(args, name, value)
            inherited.append(name)

    inherit("construction_graph", "register_categorical_blocks")
    for name in ("num_vars", "num_layers", "num_registers", "output_terms"):
        if name in template_cfg:
            inherit(name, int(template_cfg[name]))
    if "ops" in template_cfg:
        inherit("ops", ",".join(str(value) for value in template_cfg["ops"]))
    for name in ("hidden", "metadata_embedding_dim"):
        if name in model_cfg:
            inherit(name, int(model_cfg[name]))
    if "task_encoder_mode" in model_cfg:
        inherit("task_encoder_mode", str(model_cfg["task_encoder_mode"]))
    return {
        "checkpoint_architecture_inherited_count": int(len(inherited)),
        "checkpoint_architecture_inherited_fields": ",".join(inherited),
    }



def build_semantic_gt_bootstrap_examples(
    flow: FixedSymbolConditionedVelocityNet,
    train_tasks: list[TaskBundle],
    args: argparse.Namespace,
    device: torch.device,
) -> list[CycleCoupledExample]:
    """Materialize source-conditioned GT atom bridges for bootstrap training."""
    tasks = [task for task in train_tasks if task.traces]
    if not tasks:
        raise RuntimeError("bootstrap requires strictly compiled GT traces")
    sources_per_task = max(int(getattr(args, "cycle_particles_per_task", 8)), 1)
    generator = torch.Generator(device=device).manual_seed(int(args.seed) + 61_337)
    examples: list[CycleCoupledExample] = []
    for task in tasks:
        for source_index in range(sources_per_task):
            theta0 = random_theta(
                flow.template,
                scale=float(args.theta0_noise_scale),
                device=device,
                generator=generator,
            )
            trace = select_trace_for_theta0(flow.template, theta0, task, args)
            choices = [int(value) for value in trace["choices"]]
            active_mask = torch.zeros(len(flow.template.blocks), dtype=torch.bool, device=device)
            active_indices = [int(value) for value in trace.get("active_block_indices", [])]
            if active_indices:
                active_mask[torch.tensor(active_indices, dtype=torch.long, device=device)] = True
            source_probability = masked_block_softmax(
                theta0.view(len(flow.template.blocks), int(flow.template.source_count)),
                flow.template,
            )
            target_probability = source_conditioned_trace_target_probabilities(
                source_probability,
                torch.tensor(choices, dtype=torch.long, device=device),
                active_mask,
                projection_eps=float(getattr(args, "cycle_projection_eps", 0.02)),
                projection_sharpness=1.0,
            )
            theta1 = logits_from_block_probabilities(
                [target_probability[index] for index in range(int(target_probability.shape[0]))],
                flow.template,
                eps=float(args.fisher_eps),
            )
            examples.append(CycleCoupledExample(
                task=task,
                theta0=theta0.detach(),
                theta1=theta1.detach(),
                active_mask=active_mask.detach(),
                proposal_index=int(source_index),
                diagnostics={
                    "phase": "semantic_gt_bootstrap",
                    "source_index": int(source_index),
                    "target_expression": str(trace.get("expression_string", "")),
                },
                sample_weight=1.0,
                target_choices=tuple(choices),
                is_gt_anchor=True,
            ))
    uniform_weight = 1.0 / float(len(examples))
    for example in examples:
        example.sample_weight = uniform_weight
    return examples



def _semantic_crossfit_expression_energy(
    expr: Expr,
    task: TaskBundle,
    *,
    max_abs: float,
) -> dict[str, Any]:
    """Evaluate one decoded expression with a deterministic 80/20 fit split."""
    try:
        values = sanitize_values(eval_expr(expr, task.x_train), clip=float(max_abs)).float()
        target = sanitize_values(task.y_train.float(), clip=float(max_abs)).to(values.device)
        count = int(values.numel())
        if count < 4:
            raise ValueError("cross-fit energy requires at least four task points")
        generator = torch.Generator(device=values.device).manual_seed(_stable_task_seed(task.task_id) + 50_021)
        order = torch.randperm(count, generator=generator, device=values.device)
        support_count = min(max(int(round(0.8 * count)), 2), count - 1)
        support = order[:support_count]
        query = order[support_count:]
        support_values = values.index_select(0, support)
        support_target = target.index_select(0, support)
        query_values = values.index_select(0, query)
        query_target = target.index_select(0, query)
        fitted_query, coefficients = fit_affine(support_values, support_target, query_values)
        fitted_r2 = float(r2_score(query_target.detach().cpu().numpy(), fitted_query.detach().cpu().numpy()))
        raw_r2 = float(r2_score(query_target.detach().cpu().numpy(), query_values.detach().cpu().numpy()))
        if not math.isfinite(fitted_r2):
            raise ValueError("non-finite cross-fit R2")
        return {
            "semantic_energy": float(min(max(1.0 - fitted_r2, 0.0), 10.0)),
            "crossfit_fitted_r2": fitted_r2,
            "crossfit_raw_r2": raw_r2 if math.isfinite(raw_r2) else -1.0e9,
            "crossfit_scale": float(coefficients[0]),
            "crossfit_intercept": float(coefficients[1]),
            "crossfit_support_count": int(support_count),
            "crossfit_query_count": int(query.numel()),
            "semantic_invalid": 0.0,
        }
    except Exception as exc:  # noqa: BLE001 - invalid symbolic programs are finite-energy particles.
        return {
            "semantic_energy": 10.0,
            "crossfit_fitted_r2": -1.0e9,
            "crossfit_raw_r2": -1.0e9,
            "crossfit_scale": 0.0,
            "crossfit_intercept": 0.0,
            "crossfit_support_count": 0,
            "crossfit_query_count": 0,
            "semantic_invalid": 1.0,
            "semantic_error": f"{type(exc).__name__}: {exc}",
        }


def _train_semantic_base_flow(
    base: TaskConditionedVelocityNet,
    examples: list[CycleCoupledExample],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not examples:
        raise RuntimeError("current GT bootstrap requires compiled bridge examples")
    optimizer = torch.optim.AdamW(base.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = random.Random(int(args.seed) + 91_001)
    curve: list[dict[str, Any]] = []
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    schedule = [float(value) for value in str(getattr(args, "bootstrap_source_mass_schedule", "0.30,0.20,0.10")).split(",")]
    if any(value <= 0.0 or value >= 0.5 for value in schedule):
        raise ValueError("bootstrap source-mass schedule values must be in (0, 0.5)")
    stage_epochs = max(int(args.epochs), 1)
    examples_by_task: dict[str, list[CycleCoupledExample]] = {}
    for example in examples:
        examples_by_task.setdefault(example.task.task_id, []).append(example)
    task_pools = list(examples_by_task.values())
    particles_per_task = max(int(getattr(args, "cycle_particles_per_task", 8)), 1)
    global_epoch = 0
    for stage_index, source_mass in enumerate(schedule, start=1):
      for epoch in range(stage_epochs):
        global_epoch += 1
        loss_sum = torch.zeros((), device=device)
        loss_count = 0
        zero_sum = torch.zeros((), device=device)
        zero_count = 0
        inactive_sum = torch.zeros((), device=device)
        inactive_count = 0
        time_sums = {key: torch.zeros((), device=device) for key in ("low", "mid", "high")}
        time_counts = {key: 0 for key in ("low", "mid", "high")}
        relative_sums = {
            key: [torch.zeros((), device=device), torch.zeros((), device=device)]
            for key in ("global", "low", "mid", "high", "readout", "op", "arg")
        }
        kind_masks = {
            "readout": torch.tensor([str(block.kind) == "readout" for block in base.template.blocks], dtype=torch.bool, device=device),
            "op": torch.tensor([str(block.kind) == "reg_op" for block in base.template.blocks], dtype=torch.bool, device=device),
            "arg": torch.tensor([str(block.kind) == "reg_arg" for block in base.template.blocks], dtype=torch.bool, device=device),
        }
        for _step in range(max(int(args.steps_per_epoch), 1)):
            optimizer.zero_grad(set_to_none=True)
            batch: list[torch.Tensor] = []
            task_batch_size = min(max(int(args.train_batch_size), 1), len(task_pools))
            selected_pools = rng.sample(task_pools, k=task_batch_size)
            for pool in selected_pools:
                selected_examples = [pool[index % len(pool)] for index in range(particles_per_task)]
                draw = rng.random()
                t = rng.random() * 0.1 if draw < 0.2 else (0.1 + 0.8 * rng.random() if draw < 0.8 else 0.9 + 0.1 * rng.random())
                theta_rows: list[torch.Tensor] = []
                target_rows: list[torch.Tensor] = []
                for example in selected_examples:
                    source_probability = masked_block_softmax(
                        example.theta0.to(device).view(len(base.template.blocks), int(base.template.source_count)), base.template,
                    )
                    target_probability = source_conditioned_trace_target_probabilities(
                        source_probability, torch.tensor(example.target_choices, dtype=torch.long, device=device),
                        example.active_mask.to(device), projection_eps=source_mass, projection_sharpness=1.0,
                    )
                    theta1 = logits_from_block_probabilities(
                        [target_probability[index] for index in range(int(target_probability.shape[0]))], base.template, eps=float(args.fisher_eps),
                    )
                    theta_t, target = stage1_simplex_path(example.theta0.to(device), theta1, base.template, float(t))
                    if rng.random() < 0.20:
                        noise = torch.randn_like(theta_t) * 0.02
                        theta_t = center_theta(theta_t + noise, base.template)
                    theta_rows.append(theta_t)
                    target_rows.append(target)
                theta_batch = torch.stack(theta_rows)
                target_batch = torch.stack(target_rows)
                task = selected_examples[0].task
                predicted_batch = base.forward_batch(task.x_train.to(device), task.y_train.to(device), theta_batch, float(t))
                task_losses: list[torch.Tensor] = []
                for example, theta_t, predicted, target in zip(selected_examples, theta_batch, predicted_batch, target_batch):
                    active = example.active_mask.to(device)
                    weights = active.float()
                    bootstrap_gate = active.repeat_interleave(int(base.template.source_count)).to(predicted.dtype)
                    predicted = predicted * bootstrap_gate
                    active_loss, _ = stage1_velocity_loss(theta_t, predicted, target, base.template, weights, eps=float(args.fisher_eps))
                    inactive = ~active
                    inactive_loss, _ = stage1_velocity_loss(theta_t, predicted, target, base.template, inactive.float(), eps=float(args.fisher_eps))
                    loss = active_loss + float(getattr(args, "bootstrap_inactive_weight", 0.20)) * inactive_loss
                    zero, _ = stage1_velocity_loss(theta_t, torch.zeros_like(target), target, base.template, weights, eps=float(args.fisher_eps))
                    task_losses.append(loss)
                    zero_sum = zero_sum + zero.detach()
                    zero_count += 1
                    inactive_sum = inactive_sum + inactive_loss.detach()
                    inactive_count += 1
                    time_key = "low" if t < 0.1 else "high" if t >= 0.9 else "mid"
                    time_sums[time_key] = time_sums[time_key] + active_loss.detach()
                    time_counts[time_key] += 1
                    predicted_blocks = stage1_velocity_block_losses(theta_t, predicted, target, base.template, eps=float(args.fisher_eps)).detach()
                    zero_blocks = stage1_velocity_block_losses(theta_t, torch.zeros_like(predicted), target, base.template, eps=float(args.fisher_eps)).detach()
                    predicted_active = predicted_blocks[active].sum()
                    zero_active = zero_blocks[active].sum()
                    relative_sums["global"][0] += predicted_active
                    relative_sums["global"][1] += zero_active
                    relative_sums[time_key][0] += predicted_active
                    relative_sums[time_key][1] += zero_active
                    for kind, mask in kind_masks.items():
                        selected = active & mask
                        relative_sums[kind][0] = relative_sums[kind][0] + predicted_blocks[selected].sum()
                        relative_sums[kind][1] = relative_sums[kind][1] + zero_blocks[selected].sum()
                batch.append(torch.stack(task_losses).mean())
            objective = torch.stack(batch).mean()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), float(args.grad_clip))
            optimizer.step()
            loss_sum = loss_sum + objective.detach()
            loss_count += 1
        mean_loss = float((loss_sum / max(loss_count, 1)).cpu())
        mean_zero_loss = float((zero_sum / max(zero_count, 1)).cpu())
        mean_inactive_loss = float((inactive_sum / max(inactive_count, 1)).cpu())
        if mean_loss < best:
            best = mean_loss
            best_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
        row = {
            "phase": "semantic_gt_bootstrap",
            "epoch": int(global_epoch),
            "bootstrap_stage": int(stage_index),
            "bootstrap_source_mass": float(source_mass),
            "flow_fisher_velocity_loss": mean_loss,
            "flow_zero_predictor_loss": mean_zero_loss,
            "flow_relative_fisher_loss": float((relative_sums["global"][0] / relative_sums["global"][1].clamp_min(1.0e-12)).cpu()),
            "flow_inactive_fisher_loss": mean_inactive_loss,
            "flow_inactive_relative_to_active_zero": mean_inactive_loss / max(mean_zero_loss, 1.0e-12),
            "flow_low_fisher_loss": float((time_sums["low"] / time_counts["low"]).cpu()) if time_counts["low"] else None,
            "flow_mid_fisher_loss": float((time_sums["mid"] / time_counts["mid"]).cpu()) if time_counts["mid"] else None,
            "flow_high_fisher_loss": float((time_sums["high"] / time_counts["high"]).cpu()) if time_counts["high"] else None,
        }
        for key in ("low", "mid", "high", "readout", "op", "arg"):
            numerator, denominator = relative_sums[key]
            denominator_value = float(denominator.cpu())
            row[f"flow_{key}_relative_fisher_loss"] = float((numerator / denominator.clamp_min(1.0e-12)).cpu()) if denominator_value > 0.0 else None
        curve.append(row)
        if bool(args.log_epochs):
            print(json.dumps(row), flush=True)
    if best_state is not None:
        base.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    summary = dict(curve[-1])
    summary["best_flow_fisher_velocity_loss"] = float(best)
    return curve, summary


def _bootstrap_gate_decision(
    gate: str,
    bootstrap: dict[str, Any],
    population: dict[str, Any],
    *,
    raw_control: float | None = None,
) -> dict[str, Any]:
    name = str(gate).upper()
    if name == "OFF":
        return {"gate": name, "passed": True, "checks": {}}
    limits = {
        "A": {"global": 0.05, "detail": 0.10, "inactive": 0.01, "fit": 1.0, "gt": 1.0, "skeleton": 0.0, "operator": 0.0},
        "B": {"global": 0.20, "detail": 0.30, "inactive": 0.02, "fit": 0.75, "gt": 0.75, "skeleton": 0.50, "operator": 0.50},
        "C": {"global": 0.25, "detail": 0.35, "inactive": 0.02, "fit": 0.50, "gt": 0.0, "skeleton": 0.25, "operator": 0.25},
    }[name]
    checks: dict[str, bool] = {
        "global_relative": float(bootstrap.get("flow_relative_fisher_loss", math.inf)) < limits["global"],
        "inactive_drift": float(bootstrap.get("flow_inactive_relative_to_active_zero", math.inf)) < limits["inactive"],
        "fitted_r2_rate": float(population.get("fitted_r2_gt_0_95_rate", 0.0)) >= limits["fit"],
        "gt_family_hit_rate": float(population.get("gt_trace_family_hit_rate", 0.0)) >= limits["gt"],
        "skeleton_rate": float(population.get("skeleton_match_rate", 0.0)) >= limits["skeleton"],
        "operator_rate": float(population.get("operator_dependency_match_rate", 0.0)) >= limits["operator"],
    }
    for key in ("low", "mid", "high", "readout", "op", "arg"):
        value = bootstrap.get(f"flow_{key}_relative_fisher_loss")
        checks[f"{key}_relative"] = value is not None and float(value) < limits["detail"]
    if name == "A":
        checks["raw_r2_rate"] = float(population.get("raw_r2_gt_0_999_rate", 0.0)) >= 1.0
    if name == "B":
        checks["population_diversity"] = float(population.get("population_unique_expression_count_mean", 0.0)) >= 2.0
    if name == "C":
        checks["semantic_variance"] = float(population.get("population_noncollapsed_task_rate", 0.0)) >= 0.50
        checks["raw_r2_control_improvement"] = raw_control is not None and float(population.get("crossfit_raw_r2_mean", -math.inf)) >= float(raw_control) + 0.20
    return {
        "gate": name,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "bootstrap": bootstrap,
        "population": population,
        "raw_control": raw_control,
    }


def _collect_semantic_endpoint_populations(
    flow: PoissonResidualVelocity,
    tasks: list[TaskBundle],
    args: argparse.Namespace,
    device: torch.device,
    *,
    iteration: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    task_limit = int(getattr(args, "cycle_collection_task_limit", 0))
    selected_tasks = list(tasks[:task_limit]) if task_limit > 0 else list(tasks)
    particle_count = max(int(args.cycle_particles_per_task), 1)
    generator = torch.Generator(device=device).manual_seed(int(args.seed) + 100_003 * int(iteration))
    populations: dict[str, list[dict[str, Any]]] = {}
    landscape: list[dict[str, Any]] = []
    energies: list[float] = []
    fitted_r2: list[float] = []
    raw_r2: list[float] = []
    invalid: list[float] = []
    unique_counts: list[int] = []
    noncollapsed_tasks = 0
    gt_hits: list[float] = []
    skeleton_hits: list[float] = []
    operator_hits: list[float] = []
    for task_index, task in enumerate(selected_tasks):
        rows: list[dict[str, Any]] = []
        capture = task_index < max(int(getattr(args, "cycle_landscape_task_limit", 1)), 0)
        theta0_batch = torch.stack([
            random_theta(flow.template, scale=float(args.theta0_noise_scale), device=device, generator=generator)
            for _ in range(particle_count)
        ], dim=0)
        endpoint_batch, rollout_diag = rollout_batch(
            flow,
            task,
            theta0_batch,
            steps=int(args.cycle_proposer_rollout_steps),
        )
        for source_index in range(particle_count):
            theta0 = theta0_batch[source_index]
            endpoint = endpoint_batch[source_index]
            if capture and source_index < max(int(getattr(args, "cycle_landscape_sources", 4)), 0):
                _endpoint, _diag, snapshots = rollout_with_snapshots(
                    flow, task, theta0, steps=int(args.cycle_proposer_rollout_steps),
                    snapshot_count=int(args.cycle_landscape_time_points),
                )
                for snapshot_t, snapshot in snapshots:
                    landscape.append(_lineage_landscape_row(
                        flow.template, task, snapshot, iteration=int(iteration),
                        source_index=int(source_index), t=float(snapshot_t), point_kind="flow",
                    ))
            choices = hard_decode_choices(endpoint, flow.template)
            expr, terms, _layers = execute_choices(flow.template, choices)
            energy_diag = _semantic_crossfit_expression_energy(
                expr,
                task,
                max_abs=float(args.term_fit_max_abs),
            )
            probabilities = masked_block_softmax(
                endpoint.view(len(flow.template.blocks), int(flow.template.source_count)),
                flow.template,
            )
            row = {
                "task": task,
                "task_id": task.task_id,
                "source_index": int(source_index),
                "theta0": theta0.detach(),
                "reference_theta1": endpoint.detach(),
                "reference_probabilities": probabilities.detach(),
                "choices": tuple(int(value) for value in choices),
                "expression": to_string(expr, int(task.num_vars), simplify=False),
                "terms": terms,
                **energy_diag,
                **rollout_diag,
            }
            gt_choice_family = {tuple(int(value) for value in trace["choices"]) for trace in task.traces}
            row["gt_trace_family_hit"] = float(tuple(int(value) for value in choices) in gt_choice_family)
            structural = with_structural_metrics({
                "ground_truth": task.ground_truth,
                "raw_expression": str(row["expression"]),
            })
            row["skeleton_match"] = float(structural["skeleton_match"])
            row["operator_dependency_match"] = float(structural["operator_dependency_match"])
            rows.append(row)
            energies.append(float(energy_diag["semantic_energy"]))
            fitted_r2.append(float(energy_diag["crossfit_fitted_r2"]))
            raw_r2.append(float(energy_diag["crossfit_raw_r2"]))
            invalid.append(float(energy_diag["semantic_invalid"]))
            gt_hits.append(float(row["gt_trace_family_hit"]))
            skeleton_hits.append(float(row["skeleton_match"]))
            operator_hits.append(float(row["operator_dependency_match"]))
        populations[task.task_id] = rows
        task_energies = np.asarray([float(row["semantic_energy"]) for row in rows], dtype=np.float64)
        noncollapsed_tasks += int(float(task_energies.var()) > float(args.cycle_support_variance_eps))
        unique_counts.append(len({str(row["expression"]) for row in rows}))
        elapsed = float(time.perf_counter() - started)
        partial = {
            "phase": "semantic_endpoint_collection_partial",
            "iteration": int(iteration),
            "completed_tasks": int(task_index + 1),
            "total_tasks": int(len(selected_tasks)),
            "completed_particles": int(sum(len(value) for value in populations.values())),
            "runtime_sec": elapsed,
            "seconds_per_particle": elapsed / max(sum(len(value) for value in populations.values()), 1),
            "crossfit_fitted_r2_mean": float(np.mean(fitted_r2)) if fitted_r2 else None,
            "crossfit_raw_r2_mean": float(np.mean(raw_r2)) if raw_r2 else None,
            "unique_expression_count_mean": float(np.mean(unique_counts)) if unique_counts else 0.0,
        }
        partial_path = Path(args.out) / "endpoint_collection.partial.json"
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(_jsonable(partial), indent=2) + "\n")
        timeout = float(getattr(args, "cycle_collection_timeout_sec", 300.0))
        if timeout > 0.0 and elapsed > timeout:
            raise RuntimeError(f"current endpoint collection exceeded {timeout:.1f}s after {task_index + 1} tasks: {elapsed:.1f}s")
    summary = {
        "phase": "semantic_endpoint_collection",
        "iteration": int(iteration),
        "task_count": int(len(populations)),
        "particle_count": int(sum(len(rows) for rows in populations.values())),
        "expression_evaluations": int(sum(len(rows) for rows in populations.values())),
        "semantic_energy_mean": float(np.mean(energies)) if energies else None,
        "crossfit_fitted_r2_mean": float(np.mean(fitted_r2)) if fitted_r2 else None,
        "crossfit_raw_r2_mean": float(np.mean(raw_r2)) if raw_r2 else None,
        "semantic_invalid_rate": float(np.mean(invalid)) if invalid else None,
        "population_unique_expression_count_mean": float(np.mean(unique_counts)) if unique_counts else 0.0,
        "population_noncollapsed_task_rate": float(noncollapsed_tasks / max(len(populations), 1)),
        "gt_trace_family_hit_rate": float(np.mean(gt_hits)) if gt_hits else 0.0,
        "skeleton_match_rate": float(np.mean(skeleton_hits)) if skeleton_hits else 0.0,
        "operator_dependency_match_rate": float(np.mean(operator_hits)) if operator_hits else 0.0,
        "fitted_r2_gt_0_95_rate": float(np.mean(np.asarray(fitted_r2) > 0.95)) if fitted_r2 else 0.0,
        "raw_r2_gt_0_999_rate": float(np.mean(np.asarray(raw_r2) > 0.999)) if raw_r2 else 0.0,
        "collection_runtime_sec": float(time.perf_counter() - started),
    }
    return populations, landscape, summary


def _fit_semantic_potential(
    template: Any,
    populations: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    device: torch.device,
    *,
    iteration: int,
) -> tuple[ConditionalSemanticPotential, list[dict[str, Any]], dict[str, Any]]:
    potential = ConditionalSemanticPotential(template, int(args.hidden)).to(device)
    optimizer = torch.optim.Adam(potential.parameters(), lr=float(args.cycle_poisson_lr))
    rng = random.Random(int(args.seed) + 110_017 * int(iteration))
    task_rows = [rows for rows in populations.values() if rows]
    curve: list[dict[str, Any]] = []
    if not task_rows:
        raise RuntimeError("current Poisson potential requires endpoint populations")
    for step in range(max(int(args.cycle_poisson_steps), 1)):
        optimizer.zero_grad(set_to_none=True)
        batch_losses: list[torch.Tensor] = []
        batch_diags: list[dict[str, float]] = []
        for rows in rng.sample(task_rows, k=min(max(int(args.train_batch_size), 1), len(task_rows))):
            task = rows[0]["task"]
            probabilities = torch.stack([row["reference_probabilities"] for row in rows], dim=0).to(device).detach().requires_grad_(True)
            energy = torch.tensor([float(row["semantic_energy"]) for row in rows], dtype=probabilities.dtype, device=device)
            phi = potential(probabilities, task.x_train.to(device), task.y_train.to(device))
            result = weighted_poisson_loss(
                phi,
                probabilities,
                energy,
                action_mask=graph_action_mask(template, device=device),
                create_graph=True,
                eps=float(args.fisher_eps),
            )
            batch_losses.append(result.loss)
            batch_diags.append(poisson_summary(result))
        loss = torch.stack(batch_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(potential.parameters(), float(args.grad_clip))
        optimizer.step()
        row = {
            "phase": "semantic_weighted_poisson",
            "iteration": int(iteration),
            "step": int(step + 1),
            "poisson_loss": float(loss.detach().cpu()),
            "poisson_dirichlet": float(np.mean([diag["dirichlet"] for diag in batch_diags])),
            "poisson_source": float(np.mean([diag["source"] for diag in batch_diags])),
            "poisson_energy_variance": float(np.mean([diag["energy_variance"] for diag in batch_diags])),
            "poisson_tangent_mass_error": float(max(diag["tangent_mass_error"] for diag in batch_diags)),
        }
        curve.append(row)
        if bool(args.log_epochs) and (step + 1 == int(args.cycle_poisson_steps) or (step + 1) % 16 == 0):
            print(json.dumps(row), flush=True)
    return potential, curve, dict(curve[-1])


def _correct_semantic_populations(
    template: Any,
    potential: ConditionalSemanticPotential,
    populations: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    device: torch.device,
    *,
    iteration: int,
) -> tuple[list[ResidualFlowExample], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    examples: list[ResidualFlowExample] = []
    correction_rows: list[dict[str, Any]] = []
    landscape: list[dict[str, Any]] = []
    collapse_count = 0
    step_size = float(args.cycle_correction_step)
    fisher_steps: list[float] = []
    for rows in populations.values():
        task = rows[0]["task"]
        probabilities = torch.stack([row["reference_probabilities"] for row in rows], dim=0).to(device).detach().requires_grad_(True)
        energies = torch.tensor([float(row["semantic_energy"]) for row in rows], dtype=probabilities.dtype, device=device)
        variance = float(energies.var(unbiased=False).detach().cpu())
        collapsed = variance < float(args.cycle_support_variance_eps)
        if collapsed:
            collapse_count += 1
            tangent = torch.zeros_like(probabilities)
            corrected = probabilities.detach()
            poisson_diag = {"energy_variance": variance, "tangent_mass_error": 0.0}
        else:
            phi = potential(probabilities, task.x_train.to(device), task.y_train.to(device))
            result = weighted_poisson_loss(
                phi,
                probabilities,
                energies,
                action_mask=graph_action_mask(template, device=device),
                create_graph=False,
                eps=float(args.fisher_eps),
            )
            tangent = result.tangent.detach()
            corrected = exponential_fisher_correction(
                probabilities.detach(),
                tangent,
                step_size,
                action_mask=graph_action_mask(template, device=device),
                eps=float(args.fisher_eps),
            )
            poisson_diag = poisson_summary(result)
        for particle_index, row in enumerate(rows):
            corrected_theta = logits_from_block_probabilities(
                [corrected[particle_index, block] for block in range(int(corrected.shape[1]))],
                template,
                eps=float(args.fisher_eps),
            )
            source_probability = probabilities.detach()[particle_index]
            target_probability = corrected[particle_index]
            step_blocks = torch.sqrt(
                torch.stack([
                    block_fisher_squared_distance(
                        source_probability[block:block + 1],
                        target_probability[block:block + 1],
                        torch.ones(1, dtype=torch.bool, device=device),
                    )
                    for block in range(int(source_probability.shape[0]))
                ]).clamp_min(0.0)
            )
            fisher_step = float(torch.sqrt(step_blocks.square().mean()).detach().cpu())
            fisher_steps.append(fisher_step)
            examples.append(ResidualFlowExample(
                task=task,
                theta0=row["theta0"].detach(),
                reference_theta1=row["reference_theta1"].detach(),
                corrected_theta1=corrected_theta.detach(),
                source_index=int(row["source_index"]),
                energy=float(row["semantic_energy"]),
                diagnostics={
                    "support_collapsed": float(collapsed),
                    "semantic_energy_variance": variance,
                    "endpoint_correction_fr_rms": fisher_step,
                    **poisson_diag,
                },
            ))
            correction_rows.append({
                "iteration": int(iteration),
                "task_id": task.task_id,
                "source_index": int(row["source_index"]),
                "expression": str(row["expression"]),
                "semantic_energy": float(row["semantic_energy"]),
                "crossfit_fitted_r2": float(row["crossfit_fitted_r2"]),
                "crossfit_raw_r2": float(row["crossfit_raw_r2"]),
                "support_collapsed": float(collapsed),
                "endpoint_correction_fr_rms": fisher_step,
            })
            if int(row["source_index"]) < max(int(getattr(args, "cycle_landscape_sources", 4)), 0):
                landscape.append(_lineage_landscape_row(
                    template,
                    task,
                    corrected_theta,
                    iteration=int(iteration),
                    source_index=int(row["source_index"]),
                    t=1.0,
                    point_kind="semantic_target",
                ))
    summary = {
        "phase": "semantic_same_particle_correction",
        "iteration": int(iteration),
        "residual_example_count": int(len(examples)),
        "support_collapse_task_count": int(collapse_count),
        "support_collapse_task_rate": float(collapse_count / max(len(populations), 1)),
        "endpoint_correction_fr_rms_mean": float(np.mean(fisher_steps)) if fisher_steps else 0.0,
        "endpoint_correction_fr_rms_p95": float(np.quantile(fisher_steps, 0.95)) if fisher_steps else 0.0,
    }
    return examples, correction_rows, landscape, summary


def _train_semantic_residual_head(
    flow: PoissonResidualVelocity,
    examples: list[ResidualFlowExample],
    args: argparse.Namespace,
    device: torch.device,
    *,
    iteration: int,
) -> tuple[ResidualVelocityHead | None, list[dict[str, Any]], dict[str, Any]]:
    usable = [example for example in examples if float(example.diagnostics.get("support_collapsed", 0.0)) < 0.5]
    if not usable:
        return None, [], {
            "phase": "semantic_residual_skipped",
            "iteration": int(iteration),
            "skip_reason": "all_task_populations_have_zero_semantic_variance",
        }
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    head = ResidualVelocityHead(flow.template, int(args.hidden)).to(device)
    train_trunk = len(flow.residual_heads) == 0
    if train_trunk:
        for parameter in flow.residual_trunk.parameters():
            parameter.requires_grad_(True)
    trainable_parameters = list(head.parameters())
    if train_trunk:
        trainable_parameters.extend(flow.residual_trunk.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.cycle_flow_lr),
        weight_decay=float(args.weight_decay),
    )
    rng = random.Random(int(args.seed) + 120_011 * int(iteration))
    step_size = float(args.cycle_correction_step)
    curve: list[dict[str, Any]] = []
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_trunk_state: dict[str, torch.Tensor] | None = None
    for epoch in range(max(int(args.cycle_flow_epochs), 1)):
        losses: list[float] = []
        zero_losses: list[float] = []
        for step in range(max(int(args.cycle_steps_per_epoch), 1)):
            optimizer.zero_grad(set_to_none=True)
            batch: list[torch.Tensor] = []
            for batch_index in range(max(int(args.train_batch_size), 1)):
                example = rng.choice(usable)
                t = sample_cycle_time(
                    rng,
                    epoch * max(int(args.cycle_steps_per_epoch), 1) + step + batch_index,
                    str(args.cycle_time_sampling),
                    inherited_mode=str(args.time_sampling),
                    low_prob=float(args.low_t_sampling_prob),
                    low_max=float(args.low_t_max),
                )
                theta_t, corrected_velocity = stage1_simplex_path(
                    example.theta0.to(device),
                    example.corrected_theta1.to(device),
                    flow.template,
                    float(t),
                )
                with torch.no_grad():
                    reference_velocity = flow(
                        example.task.x_train.to(device),
                        example.task.y_train.to(device),
                        theta_t,
                        float(t),
                    )
                    target_residual = (corrected_velocity - reference_velocity) / max(step_size, 1.0e-8)
                features = flow.residual_trunk(
                    example.task.x_train.to(device),
                    example.task.y_train.to(device),
                    theta_t,
                    float(t),
                )
                predicted = head(features)
                block_weights = torch.ones(len(flow.template.blocks), dtype=torch.float32, device=device)
                loss, _ = stage1_velocity_loss(
                    theta_t,
                    predicted,
                    target_residual,
                    flow.template,
                    block_weights,
                    eps=float(args.fisher_eps),
                )
                zero, _ = stage1_velocity_loss(
                    theta_t,
                    torch.zeros_like(target_residual),
                    target_residual,
                    flow.template,
                    block_weights,
                    eps=float(args.fisher_eps),
                )
                batch.append(loss)
                zero_losses.append(float(zero.detach().cpu()))
            objective = torch.stack(batch).mean()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, float(args.grad_clip))
            optimizer.step()
            losses.append(float(objective.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        zero_mean = float(np.mean(zero_losses)) if zero_losses else 0.0
        if mean_loss < best:
            best = mean_loss
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            if train_trunk:
                best_trunk_state = {
                    key: value.detach().cpu().clone()
                    for key, value in flow.residual_trunk.state_dict().items()
                }
        row = {
            "phase": "semantic_residual_fisher_matching",
            "iteration": int(iteration),
            "epoch": int(epoch + 1),
            "residual_fisher_loss": mean_loss,
            "residual_zero_predictor_loss": zero_mean,
            "residual_relative_fisher_loss": float(mean_loss / max(zero_mean, 1.0e-12)),
        }
        curve.append(row)
        if bool(args.log_epochs):
            print(json.dumps(row), flush=True)
    if best_state is not None:
        head.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    if best_trunk_state is not None:
        flow.residual_trunk.load_state_dict({key: value.to(device) for key, value in best_trunk_state.items()})
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    return head, curve, dict(curve[-1])



def run_one_step_semantic_fisher_cycle(
    args: argparse.Namespace,
    *,
    template: Any,
    graph_family: str,
    train_tasks: list[TaskBundle],
    eval_tasks: list[TaskBundle],
    source_counts: dict[str, int],
    loaded_ckpt: dict[str, Any] | None,
    device: torch.device,
) -> dict[str, Any]:
    """Run the route-free semantic weighted-Poisson residual Fisher cycle."""
    if not _is_register_template(template):
        raise ValueError("semantic flow requires register_categorical_blocks")
    if int(template.output_terms) != 1:
        raise ValueError("semantic flow requires --output-terms 1")
    if str(getattr(args, "velocity_parameterization", "direct_velocity")) != "direct_velocity":
        raise ValueError("semantic flow supports direct_velocity only")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs = {
        "global_state_mode": "full",
        "metadata_embedding_dim": int(args.metadata_embedding_dim),
        "task_encoder_mode": str(args.task_encoder_mode),
        "task_conditioning": "xy",
    }
    base = TaskConditionedVelocityNet(template, int(args.hidden), **model_kwargs).to(device)
    flow = PoissonResidualVelocity(base).to(device)
    train_curve: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    landscape_rows: list[dict[str, Any]] = []
    cycle_history: list[dict[str, Any]] = []
    iteration_eval_records: list[dict[str, Any]] = []
    iteration_eval_summaries: list[dict[str, Any]] = []
    bootstrap_summary: dict[str, Any] = {}
    bootstrap_count = 0
    bootstrap_gate_result: dict[str, Any] = {"gate": "OFF", "passed": True}

    if loaded_ckpt is not None:
        objective = str(loaded_ckpt.get("objective_name", ""))
        if objective != OBJECTIVE_NAME:
            raise ValueError("only semantic_poisson_residual_fisher_bootstrap checkpoints can resume training")
        stage_count = int(loaded_ckpt.get("residual_stage_count", 0))
        raw_steps = loaded_ckpt.get("residual_step_sizes", [float(args.cycle_correction_step)] * stage_count)
        for index in range(stage_count):
            flow.add_residual(
                ResidualVelocityHead(template, int(args.hidden)).to(device),
                float(raw_steps[index]),
            )
        flow.load_state_dict(loaded_ckpt["flow_model"])
        loaded_summary = loaded_ckpt.get("summary", {})
        if isinstance(loaded_summary, dict):
            cycle_history = list(loaded_summary.get("cycle_history", []))
            bootstrap_summary = dict(loaded_summary.get("bootstrap_training_summary", {}))
    elif bool(args.eval_only):
        raise ValueError("--eval-only requires a semantic-flow checkpoint")
    else:
        bootstrap_examples = build_semantic_gt_bootstrap_examples(base, train_tasks, args, device)
        bootstrap_count = len(bootstrap_examples)
        gate_name = str(getattr(args, "bootstrap_gate", "off")).upper()
        raw_control = None
        if gate_name == "C":
            control_args = argparse.Namespace(**vars(args))
            control_args.cycle_landscape_task_limit = 0
            control_args.cycle_collection_task_limit = 0
            _control_population, _control_landscape, control_summary = _collect_semantic_endpoint_populations(
                flow, eval_tasks[:8], control_args, device, iteration=-1,
            )
            raw_control = float(control_summary.get("crossfit_raw_r2_mean", 0.0))
        bootstrap_curve, bootstrap_summary = _train_semantic_base_flow(base, bootstrap_examples, args, device)
        train_curve.extend(bootstrap_curve)
        if gate_name != "OFF":
            gate_args = argparse.Namespace(**vars(args))
            gate_args.cycle_landscape_task_limit = 0
            gate_args.cycle_collection_task_limit = 0
            if gate_name == "A":
                gate_tasks = train_tasks[:1]
            elif gate_name == "B":
                gate_tasks = train_tasks[:8]
            else:
                gate_tasks = eval_tasks[:8]
            gate_population, gate_landscape, gate_summary = _collect_semantic_endpoint_populations(
                flow, gate_tasks, gate_args, device, iteration=0,
            )
            del gate_population, gate_landscape
            bootstrap_gate_result = _bootstrap_gate_decision(
                gate_name, bootstrap_summary, gate_summary, raw_control=raw_control,
            )
            (out_dir / "bootstrap_gate.json").write_text(json.dumps(_jsonable(bootstrap_gate_result), indent=2) + "\n")
            torch.save({
                "objective_name": OBJECTIVE_NAME,
                "bootstrap_gate": bootstrap_gate_result,
                "flow_model": flow.state_dict(),
            }, out_dir / "bootstrap_gate_checkpoint.pt")
            if not bool(bootstrap_gate_result["passed"]):
                raise RuntimeError(f"bootstrap {gate_name} gate failed; Poisson outer loop is disabled")
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        for iteration in range(1, max(int(args.cycle_iterations), 0) + 1):
            populations, iteration_landscape, collection_summary = _collect_semantic_endpoint_populations(
                flow,
                train_tasks,
                args,
                device,
                iteration=iteration,
            )
            landscape_rows.extend(iteration_landscape)
            timeout = float(getattr(args, "cycle_collection_timeout_sec", 300.0))
            if timeout > 0 and float(collection_summary["collection_runtime_sec"]) > timeout:
                raise RuntimeError(
                    f"current endpoint collection exceeded {timeout:.1f}s: "
                    f"{float(collection_summary['collection_runtime_sec']):.1f}s"
                )
            population_gate = {
                "population_noncollapsed_task_rate": float(collection_summary.get("population_noncollapsed_task_rate", 0.0)) >= 0.50,
                "population_unique_expression_count_mean": float(collection_summary.get("population_unique_expression_count_mean", 0.0)) >= 2.0,
                "semantic_invalid_rate": float(collection_summary.get("semantic_invalid_rate", 1.0)) < 0.10,
            }
            if not all(population_gate.values()):
                failure = {
                    "status": "bootstrap_population_collapse",
                    "iteration": int(iteration),
                    "checks": population_gate,
                    "collection": collection_summary,
                }
                (out_dir / "bootstrap_population_collapse.json").write_text(json.dumps(_jsonable(failure), indent=2) + "\n")
                raise RuntimeError("bootstrap population gate failed; Poisson outer loop is disabled")
            potential, potential_curve, potential_summary = _fit_semantic_potential(
                template,
                populations,
                args,
                device,
                iteration=iteration,
            )
            train_curve.extend(potential_curve)
            residual_examples, iteration_corrections, correction_landscape, correction_summary = _correct_semantic_populations(
                template,
                potential,
                populations,
                args,
                device,
                iteration=iteration,
            )
            correction_rows.extend(iteration_corrections)
            landscape_rows.extend(correction_landscape)
            head, residual_curve, residual_summary = _train_semantic_residual_head(
                flow,
                residual_examples,
                args,
                device,
                iteration=iteration,
            )
            train_curve.extend(residual_curve)
            if head is not None:
                for parameter in head.parameters():
                    parameter.requires_grad_(False)
                flow.add_residual(head, float(args.cycle_correction_step))
            cycle_summary = {
                "iteration": int(iteration),
                **collection_summary,
                **potential_summary,
                **correction_summary,
                **residual_summary,
                "residual_stage_count": int(len(flow.residual_heads)),
            }
            if not bool(args.train_only) and bool(args.cycle_eval_each_iteration):
                records, _endpoints, _sweep = evaluate_model(
                    flow,
                    None,
                    eval_tasks,
                    args,
                    device,
                    progress_out_dir=out_dir / f"cycle_eval_iter_{iteration}_progress",
                    progress_prefix=f"cycle_eval_iter_{iteration}",
                )
                records = [
                    {**_cycle_eval_record(row, model_role="semantic_poisson_residual_field"), "cycle_iteration": int(iteration)}
                    for row in records
                ]
                iteration_summary = _cycle_eval_summary(records, model_role="semantic_poisson_residual_field")
                iteration_summary["iteration"] = int(iteration)
                iteration_eval_records.extend(records)
                iteration_eval_summaries.append(iteration_summary)
                cycle_summary["iteration_eval"] = iteration_summary
                _write_cycle_jsonl(out_dir / f"cycle_eval_iter_{iteration}_samples.jsonl", records)
                (out_dir / f"cycle_eval_iter_{iteration}_summary.json").write_text(
                    json.dumps(_jsonable(iteration_summary), indent=2, ensure_ascii=False) + "\n"
                )
            cycle_history.append(cycle_summary)
            if bool(args.log_epochs):
                print(json.dumps({"phase": "semantic_cycle_complete", **cycle_summary}), flush=True)

    flow_records: list[dict[str, Any]] = []
    if not bool(args.train_only):
        flow_records, _flow_endpoints, _flow_sweep = evaluate_model(
            flow,
            None,
            eval_tasks,
            args,
            device,
            progress_out_dir=out_dir / "flow_eval",
            progress_prefix="flow",
        )
        flow_records = [_cycle_eval_record(row, model_role="semantic_poisson_residual_field") for row in flow_records]
    flow_summary = _cycle_eval_summary(flow_records, model_role="semantic_poisson_residual_field")
    summary = {
        "algorithm": "complete_expression_semantic_fm",
        "training_flow": "semantic_poisson_residual_fisher_cycle",
        "objective_name": OBJECTIVE_NAME,
        "construction_graph": str(args.construction_graph),
        "construction_family": str(graph_family),
        "task_conditioning": "xy",
        "theta0_conditioning": "none",
        "route_conditioning": "none",
        "velocity_parameterization": "direct_velocity_residual",
        "semantic_update": "weighted_poisson_minimum_fisher_kinetic",
        "coupling": "same_particle_no_matching",
        "matching": "none",
        "output_terms": 1,
        "cycle_particles_per_task": int(args.cycle_particles_per_task),
        "expressions_per_endpoint": 1,
        "cycle_correction_step": float(args.cycle_correction_step),
        "cycle_poisson_steps": int(args.cycle_poisson_steps),
        "cycle_poisson_lr": float(args.cycle_poisson_lr),
        "bootstrap_example_count": int(bootstrap_count),
        "bootstrap_training_summary": bootstrap_summary,
        "bootstrap_gate": bootstrap_gate_result,
        "residual_stage_count": int(len(flow.residual_heads)),
        "cycle_history": cycle_history,
        "cycle_eval_history": iteration_eval_summaries,
        "flow_eval": flow_summary,
        "train_compiler": _cycle_compiler_metrics(train_tasks),
        "eval_compiler": _cycle_compiler_metrics(eval_tasks),
        "data_source_counts": source_counts,
        "eval_terminal_retraction": bool(args.eval_terminal_retraction),
        "eval_oracle_free_selection_mode": str(args.eval_oracle_free_selection_mode),
        "train_only": bool(args.train_only),
        "eval_only": bool(args.eval_only),
    }
    task_split = make_task_split(train_tasks, eval_tasks)
    checkpoint = {
        "algorithm": "complete_expression_semantic_fm",
        "objective_name": OBJECTIVE_NAME,
        "template": {
            "construction_graph": str(args.construction_graph),
            "num_vars": int(template.num_vars),
            "num_layers": int(template.num_layers),
            "num_registers": int(template.register_count),
            "ops": list(template.ops),
            "output_terms": 1,
        },
        "model_cfg": {
            "hidden": int(args.hidden),
            "metadata_embedding_dim": int(args.metadata_embedding_dim),
            "task_encoder_mode": str(args.task_encoder_mode),
            "theta0_conditioning": "none",
        },
        "residual_stage_count": int(len(flow.residual_heads)),
        "residual_step_sizes": [float(value) for value in flow.residual_step_sizes.detach().cpu().tolist()],
        "flow_model": flow.state_dict(),
        "summary": summary,
    }
    _write_cycle_jsonl(out_dir / "one_step_cycle_train_curve.jsonl", train_curve)
    _write_cycle_jsonl(out_dir / "semantic_corrections.jsonl", correction_rows)
    _write_cycle_jsonl(out_dir / "outer_iteration_flow_landscape.jsonl", landscape_rows)
    _write_cycle_jsonl(out_dir / "cycle_eval_iterations.jsonl", iteration_eval_records)
    _write_cycle_jsonl(out_dir / "one_step_cycle_samples.jsonl", flow_records)
    summary["outer_iteration_flow_landscape"] = write_outer_iteration_coupling_figure(out_dir, landscape_rows)
    (out_dir / "one_step_cycle_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n")
    (out_dir / "task_split.json").write_text(json.dumps(_jsonable(task_split), indent=2, ensure_ascii=False) + "\n")
    torch.save(checkpoint, out_dir / "one_step_cycle_checkpoint.pt")
    return {"summary": summary}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.training_flow) != "one_step_semantic_fisher_cycle":
        raise ValueError("only one_step_semantic_fisher_cycle is supported")
    _seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    loaded_ckpt: dict[str, Any] | None = None
    if str(args.load_checkpoint):
        loaded_ckpt = torch.load(str(args.load_checkpoint), map_location=device)
        _inherit_graph_architecture_from_checkpoint(args, loaded_ckpt)
    graph_family = canonical_construction_graph(str(args.construction_graph))
    template = make_construction_template(args, graph_family)
    cached_records = None
    require_trace_cache = str(getattr(args, "trace_cache_mode", "off")) == "require"
    if require_trace_cache:
        cached_records, _cache_manifest = load_trace_cache(str(args.trace_cache_root), template)
    setup_started = time.perf_counter()
    train_raw, eval_raw, source_counts = load_all_task_sources(
        args,
        template.num_vars,
        device=torch.device("cpu"),
    )
    if require_trace_cache:
        cached_ids = set(cached_records or {})
        missing_train = [str(task.name) for task in train_raw if str(task.name) not in cached_ids]
        missing_eval = [str(task.name) for task in eval_raw if str(task.name) not in cached_ids]
        train_raw = [task for task in train_raw if str(task.name) in cached_ids]
        eval_raw = [task for task in eval_raw if str(task.name) in cached_ids]
        source_counts["trace_cache_skipped_train_task_count"] = len(missing_train)
        source_counts["trace_cache_skipped_eval_task_count"] = len(missing_eval)
        source_counts["trace_cache_skipped_task_ids"] = missing_train + missing_eval
        if not train_raw:
            raise RuntimeError("compiled trace cache leaves no training tasks")
    print(
        json.dumps(
            {
                "phase": "semantic_task_sources_loaded",
                "train_tasks": len(train_raw),
                "eval_tasks": len(eval_raw),
                "runtime_sec": time.perf_counter() - setup_started,
            }
        ),
        flush=True,
    )
    compile_started = time.perf_counter()
    train_tasks = build_task_bundles(
        train_raw,
        template,
        traces_per_task=int(args.gt_traces_per_task),
        max_train_points=int(args.max_train_points),
        max_eval_points=int(args.max_eval_points),
        device=device,
        seed=int(getattr(args, "trace_cache_seed", args.seed)) if require_trace_cache else int(args.seed),
        split="train",
        copy_assignment=str(args.trace_copy_assignment),
        cached_records=cached_records,
        require_cache=require_trace_cache,
    )
    eval_tasks = build_task_bundles(
        eval_raw,
        template,
        traces_per_task=int(args.gt_traces_per_task),
        max_train_points=int(args.max_train_points),
        max_eval_points=int(args.max_eval_points),
        device=device,
        seed=(int(getattr(args, "trace_cache_seed", args.seed)) + 12_345) if require_trace_cache else int(args.seed) + 12_345,
        split="eval",
        copy_assignment=str(args.trace_copy_assignment),
        cached_records=cached_records,
        require_cache=require_trace_cache,
    )
    print(
        json.dumps(
            {
                "phase": "semantic_task_bundles_compiled",
                "train_tasks": len(train_tasks),
                "eval_tasks": len(eval_tasks),
                "gt_traces_per_task": int(args.gt_traces_per_task),
                "runtime_sec": time.perf_counter() - compile_started,
            }
        ),
        flush=True,
    )
    return run_one_step_semantic_fisher_cycle(
        args,
        template=template,
        graph_family=graph_family,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        source_counts=source_counts,
        loaded_ckpt=loaded_ckpt,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/clean_benchmark/semantic_flow/run")
    parser.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    parser.add_argument("--manifest-root", default="data/benchmark_suites")
    parser.add_argument("--suites", nargs="+", default=["nguyen", "constant", "livermore", "jin"])
    parser.add_argument("--symbolicgpt-root", default="")
    parser.add_argument("--symbolicgpt-train-limit", type=int, default=0)
    parser.add_argument("--symbolicgpt-eval-limit", type=int, default=0)
    parser.add_argument("--symbolicgpt-eval-splits", default="val,test")
    parser.add_argument("--symbolicgpt-point-train-fraction", type=float, default=0.8)
    parser.add_argument("--task-id-filter", default="")
    parser.add_argument("--allow-empty-eval", action="store_true")
    parser.add_argument("--trace-cache-root", default="data/cache/semantic_flow")
    parser.add_argument("--trace-cache-mode", choices=["off", "require"], default="off")
    parser.add_argument("--trace-cache-seed", type=int, default=20260711)
    parser.add_argument("--bootstrap-source-mass-schedule", default="0.30,0.20,0.10")
    parser.add_argument("--bootstrap-inactive-weight", type=float, default=0.20)
    parser.add_argument("--bootstrap-gate", choices=["off", "A", "B", "C"], default="off")
    parser.add_argument("--training-flow", choices=["one_step_semantic_fisher_cycle"], default="one_step_semantic_fisher_cycle")
    parser.add_argument("--construction-graph", choices=list(CONSTRUCTION_GRAPHS), default="register_categorical_blocks")
    parser.add_argument("--task-conditioning", choices=["auto", "off", "xy", "xy_residual"], default="xy")
    parser.add_argument("--num-vars", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-registers", type=int, default=0)
    parser.add_argument("--ops", default=",".join(DEFAULT_OPS))
    parser.add_argument("--op-copies", type=int, default=1)
    parser.add_argument("--trace-copy-assignment", choices=["canonical", "random"], default="canonical")
    parser.add_argument("--output-terms", type=int, default=1)
    parser.add_argument("--gt-traces-per-task", type=int, default=1)
    parser.add_argument("--latent-components", type=int, default=4)
    parser.add_argument("--latent-set-layers", type=int, default=2)
    parser.add_argument("--latent-attention-heads", type=int, default=4)
    parser.add_argument("--latent-tangent-weight", default="auto")
    parser.add_argument("--latent-p0-samples", type=int, default=1)
    parser.add_argument("--syntax-prior-random-trace-count", type=int, default=1024)
    parser.add_argument("--syntax-prior-endpoint-mode", choices=["sampled_trace", "trace_family_marginal"], default="sampled_trace")
    parser.add_argument("--endpoint-target-mode", choices=["trace_family_marginal", "sampled_trace"], default="sampled_trace")
    parser.add_argument("--endpoint-target-smoothing", type=float, default=0.01)
    parser.add_argument("--reference-vector-field", choices=["bridge_path", "endpoint_attractor"], default="bridge_path")
    parser.add_argument("--reference-state-sampler", choices=["bridge_path", "bridge_plus_random"], default="bridge_path")
    parser.add_argument("--reference-random-state-prob", type=float, default=0.0)
    parser.add_argument("--endpoint-attractor-min-remaining", type=float, default=0.05)
    parser.add_argument("--velocity-parameterization", choices=["direct_velocity", "endpoint_bridge"], default="direct_velocity")
    parser.add_argument("--global-state-mode", choices=["summary", "full"], default="full")
    parser.add_argument("--metadata-embedding-dim", type=int, default=0)
    parser.add_argument("--task-encoder-mode", choices=["point_mlp", "stats", "hybrid_stats"], default="hybrid_stats")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--train-task-limit", type=int, default=20)
    parser.add_argument("--eval-task-limit", type=int, default=2)
    parser.add_argument("--max-train-points", type=int, default=64)
    parser.add_argument("--max-eval-points", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr-min-factor", type=float, default=0.1)
    parser.add_argument("--score-lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--theta0-noise-scale", type=float, default=1.0)
    parser.add_argument("--theta0-endpoint-coupling", choices=["none", "active_choice_bias"], default="none")
    parser.add_argument("--theta0-endpoint-bias", type=float, default=5.0)
    parser.add_argument("--target-high", type=float, default=4.0)
    parser.add_argument("--target-low", type=float, default=-4.0)
    parser.add_argument("--inactive-block-target-mode", choices=["start", "zero"], default="start")
    parser.add_argument("--inactive-block-loss-weight", type=float, default=0.0)
    parser.add_argument("--fisher-eps", type=float, default=1.0e-4)
    parser.add_argument("--time-sampling", choices=["uniform", "low_t_mixture"], default="uniform")
    parser.add_argument("--low-t-sampling-prob", type=float, default=0.4)
    parser.add_argument("--low-t-max", type=float, default=0.35)
    parser.add_argument("--cycle-iterations", type=int, default=1)
    parser.add_argument("--cycle-collection-task-limit", type=int, default=0)
    parser.add_argument("--cycle-particles-per-task", type=int, default=8)
    parser.add_argument("--cycle-expression-samples", type=int, default=0)
    parser.add_argument("--cycle-poisson-steps", type=int, default=64)
    parser.add_argument("--cycle-poisson-lr", type=float, default=1.0e-3)
    parser.add_argument("--cycle-correction-step", type=float, default=0.10)
    parser.add_argument("--cycle-support-variance-eps", type=float, default=1.0e-6)
    parser.add_argument("--cycle-collection-timeout-sec", type=float, default=300.0)
    parser.add_argument("--cycle-landscape-sources", type=int, default=4)
    parser.add_argument("--cycle-landscape-task-limit", type=int, default=1)
    parser.add_argument("--cycle-landscape-time-points", type=int, default=5)
    parser.add_argument("--cycle-eval-each-iteration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cycle-proposer-rollout-steps", type=int, default=8)
    parser.add_argument("--cycle-graph-visualization-sources", type=int, default=4)
    parser.add_argument("--cycle-flow-epochs", type=int, default=4)
    parser.add_argument("--cycle-steps-per-epoch", type=int, default=50)
    parser.add_argument("--cycle-flow-lr", type=float, default=5.0e-4)
    parser.add_argument("--cycle-time-sampling", choices=["stratified_fisher", "inherit"], default="stratified_fisher")
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--ode-sweep-steps", default="32,64")
    parser.add_argument("--eval-theta0-samples", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--eval-flow-gt-probe-samples", type=int, default=4)
    parser.add_argument("--eval-theta0-mode", choices=["deterministic_random", "random"], default="deterministic_random")
    parser.add_argument("--eval-endpoint-decode-mode", choices=["hard_argmax", "soft_sample"], default="hard_argmax")
    parser.add_argument(
        "--eval-oracle-free-selection-mode",
        choices=["train_fit", "semantic_medoid"],
        default="train_fit",
        help=(
            "oracle-free candidate selection from multiple theta0 rollouts: "
            "train_fit selects by train-split affine/term-fit score; semantic_medoid keeps the old population medoid diagnostic"
        ),
    )
    parser.add_argument("--eval-terminal-retraction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-terminal-retraction-eps", type=float, default=-1.0)
    parser.add_argument("--eval-progress-interval", type=int, default=16)
    parser.add_argument("--eval-reference-field-oracle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-reference-projection-sharpness", type=float, default=-1.0)
    parser.add_argument("--term-fit-ridge", type=float, default=1.0e-8)
    parser.add_argument("--term-fit-max-abs", type=float, default=1.0e6)
    parser.add_argument("--temporal-visualization-steps", type=int, default=16)
    parser.add_argument("--eval-theta0-use-gt-trace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixed-batch-overfit", action="store_true")
    parser.add_argument("--fixed-batch-size", type=int, default=64)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--early-stop-loss", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--log-epochs", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(_jsonable(result["summary"]), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
