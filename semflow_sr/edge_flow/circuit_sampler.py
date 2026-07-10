"""Complete expression sampling from edge distributions."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from ..sr.ast import Expr
from ..sr.ops import NAME_TO_ID, get_op
from .semantic_teacher import DecisionTrace
from .edge_distribution import EdgeDistribution
from .template import RegisterOperatorTemplate


@dataclass
class CircuitSample:
    sample_id: int
    mode: int
    edge_choices: dict[str, int]
    expression: Expr
    log_prob: float
    complexity: int
    canonical: str = ""
    head_terms: tuple[Expr, ...] = ()
    log_prob_tensor: torch.Tensor | None = None
    active_log_prob_tensor: torch.Tensor | None = None
    entropy_tensor: torch.Tensor | None = None
    decision_traces: tuple[DecisionTrace, ...] = ()
    semantic_teacher_loss_tensor: torch.Tensor | None = None
    diagnostics: dict | None = None


class CircuitSampler:
    def __init__(self, template: RegisterOperatorTemplate):
        self.template = template
        self.groups = template.groups

    def sample(
        self,
        edge_dist: EdgeDistribution,
        *,
        batch_size: int,
        rng: random.Random,
        mode_policy: str = "stratified",
    ) -> list[CircuitSample]:
        quotas = self._mode_quotas(edge_dist, batch_size, mode_policy, rng)
        out: list[CircuitSample] = []
        for mode, count in enumerate(quotas):
            for _ in range(count):
                choices, log_prob, log_prob_tensor = self._sample_choices(edge_dist, mode, rng)
                expr, head_terms = self._execute_choices_with_terms(choices)
                out.append(CircuitSample(
                    sample_id=len(out),
                    mode=mode,
                    edge_choices=choices,
                    expression=expr,
                    log_prob=float(log_prob),
                    complexity=int(expr.complexity),
                    head_terms=head_terms,
                    log_prob_tensor=log_prob_tensor,
                    active_log_prob_tensor=log_prob_tensor,
                ))
        return out

    def _mode_quotas(
        self,
        edge_dist: EdgeDistribution,
        batch_size: int,
        mode_policy: str,
        rng: random.Random,
    ) -> list[int]:
        H = int(self.template.mixture_modes)
        B = max(int(batch_size), 0)
        if H == 1:
            return [B]
        if str(mode_policy).lower() != "stratified":
            probs = edge_dist.mixture_probs.detach().cpu().tolist()
            counts = [0 for _ in range(H)]
            for _ in range(B):
                counts[_sample_index(probs, rng)] += 1
            return counts
        base = B // H
        counts = [base for _ in range(H)]
        remaining = B - base * H
        if remaining > 0:
            probs = edge_dist.mixture_probs.detach().cpu()
            order = torch.argsort(probs, descending=True).tolist()
            for idx in order[:remaining]:
                counts[int(idx)] += 1
        return counts

    def _sample_choices(
        self,
        edge_dist: EdgeDistribution,
        mode: int,
        rng: random.Random,
    ) -> tuple[dict[str, int], float, torch.Tensor]:
        choices: dict[str, int] = {}
        log_terms: list[torch.Tensor] = []
        mix_prob = edge_dist.mixture_probs[int(mode)].clamp_min(1e-12)
        log_terms.append(mix_prob.log())
        log_prob = math.log(max(float(mix_prob.detach().cpu().item()), 1e-12))
        for group in self.groups:
            probs = edge_dist.group_probs[group.group_id][mode].detach().cpu().tolist()
            pos = _sample_index(probs, rng)
            choices[group.group_id] = int(pos)
            selected = edge_dist.group_probs[group.group_id][int(mode), int(pos)].clamp_min(1e-12)
            log_terms.append(selected.log())
            log_prob += math.log(max(float(selected.detach().cpu().item()), 1e-12))
        log_prob_tensor = torch.stack(log_terms).sum() if log_terms else torch.zeros((), dtype=edge_dist.mixture_probs.dtype)
        return choices, log_prob, log_prob_tensor

    def _execute_choices(self, choices: dict[str, int]) -> Expr:
        expr, _head_terms = self._execute_choices_with_terms(choices)
        return expr

    def _execute_choices_with_terms(self, choices: dict[str, int]) -> tuple[Expr, tuple[Expr, ...]]:
        regs = _initial_registers(self.template.num_vars, self.template.num_registers)
        primitive_count = len(self.template.primitives)
        for layer in range(self.template.num_layers):
            images: list[Expr] = []
            for op_index, primitive in enumerate(self.template.primitives):
                op = get_op(NAME_TO_ID[primitive])
                children = []
                for slot in range(op.arity):
                    group_id = f"L{layer}:OP{op_index}:{primitive}:ARG{slot}"
                    children.append(regs[int(choices[group_id])])
                images.append(Expr.op(NAME_TO_ID[primitive], tuple(children)))
            pool = regs + images
            next_regs = []
            for reg in range(self.template.num_registers):
                group_id = f"L{layer}:REG{reg}:UPDATE"
                next_regs.append(pool[int(choices[group_id])])
            regs = next_regs
            if len(regs) != self.template.num_registers or primitive_count < 0:
                raise RuntimeError("invalid register update")
        terms: list[Expr] = []
        seen_terms: set[Expr] = set()
        output_groups = [group for group in self.template.groups if group.group_type == "OUTPUT_SELECT"]
        for group in output_groups:
            choice = int(choices.get(group.group_id, 0))
            choice = max(0, min(choice, int(group.num_candidates) - 1))
            term = Expr.const(0.0) if int(choice) >= len(regs) else regs[int(choice)]
            if not _is_zero_const(term) and term not in seen_terms:
                terms.append(term)
                seen_terms.add(term)
        if not terms:
            terms.append(Expr.const(0.0))
        return _sum_exprs(tuple(terms)), tuple(terms)


def _initial_registers(num_vars: int, num_registers: int) -> list[Expr]:
    regs: list[Expr] = []
    for idx in range(num_registers):
        if idx < num_vars:
            regs.append(Expr.var(idx))
        elif idx == num_vars:
            regs.append(Expr.const(1.0))
        else:
            regs.append(Expr.const(0.0))
    return regs


def _is_zero_const(expr: Expr) -> bool:
    return expr.kind == "const" and abs(float(expr.value)) < 1.0e-12


def _sum_exprs(terms: tuple[Expr, ...]) -> Expr:
    if not terms:
        return Expr.const(0.0)
    out = terms[0]
    for term in terms[1:]:
        if _is_zero_const(term):
            continue
        if _is_zero_const(out):
            out = term
        else:
            out = Expr.op(NAME_TO_ID["add"], (out, term))
    return out


def _sample_index(probs: list[float], rng: random.Random) -> int:
    r = rng.random()
    total = 0.0
    for idx, value in enumerate(probs):
        total += max(float(value), 0.0)
        if r <= total:
            return idx
    return max(len(probs) - 1, 0)
