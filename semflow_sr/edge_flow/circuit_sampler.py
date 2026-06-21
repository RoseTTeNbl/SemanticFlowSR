"""Complete expression sampling from edge distributions."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from ..sr.ast import Expr
from ..sr.ops import NAME_TO_ID, get_op
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
                choices, log_prob = self._sample_choices(edge_dist, mode, rng)
                expr = self._execute_choices(choices)
                out.append(CircuitSample(
                    sample_id=len(out),
                    mode=mode,
                    edge_choices=choices,
                    expression=expr,
                    log_prob=float(log_prob),
                    complexity=int(expr.complexity),
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

    def _sample_choices(self, edge_dist: EdgeDistribution, mode: int, rng: random.Random) -> tuple[dict[str, int], float]:
        choices: dict[str, int] = {}
        log_prob = math.log(max(float(edge_dist.mixture_probs[mode].item()), 1e-12))
        for group in self.groups:
            probs = edge_dist.group_probs[group.group_id][mode].detach().cpu().tolist()
            pos = _sample_index(probs, rng)
            choices[group.group_id] = int(pos)
            log_prob += math.log(max(float(probs[pos]), 1e-12))
        return choices, log_prob

    def _execute_choices(self, choices: dict[str, int]) -> Expr:
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
        return regs[int(choices["OUT:SELECT"])]


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


def _sample_index(probs: list[float], rng: random.Random) -> int:
    r = rng.random()
    total = 0.0
    for idx, value in enumerate(probs):
        total += max(float(value), 0.0)
        if r <= total:
            return idx
    return max(len(probs) - 1, 0)
