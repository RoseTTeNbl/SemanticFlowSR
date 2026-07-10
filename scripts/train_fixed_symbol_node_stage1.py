#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from semflow_sr.sr.ast import Expr, eval_expr
from semflow_sr.sr.ops import NAME_TO_ID, get_op
from semflow_sr.sr.printer import to_string


UNARY_OPS = {"copy", "sin", "cos", "square", "cube", "exp", "protected_log", "protected_sqrt"}
BINARY_OPS = {"add", "sub", "mul", "protected_div"}


@dataclass(frozen=True)
class Block:
    kind: str
    layer: int
    node: int = -1
    slot: int = -1
    term: int = -1
    size: int = 0


@dataclass(frozen=True)
class FixedSymbolTemplate:
    num_vars: int
    num_layers: int
    ops: tuple[str, ...]
    output_terms: int

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
    def node_count(self) -> int:
        return len(self.ops)

    @property
    def source_count(self) -> int:
        return self.base_count + self.node_count

    @property
    def blocks(self) -> tuple[Block, ...]:
        rows: list[Block] = []
        for layer in range(int(self.num_layers)):
            for node, op in enumerate(self.ops):
                arity = op_arity(op)
                for slot in range(arity):
                    rows.append(Block("edge", layer=layer, node=node, slot=slot, size=self.source_count))
        for term in range(int(self.output_terms)):
            rows.append(Block("readout", layer=int(self.num_layers), term=term, size=self.source_count))
        return tuple(rows)


def op_arity(op: str) -> int:
    if op == "copy":
        return 1
    return int(get_op(NAME_TO_ID[str(op)]).arity)


def theta_dim(template: FixedSymbolTemplate) -> int:
    return sum(int(block.size) for block in template.blocks)


def split_blocks(theta: torch.Tensor, template: FixedSymbolTemplate) -> list[torch.Tensor]:
    value = torch.as_tensor(theta).float().flatten()
    out = []
    cursor = 0
    for block in template.blocks:
        out.append(value[cursor: cursor + int(block.size)])
        cursor += int(block.size)
    if cursor != int(value.numel()):
        raise ValueError(f"theta dim mismatch: got {int(value.numel())}, expected {cursor}")
    return out


def pack_blocks(blocks: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([torch.as_tensor(block).float().flatten() for block in blocks], dim=0)


def center_theta(theta: torch.Tensor, template: FixedSymbolTemplate) -> torch.Tensor:
    return pack_blocks([block - block.mean() for block in split_blocks(theta, template)])


def masked_probs(logits: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.clamp_min(float(eps)) / probs.clamp_min(float(eps)).sum()


def initial_exprs(template: FixedSymbolTemplate) -> list[Expr]:
    regs = [Expr.var(i) for i in range(int(template.num_vars))]
    regs.append(Expr.const(0.0))
    regs.append(Expr.const(1.0))
    return regs


def apply_op(op: str, children: tuple[Expr, ...]) -> Expr:
    if op == "copy":
        return children[0]
    return Expr.op(NAME_TO_ID[str(op)], children)


def sum_exprs(terms: list[Expr]) -> Expr:
    kept = [term for term in terms if not (term.kind == "const" and abs(float(term.value)) < 1.0e-12)]
    if not kept:
        return Expr.const(0.0)
    out = kept[0]
    for term in kept[1:]:
        out = Expr.op(NAME_TO_ID["add"], (out, term))
    return out


def execute_choices(template: FixedSymbolTemplate, choices: list[int]) -> tuple[Expr, list[Expr], list[list[Expr]]]:
    base = initial_exprs(template)
    prev_nodes = [Expr.const(0.0) for _ in range(template.node_count)]
    all_layers: list[list[Expr]] = []
    cursor = 0
    for _layer in range(int(template.num_layers)):
        bank = base + prev_nodes
        current: list[Expr] = []
        for op in template.ops:
            arity = op_arity(op)
            srcs = []
            for _slot in range(arity):
                src = int(choices[cursor]) if cursor < len(choices) else 0
                cursor += 1
                srcs.append(max(0, min(src, len(bank) - 1)))
            current.append(apply_op(op, tuple(bank[src] for src in srcs)))
        prev_nodes = current
        all_layers.append(current)
    final_bank = base + prev_nodes
    terms: list[Expr] = []
    for _term in range(int(template.output_terms)):
        src = int(choices[cursor]) if cursor < len(choices) else 0
        cursor += 1
        terms.append(final_bank[max(0, min(src, len(final_bank) - 1))])
    return sum_exprs(terms), terms, all_layers


def block_index(template: FixedSymbolTemplate, *, layer: int, node: int, slot: int) -> int:
    idx = 0
    for l in range(int(layer)):
        for op in template.ops:
            idx += op_arity(op)
    for n in range(int(node)):
        idx += op_arity(template.ops[n])
    return idx + int(slot)


def readout_block_index(template: FixedSymbolTemplate, term: int) -> int:
    return sum(op_arity(op) for _ in range(int(template.num_layers)) for op in template.ops) + int(term)


def active_block_indices_for_choices(template: FixedSymbolTemplate, choices: list[int]) -> list[int]:
    active: set[int] = set()

    def visit_source(layer: int, src: int) -> None:
        if int(src) < template.base_count or int(layer) <= 0:
            return
        node = int(src) - template.base_count
        if not (0 <= node < template.node_count):
            return
        visit_node(int(layer) - 1, node)

    def visit_node(layer: int, node: int) -> None:
        if int(layer) < 0:
            return
        arity = op_arity(template.ops[int(node)])
        for slot in range(arity):
            bidx = block_index(template, layer=int(layer), node=int(node), slot=slot)
            active.add(bidx)
            visit_source(int(layer), int(choices[bidx]))

    for term in range(int(template.output_terms)):
        bidx = readout_block_index(template, term)
        active.add(bidx)
        visit_source(int(template.num_layers), int(choices[bidx]))
    return sorted(active)


def random_trace(template: FixedSymbolTemplate, rng: random.Random, *, max_depth_bias: float = 0.7) -> dict:
    choices = [0 for _ in template.blocks]
    for block_idx, block in enumerate(template.blocks):
        if block.kind == "edge":
            if int(block.layer) == 0 or rng.random() < float(max_depth_bias):
                # Prefer already meaningful base values early, but allow previous nodes later.
                upper = template.source_count if int(block.layer) > 0 and rng.random() < 0.55 else template.base_count
            else:
                upper = template.source_count
            choices[block_idx] = rng.randrange(max(upper, 1))
    output_sources = list(range(template.base_count, template.source_count))
    for term in range(int(template.output_terms)):
        idx = readout_block_index(template, term)
        if term == 0 or rng.random() < 0.65:
            choices[idx] = rng.choice(output_sources)
        else:
            choices[idx] = int(template.zero_source_index)
    active = set(active_block_indices_for_choices(template, choices))
    expr, terms, _layers = execute_choices(template, choices)
    return {
        "choices": choices,
        "active_block_indices": sorted(active),
        "block_weights": [1.0 if idx in active else 0.0 for idx in range(len(template.blocks))],
        "expression": expr,
        "expression_string": to_string(expr, int(template.num_vars), simplify=False),
        "term_count": int(len(terms)),
        "active_block_count": int(len(active)),
    }


def random_theta(
    template: FixedSymbolTemplate,
    *,
    device: torch.device,
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    blocks = [float(scale) * torch.randn(int(block.size), generator=generator, device=device) for block in template.blocks]
    return center_theta(pack_blocks(blocks), template)


def random_theta_for_trace(
    template: FixedSymbolTemplate,
    trace: dict,
    *,
    device: torch.device,
    scale: float,
    generator: torch.Generator,
    coupling: str,
    target_bias: float,
) -> torch.Tensor:
    blocks = [float(scale) * torch.randn(int(block.size), generator=generator, device=device) for block in template.blocks]
    if str(coupling) == "choice_bias":
        choices = list(trace["choices"])
        active = set(int(v) for v in trace["active_block_indices"])
        for idx in active:
            block = blocks[idx]
            block[int(choices[idx])] = block[int(choices[idx])] + float(target_bias)
    elif str(coupling) != "none":
        raise ValueError(f"unknown theta0 endpoint coupling: {coupling}")
    return center_theta(pack_blocks(blocks), template)


def target_theta(template: FixedSymbolTemplate, start: torch.Tensor, trace: dict, *, high: float, low: float) -> tuple[torch.Tensor, torch.Tensor]:
    start_blocks = split_blocks(start, template)
    choices = list(trace["choices"])
    active = set(int(v) for v in trace["active_block_indices"])
    out: list[torch.Tensor] = []
    weights: list[float] = []
    for idx, block in enumerate(template.blocks):
        if idx in active:
            logits = torch.full((int(block.size),), float(low), device=start.device)
            logits[int(choices[idx])] = float(high)
            out.append(logits - logits.mean())
            weights.append(1.0)
        else:
            out.append(start_blocks[idx].clone())
            weights.append(0.0)
    return pack_blocks(out).detach(), torch.tensor(weights, dtype=torch.float32, device=start.device)


def simplex_path(theta0: torch.Tensor, theta1: torch.Tensor, template: FixedSymbolTemplate, t: float, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor]:
    out_theta: list[torch.Tensor] = []
    out_velocity: list[torch.Tensor] = []
    for start, end in zip(split_blocks(theta0, template), split_blocks(theta1, template)):
        p0 = torch.softmax(start, dim=-1).clamp_min(eps)
        p0 = p0 / p0.sum()
        p1 = torch.softmax(end, dim=-1).clamp_min(eps)
        p1 = p1 / p1.sum()
        r0 = p0.sqrt()
        r1 = p1.sqrt()
        dot = (r0 * r1).sum().clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega).clamp_min(1.0e-6)
        tt = torch.as_tensor(float(t), device=start.device)
        a = torch.sin((1.0 - tt) * omega) / sin_omega
        b = torch.sin(tt * omega) / sin_omega
        da = -omega * torch.cos((1.0 - tt) * omega) / sin_omega
        db = omega * torch.cos(tt * omega) / sin_omega
        r = a * r0 + b * r1
        dr = da * r0 + db * r1
        p = (r * r).clamp_min(eps)
        p = p / p.sum()
        dp = 2.0 * r * dr
        dp = dp - p * dp.sum()
        velocity = dp / p.clamp_min(eps)
        velocity = velocity - velocity.mean()
        out_theta.append(p.clamp_min(eps).log() - p.clamp_min(eps).log().mean())
        out_velocity.append(velocity)
    return pack_blocks(out_theta).detach(), pack_blocks(out_velocity).detach()


def velocity_loss(theta_t: torch.Tensor, pred_v: torch.Tensor, target_v: torch.Tensor, template: FixedSymbolTemplate, weights: torch.Tensor, eps: float = 1.0e-4) -> tuple[torch.Tensor, dict]:
    losses: list[torch.Tensor] = []
    active_probs: list[torch.Tensor] = []
    for idx, (logits, pred, target) in enumerate(zip(split_blocks(theta_t, template), split_blocks(pred_v, template), split_blocks(target_v, template))):
        p = torch.softmax(logits, dim=-1)
        pred_dot = p * (pred - (p * pred).sum())
        target_dot = p * (target - (p * target).sum())
        diff = pred_dot - target_dot.detach()
        block_loss = ((diff * diff) / p.clamp_min(float(eps))).sum()
        losses.append(weights[idx].to(logits.device) * block_loss)
        if float(weights[idx].detach().cpu().item()) > 0:
            active_probs.append(p.max().detach())
    denom = weights.sum().clamp_min(1.0)
    loss = torch.stack(losses).sum() / denom
    return loss, {
        "active_block_count": float(weights.sum().detach().cpu().item()),
        "active_max_prob_mean": float(torch.stack(active_probs).mean().detach().cpu().item()) if active_probs else 0.0,
    }


class FixedSymbolVelocityNet(nn.Module):
    def __init__(self, template: FixedSymbolTemplate, hidden: int, *, condition_on_theta0: bool = True):
        super().__init__()
        self.template = template
        self.theta_dim = theta_dim(template)
        self.condition_on_theta0 = bool(condition_on_theta0)
        block_meta, action_meta = self.meta_rows()
        self.register_buffer("block_meta", block_meta, persistent=False)
        self.register_buffer("action_meta", action_meta, persistent=False)
        self.source_count = int(template.source_count)
        global_in = self.theta_dim + 8 + 1
        if self.condition_on_theta0:
            global_in += self.theta_dim + 8
        self.global_net = nn.Sequential(
            nn.Linear(global_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(
                hidden
                + block_meta.shape[1]
                + action_meta.shape[1]
                + 5
                + 1
                + 5
                + 1
                + 2 * self.source_count,
                hidden,
            ),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def meta_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        block_rows = []
        action_rows = []
        max_layer = max(float(self.template.num_layers), 1.0)
        max_node = max(float(self.template.node_count - 1), 1.0)
        max_source = max(float(self.template.source_count - 1), 1.0)
        op_vocab = list(dict.fromkeys(str(op) for op in self.template.ops))
        op_to_idx = {op: idx for idx, op in enumerate(op_vocab)}
        for block in self.template.blocks:
            for action in range(int(block.size)):
                block_op = [0.0 for _ in op_vocab]
                block_arity = 0.0
                if block.kind == "edge" and int(block.node) >= 0:
                    op_name = str(self.template.ops[int(block.node)])
                    block_op[op_to_idx[op_name]] = 1.0
                    block_arity = float(op_arity(op_name)) / 2.0
                block_rows.append([
                    float(block.layer) / max_layer,
                    1.0 if block.kind == "edge" else 0.0,
                    1.0 if block.kind == "readout" else 0.0,
                    float(block.node) / max_node if block.node >= 0 else -1.0,
                    float(block.slot) / 2.0 if block.slot >= 0 else -1.0,
                    float(block.term) / max(float(self.template.output_terms - 1), 1.0) if block.term >= 0 else -1.0,
                    block_arity,
                ] + block_op)
                src = int(action)
                source_op = [0.0 for _ in op_vocab]
                if src >= self.template.base_count:
                    node = int(src) - int(self.template.base_count)
                    if 0 <= node < int(self.template.node_count):
                        source_op[op_to_idx[str(self.template.ops[node])]] = 1.0
                action_rows.append([
                    float(src) / max_source,
                    1.0 if src < int(self.template.num_vars) else 0.0,
                    1.0 if src == int(self.template.zero_source_index) else 0.0,
                    1.0 if src == int(self.template.one_source_index) else 0.0,
                    1.0 if src >= self.template.base_count else 0.0,
                    float(src - self.template.base_count) / max_node if src >= self.template.base_count else -1.0,
                ] + source_op)
        return torch.tensor(block_rows, dtype=torch.float32), torch.tensor(action_rows, dtype=torch.float32)

    def theta_summary(self, theta: torch.Tensor) -> torch.Tensor:
        ent = []
        mx = []
        for logits in split_blocks(theta, self.template):
            p = torch.softmax(logits, dim=-1)
            ent.append((-(p * p.clamp_min(1.0e-8).log()).sum() / math.log(max(int(p.numel()), 2))).float())
            mx.append(p.max().float())
        return torch.stack([
            torch.stack(ent).mean(),
            torch.stack(mx).mean(),
            torch.stack(ent[: -self.template.output_terms]).mean(),
            torch.stack(mx[: -self.template.output_terms]).mean(),
            torch.stack(ent[-self.template.output_terms:]).mean(),
            torch.stack(mx[-self.template.output_terms:]).mean(),
            torch.tensor(float(self.template.num_layers), device=theta.device) / 32.0,
            torch.tensor(float(self.template.node_count), device=theta.device) / 16.0,
        ])

    def probability_meta(self, theta: torch.Tensor) -> torch.Tensor:
        rows = []
        for logits in split_blocks(theta, self.template):
            p = torch.softmax(logits, dim=-1)
            ent = (-(p * p.clamp_min(1.0e-8).log()).sum() / math.log(max(int(p.numel()), 2))).float()
            mx = p.max().float()
            for idx in range(int(logits.numel())):
                prob = p[idx]
                rows.append(torch.stack([prob, prob.clamp_min(1.0e-8).log() / 8.0, ent, mx, mx - prob]))
        return torch.stack(rows, dim=0)

    def block_context_meta(self, theta: torch.Tensor, theta0: torch.Tensor | None) -> torch.Tensor:
        rows = []
        seed_blocks = split_blocks(torch.zeros_like(theta) if theta0 is None else theta0, self.template)
        for logits, seed_logits in zip(split_blocks(theta, self.template), seed_blocks):
            current_ctx = logits.float() / 8.0
            seed_ctx = seed_logits.float() / 8.0
            ctx = torch.cat([current_ctx, seed_ctx], dim=0)
            for _idx in range(int(logits.numel())):
                rows.append(ctx)
        return torch.stack(rows, dim=0)

    def forward(self, theta: torch.Tensor, t: float, theta0: torch.Tensor | None = None) -> torch.Tensor:
        theta = theta.float().flatten()
        summary = self.theta_summary(theta)
        global_parts = [theta, summary, torch.tensor([float(t)], device=theta.device)]
        if self.condition_on_theta0:
            seed = torch.zeros_like(theta) if theta0 is None else theta0.to(theta.device).float().flatten()
            global_parts.extend([seed, self.theta_summary(seed)])
        g = self.global_net(torch.cat(global_parts, dim=0).unsqueeze(0)).squeeze(0)
        state = g.unsqueeze(0).expand(self.theta_dim, -1)
        prob_meta = self.probability_meta(theta)
        seed = torch.zeros_like(theta) if theta0 is None else theta0.to(theta.device).float().flatten()
        seed_prob_meta = self.probability_meta(seed)
        block_ctx = self.block_context_meta(theta, seed)
        out = self.head(torch.cat([
            state,
            self.block_meta.to(theta.device),
            self.action_meta.to(theta.device),
            prob_meta,
            theta[:, None] / 8.0,
            seed_prob_meta,
            seed[:, None] / 8.0,
            block_ctx,
        ], dim=-1)).squeeze(-1)
        return center_theta(20.0 * torch.tanh(out / 20.0), self.template)


def integrate(theta: torch.Tensor, velocity: torch.Tensor, template: FixedSymbolTemplate, dt: float) -> torch.Tensor:
    rows = []
    for logits, v in zip(split_blocks(theta, template), split_blocks(velocity, template)):
        next_logits = logits + float(dt) * v
        rows.append(next_logits - next_logits.mean())
    return pack_blocks(rows)


def decode_argmax(theta: torch.Tensor, template: FixedSymbolTemplate) -> tuple[Expr, list[int]]:
    choices = [int(torch.argmax(block).detach().cpu().item()) for block in split_blocks(theta, template)]
    expr, _terms, _layers = execute_choices(template, choices)
    return expr, choices


def terminal_summary(theta: torch.Tensor, template: FixedSymbolTemplate, trace: dict | None = None) -> dict:
    entropies = []
    max_probs = []
    argmax = []
    active_probs = []
    blocks = split_blocks(theta, template)
    for block in blocks:
        p = torch.softmax(block, dim=-1)
        entropies.append(float((-(p * p.clamp_min(1.0e-8).log()).sum() / math.log(max(int(p.numel()), 2))).detach().cpu().item()))
        max_probs.append(float(p.max().detach().cpu().item()))
        argmax.append(int(torch.argmax(p).detach().cpu().item()))
    out = {
        "terminal_entropy_mean": float(sum(entropies) / max(len(entropies), 1)),
        "terminal_max_prob_mean": float(sum(max_probs) / max(len(max_probs), 1)),
    }
    if trace is not None:
        active = [int(v) for v in trace["active_block_indices"]]
        choices = list(trace["choices"])
        matches = []
        for idx in active:
            p = torch.softmax(blocks[idx], dim=-1)
            active_probs.append(float(p[int(choices[idx])].detach().cpu().item()))
            matches.append(float(argmax[idx] == int(choices[idx])))
        out.update({
            "active_target_prob_mean": float(sum(active_probs) / max(len(active_probs), 1)),
            "active_argmax_match_mean": float(sum(matches) / max(len(matches), 1)),
        })
    return out


def sanitize_values(values: torch.Tensor, clip: float = 1.0e6) -> torch.Tensor:
    return torch.nan_to_num(values.float(), nan=0.0, posinf=float(clip), neginf=-float(clip)).clamp(-float(clip), float(clip))


def make_task_data(trace: dict, template: FixedSymbolTemplate, *, points: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = 2.0 * torch.rand((int(points), int(template.num_vars)), generator=gen, device=device) - 1.0
    y = sanitize_values(eval_expr(trace["expression"], x))
    y = y - y.mean()
    scale = y.std().clamp_min(1.0e-6)
    y = y / scale
    return x, y


def expression_energy(expr: Expr, x: torch.Tensor, y: torch.Tensor, *, complexity_weight: float) -> tuple[float, dict]:
    pred = sanitize_values(eval_expr(expr, x))
    pred = pred - pred.mean()
    pred_scale = pred.std().clamp_min(1.0e-6)
    pred = pred / pred_scale
    diff = pred - y
    mse = float((diff * diff).mean().detach().cpu().item())
    complexity = float(expr.complexity)
    energy = mse + float(complexity_weight) * complexity
    ss_res = float(((pred - y) ** 2).sum().detach().cpu().item())
    ss_tot = float(((y - y.mean()) ** 2).sum().clamp_min(1.0e-8).detach().cpu().item())
    return float(energy), {
        "mse": float(mse),
        "complexity": float(complexity),
        "r2": float(1.0 - ss_res / ss_tot),
    }


def sample_choices_from_theta(theta: torch.Tensor, template: FixedSymbolTemplate, generator: torch.Generator) -> list[int]:
    choices: list[int] = []
    for logits in split_blocks(theta, template):
        probs = torch.softmax(logits.float(), dim=-1)
        action = torch.multinomial(probs, num_samples=1, generator=generator).item()
        choices.append(int(action))
    return choices


def posterior_marginal_guidance(
    theta: torch.Tensor,
    template: FixedSymbolTemplate,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    samples: int,
    temperature: float,
    complexity_weight: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict]:
    sampled: list[dict] = []
    for _ in range(int(samples)):
        choices = sample_choices_from_theta(theta, template, generator)
        expr, _terms, _layers = execute_choices(template, choices)
        energy, ed = expression_energy(expr, x, y, complexity_weight=float(complexity_weight))
        sampled.append({
            "choices": choices,
            "active": active_block_indices_for_choices(template, choices),
            "energy": float(energy),
            **ed,
        })
    energies = torch.tensor([row["energy"] for row in sampled], dtype=torch.float32, device=theta.device)
    finite = torch.isfinite(energies)
    if not bool(finite.all()):
        energies = torch.where(finite, energies, torch.full_like(energies, float(energies[finite].max().item()) if bool(finite.any()) else 1.0e6))
    temp = max(float(temperature), 1.0e-6)
    weights = torch.softmax(-(energies - energies.min()) / temp, dim=0)
    theta_blocks = split_blocks(theta, template)
    out_blocks: list[torch.Tensor] = []
    active_mass_values: list[float] = []
    changed_blocks = 0
    for bidx, block in enumerate(template.blocks):
        probs = torch.softmax(theta_blocks[bidx], dim=-1)
        q = torch.zeros_like(probs)
        mass = torch.zeros((), dtype=torch.float32, device=theta.device)
        for sidx, row in enumerate(sampled):
            if bidx not in row["active"]:
                continue
            action = int(row["choices"][bidx])
            q[action] = q[action] + weights[sidx]
            mass = mass + weights[sidx]
        if float(mass.detach().cpu().item()) > 1.0e-6:
            q = q / mass.clamp_min(1.0e-6)
            out_blocks.append(q - probs)
            active_mass_values.append(float(mass.detach().cpu().item()))
            changed_blocks += 1
        else:
            out_blocks.append(torch.zeros_like(probs))
    best_idx = int(torch.argmin(energies).detach().cpu().item())
    mean_energy = float(energies.mean().detach().cpu().item())
    best_energy = float(energies[best_idx].detach().cpu().item())
    posterior_entropy = float((-(weights * weights.clamp_min(1.0e-8).log()).sum() / math.log(max(int(samples), 2))).detach().cpu().item())
    return pack_blocks(out_blocks), {
        "sample_energy_mean": mean_energy,
        "sample_energy_best": best_energy,
        "sample_mse_best": float(sampled[best_idx]["mse"]),
        "sample_r2_best": float(sampled[best_idx]["r2"]),
        "sample_complexity_best": float(sampled[best_idx]["complexity"]),
        "posterior_entropy": posterior_entropy,
        "posterior_changed_block_count": float(changed_blocks),
        "posterior_active_mass_mean": float(sum(active_mass_values) / max(len(active_mass_values), 1)),
    }


def probability_correction_to_logit_velocity(
    theta: torch.Tensor,
    correction_dot: torch.Tensor,
    base_velocity: torch.Tensor,
    template: FixedSymbolTemplate,
    *,
    relative_cap: float,
    absolute_cap: float,
    eps: float = 1.0e-5,
) -> tuple[torch.Tensor, dict]:
    rows: list[torch.Tensor] = []
    cap_ratios: list[float] = []
    corr_norms: list[float] = []
    for logits, corr, base_v in zip(split_blocks(theta, template), split_blocks(correction_dot, template), split_blocks(base_velocity, template)):
        p = torch.softmax(logits, dim=-1)
        base_dot = p * (base_v - (p * base_v).sum())
        corr = corr - corr.sum() * p
        base_fr = torch.sqrt(((base_dot * base_dot) / p.clamp_min(eps)).sum()).detach()
        corr_fr = torch.sqrt(((corr * corr) / p.clamp_min(eps)).sum()).detach()
        cap = float(absolute_cap) + float(relative_cap) * float(base_fr.detach().cpu().item())
        ratio = 1.0
        if float(corr_fr.detach().cpu().item()) > cap > 0.0:
            ratio = cap / float(corr_fr.detach().cpu().item())
            corr = corr * ratio
        v = corr / p.clamp_min(eps)
        v = v - v.mean()
        rows.append(v)
        cap_ratios.append(float(ratio))
        corr_norms.append(float(corr_fr.detach().cpu().item()))
    return pack_blocks(rows), {
        "guidance_cap_ratio_mean": float(sum(cap_ratios) / max(len(cap_ratios), 1)),
        "guidance_correction_fr_norm_mean": float(sum(corr_norms) / max(len(corr_norms), 1)),
    }


def rollout_with_optional_guidance(
    model: FixedSymbolVelocityNet,
    template: FixedSymbolTemplate,
    theta0: torch.Tensor,
    trace: dict,
    args,
    *,
    mode: str,
    generator: torch.Generator,
    task_seed: int,
) -> dict:
    device = theta0.device
    x, y = make_task_data(trace, template, points=int(args.stage2_points), seed=int(task_seed), device=device)
    theta = theta0.clone()
    steps = max(int(args.stage2_ode_steps), 1)
    step_metrics: list[dict] = []
    with torch.no_grad():
        for step in range(steps):
            t = float(step) / float(steps)
            base_v = model(theta, t, theta0=theta0)
            v = base_v
            if str(mode) == "online_semantic":
                corr_dot, md = posterior_marginal_guidance(
                    theta,
                    template,
                    x,
                    y,
                    samples=int(args.online_guidance_samples),
                    temperature=float(args.online_guidance_temperature),
                    complexity_weight=float(args.stage2_complexity_weight),
                    generator=generator,
                )
                gate = math.sin(math.pi * t) ** 2 if str(args.online_guidance_time_gate) == "sin2" else 1.0
                corr_dot = corr_dot * (float(args.online_guidance_strength) * float(gate))
                corr_v, cap_md = probability_correction_to_logit_velocity(
                    theta,
                    corr_dot,
                    base_v,
                    template,
                    relative_cap=float(args.online_guidance_relative_cap),
                    absolute_cap=float(args.online_guidance_absolute_cap),
                )
                v = base_v + corr_v
                step_metrics.append({"step": int(step), "t": float(t), "gate": float(gate), **md, **cap_md})
            theta = integrate(theta, v, template, 1.0 / float(steps))
    expr, choices = decode_argmax(theta, template)
    energy, ed = expression_energy(expr, x, y, complexity_weight=float(args.stage2_complexity_weight))
    term = terminal_summary(theta, template, trace)
    base = {
        "mode": str(mode),
        "expression": to_string(expr, int(template.num_vars), simplify=False),
        "energy": float(energy),
        **ed,
        **term,
    }
    if step_metrics:
        for key in [
            "sample_energy_mean",
            "sample_energy_best",
            "sample_r2_best",
            "posterior_entropy",
            "posterior_changed_block_count",
            "posterior_active_mass_mean",
            "guidance_cap_ratio_mean",
            "guidance_correction_fr_norm_mean",
        ]:
            vals = [float(row[key]) for row in step_metrics if key in row]
            base[f"{key}_mean"] = float(sum(vals) / max(len(vals), 1))
        base["guidance_step_count"] = int(len(step_metrics))
    else:
        base["guidance_step_count"] = 0
    return base


def run_stage2_energy_and_guidance(
    model: FixedSymbolVelocityNet,
    template: FixedSymbolTemplate,
    traces: list[dict],
    args,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    if int(args.stage2_tasks) <= 0:
        return {}, []
    gen = torch.Generator(device=device).manual_seed(int(args.seed) + 5011)
    rows: list[dict] = []
    task_count = min(int(args.stage2_tasks), len(traces))
    for idx, trace in enumerate(traces[:task_count]):
        theta0 = random_theta_for_trace(
            template,
            trace,
            device=device,
            scale=float(args.theta0_noise_scale),
            generator=gen,
            coupling=str(args.theta0_endpoint_coupling),
            target_bias=float(args.theta0_target_bias),
        )
        modes = ["off", "online_semantic"] if str(args.rollout_guidance_mode) == "both" else [str(args.rollout_guidance_mode)]
        for mode in modes:
            row = rollout_with_optional_guidance(
                model,
                template,
                theta0,
                trace,
                args,
                mode=mode,
                generator=gen,
                task_seed=int(args.seed) + 7000 + idx,
            )
            row.update({
                "task_index": int(idx),
                "target_expression": str(trace["expression_string"]),
                "target_active_block_count": int(trace["active_block_count"]),
            })
            rows.append(row)
    summary: dict[str, float | int | str] = {
        "stage2_task_count": int(task_count),
        "stage2_rollout_guidance_mode": str(args.rollout_guidance_mode),
        "stage2_ode_steps": int(args.stage2_ode_steps),
        "online_guidance_samples": int(args.online_guidance_samples),
        "online_guidance_strength": float(args.online_guidance_strength),
        "online_guidance_temperature": float(args.online_guidance_temperature),
    }
    for mode in sorted({str(row["mode"]) for row in rows}):
        mrows = [row for row in rows if str(row["mode"]) == mode]
        for key in ["energy", "mse", "r2", "active_target_prob_mean", "active_argmax_match_mean", "terminal_entropy_mean", "terminal_max_prob_mean"]:
            vals = [float(row[key]) for row in mrows if key in row]
            summary[f"stage2_{mode}_{key}_mean"] = float(sum(vals) / max(len(vals), 1))
        for key in ["sample_energy_best_mean", "sample_r2_best_mean", "posterior_entropy_mean", "posterior_changed_block_count_mean", "guidance_correction_fr_norm_mean"]:
            vals = [float(row[key]) for row in mrows if key in row]
            if vals:
                summary[f"stage2_{mode}_{key}"] = float(sum(vals) / max(len(vals), 1))
    if {"off", "online_semantic"}.issubset({str(row["mode"]) for row in rows}):
        off_by_task = {int(row["task_index"]): row for row in rows if str(row["mode"]) == "off"}
        on_by_task = {int(row["task_index"]): row for row in rows if str(row["mode"]) == "online_semantic"}
        shared = sorted(set(off_by_task) & set(on_by_task))
        if shared:
            summary["stage2_online_minus_off_energy_mean"] = float(sum(float(on_by_task[i]["energy"]) - float(off_by_task[i]["energy"]) for i in shared) / len(shared))
            summary["stage2_online_minus_off_r2_mean"] = float(sum(float(on_by_task[i]["r2"]) - float(off_by_task[i]["r2"]) for i in shared) / len(shared))
    return summary, rows


def sample_t(args, rng: random.Random) -> float:
    if args.time_sampling == "low_t_mixture" and rng.random() < float(args.low_t_sampling_prob):
        return rng.random() * float(args.low_t_max)
    return rng.random()


def train(args) -> dict:
    rng = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    template = FixedSymbolTemplate(
        num_vars=int(args.num_vars),
        num_layers=int(args.num_layers),
        ops=tuple(args.ops),
        output_terms=int(args.output_terms),
    )
    gen = torch.Generator(device=device).manual_seed(int(args.seed) + 17)
    traces = [random_trace(template, rng) for _ in range(int(args.trace_count))]
    if int(args.fixed_overfit_examples) > 0:
        traces = traces[: int(args.fixed_overfit_examples)]
    model = FixedSymbolVelocityNet(template, hidden=int(args.hidden), condition_on_theta0=bool(args.condition_on_theta0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    curve: list[dict] = []
    fixed_examples = None
    stopped_epoch = int(args.epochs) - 1
    for epoch in range(int(args.epochs)):
        examples = []
        for trace_idx, trace in enumerate(traces):
            theta0 = random_theta_for_trace(
                template,
                trace,
                device=device,
                scale=float(args.theta0_noise_scale),
                generator=gen,
                coupling=str(args.theta0_endpoint_coupling),
                target_bias=float(args.theta0_target_bias),
            )
            t = sample_t(args, rng)
            examples.append((trace_idx, trace, theta0, t))
        if int(args.fixed_overfit_examples) > 0:
            if fixed_examples is None:
                fixed_examples = examples
            examples = fixed_examples
        rng.shuffle(examples)
        losses = []
        for start in range(0, len(examples), int(args.batch_size)):
            batch = examples[start: start + int(args.batch_size)]
            batch_losses = []
            batch_metrics = []
            for _trace_idx, trace, theta0, t in batch:
                p1, weights = target_theta(template, theta0, trace, high=float(args.target_high), low=float(args.target_low))
                theta_t, target_v = simplex_path(theta0, p1, template, float(t))
                pred = model(theta_t.to(device), float(t), theta0=theta0.to(device))
                loss, metrics = velocity_loss(theta_t.to(device), pred, target_v.to(device), template, weights.to(device), eps=float(args.fisher_eps))
                batch_losses.append(loss)
                batch_metrics.append(metrics)
            loss = torch.stack(batch_losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            if bool(args.log_epochs) and (len(curve) == 0 or (len(curve) + 1) % int(args.log_interval) == 0):
                print(f"step {len(curve)+1} epoch={epoch} loss={losses[-1]:.6f}", flush=True)
            curve.append({
                "step": int(len(curve) + 1),
                "epoch": int(epoch),
                "loss": float(loss.detach().cpu().item()),
                "active_block_count": float(sum(m["active_block_count"] for m in batch_metrics) / max(len(batch_metrics), 1)),
                "active_max_prob_mean": float(sum(m["active_max_prob_mean"] for m in batch_metrics) / max(len(batch_metrics), 1)),
            })
        epoch_loss = sum(losses) / max(len(losses), 1)
        if bool(args.log_epochs):
            print(f"epoch {epoch} mean_loss={epoch_loss:.6f}", flush=True)
        if float(args.early_stop_loss) > 0.0 and epoch_loss <= float(args.early_stop_loss):
            stopped_epoch = int(epoch)
            if bool(args.log_epochs):
                print(f"early stop at epoch {epoch}: mean_loss={epoch_loss:.6f}", flush=True)
            break
        stopped_epoch = int(epoch)
    model.eval()
    eval_rows = []
    eval_gen = torch.Generator(device=device).manual_seed(int(args.seed) + 1009)
    with torch.no_grad():
        for idx, trace in enumerate(traces[: int(args.eval_traces)]):
            theta0 = random_theta_for_trace(
                template,
                trace,
                device=device,
                scale=float(args.theta0_noise_scale),
                generator=eval_gen,
                coupling=str(args.theta0_endpoint_coupling),
                target_bias=float(args.theta0_target_bias),
            )
            theta = theta0.clone()
            steps = max(int(args.ode_steps), 1)
            for step in range(steps):
                t = float(step) / float(steps)
                v = model(theta, t, theta0=theta0)
                theta = integrate(theta, v, template, 1.0 / float(steps))
            expr, _choices = decode_argmax(theta, template)
            row = {
                "trace_index": int(idx),
                "expression": to_string(expr, int(template.num_vars), simplify=False),
                **terminal_summary(theta, template, trace),
            }
            eval_rows.append(row)
    stage2_summary, stage2_rows = run_stage2_energy_and_guidance(model, template, traces, args, device)
    summary = {
        "training_flow": "fixed_symbol_node_stage1_syntax_prior",
        "construction_graph": "fixed_symbol_node_edges",
        "device": str(device),
        "theta_dim": int(theta_dim(template)),
        "num_layers": int(template.num_layers),
        "node_count": int(template.node_count),
        "source_count": int(template.source_count),
        "theta0_endpoint_coupling": str(args.theta0_endpoint_coupling),
        "theta0_target_bias": float(args.theta0_target_bias),
        "condition_on_theta0": bool(args.condition_on_theta0),
        "trace_count": int(len(traces)),
        "fixed_overfit_examples": int(args.fixed_overfit_examples),
        "final_loss_mean": float(sum(row["loss"] for row in curve[-max(1, min(len(curve), 20)):]) / max(1, min(len(curve), 20))),
        "last_train_loss": float(curve[-1]["loss"]) if curve else 0.0,
        "best_train_loss": float(min(row["loss"] for row in curve)) if curve else 0.0,
        "stopped_epoch": int(stopped_epoch),
        "early_stop_loss": float(args.early_stop_loss),
        "early_stopped": bool(float(args.early_stop_loss) > 0.0 and curve and float(curve[-1]["loss"]) <= float(args.early_stop_loss)),
        "eval_terminal_entropy_mean": float(sum(row["terminal_entropy_mean"] for row in eval_rows) / max(len(eval_rows), 1)),
        "eval_terminal_max_prob_mean": float(sum(row["terminal_max_prob_mean"] for row in eval_rows) / max(len(eval_rows), 1)),
        "eval_active_target_prob_mean": float(sum(row["active_target_prob_mean"] for row in eval_rows) / max(len(eval_rows), 1)),
        "eval_active_argmax_match_mean": float(sum(row["active_argmax_match_mean"] for row in eval_rows) / max(len(eval_rows), 1)),
        "active_block_count_mean": float(sum(trace["active_block_count"] for trace in traces) / max(len(traces), 1)),
    }
    summary.update(stage2_summary)
    return {"summary": summary, "curve": curve, "eval_rows": eval_rows, "stage2_rows": stage2_rows, "template": template, "model": model}


def write_outputs(args, result: dict) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": result["summary"]}
    (out_dir / "fixed_symbol_stage1_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (out_dir / "fixed_symbol_stage1_train_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in result["curve"] for k in row}))
        writer.writeheader()
        for row in result["curve"]:
            writer.writerow(row)
    with (out_dir / "fixed_symbol_stage1_eval_samples.jsonl").open("w") as f:
        for row in result["eval_rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "fixed_symbol_stage2_guidance_samples.jsonl").open("w") as f:
        for row in result.get("stage2_rows", []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    torch.save(
        {
            "model_state_dict": result["model"].state_dict(),
            "template": {
                "num_vars": result["template"].num_vars,
                "num_layers": result["template"].num_layers,
                "ops": result["template"].ops,
                "output_terms": result["template"].output_terms,
            },
            "args": vars(args),
        },
        out_dir / "fixed_symbol_stage1_checkpoint.pt",
    )
    lines = [
        "# Fixed Symbol Node Stage1",
        "",
        f"- final loss mean: {result['summary']['final_loss_mean']:.6f}",
        f"- terminal entropy mean: {result['summary']['eval_terminal_entropy_mean']:.6f}",
        f"- active target prob mean: {result['summary']['eval_active_target_prob_mean']:.6f}",
        f"- active argmax match mean: {result['summary']['eval_active_argmax_match_mean']:.6f}",
    ]
    if result.get("stage2_rows"):
        lines.extend([
            "",
            "## Stage2 Online Guidance",
            "",
            f"- stage2 tasks: {int(result['summary'].get('stage2_task_count', 0))}",
            f"- off energy mean: {float(result['summary'].get('stage2_off_energy_mean', 0.0)):.6f}",
            f"- online energy mean: {float(result['summary'].get('stage2_online_semantic_energy_mean', 0.0)):.6f}",
            f"- online-minus-off energy: {float(result['summary'].get('stage2_online_minus_off_energy_mean', 0.0)):.6f}",
            f"- off R2 mean: {float(result['summary'].get('stage2_off_r2_mean', 0.0)):.6f}",
            f"- online R2 mean: {float(result['summary'].get('stage2_online_semantic_r2_mean', 0.0)):.6f}",
        ])
    (out_dir / "fixed_symbol_stage1_results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-vars", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--output-terms", type=int, default=2)
    parser.add_argument("--ops", nargs="+", default=["copy", "add", "sub", "mul", "protected_div", "sin", "cos", "square"])
    parser.add_argument("--trace-count", type=int, default=512)
    parser.add_argument("--fixed-overfit-examples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--theta0-noise-scale", type=float, default=1.0)
    parser.add_argument("--target-high", type=float, default=4.0)
    parser.add_argument("--target-low", type=float, default=-4.0)
    parser.add_argument("--fisher-eps", type=float, default=1.0e-4)
    parser.add_argument("--theta0-endpoint-coupling", choices=["none", "choice_bias"], default="choice_bias")
    parser.add_argument("--theta0-target-bias", type=float, default=5.0)
    parser.add_argument("--condition-on-theta0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-sampling", choices=["uniform", "low_t_mixture"], default="uniform")
    parser.add_argument("--low-t-sampling-prob", type=float, default=0.4)
    parser.add_argument("--low-t-max", type=float, default=0.35)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--eval-traces", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--log-epochs", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--early-stop-loss", type=float, default=0.0)
    parser.add_argument("--stage2-tasks", type=int, default=0)
    parser.add_argument("--stage2-points", type=int, default=128)
    parser.add_argument("--stage2-ode-steps", type=int, default=64)
    parser.add_argument("--stage2-complexity-weight", type=float, default=1.0e-3)
    parser.add_argument("--rollout-guidance-mode", choices=["off", "online_semantic", "both"], default="off")
    parser.add_argument("--online-guidance-samples", type=int, default=32)
    parser.add_argument("--online-guidance-strength", type=float, default=0.05)
    parser.add_argument("--online-guidance-temperature", type=float, default=0.5)
    parser.add_argument("--online-guidance-relative-cap", type=float, default=0.25)
    parser.add_argument("--online-guidance-absolute-cap", type=float, default=0.02)
    parser.add_argument("--online-guidance-time-gate", choices=["none", "sin2"], default="sin2")
    args = parser.parse_args()
    result = train(args)
    write_outputs(args, result)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
