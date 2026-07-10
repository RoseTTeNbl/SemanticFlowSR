"""Semantic Pullback Fisher Flow conditional construction model.

The current SPFF mainline does not learn one global edge-probability table.
It encodes the current register roots and scores source choices conditioned on
the target register root being expanded. Register semantics are normalized and
encoded as curves; they are not exposed as hand-written candidate statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
import random

import torch
from torch import nn

from ..sr.ast import Expr
from ..sr.ops import NAME_TO_ID, N_OPS, get_op
from .circuit_sampler import CircuitSample
from .pullback_chart import (
    IdentitySemanticChart,
    LowRankSemanticChart,
    NeuralODESemanticChart,
    normalize_sphere,
    project_tangent,
    simplex_to_sphere,
    sphere_to_simplex,
)
from .reward import prune_tiny_coefficients
from .semantic_teacher import DecisionTrace
from .template import RegisterOperatorTemplate


@dataclass(frozen=True)
class ConditionalEdgeFlowConfig:
    num_vars: int
    hidden: int = 96
    head_terms: int = 3
    branches_per_register: int = 1
    update_mode: str = "carry_write"
    write_registers_per_layer: int = 0
    exclude_base_head_candidates: bool = False
    enable_keep_option: bool = False
    mask_duplicate_branches: bool = False
    include_base_source_pool: bool = True
    task_encoder: str = "mean"
    min_prob: float = 1e-6
    term_factorized: bool = False
    term_num_heads: int = 0
    term_prior_strength: float = 0.0
    term_prior_type: str = "natural"
    spff_enabled: bool = False
    spff_geometry: str = "pullback"
    spff_chart_type: str = "ode"
    spff_context_mode: str = "semantic"
    spff_num_candidates: int = 0
    spff_sem_dim: int = 96
    spff_chart_hidden: int = 96
    spff_velocity_hidden: int = 96
    spff_chart_rank: int = 2
    spff_ode_steps: int = 4
    spff_inference_steps: int = 4
    spff_max_chart_velocity: float = 0.05
    spff_max_velocity: float = 1.0
    constant_values: tuple[float, ...] = field(default_factory=lambda: (1.0,))


class ConditionalEdgeFlowModel(nn.Module):
    """Register-root conditioned edge model.

    The model first encodes every register root as a fused semantic/tree token.
    Edge probabilities are then scored pairwise from a target register token and
    each candidate source register token. The same scorer is reused for all
    layers, branches, argument slots, and head slots.
    """

    def __init__(self, cfg: ConditionalEdgeFlowConfig):
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.hidden)
        self.task_encoder = nn.Sequential(
            nn.Linear(int(cfg.num_vars) + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.pointnet_input_norm = nn.GroupNorm(1, int(cfg.num_vars) + 1)
        self.pointnet_conv1 = nn.Conv1d(int(cfg.num_vars) + 1, hidden, 1)
        self.pointnet_conv2 = nn.Conv1d(hidden, 2 * hidden, 1)
        self.pointnet_conv3 = nn.Conv1d(2 * hidden, 4 * hidden, 1)
        self.pointnet_fc = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(int(cfg.num_vars) + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.tree_encoder = nn.Sequential(
            nn.Linear(8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.register_fuse = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_context_fuse = nn.Sequential(
            nn.Linear(4 * hidden + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.op_embedding = nn.Embedding(N_OPS, hidden)
        self.update_embedding = nn.Embedding(2, hidden)
        self.operator_scorer = nn.Sequential(
            nn.Linear(2 * hidden + 9, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.source_target_scorer = nn.Sequential(
            nn.Linear(2 * hidden + 9, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.spff_num_candidates = int(cfg.spff_num_candidates or 0)
        self.spff_enabled = bool(cfg.spff_enabled and self.spff_num_candidates > 1)
        if self.spff_enabled:
            sem_dim = int(cfg.spff_sem_dim)
            spff_feature_dim = 4 * self.spff_num_candidates + 6
            self.spff_context_encoder = nn.Sequential(
                nn.Linear(spff_feature_dim, sem_dim),
                nn.SiLU(),
                nn.Linear(sem_dim, sem_dim),
                nn.SiLU(),
            )
            self.spff_velocity_head = nn.Sequential(
                nn.Linear(sem_dim + self.spff_num_candidates + 1, int(cfg.spff_velocity_hidden)),
                nn.SiLU(),
                nn.Linear(int(cfg.spff_velocity_hidden), int(cfg.spff_velocity_hidden)),
                nn.SiLU(),
                nn.Linear(int(cfg.spff_velocity_hidden), self.spff_num_candidates),
            )
            chart_type = str(cfg.spff_chart_type).lower()
            if chart_type in {"identity", "none"}:
                self.spff_chart = IdentitySemanticChart(self.spff_num_candidates, sem_dim)
            elif chart_type in {"low_rank", "lowrank"}:
                self.spff_chart = LowRankSemanticChart(
                    self.spff_num_candidates,
                    sem_dim,
                    rank=int(cfg.spff_chart_rank),
                    hidden_dim=int(cfg.spff_chart_hidden),
                )
            elif chart_type in {"ode", "neural_ode", "neural-ode"}:
                self.spff_chart = NeuralODESemanticChart(
                    self.spff_num_candidates,
                    sem_dim,
                    hidden_dim=int(cfg.spff_chart_hidden),
                    ode_steps=int(cfg.spff_ode_steps),
                    max_velocity=float(cfg.spff_max_chart_velocity),
                )
            else:
                raise ValueError(f"unknown SPFF chart type: {cfg.spff_chart_type}")
        else:
            self.spff_context_encoder = None
            self.spff_velocity_head = None
            self.spff_chart = None

    def task_embedding(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_used = _pad_x(x.float(), self.cfg.num_vars)
        y_norm = _normalize_vector(y.float())
        if str(self.cfg.task_encoder).lower() in {"pointnet", "tnet", "t-net"}:
            tokens = torch.cat([x_used, y_norm.unsqueeze(1)], dim=1).transpose(0, 1).unsqueeze(0)
            z = self.pointnet_input_norm(tokens)
            z = torch.nn.functional.silu(self.pointnet_conv1(z))
            z = torch.nn.functional.silu(self.pointnet_conv2(z))
            z = torch.nn.functional.silu(self.pointnet_conv3(z))
            z = z.max(dim=2).values.squeeze(0)
            return self.pointnet_fc(z)
        tokens = torch.cat([x_used, y_norm.unsqueeze(1)], dim=1)
        return self.task_encoder(tokens).mean(dim=0)

    def register_tokens(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regs: list[Expr],
        register_semantics: torch.Tensor,
        *,
        layer_id: int,
    ) -> torch.Tensor:
        if len(regs) != int(register_semantics.shape[1]):
            raise ValueError("register expression count must match semantic columns")
        x_used = _pad_x(x.float(), self.cfg.num_vars)
        y_norm = _normalize_vector(y.float())
        task = self.task_embedding(x, y)
        tokens: list[torch.Tensor] = []
        denom_reg = max(len(regs) - 1, 1)
        for reg_idx, expr in enumerate(regs):
            b_norm = _normalize_vector(register_semantics[:, reg_idx].float())
            point = torch.cat([x_used, b_norm.unsqueeze(1), y_norm.unsqueeze(1)], dim=1)
            sem = self.semantic_encoder(point).mean(dim=0)
            tree = self.tree_encoder(_tree_features(
                expr,
                reg_index=float(reg_idx) / float(denom_reg),
                layer_id=float(layer_id),
                device=point.device,
                dtype=point.dtype,
            ))
            tokens.append(self.register_fuse(torch.cat([task, sem, tree], dim=0)))
        return torch.stack(tokens, dim=0)

    def head_context_target_token(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        head_exprs: list[Expr] | tuple[Expr, ...],
        head_semantics: torch.Tensor,
        *,
        head_index: int,
        layer_id: int,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        head_list = list(head_exprs)
        if len(head_list) != int(head_semantics.shape[1]):
            raise ValueError("head context expression count must match semantic columns")
        if not 0 <= int(head_index) < len(head_list):
            raise ValueError("head index is out of range for the head context")
        tokens = self.register_tokens(
            x,
            y,
            head_list,
            head_semantics,
            layer_id=layer_id,
        )
        d = int(tokens.shape[0])
        mask = _candidate_mask(active_mask, d=d, device=tokens.device)
        if bool(mask.any()):
            active_tokens = tokens[mask]
            pooled_mean = active_tokens.mean(dim=0)
            pooled_max = active_tokens.max(dim=0).values
        else:
            pooled_mean = torch.zeros(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
            pooled_max = torch.zeros_like(pooled_mean)
        denom_heads = max(d - 1, 1)
        task = self.task_embedding(x, y).to(tokens.device, tokens.dtype)
        meta = torch.tensor([
            float(head_index) / float(denom_heads),
            float(layer_id) / 10.0,
            float(mask.to(dtype=tokens.dtype).mean().detach().cpu().item()),
        ], device=tokens.device, dtype=tokens.dtype)
        return self.head_context_fuse(torch.cat([
            tokens[int(head_index)],
            pooled_mean,
            pooled_max,
            task,
            meta,
        ], dim=0))

    def endpoint_from_logits(
        self,
        logits_fn,
        *,
        num_candidates: int,
        device: torch.device,
        dtype: torch.dtype,
        method: str,
        flow_steps: int,
        mask: torch.Tensor | None = None,
        return_details: bool = False,
        flow_time: float | None = None,
    ) -> torch.Tensor | dict:
        d = int(num_candidates)
        if d <= 0:
            raise ValueError("conditional edge group must have candidates")
        mask = _candidate_mask(mask, d=d, device=device)
        p = mask.to(dtype)
        p = p / p.sum().clamp_min(float(self.cfg.min_prob))
        initial_p = p.clone()

        def velocity_fn(probs: torch.Tensor, t: float) -> torch.Tensor:
            p_eval = probs.to(device=device, dtype=dtype).flatten()
            if int(p_eval.numel()) != d:
                raise ValueError(f"velocity state has {int(p_eval.numel())} entries for {d} candidates")
            p_eval = p_eval * mask.to(dtype)
            p_eval = p_eval.clamp_min(0.0)
            p_eval = p_eval / p_eval.sum().clamp_min(float(self.cfg.min_prob))
            raw_eval = logits_fn(p_eval, float(t)).masked_fill(~mask, 0.0)
            centered_eval = raw_eval - (p_eval * raw_eval).sum()
            return 0.5 * p_eval.sqrt() * centered_eval

        method = str(method).lower()
        if method in {"policy", "natural", "natural_policy", "ng_policy"}:
            t_value = 1.0 if flow_time is None else float(flow_time)
            logits = logits_fn(p, t_value)
            logits = logits.masked_fill(~mask, -1.0e9)
            raw = logits.masked_fill(~mask, 0.0)
            centered = raw - (p * raw).sum()
            pred_sqrt_velocity = 0.5 * p.sqrt() * centered
            probs = _simplex_probs(logits, self.cfg.min_prob)
            probs = probs * mask.to(probs.dtype)
            probs = probs / probs.sum().clamp_min(float(self.cfg.min_prob))
            if return_details:
                return {
                    "probs": probs,
                    "current_probs": p,
                    "initial_probs": initial_p,
                    "predicted_sqrt_velocity": pred_sqrt_velocity,
                    "velocity_fn": velocity_fn,
                    "flow_time": float(t_value),
                }
            return probs
        if method != "ode":
            raise ValueError(f"unknown conditional edge sampler method: {method}")
        steps = max(int(flow_steps), 1)
        for step in range(steps):
            t = float(step) / float(steps)
            raw = logits_fn(p, t).masked_fill(~mask, 0.0)
            w = raw - (p * raw).sum()
            pred_sqrt_velocity = 0.5 * p.sqrt() * w
            p = _sqrt_step(p, w, dt=1.0 / float(steps), min_prob=float(self.cfg.min_prob))
            p = p * mask.to(p.dtype)
            p = p / p.sum().clamp_min(float(self.cfg.min_prob))
        if return_details:
            return {
                "probs": p,
                "current_probs": p,
                "initial_probs": initial_p,
                "predicted_sqrt_velocity": pred_sqrt_velocity,
                "velocity_fn": velocity_fn,
                "flow_time": float(1.0),
            }
        return p

    def spff_context_from_semantics(
        self,
        candidate_semantics: torch.Tensor,
        target_semantics: torch.Tensor,
        *,
        layer_id: int,
        target_register: int,
        branch_id: int,
        arity_slot: int,
        primitive_id: int,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.spff_enabled or self.spff_context_encoder is None:
            raise RuntimeError("SPFF context requested while SPFF is disabled")
        sem = torch.as_tensor(candidate_semantics).float()
        if sem.ndim == 1:
            sem = sem.unsqueeze(1)
        if int(sem.shape[1]) != int(self.spff_num_candidates):
            raise ValueError(
                f"SPFF candidate semantics has {int(sem.shape[1])} candidates; "
                f"expected {int(self.spff_num_candidates)}"
            )
        device = sem.device
        dtype = sem.dtype
        y = torch.as_tensor(target_semantics, device=device, dtype=dtype).flatten()
        if int(y.numel()) != int(sem.shape[0]):
            y = torch.zeros(int(sem.shape[0]), device=device, dtype=dtype)
        sem_clean = torch.nan_to_num(sem, nan=0.0, posinf=0.0, neginf=0.0)
        sem_norm = sem_clean - sem_clean.mean(dim=0, keepdim=True)
        sem_std = sem_norm.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        sem_norm = sem_norm / sem_std
        y_norm = _normalize_vector(y).to(device=device, dtype=dtype)
        corr = (sem_norm * y_norm.unsqueeze(1)).mean(dim=0)
        means = sem_clean.mean(dim=0)
        stds = sem_clean.std(dim=0, unbiased=False)
        if source_mask is None:
            mask = torch.ones(int(self.spff_num_candidates), device=device, dtype=dtype)
        else:
            mask = torch.as_tensor(source_mask, device=device).flatten().to(dtype=dtype)
            if int(mask.numel()) != int(self.spff_num_candidates):
                raise ValueError("SPFF source mask candidate count mismatch")
        active_fraction = mask.mean()
        denom_regs = max(int(self.spff_num_candidates) - 1, 1)
        meta = torch.tensor([
            float(layer_id) / 10.0,
            float(max(target_register, 0)) / float(denom_regs),
            float(max(branch_id, 0)) / 10.0,
            float(max(arity_slot, 0)) / 2.0,
            float(max(primitive_id, 0)) / float(max(N_OPS - 1, 1)),
            float(active_fraction.detach().cpu().item()),
        ], device=device, dtype=dtype)
        features = torch.cat([corr, means, stds, mask, meta], dim=0).unsqueeze(0)
        if str(self.cfg.spff_context_mode).lower() in {"constant", "none", "zero", "no_semantic"}:
            features = torch.zeros_like(features)
        return self.spff_context_encoder(features)

    def spff_velocity(
        self,
        sem_context: torch.Tensor,
        r_t: torch.Tensor,
        t: torch.Tensor | float,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.spff_enabled or self.spff_velocity_head is None:
            raise RuntimeError("SPFF velocity requested while SPFF is disabled")
        state = normalize_sphere(r_t)
        if sem_context.ndim == 1:
            sem_context = sem_context.unsqueeze(0)
        if sem_context.shape[0] == 1 and state.shape[0] > 1:
            sem_context = sem_context.expand(state.shape[0], -1)
        if isinstance(t, torch.Tensor):
            tau = t.to(device=state.device, dtype=state.dtype)
            if tau.ndim == 0:
                tau = tau.expand(state.shape[0])
            tau = tau.reshape(state.shape[0], 1)
        else:
            tau = torch.full((state.shape[0], 1), float(t), device=state.device, dtype=state.dtype)
        raw = self.spff_velocity_head(torch.cat([sem_context.to(state.device, state.dtype), state, tau], dim=-1))
        raw = torch.tanh(raw) * float(self.cfg.spff_max_velocity)
        return project_tangent(raw, state, mask=mask)

    def spff_source_probs_from_context(
        self,
        sem_context: torch.Tensor,
        *,
        source_mask: torch.Tensor | None,
        flow_steps: int | None = None,
        p0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d = int(self.spff_num_candidates)
        device = sem_context.device
        dtype = sem_context.dtype
        mask = _candidate_mask(source_mask, d=d, device=device)
        if p0 is None:
            p0_value = mask.to(dtype)
            p0_value = p0_value / p0_value.sum().clamp_min(float(self.cfg.min_prob))
        else:
            p0_value = torch.as_tensor(p0, device=device, dtype=dtype).flatten()
            if int(p0_value.numel()) != d:
                raise ValueError("SPFF p0 candidate count mismatch")
            p0_value = torch.where(mask, p0_value.clamp_min(0.0), torch.zeros_like(p0_value))
            p0_value = p0_value / p0_value.sum().clamp_min(float(self.cfg.min_prob))
        geometry = str(self.cfg.spff_geometry).lower()
        if geometry in {"simplex", "fisher", "plain_simplex"}:
            r = simplex_to_sphere(p0_value.unsqueeze(0), mask=mask.unsqueeze(0))
            use_chart = False
        else:
            if self.spff_chart is None:
                raise RuntimeError("SPFF chart is not initialized")
            r = self.spff_chart(simplex_to_sphere(p0_value.unsqueeze(0)), sem_context, mask=mask.unsqueeze(0))
            use_chart = True
        steps = max(int(flow_steps or self.cfg.spff_inference_steps), 1)
        for step in range(steps):
            t = float(step) / float(steps)
            v = self.spff_velocity(sem_context, r, t, mask=mask.unsqueeze(0))
            r = normalize_sphere(r + (1.0 / float(steps)) * v, mask=mask.unsqueeze(0))
        s = self.spff_chart.inverse(r, sem_context, mask=mask.unsqueeze(0)) if use_chart else r
        probs = sphere_to_simplex(s, mask=mask.unsqueeze(0)).squeeze(0)
        probs = probs.clamp_min(float(self.cfg.min_prob)) * mask.to(dtype)
        return probs / probs.sum().clamp_min(float(self.cfg.min_prob))

    def operator_probs(
        self,
        *,
        target_token: torch.Tensor,
        primitive_ids: list[int],
        layer_id: int,
        target_register: int,
        branch_id: int,
        method: str,
        flow_steps: int,
        return_details: bool = False,
        flow_time: float | None = None,
    ) -> torch.Tensor:
        ids = torch.tensor(primitive_ids, dtype=torch.long, device=target_token.device)
        if int(ids.numel()) <= 0:
            raise ValueError("operator probability group must have candidates")
        op_tokens = torch.zeros(
            (int(ids.numel()), int(target_token.numel())),
            dtype=target_token.dtype,
            device=target_token.device,
        )
        real_mask = ids >= 0
        if bool(real_mask.any()):
            op_tokens[real_mask] = self.op_embedding(ids[real_mask])
        if bool((~real_mask).any()):
            stop_ids = torch.zeros(int((~real_mask).sum().item()), dtype=torch.long, device=target_token.device)
            op_tokens[~real_mask] = self.update_embedding(stop_ids)
        d = int(ids.numel())
        denom_ops = max(N_OPS - 1, 1)

        def logits_fn(p: torch.Tensor, t: float) -> torch.Tensor:
            meta = _meta_columns(
                d,
                device=target_token.device,
                dtype=target_token.dtype,
                values=[
                    float(layer_id),
                    float(target_register),
                    float(branch_id),
                    -1.0,
                    -1.0,
                    float(t),
                ],
                probs=p,
            )
            op_norm = torch.where(
                ids >= 0,
                ids.to(target_token.dtype) / float(denom_ops),
                torch.full_like(ids.to(target_token.dtype), -1.0),
            ).unsqueeze(1)
            target = target_token.expand(d, -1)
            return self.operator_scorer(torch.cat([target, op_tokens, meta, op_norm], dim=1)).squeeze(1)

        return self.endpoint_from_logits(
            logits_fn,
            num_candidates=d,
            device=target_token.device,
            dtype=target_token.dtype,
            method=method,
            flow_steps=flow_steps,
            return_details=return_details,
            flow_time=flow_time,
        )

    def update_action_probs(
        self,
        *,
        target_token: torch.Tensor,
        layer_id: int,
        target_register: int,
        method: str,
        flow_steps: int,
        return_details: bool = False,
        flow_time: float | None = None,
    ) -> torch.Tensor:
        ids = torch.tensor([0, 1], dtype=torch.long, device=target_token.device)
        action_tokens = self.update_embedding(ids)
        d = 2

        def logits_fn(p: torch.Tensor, t: float) -> torch.Tensor:
            meta = _meta_columns(
                d,
                device=target_token.device,
                dtype=target_token.dtype,
                values=[
                    float(layer_id),
                    float(target_register),
                    -1.0,
                    -1.0,
                    -1.0,
                    float(t),
                ],
                probs=p,
            )
            action_norm = ids.to(target_token.dtype).unsqueeze(1)
            target = target_token.expand(d, -1)
            return self.operator_scorer(torch.cat([target, action_tokens, meta, action_norm], dim=1)).squeeze(1)

        return self.endpoint_from_logits(
            logits_fn,
            num_candidates=d,
            device=target_token.device,
            dtype=target_token.dtype,
            method=method,
            flow_steps=flow_steps,
            return_details=return_details,
            flow_time=flow_time,
        )

    def source_probs(
        self,
        *,
        target_token: torch.Tensor,
        source_tokens: torch.Tensor,
        layer_id: int,
        target_register: int,
        branch_id: int,
        arity_slot: int,
        primitive_id: int,
        method: str,
        flow_steps: int,
        source_mask: torch.Tensor | None = None,
        source_p0: torch.Tensor | None = None,
        candidate_semantics: torch.Tensor | None = None,
        target_semantics: torch.Tensor | None = None,
        return_details: bool = False,
        flow_time: float | None = None,
    ) -> torch.Tensor:
        d = int(source_tokens.shape[0])
        has_spff_context = (
            self.spff_enabled
            and int(target_register) >= 0
            and int(branch_id) >= 0
            and int(arity_slot) >= 0
            and int(primitive_id) >= 0
            and candidate_semantics is not None
            and target_semantics is not None
        )
        if has_spff_context and int(d) > int(self.spff_num_candidates):
            raise ValueError(
                f"SPFF runtime candidates {int(d)} exceed chart width {int(self.spff_num_candidates)}"
            )
        if has_spff_context:
            padded_semantics = _pad_candidate_semantics(
                candidate_semantics,
                d=d,
                target_d=int(self.spff_num_candidates),
                device=source_tokens.device,
            )
            padded_mask = _pad_candidate_mask(
                source_mask,
                d=d,
                target_d=int(self.spff_num_candidates),
                device=source_tokens.device,
            )
            padded_p0 = _pad_candidate_distribution(
                source_p0,
                mask=source_mask,
                d=d,
                target_d=int(self.spff_num_candidates),
                device=source_tokens.device,
                dtype=source_tokens.dtype,
                min_prob=float(self.cfg.min_prob),
            )
            sem_context = self.spff_context_from_semantics(
                padded_semantics,
                target_semantics.to(source_tokens.device),
                layer_id=int(layer_id),
                target_register=int(target_register),
                branch_id=int(branch_id),
                arity_slot=int(arity_slot),
                primitive_id=int(primitive_id),
                source_mask=padded_mask,
            )
            full_probs = self.spff_source_probs_from_context(
                sem_context,
                source_mask=padded_mask,
                flow_steps=flow_steps,
                p0=padded_p0,
            )
            probs = full_probs[:d]
            mask = _candidate_mask(source_mask, d=d, device=source_tokens.device)
            probs = torch.where(mask, probs, torch.zeros_like(probs))
            probs = probs / probs.sum().clamp_min(float(self.cfg.min_prob))
            mask = _candidate_mask(source_mask, d=d, device=source_tokens.device)
            p0 = padded_p0[:d].to(source_tokens.dtype)
            p0 = torch.where(mask, p0, torch.zeros_like(p0))
            p0 = p0 / p0.sum().clamp_min(float(self.cfg.min_prob))
            if return_details:
                return {
                    "probs": probs,
                    "current_probs": p0,
                    "initial_probs": p0.clone(),
                    "predicted_sqrt_velocity": torch.zeros_like(probs),
                    "velocity_fn": None,
                    "flow_time": 1.0 if flow_time is None else float(flow_time),
                    "spff_sem_context": sem_context,
                    "spff_full_probs": full_probs,
                    "spff_full_mask": padded_mask,
                    "spff_full_p0": padded_p0,
                    "spff_padded_candidate_count": int(self.spff_num_candidates) - int(d),
                }
            return probs
        denom_src = max(d - 1, 1)
        denom_ops = max(N_OPS - 1, 1)

        def logits_fn(p: torch.Tensor, t: float) -> torch.Tensor:
            src_idx = torch.arange(d, dtype=source_tokens.dtype, device=source_tokens.device).unsqueeze(1)
            meta = torch.cat([
                torch.full((d, 1), float(layer_id), dtype=source_tokens.dtype, device=source_tokens.device),
                torch.full((d, 1), float(target_register), dtype=source_tokens.dtype, device=source_tokens.device),
                torch.full((d, 1), float(branch_id), dtype=source_tokens.dtype, device=source_tokens.device),
                torch.full((d, 1), float(arity_slot), dtype=source_tokens.dtype, device=source_tokens.device),
                torch.full((d, 1), float(primitive_id) / float(denom_ops), dtype=source_tokens.dtype, device=source_tokens.device),
                src_idx / float(denom_src),
                p.unsqueeze(1),
                p.clamp_min(float(self.cfg.min_prob)).log().unsqueeze(1),
                torch.full((d, 1), float(t), dtype=source_tokens.dtype, device=source_tokens.device),
            ], dim=1)
            target = target_token.expand(d, -1)
            return self.source_target_scorer(torch.cat([target, source_tokens, meta], dim=1)).squeeze(1)

        return self.endpoint_from_logits(
            logits_fn,
            num_candidates=d,
            device=source_tokens.device,
            dtype=source_tokens.dtype,
            method=method,
            flow_steps=flow_steps,
            mask=source_mask,
            return_details=return_details,
            flow_time=flow_time,
        )


class ConditionalEdgeFlowSampler:
    """Sequential sampler for target-register-conditioned expression paths."""

    def __init__(
        self,
        template: RegisterOperatorTemplate,
        model: ConditionalEdgeFlowModel,
        *,
        method: str = "policy",
        flow_steps: int = 4,
        time_sampling: str | float | None = None,
    ):
        self.template = template
        self.model = model
        self.method = str(method)
        self.flow_steps = int(flow_steps)
        self.time_sampling = time_sampling

    def sample(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        batch_size: int,
        rng: random.Random,
        active_variable_count: int | None = None,
    ) -> list[CircuitSample]:
        if bool(getattr(self.model.cfg, "term_factorized", False)):
            from .term_graph import TermGraphSampler

            return TermGraphSampler(
                self.template,
                self.model,
                method=self.method,
                flow_steps=self.flow_steps,
                time_sampling=self.time_sampling,
            ).sample(
                x,
                y,
                batch_size=batch_size,
                rng=rng,
                active_variable_count=active_variable_count,
            )
        out: list[CircuitSample] = []
        for sample_id in range(max(int(batch_size), 0)):
            out.append(self._sample_one(
                sample_id,
                x.float(),
                y.float(),
                rng,
                active_variable_count=active_variable_count,
            ))
        return out

    def _sample_one(
        self,
        sample_id: int,
        x: torch.Tensor,
        y: torch.Tensor,
        rng: random.Random,
        active_variable_count: int | None = None,
    ) -> CircuitSample:
        active_vars = _active_variable_count(
            x,
            template_num_vars=self.template.num_vars,
            explicit=active_variable_count,
        )
        regs, reg_sem, source_active = _initial_registers_and_semantics(
            x,
            self.template.num_vars,
            self.template.num_registers,
            active_vars,
        )
        base_regs = list(regs)
        base_sem = reg_sem.clone()
        base_active = source_active.clone()
        base_expr_ids = {
            id(expr)
            for expr, active in zip(base_regs, base_active.tolist())
            if bool(active)
        }
        head_pool: list[Expr] = [reg for idx, reg in enumerate(regs) if bool(source_active[idx])]
        head_semantics = [reg_sem[:, idx] for idx in range(reg_sem.shape[1]) if bool(source_active[idx])]
        head_active: list[bool] = [True for _ in head_pool]
        head_base_flags: list[bool] = [True for _ in head_pool]
        choices: dict[str, int] = {}
        log_prob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        additive_update_count = 0
        carried_register_count = 0
        written_register_count = 0
        keep_decision_count = 0
        update_choice_count = 0
        duplicate_branch_resample_count = 0
        sampled_branch_keys: list[str] = []
        ancestry_log_probs: dict[int, torch.Tensor] = {}
        ancestry_decision_counts: dict[int, int] = {}
        ancestry_trace_ids: dict[int, list[int]] = {}
        decision_traces: list[DecisionTrace] = []
        for idx, expr in enumerate(regs):
            ancestry_log_probs[id(expr)] = torch.zeros((), dtype=x.dtype, device=x.device)
            ancestry_decision_counts[id(expr)] = 0
            ancestry_trace_ids[id(expr)] = []
        primitive_ids = [NAME_TO_ID[name] for name in self.template.primitives]
        branch_count = max(int(self.model.cfg.branches_per_register), 1)
        for layer in range(self.template.num_layers):
            reg_tokens = self.model.register_tokens(x, y, regs, reg_sem, layer_id=layer)
            source_exprs, source_sem, source_mask = _source_pool(
                regs,
                reg_sem,
                source_active,
                base_regs=base_regs,
                base_sem=base_sem,
                base_active=base_active,
                layer_id=layer,
                include_base_source_pool=bool(self.model.cfg.include_base_source_pool),
            )
            source_tokens = self.model.register_tokens(x, y, source_exprs, source_sem, layer_id=layer)
            next_regs: list[Expr] = []
            next_semantics: list[torch.Tensor] = []
            next_active: list[bool] = []
            layer_branches: list[Expr] = []
            layer_branch_semantics: list[torch.Tensor] = []
            layer_branch_active_logs: list[torch.Tensor] = []
            layer_branch_active_counts: list[int] = []
            layer_branch_active_trace_ids: list[list[int]] = []
            write_targets = _write_targets(
                self.template.num_registers,
                layer_id=layer,
                configured=int(self.model.cfg.write_registers_per_layer),
                update_mode=str(self.model.cfg.update_mode),
            )
            used_branch_keys: set[str] = set()
            for target_reg in range(self.template.num_registers):
                if target_reg not in write_targets and _is_carry_write_mode(str(self.model.cfg.update_mode)):
                    next_regs.append(regs[target_reg])
                    next_semantics.append(reg_sem[:, target_reg])
                    ancestry_log_probs[id(regs[target_reg])] = ancestry_log_probs.get(
                        id(regs[target_reg]),
                        torch.zeros((), dtype=x.dtype, device=x.device),
                    )
                    ancestry_decision_counts[id(regs[target_reg])] = ancestry_decision_counts.get(id(regs[target_reg]), 0)
                    ancestry_trace_ids[id(regs[target_reg])] = ancestry_trace_ids.get(id(regs[target_reg]), [])
                    next_active.append(bool(source_active[target_reg]))
                    carried_register_count += 1
                    continue
                if bool(self.model.cfg.enable_keep_option):
                    group_id = f"L{layer}:TARGET{target_reg}:UPDATE_ACTION"
                    update_probs = self.model.update_action_probs(
                        target_token=reg_tokens[target_reg],
                        layer_id=layer,
                        target_register=target_reg,
                        method=self.method,
                        flow_steps=self.flow_steps,
                        flow_time=self._sample_flow_time(rng),
                    )
                    update_idx = _sample_index(update_probs, rng)
                    choices[group_id] = int(update_idx)
                    update_log = update_probs[int(update_idx)].clamp_min(1e-12).log()
                    log_prob_terms.append(update_log)
                    entropy_terms.append(_entropy(update_probs))
                    update_choice_count += 1
                    if int(update_idx) == 0:
                        current_expr = regs[target_reg]
                        current_count = int(ancestry_decision_counts.get(id(current_expr), 0))
                        current_log = ancestry_log_probs.get(
                            id(current_expr),
                            torch.zeros((), dtype=x.dtype, device=x.device),
                        )
                        ancestry_log_probs[id(current_expr)] = current_log + update_log
                        ancestry_decision_counts[id(current_expr)] = current_count + 1
                        ancestry_trace_ids[id(current_expr)] = ancestry_trace_ids.get(id(current_expr), [])
                        next_regs.append(current_expr)
                        next_semantics.append(reg_sem[:, target_reg])
                        next_active.append(bool(source_active[target_reg]))
                        keep_decision_count += 1
                        carried_register_count += 1
                        continue
                branches: list[Expr] = []
                branch_semantics: list[torch.Tensor] = []
                branch_active_logs: list[torch.Tensor] = []
                branch_active_counts: list[int] = []
                branch_active_trace_ids: list[list[int]] = []
                for branch in range(branch_count):
                    max_attempts = 4 if bool(self.model.cfg.mask_duplicate_branches) else 1
                    accepted = None
                    for attempt in range(max_attempts):
                        temp_choices: dict[str, int] = {}
                        temp_logs: list[torch.Tensor] = []
                        temp_entropies: list[torch.Tensor] = []
                        op_group = f"L{layer}:TARGET{target_reg}:BRANCH{branch}:OP"
                        op_dist = self.model.operator_probs(
                            target_token=reg_tokens[target_reg],
                            primitive_ids=primitive_ids,
                            layer_id=layer,
                            target_register=target_reg,
                            branch_id=branch,
                            method=self.method,
                            flow_steps=self.flow_steps,
                            return_details=True,
                            flow_time=self._sample_flow_time(rng),
                        )
                        op_probs = op_dist["probs"]
                        op_pos = _sample_index(op_probs, rng)
                        op_id = primitive_ids[int(op_pos)]
                        op = get_op(op_id)
                        temp_choices[op_group] = int(op_pos)
                        op_log_prob = op_probs[int(op_pos)].clamp_min(1e-12).log()
                        temp_logs.append(op_log_prob)
                        op_trace_id = len(decision_traces)
                        decision_traces.append(DecisionTrace(
                            group_id=op_group,
                            choice=int(op_pos),
                            current_probs=op_dist["current_probs"],
                            candidate_semantics=_operator_output_semantics(
                                primitive_ids,
                                reg_sem[:, target_reg],
                                fallback=y,
                            ).detach(),
                            predicted_sqrt_velocity=op_dist["predicted_sqrt_velocity"],
                            initial_probs=op_dist["initial_probs"],
                            velocity_fn=op_dist["velocity_fn"],
                            flow_time=float(op_dist["flow_time"]),
                            candidate_keys=tuple(get_op(op_id_value).name for op_id_value in primitive_ids),
                        ))
                        children: list[Expr] = []
                        child_semantics: list[torch.Tensor] = []
                        source_indices: list[int] = []
                        active_log = op_log_prob
                        active_count = 1
                        active_traces: list[int] = [op_trace_id]
                        for slot in range(op.arity):
                            src_group = f"L{layer}:TARGET{target_reg}:BRANCH{branch}:ARG{slot}:SRC"
                            source_candidate_semantics = _source_output_semantics(
                                op_id,
                                slot,
                                source_sem,
                                child_semantics,
                                fallback=y,
                            ).detach()
                            dist = self.model.source_probs(
                                target_token=reg_tokens[target_reg],
                                source_tokens=source_tokens,
                                layer_id=layer,
                                target_register=target_reg,
                                branch_id=branch,
                                arity_slot=slot,
                                primitive_id=op_id,
                                method=self.method,
                                flow_steps=self.flow_steps,
                                source_mask=source_mask.to(source_tokens.device),
                                candidate_semantics=source_candidate_semantics,
                                target_semantics=y,
                                return_details=True,
                                flow_time=self._sample_flow_time(rng),
                            )
                            probs = dist["probs"]
                            src_idx = _sample_index(probs, rng)
                            trace_id = len(decision_traces)
                            decision_traces.append(DecisionTrace(
                                group_id=src_group,
                                choice=int(src_idx),
                                current_probs=dist["current_probs"],
                                candidate_semantics=source_candidate_semantics,
                                predicted_sqrt_velocity=dist["predicted_sqrt_velocity"],
                                initial_probs=dist["initial_probs"],
                                velocity_fn=dist["velocity_fn"],
                                flow_time=float(dist["flow_time"]),
                                candidate_keys=_expr_keys(source_exprs),
                            ))
                            source_indices.append(int(src_idx))
                            temp_choices[src_group] = int(src_idx)
                            src_log_prob = probs[int(src_idx)].clamp_min(1e-12).log()
                            temp_logs.append(src_log_prob)
                            temp_entropies.append(_entropy(probs))
                            child_expr = source_exprs[int(src_idx)]
                            children.append(child_expr)
                            child_semantics.append(source_sem[:, int(src_idx)])
                            active_traces.append(trace_id)
                            active_traces.extend(ancestry_trace_ids.get(id(child_expr), []))
                            active_log = active_log + src_log_prob + ancestry_log_probs.get(
                                id(child_expr),
                                torch.zeros((), dtype=x.dtype, device=x.device),
                            )
                            active_count += 1 + int(ancestry_decision_counts.get(id(child_expr), 0))
                        branch_key = _branch_key(op_id, source_indices)
                        if (
                            not bool(self.model.cfg.mask_duplicate_branches)
                            or branch_key not in used_branch_keys
                            or attempt == max_attempts - 1
                        ):
                            accepted = (
                                temp_choices,
                                temp_logs,
                                temp_entropies,
                                op_id,
                                children,
                                child_semantics,
                                active_log,
                                active_count,
                                list(active_traces),
                                branch_key,
                            )
                            if branch_key in used_branch_keys and attempt == max_attempts - 1:
                                branch_key = f"{branch_key}#retry{attempt}"
                                accepted = (*accepted[:-1], branch_key)
                            break
                        duplicate_branch_resample_count += 1
                    if accepted is None:
                        raise RuntimeError("failed to sample conditional branch")
                    (
                        temp_choices,
                        temp_logs,
                        temp_entropies,
                        op_id,
                        children,
                        child_semantics,
                        active_log,
                        active_count,
                        active_traces,
                        branch_key,
                    ) = accepted
                    choices.update(temp_choices)
                    log_prob_terms.extend(temp_logs)
                    entropy_terms.extend(temp_entropies)
                    used_branch_keys.add(str(branch_key))
                    sampled_branch_keys.append(f"L{layer}:{branch_key}")
                    branch_expr = Expr.op(op_id, tuple(children))
                    branch_sem = _eval_op_semantics(op_id, child_semantics, fallback=y)
                    branches.append(branch_expr)
                    branch_semantics.append(branch_sem)
                    branch_active_logs.append(active_log)
                    branch_active_counts.append(active_count)
                    branch_active_trace_ids.append(list(active_traces))
                    layer_branches.append(branch_expr)
                    layer_branch_semantics.append(branch_sem)
                    layer_branch_active_logs.append(active_log)
                    layer_branch_active_counts.append(active_count)
                    layer_branch_active_trace_ids.append(list(active_traces))
                selected_idx = self._select_branch(
                    choices,
                    log_prob_terms,
                    x,
                    y,
                    reg_tokens[target_reg],
                    branches,
                    branch_semantics,
                    layer_id=layer,
                    target_register=target_reg,
                    rng=rng,
                )
                selected_expr = branches[int(selected_idx)]
                selected_sem = branch_semantics[int(selected_idx)]
                if branch_count > 1:
                    branch_log = log_prob_terms[-1]
                    branch_active_logs[int(selected_idx)] = branch_active_logs[int(selected_idx)] + branch_log
                    branch_active_counts[int(selected_idx)] += 1
                next_expr, next_sem = _combine_register_update(
                    regs[target_reg],
                    reg_sem[:, target_reg],
                    selected_expr,
                    selected_sem,
                    str(self.model.cfg.update_mode),
                )
                if str(self.model.cfg.update_mode).lower() in {"add", "additive", "residual"}:
                    additive_update_count += 1
                ancestry_log_probs[id(next_expr)] = branch_active_logs[int(selected_idx)]
                ancestry_decision_counts[id(next_expr)] = int(branch_active_counts[int(selected_idx)])
                ancestry_trace_ids[id(next_expr)] = list(branch_active_trace_ids[int(selected_idx)])
                next_regs.append(next_expr)
                next_semantics.append(next_sem)
                next_active.append(True)
                written_register_count += 1
            for branch_expr, branch_log, branch_count_value in zip(
                layer_branches,
                layer_branch_active_logs,
                layer_branch_active_counts,
            ):
                ancestry_log_probs[id(branch_expr)] = branch_log
                ancestry_decision_counts[id(branch_expr)] = int(branch_count_value)
            for branch_expr, trace_ids in zip(layer_branches, layer_branch_active_trace_ids):
                ancestry_trace_ids[id(branch_expr)] = list(trace_ids)
            regs = next_regs
            reg_sem = torch.stack(next_semantics, dim=1)
            source_active = torch.tensor(next_active, dtype=torch.bool, device=reg_sem.device)
            head_pool.extend(layer_branches)
            head_pool.extend(regs)
            head_semantics.extend(layer_branch_semantics)
            head_semantics.extend(next_semantics)
            head_active.extend(True for _ in layer_branches)
            head_active.extend(bool(flag) for flag in next_active)
            head_base_flags.extend(False for _ in layer_branches)
            head_base_flags.extend(id(expr) in base_expr_ids for expr in regs)
        final_tokens = self.model.register_tokens(x, y, regs, reg_sem, layer_id=self.template.num_layers)
        final_query = final_tokens.mean(dim=0)
        head_matrix = torch.stack(head_semantics, dim=1)
        head_tokens = self.model.register_tokens(
            x,
            y,
            head_pool,
            head_matrix,
            layer_id=self.template.num_layers,
        )
        head_mask = torch.tensor(head_active, dtype=torch.bool, device=head_tokens.device)
        head_base_mask = torch.tensor(head_base_flags, dtype=torch.bool, device=head_tokens.device)
        base_head_candidate_count = int((head_mask & head_base_mask).sum().item())
        if bool(self.model.cfg.exclude_base_head_candidates):
            candidate_mask = head_mask & ~head_base_mask
            if bool(candidate_mask.any()):
                head_mask = candidate_mask
        head_terms: list[Expr] = []
        head_active_logs: list[torch.Tensor] = []
        head_active_counts: list[int] = []
        active_trace_ids: list[int] = []
        base_head_selected_count = 0
        for slot in range(max(int(self.model.cfg.head_terms), 1)):
            group_id = f"HEAD:TERM{slot}:SRC"
            dist = self.model.source_probs(
                target_token=final_query,
                source_tokens=head_tokens,
                layer_id=self.template.num_layers,
                target_register=-1,
                branch_id=slot,
                arity_slot=slot,
                primitive_id=-1,
                method=self.method,
                flow_steps=self.flow_steps,
                source_mask=head_mask,
                return_details=True,
                flow_time=self._sample_flow_time(rng),
            )
            probs = dist["probs"]
            idx = _sample_index(probs, rng)
            trace_id = len(decision_traces)
            decision_traces.append(DecisionTrace(
                group_id=group_id,
                choice=int(idx),
                current_probs=dist["current_probs"],
                candidate_semantics=head_matrix.detach(),
                predicted_sqrt_velocity=dist["predicted_sqrt_velocity"],
                initial_probs=dist["initial_probs"],
                velocity_fn=dist["velocity_fn"],
                flow_time=float(dist["flow_time"]),
                candidate_keys=_expr_keys(head_pool),
            ))
            choices[group_id] = int(idx)
            head_log_prob = probs[int(idx)].clamp_min(1e-12).log()
            log_prob_terms.append(head_log_prob)
            entropy_terms.append(_entropy(probs))
            head_expr = head_pool[int(idx)]
            if bool(head_base_mask[int(idx)].item()):
                base_head_selected_count += 1
            head_terms.append(head_expr)
            head_active_logs.append(head_log_prob + ancestry_log_probs.get(
                id(head_expr),
                torch.zeros((), dtype=x.dtype, device=x.device),
            ))
            head_active_counts.append(1 + int(ancestry_decision_counts.get(id(head_expr), 0)))
            active_trace_ids.append(trace_id)
            active_trace_ids.extend(ancestry_trace_ids.get(id(head_expr), []))
        for trace_id in set(active_trace_ids):
            if 0 <= int(trace_id) < len(decision_traces):
                decision_traces[int(trace_id)].active = True
        expr = _sum_exprs(tuple(head_terms))
        log_prob_tensor = torch.stack(log_prob_terms).sum() if log_prob_terms else torch.tensor(0.0, device=x.device)
        active_log_prob_tensor = torch.stack(head_active_logs).sum() if head_active_logs else log_prob_tensor
        active_decision_count = int(sum(head_active_counts)) if head_active_counts else int(len(log_prob_terms))
        entropy_tensor = torch.stack(entropy_terms).mean() if entropy_terms else torch.tensor(0.0, device=x.device)
        return CircuitSample(
            sample_id=int(sample_id),
            mode=0,
            edge_choices=choices,
            expression=expr,
            log_prob=float(log_prob_tensor.detach().cpu().item()),
            complexity=int(sum(term.complexity for term in head_terms) + max(len(head_terms) - 1, 0)),
            head_terms=tuple(head_terms),
            log_prob_tensor=log_prob_tensor,
            active_log_prob_tensor=active_log_prob_tensor,
            entropy_tensor=entropy_tensor,
            decision_traces=tuple(decision_traces),
            diagnostics={
                "sampler_method": self.method,
                "head_terms": int(len(head_terms)),
                "conditional_groups": int(len(choices)),
                "decision_count": int(len(log_prob_terms)),
                "branches_per_register": int(branch_count),
                "update_mode": str(self.model.cfg.update_mode),
                "additive_update_count": int(additive_update_count),
                "carried_register_count": int(carried_register_count),
                "written_register_count": int(written_register_count),
                "keep_decision_count": int(keep_decision_count),
                "update_choice_count": int(update_choice_count),
                "keep_option_enabled": bool(self.model.cfg.enable_keep_option),
                "mask_duplicate_branches": bool(self.model.cfg.mask_duplicate_branches),
                "duplicate_branch_resample_count": int(duplicate_branch_resample_count),
                "sampled_branch_keys": list(sampled_branch_keys),
                "include_base_source_pool": bool(self.model.cfg.include_base_source_pool),
                "exclude_base_head_candidates": bool(self.model.cfg.exclude_base_head_candidates),
                "base_head_candidate_count": int(base_head_candidate_count),
                "base_head_selected_count": int(base_head_selected_count),
                "base_head_selected_rate": float(base_head_selected_count / max(len(head_terms), 1)),
                "active_variable_count": int(active_vars),
                "active_decision_count": int(active_decision_count),
                "base_source_count": int(base_active.sum().item()),
            },
        )

    def _sample_flow_time(self, rng: random.Random) -> float | None:
        return _sample_flow_time(self.time_sampling, rng)

    def _select_branch(
        self,
        choices: dict[str, int],
        log_prob_terms: list[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        target_token: torch.Tensor,
        branches: list[Expr],
        branch_semantics: list[torch.Tensor],
        *,
        layer_id: int,
        target_register: int,
        rng: random.Random,
    ) -> int:
        if len(branches) <= 1:
            return 0
        branch_matrix = torch.stack(branch_semantics, dim=1)
        branch_tokens = self.model.register_tokens(x, y, branches, branch_matrix, layer_id=layer_id)
        group_id = f"L{layer_id}:TARGET{target_register}:BRANCH_SELECT"
        probs = self.model.source_probs(
            target_token=target_token,
            source_tokens=branch_tokens,
            layer_id=layer_id,
            target_register=target_register,
            branch_id=-1,
            arity_slot=-1,
            primitive_id=-1,
            method=self.method,
            flow_steps=self.flow_steps,
            flow_time=self._sample_flow_time(rng),
        )
        idx = _sample_index(probs, rng)
        choices[group_id] = int(idx)
        log_prob_terms.append(probs[int(idx)].clamp_min(1e-12).log())
        return int(idx)


def conditional_elite_policy_loss(
    samples: list[CircuitSample],
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    elite_k: int,
    rank_temperature: float | None = None,
    entropy_bonus: float = 0.0,
    unique_elites: bool = False,
    gt_samples: list[CircuitSample] | None = None,
) -> tuple[torch.Tensor, dict]:
    trainable = torch.tensor(
        [sample.log_prob_tensor is not None for sample in samples],
        dtype=torch.bool,
        device=valid_mask.device,
    )
    valid_indices = torch.nonzero(valid_mask & trainable, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        zero = torch.zeros((), dtype=rewards.dtype, device=rewards.device, requires_grad=True)
        return zero, {"conditional_elite_count": 0, "conditional_best_reward": 0.0}
    k = min(max(int(elite_k), 1), int(valid_indices.numel()))
    valid_rewards = rewards[valid_indices]
    _, order = torch.topk(valid_rewards, k=min(k, int(valid_rewards.numel())))
    ordered_indices = [int(valid_indices[pos].item()) for pos in order]
    elite_indices = _select_diverse_indices(samples, ordered_indices, k=k) if bool(unique_elites) else ordered_indices[:k]
    elite_samples = [samples[int(idx)] for idx in elite_indices]
    if gt_samples:
        elite_samples.extend(sample for sample in gt_samples if sample.log_prob_tensor is not None)
    losses = []
    entropies = []
    for sample in elite_samples:
        log_prob = sample.active_log_prob_tensor if sample.active_log_prob_tensor is not None else sample.log_prob_tensor
        if log_prob is not None:
            decision_count = int((sample.diagnostics or {}).get(
                "active_decision_count" if sample.active_log_prob_tensor is not None else "decision_count",
                len(sample.edge_choices),
            ))
            losses.append(-log_prob / max(float(decision_count), 1.0))
            if sample.entropy_tensor is not None:
                entropies.append(sample.entropy_tensor)
    if not losses:
        zero = torch.zeros((), requires_grad=True)
        return zero, {"conditional_elite_count": 0, "conditional_best_reward": 0.0}
    loss_values = torch.stack(losses)
    tau = float(rank_temperature or 0.0)
    if tau > 0.0 and len(losses) > 1:
        rank_scores = torch.linspace(1.0, 0.0, steps=len(losses), dtype=loss_values.dtype, device=loss_values.device)
        weights = torch.softmax(rank_scores / max(tau, 1e-6), dim=0)
        loss = (weights * loss_values).sum()
        ess = float(1.0 / (weights.detach().cpu().pow(2).sum().item()))
    else:
        weights = torch.full_like(loss_values, 1.0 / max(int(loss_values.numel()), 1))
        loss = loss_values.mean()
        ess = float(loss_values.numel())
    entropy_value = torch.stack(entropies).mean() if entropies else torch.zeros((), dtype=loss.dtype, device=loss.device)
    if float(entropy_bonus) != 0.0:
        loss = loss - float(entropy_bonus) * entropy_value
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "loss_per_decision": float(loss.detach().cpu().item()),
        "conditional_elite_count": int(len(losses)),
        "conditional_unique_elite_count": int(len({_sample_key(sample) for sample in elite_samples})),
        "gt_elite_count": int(len(gt_samples or [])),
        "conditional_best_reward": float(valid_rewards.max().detach().cpu().item()),
        "elite_weight_ess": ess,
        "entropy_bonus": float(entropy_value.detach().cpu().item()),
        "active_ancestry_loss": bool(any(sample.active_log_prob_tensor is not None for sample in elite_samples)),
        "active_decision_count_mean": float(sum(
            int((sample.diagnostics or {}).get("active_decision_count", 0))
            for sample in elite_samples
            if sample.active_log_prob_tensor is not None
        ) / max(sum(1 for sample in elite_samples if sample.active_log_prob_tensor is not None), 1)),
    }


def render_sparse_expression(
    sample: CircuitSample,
    coef: torch.Tensor | list[float],
    *,
    prune_rel: float = 1.0e-4,
    prune_abs: float = 1.0e-6,
) -> Expr:
    pruned = prune_tiny_coefficients(
        torch.as_tensor(coef).detach().float().cpu().flatten(),
        rel=float(prune_rel),
        abs_threshold=float(prune_abs),
    )
    coeffs = [float(v) for v in pruned.tolist()]
    terms = tuple(sample.head_terms) if sample.head_terms else (sample.expression,)
    if len(coeffs) < len(terms) + 1:
        coeffs = coeffs + [0.0 for _ in range(len(terms) + 1 - len(coeffs))]
    pieces: list[Expr] = []
    for c, term in zip(coeffs[:-1], terms):
        if abs(float(c)) < 1e-10:
            continue
        pieces.append(Expr.op(NAME_TO_ID["mul"], (Expr.const(float(c)), term)))
    if abs(float(coeffs[-1])) >= 1e-10 or not pieces:
        pieces.append(Expr.const(float(coeffs[-1])))
    return _sum_exprs(tuple(pieces))


def _initial_registers_and_semantics(
    x: torch.Tensor,
    num_vars: int,
    num_registers: int,
    active_variable_count: int | None = None,
) -> tuple[list[Expr], torch.Tensor, torch.Tensor]:
    active_vars = _active_variable_count(x, template_num_vars=num_vars, explicit=active_variable_count)
    regs: list[Expr] = []
    sems: list[torch.Tensor] = []
    active: list[bool] = []
    for idx in range(num_registers):
        if idx < num_vars:
            regs.append(Expr.var(idx))
            if idx < active_vars and idx < int(x.shape[1]):
                sems.append(x[:, idx])
                active.append(True)
            else:
                sems.append(torch.zeros_like(x[:, 0]))
                active.append(False)
        elif idx == num_vars:
            regs.append(Expr.const(1.0))
            sems.append(torch.ones_like(x[:, 0]))
            active.append(True)
        else:
            regs.append(Expr.const(0.0))
            sems.append(torch.zeros_like(x[:, 0]))
            active.append(False)
    return regs, torch.stack(sems, dim=1), torch.tensor(active, dtype=torch.bool, device=x.device)


def _sum_exprs(terms: tuple[Expr, ...]) -> Expr:
    if not terms:
        return Expr.const(0.0)
    out = terms[0]
    for term in terms[1:]:
        out = Expr.op(NAME_TO_ID["add"], (out, term))
    return out


def _source_pool(
    regs: list[Expr],
    reg_sem: torch.Tensor,
    source_active: torch.Tensor,
    *,
    base_regs: list[Expr],
    base_sem: torch.Tensor,
    base_active: torch.Tensor,
    layer_id: int,
    include_base_source_pool: bool = True,
) -> tuple[list[Expr], torch.Tensor, torch.Tensor]:
    if int(layer_id) <= 0 or not bool(include_base_source_pool):
        return regs, reg_sem, source_active
    active_base_indices = [idx for idx in range(len(base_regs)) if bool(base_active[idx])]
    if not active_base_indices:
        return regs, reg_sem, torch.ones(len(regs), dtype=torch.bool, device=reg_sem.device)
    exprs = list(regs) + [base_regs[idx] for idx in active_base_indices]
    sems = [reg_sem[:, idx] for idx in range(reg_sem.shape[1])]
    sems.extend(base_sem[:, idx] for idx in active_base_indices)
    active_values = [bool(source_active[idx]) for idx in range(len(regs))]
    active_values.extend(True for _ in active_base_indices)
    mask = torch.tensor(active_values, dtype=torch.bool, device=reg_sem.device)
    return exprs, torch.stack(sems, dim=1), mask


def _combine_register_update(
    current_expr: Expr,
    current_sem: torch.Tensor,
    branch_expr: Expr,
    branch_sem: torch.Tensor,
    update_mode: str,
) -> tuple[Expr, torch.Tensor]:
    mode = str(update_mode).lower()
    if mode in {"replace", "branch", "overwrite", "affine", "linear", "carry_write", "partial_write"}:
        return branch_expr, branch_sem
    if mode in {"add", "additive", "residual"}:
        return Expr.op(NAME_TO_ID["add"], (current_expr, branch_expr)), torch.nan_to_num(current_sem + branch_sem)
    raise ValueError(f"unknown conditional register update mode: {update_mode}")


def _is_carry_write_mode(update_mode: str) -> bool:
    return str(update_mode).lower() in {"carry_write", "partial_write", "carry", "partial"}


def _branch_key(op_id: int, source_indices: list[int]) -> str:
    op_name = get_op(int(op_id)).name
    values = [int(idx) for idx in source_indices]
    if op_name in {"add", "mul"}:
        values = sorted(values)
    return op_name + "(" + ",".join(str(value) for value in values) + ")"


def _write_targets(
    num_registers: int,
    *,
    layer_id: int,
    configured: int,
    update_mode: str,
) -> set[int]:
    if not _is_carry_write_mode(update_mode):
        return set(range(int(num_registers)))
    n = int(num_registers)
    count = int(configured)
    if count <= 0:
        count = max(1, n // 2)
    count = min(max(count, 1), n)
    start = (int(layer_id) * count) % n
    return {int((start + offset) % n) for offset in range(count)}


def _eval_op_semantics(op_id: int, child_semantics: list[torch.Tensor], *, fallback: torch.Tensor) -> torch.Tensor:
    try:
        sem = get_op(op_id).fn(*child_semantics)
        return torch.nan_to_num(sem, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return torch.zeros_like(fallback)


def _operator_output_semantics(
    primitive_ids: list[int],
    anchor_semantics: torch.Tensor,
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    values = []
    for op_id in primitive_ids:
        op = get_op(int(op_id))
        if int(op.arity) <= 1:
            children = [anchor_semantics]
        else:
            children = [anchor_semantics, _neutral_semantics_for_op(int(op_id), anchor_semantics, right=True)]
        values.append(_eval_op_semantics(int(op_id), children, fallback=fallback))
    return torch.stack(values, dim=1)


def _source_output_semantics(
    op_id: int,
    slot: int,
    source_semantics: torch.Tensor,
    fixed_child_semantics: list[torch.Tensor],
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    op = get_op(int(op_id))
    if source_semantics.ndim != 2:
        raise ValueError("source_semantics must be [n, candidates]")
    values = []
    for idx in range(int(source_semantics.shape[1])):
        candidate = source_semantics[:, idx]
        if int(op.arity) <= 1:
            children = [candidate]
        elif int(slot) == 0:
            children = [candidate, _neutral_semantics_for_op(int(op_id), candidate, right=True)]
        else:
            left = fixed_child_semantics[0] if fixed_child_semantics else _neutral_semantics_for_op(
                int(op_id), candidate, right=False
            )
            children = [left, candidate]
        values.append(_eval_op_semantics(int(op_id), children, fallback=fallback))
    return torch.stack(values, dim=1)


def _neutral_semantics_for_op(op_id: int, reference: torch.Tensor, *, right: bool) -> torch.Tensor:
    name = get_op(int(op_id)).name
    if name in {"mul", "protected_div"}:
        return torch.ones_like(reference)
    return torch.zeros_like(reference)


def _sample_index(probs: torch.Tensor, rng: random.Random) -> int:
    values = probs.detach().cpu().tolist()
    r = rng.random()
    total = 0.0
    for idx, value in enumerate(values):
        total += max(float(value), 0.0)
        if r <= total:
            return idx
    return max(len(values) - 1, 0)


def _sample_flow_time(time_sampling: str | float | None, rng: random.Random) -> float | None:
    if time_sampling is None:
        return None
    if isinstance(time_sampling, (float, int)):
        return float(max(0.0, min(1.0, float(time_sampling))))
    mode = str(time_sampling).strip().lower()
    if mode in {"", "none", "endpoint", "final", "one", "1", "deterministic"}:
        return None
    if mode in {"uniform", "random", "u01"}:
        return float(max(0.0, min(1.0, rng.random())))
    try:
        return float(max(0.0, min(1.0, float(mode))))
    except ValueError as exc:
        raise ValueError(f"unknown teacher_time_sampling mode: {time_sampling}") from exc


def _expr_keys(exprs: list[Expr]) -> tuple[str, ...]:
    return tuple(str(expr) for expr in exprs)


def _sqrt_step(p: torch.Tensor, w: torch.Tensor, *, dt: float, min_prob: float) -> torch.Tensor:
    z = p.clamp_min(min_prob).sqrt()
    zdot = 0.5 * z * w
    z = (z + float(dt) * zdot).clamp_min(math.sqrt(min_prob))
    z = z / z.norm().clamp_min(min_prob)
    out = z * z
    return out / out.sum().clamp_min(min_prob)


def _simplex_probs(logits: torch.Tensor, min_prob: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=0)
    probs = probs.clamp_min(float(min_prob))
    return probs / probs.sum().clamp_min(float(min_prob))


def _candidate_mask(mask: torch.Tensor | None, *, d: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(int(d), dtype=torch.bool, device=device)
    out = mask.to(device=device, dtype=torch.bool).flatten()
    if int(out.numel()) != int(d):
        raise ValueError(f"candidate mask has {int(out.numel())} entries for {d} candidates")
    if not bool(out.any()):
        out = torch.ones(int(d), dtype=torch.bool, device=device)
    return out


def _pad_candidate_mask(
    mask: torch.Tensor | None,
    *,
    d: int,
    target_d: int,
    device: torch.device,
) -> torch.Tensor:
    if int(d) > int(target_d):
        raise ValueError(f"SPFF runtime candidates {int(d)} exceed chart width {int(target_d)}")
    out = torch.zeros(int(target_d), dtype=torch.bool, device=device)
    out[:int(d)] = _candidate_mask(mask, d=int(d), device=device)
    return out


def _pad_candidate_semantics(
    candidate_semantics: torch.Tensor,
    *,
    d: int,
    target_d: int,
    device: torch.device,
) -> torch.Tensor:
    if int(d) > int(target_d):
        raise ValueError(f"SPFF runtime candidates {int(d)} exceed chart width {int(target_d)}")
    sem = torch.as_tensor(candidate_semantics, device=device).float()
    if sem.ndim == 1:
        sem = sem.unsqueeze(1)
    if int(sem.shape[1]) != int(d):
        raise ValueError(f"candidate semantics has {int(sem.shape[1])} columns for {int(d)} source candidates")
    if int(d) == int(target_d):
        return sem
    out = torch.zeros((int(sem.shape[0]), int(target_d)), device=device, dtype=sem.dtype)
    out[:, :int(d)] = sem
    return out


def _pad_candidate_distribution(
    probs: torch.Tensor | None,
    *,
    mask: torch.Tensor | None,
    d: int,
    target_d: int,
    device: torch.device,
    dtype: torch.dtype,
    min_prob: float,
) -> torch.Tensor:
    active = _pad_candidate_mask(mask, d=int(d), target_d=int(target_d), device=device)
    if probs is None:
        out = active.to(dtype=dtype)
        return out / out.sum().clamp_min(float(min_prob))
    raw = torch.as_tensor(probs, device=device, dtype=dtype).flatten()
    if int(raw.numel()) != int(d):
        raise ValueError(f"source_p0 has {int(raw.numel())} entries for {int(d)} source candidates")
    out = torch.zeros(int(target_d), device=device, dtype=dtype)
    out[:int(d)] = raw.clamp_min(0.0)
    out = torch.where(active, out, torch.zeros_like(out))
    if not bool(out.sum() > 0):
        out = active.to(dtype=dtype)
    return out / out.sum().clamp_min(float(min_prob))


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum()


def _active_variable_count(
    x: torch.Tensor,
    *,
    template_num_vars: int,
    explicit: int | None = None,
) -> int:
    if explicit is not None:
        return max(0, min(int(explicit), int(template_num_vars), int(x.shape[1])))
    used = min(int(template_num_vars), int(x.shape[1]))
    if used <= 0:
        return 0
    active = 0
    for idx in range(used):
        col = torch.nan_to_num(x[:, idx].float())
        if bool((col.abs().max() > 1e-12) or (col.std(unbiased=False) > 1e-12)):
            active = idx + 1
    return active


def _sample_key(sample: CircuitSample) -> str:
    return str(sample.expression)


def _select_diverse_indices(samples: list[CircuitSample], ordered_indices: list[int], *, k: int) -> list[int]:
    selected: list[int] = []
    seen: set[str] = set()
    for idx in ordered_indices:
        key = _sample_key(samples[int(idx)])
        if key in seen:
            continue
        selected.append(int(idx))
        seen.add(key)
        if len(selected) >= int(k):
            return selected
    return selected


def _normalize_vector(v: torch.Tensor) -> torch.Tensor:
    v = torch.nan_to_num(v.float())
    return (v - v.mean()) / v.std(unbiased=False).clamp_min(1e-6)


def _pad_x(x: torch.Tensor, num_vars: int) -> torch.Tensor:
    x = x.float()
    used = min(int(num_vars), int(x.shape[1]))
    out = torch.zeros((x.shape[0], int(num_vars)), dtype=x.dtype, device=x.device)
    if used:
        out[:, :used] = x[:, :used]
    return out


def _tree_features(
    expr: Expr,
    *,
    reg_index: float,
    layer_id: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    root_op = -1.0
    kind = 0.0
    if expr.kind == "var":
        kind = 1.0
    elif expr.kind == "const":
        kind = 2.0
    elif expr.kind == "op" and expr.op_id is not None:
        kind = 3.0
        root_op = float(expr.op_id) / float(max(N_OPS - 1, 1))
    var_count = float(len(_expr_vars(expr))) / float(max(1, 8))
    return torch.tensor([
        kind / 3.0,
        root_op,
        math.log1p(float(expr.complexity)),
        float(expr.depth),
        var_count,
        1.0 if expr.kind == "const" else 0.0,
        float(reg_index),
        float(layer_id),
    ], dtype=dtype, device=device)


def _expr_vars(expr: Expr) -> set[int]:
    if expr.kind == "var" and expr.var_index is not None:
        return {int(expr.var_index)}
    out: set[int] = set()
    for child in expr.children:
        out.update(_expr_vars(child))
    return out


def _meta_columns(
    d: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    values: list[float],
    probs: torch.Tensor,
) -> torch.Tensor:
    cols = [torch.full((d, 1), float(value), dtype=dtype, device=device) for value in values]
    cols.append(probs.unsqueeze(1))
    cols.append(probs.clamp_min(1e-6).log().unsqueeze(1))
    return torch.cat(cols, dim=1)
