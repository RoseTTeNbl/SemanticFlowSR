"""Latent complete-trace endpoints with closed-form Fisher transport."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import nn


LATENT_ENDPOINT_OBJECTIVE_VERSION = "semantic_latent_endpoint_family_fm_v1"
DEFAULT_TANGENT_BINS = ((0.0, 0.001), (0.001, 0.005), (0.005, 0.02), (0.02, 0.1), (0.1, 1.0))


@dataclass
class LatentEndpointOutput:
    mixture_logits: torch.Tensor
    endpoint_logits: torch.Tensor

    @property
    def mixture_weights(self) -> torch.Tensor:
        return torch.softmax(self.mixture_logits, dim=-1)


class LatentEndpointSetModel(nn.Module):
    """Permutation-invariant task encoder with learned component queries."""

    def __init__(
        self,
        *,
        num_vars: int,
        block_count: int,
        source_count: int,
        hidden: int,
        components: int = 4,
        set_layers: int = 2,
        attention_heads: int = 4,
        action_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if hidden % attention_heads != 0:
            raise ValueError("hidden must be divisible by attention_heads")
        self.num_vars = int(num_vars)
        self.block_count = int(block_count)
        self.source_count = int(source_count)
        self.hidden = int(hidden)
        self.components = int(components)
        self.set_layers = int(set_layers)
        self.attention_heads = int(attention_heads)
        self.point_projection = nn.Sequential(
            nn.Linear(self.num_vars + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=attention_heads,
            dim_feedforward=4 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=set_layers)
        self.component_queries = nn.Parameter(torch.randn(components, hidden) / hidden**0.5)
        self.cross_attention = nn.MultiheadAttention(hidden, attention_heads, dropout=0.0, batch_first=True)
        self.query_norm = nn.LayerNorm(hidden)
        self.mixture_head = nn.Linear(hidden, 1)
        self.endpoint_head = nn.Linear(hidden, block_count * source_count)
        if action_mask is None:
            action_mask = torch.ones(block_count, source_count, dtype=torch.bool)
        self.register_buffer("action_mask", action_mask.bool(), persistent=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> LatentEndpointOutput:
        x = x.float()
        y = y.float()
        y_std = y.std(unbiased=False).clamp_min(1.0e-6)
        y_norm = (y - y.mean()) / y_std
        tokens = self.point_projection(torch.cat([x, y_norm[:, None]], dim=-1))[None, :, :]
        encoded = self.set_encoder(tokens)
        queries = self.component_queries[None, :, :]
        attended, _ = self.cross_attention(queries, encoded, encoded, need_weights=False)
        states = self.query_norm(queries + attended).squeeze(0)
        mixture_logits = self.mixture_head(states).squeeze(-1)
        endpoint_logits = self.endpoint_head(states).view(self.components, self.block_count, self.source_count)
        endpoint_logits = endpoint_logits.masked_fill(~self.action_mask[None, :, :], -1.0e9)
        return LatentEndpointOutput(mixture_logits=mixture_logits, endpoint_logits=endpoint_logits)


def sharp_endpoint_probabilities(
    endpoint_logits: torch.Tensor,
    p0: torch.Tensor,
    active_indices_fn: Callable[[list[int]], Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Make active blocks one-hot and leave inactive blocks equal to ``p0``."""
    choices = endpoint_logits.argmax(dim=-1)
    endpoints = p0.unsqueeze(0).expand(endpoint_logits.shape[0], -1, -1).clone()
    active_masks = torch.zeros(endpoint_logits.shape[:2], dtype=torch.bool, device=endpoint_logits.device)
    active_sets: list[list[int]] = []
    for component in range(endpoint_logits.shape[0]):
        active = sorted({int(v) for v in active_indices_fn(choices[component].tolist())})
        active_sets.append(active)
        if active:
            rows = torch.tensor(active, dtype=torch.long, device=endpoint_logits.device)
            actions = choices[component, rows]
            endpoints[component, rows] = 0.0
            endpoints[component, rows, actions] = 1.0
            active_masks[component, rows] = True
    return endpoints, active_masks, active_sets


def trace_active_mask(trace: dict[str, Any], block_count: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(block_count, dtype=torch.bool, device=device)
    active = trace.get("active_blocks", trace.get("active_block_indices", []))
    if active:
        mask[torch.tensor(list(active), dtype=torch.long, device=device)] = True
    return mask


def trace_sharp_endpoint(trace: dict[str, Any], p0: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    endpoint = p0.clone()
    rows = active_mask.nonzero(as_tuple=False).flatten()
    if rows.numel():
        choices = torch.tensor(trace["choices"], dtype=torch.long, device=p0.device)
        endpoint[rows] = 0.0
        endpoint[rows, choices[rows]] = 1.0
    return endpoint


def fisher_sphere_transport(
    p0: torch.Tensor,
    endpoint: torch.Tensor,
    t: float | torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form spherical interpolation and analytic probability tangent."""
    tt = torch.as_tensor(t, dtype=p0.dtype, device=p0.device)
    r0 = p0.clamp_min(0.0).sqrt()
    r1 = endpoint.clamp_min(0.0).sqrt()
    dot = (r0 * r1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    regular = sin_omega.abs() > 1.0e-7
    safe_sin = torch.where(regular, sin_omega, torch.ones_like(sin_omega))
    a = torch.sin((1.0 - tt) * omega) / safe_sin
    b = torch.sin(tt * omega) / safe_sin
    da = -omega * torch.cos((1.0 - tt) * omega) / safe_sin
    db = omega * torch.cos(tt * omega) / safe_sin
    r = torch.where(regular, a * r0 + b * r1, (1.0 - tt) * r0 + tt * r1)
    dr = torch.where(regular, da * r0 + db * r1, r1 - r0)
    probability = r.square()
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(eps)
    tangent = 2.0 * r * dr
    tangent = tangent - probability * tangent.sum(dim=-1, keepdim=True)
    if active_mask is not None:
        inactive = ~active_mask.bool()
        probability = torch.where(inactive[:, None], p0, probability)
        tangent = torch.where(inactive[:, None], torch.zeros_like(tangent), tangent)
    if float(tt.detach().cpu()) >= 1.0:
        probability = torch.where(active_mask[:, None], endpoint, p0) if active_mask is not None else endpoint
    return probability, tangent


def fisher_tangent_error(
    predicted_endpoint: torch.Tensor,
    target_endpoint: torch.Tensor,
    p0: torch.Tensor,
    active_mask: torch.Tensor,
    times: Sequence[float],
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    losses = []
    for time in times:
        probability, pred_tangent = fisher_sphere_transport(p0, predicted_endpoint, time, active_mask=active_mask, eps=eps)
        _, target_tangent = fisher_sphere_transport(p0, target_endpoint, time, active_mask=active_mask, eps=eps)
        block_error = ((pred_tangent - target_tangent).square() / probability.clamp_min(eps)).sum(dim=-1)
        losses.append(block_error[active_mask].mean() if bool(active_mask.any()) else block_error.sum() * 0.0)
    return torch.stack(losses).mean()


def stratified_tangent_times(generator: torch.Generator | None = None) -> list[float]:
    values = []
    for low, high in DEFAULT_TANGENT_BINS:
        sample = torch.rand((), generator=generator).item()
        values.append(float(low + (high - low) * sample))
    return values


def minimum_cost_assignment(cost: torch.Tensor) -> list[tuple[int, int]]:
    """Exact rectangular assignment for K<=4 without a SciPy dependency."""
    components, traces = cost.shape
    if traces == 0:
        return []
    if traces > components:
        raise ValueError("trace family cannot exceed latent component count")
    best_pairs: list[tuple[int, int]] | None = None
    best_value: float | None = None
    for selected in itertools.permutations(range(components), traces):
        value = sum(float(cost[selected[trace], trace].detach().cpu()) for trace in range(traces))
        if best_value is None or value < best_value:
            best_value = value
            best_pairs = [(selected[trace], trace) for trace in range(traces)]
    return best_pairs or []


def family_matching_loss(
    output: LatentEndpointOutput,
    traces: Sequence[dict[str, Any]],
    p0_samples: Sequence[torch.Tensor],
    *,
    tangent_weight: float,
    tangent_times: Sequence[float],
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    components, block_count, _ = output.endpoint_logits.shape
    if not traces:
        zero = output.endpoint_logits.sum() * 0.0
        return zero, {"assignment": [], "endpoint_loss": 0.0, "tangent_loss": 0.0, "mixture_loss": 0.0}
    log_probs = torch.log_softmax(output.endpoint_logits, dim=-1)
    cost_rows = []
    endpoint_terms: list[list[torch.Tensor]] = []
    tangent_terms: list[list[torch.Tensor]] = []
    for component in range(components):
        component_endpoint = torch.softmax(output.endpoint_logits[component], dim=-1)
        component_endpoint_costs = []
        component_tangent_costs = []
        for trace in traces:
            active_mask = trace_active_mask(trace, block_count, output.endpoint_logits.device)
            choices = torch.tensor(trace["choices"], dtype=torch.long, device=output.endpoint_logits.device)
            rows = active_mask.nonzero(as_tuple=False).flatten()
            endpoint_loss = -log_probs[component, rows, choices[rows]].mean() if rows.numel() else log_probs.sum() * 0.0
            tangent_losses = []
            for p0 in p0_samples:
                target_endpoint = trace_sharp_endpoint(trace, p0, active_mask)
                predicted_endpoint = torch.where(active_mask[:, None], component_endpoint, p0)
                tangent_losses.append(
                    fisher_tangent_error(predicted_endpoint, target_endpoint, p0, active_mask, tangent_times, eps=eps)
                )
            tangent_loss = torch.stack(tangent_losses).mean()
            component_endpoint_costs.append(endpoint_loss)
            component_tangent_costs.append(tangent_loss)
        endpoint_terms.append(component_endpoint_costs)
        tangent_terms.append(component_tangent_costs)
        cost_rows.append(torch.stack([e + float(tangent_weight) * t for e, t in zip(component_endpoint_costs, component_tangent_costs)]))
    cost = torch.stack(cost_rows)
    assignment = minimum_cost_assignment(cost.detach())
    endpoint_loss = torch.stack([endpoint_terms[c][z] for c, z in assignment]).mean()
    tangent_loss = torch.stack([tangent_terms[c][z] for c, z in assignment]).mean()
    mixture_target = torch.zeros(components, dtype=output.mixture_logits.dtype, device=output.mixture_logits.device)
    mixture_target[[c for c, _ in assignment]] = 1.0 / len(assignment)
    mixture_loss = -(mixture_target * torch.log_softmax(output.mixture_logits, dim=-1)).sum()
    total = endpoint_loss + float(tangent_weight) * tangent_loss + mixture_loss
    return total, {
        "assignment": assignment,
        "cost_matrix": cost.detach(),
        "endpoint_loss": float(endpoint_loss.detach().cpu()),
        "tangent_loss": float(tangent_loss.detach().cpu()),
        "mixture_loss": float(mixture_loss.detach().cpu()),
        "endpoint_loss_tensor": endpoint_loss,
        "tangent_loss_tensor": tangent_loss,
        "mixture_loss_tensor": mixture_loss,
    }


def sample_component(weights: torch.Tensor, generator: torch.Generator | None = None) -> int:
    return int(torch.multinomial(weights, 1, generator=generator).item())
