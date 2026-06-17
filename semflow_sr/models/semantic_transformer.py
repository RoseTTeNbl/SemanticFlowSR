"""Semantic Transformer: assembles row/register/action encoders + main log-rate head.

The main output is a support-local semantic-Fisher log-rate. The potential head remains
available only for the plain Fisher closed-form ablation.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from .row_encoder import RowEncoder
from .register_encoder import RegisterEncoder
from .action_encoder import ActionEncoder
from .velocity_model import VelocityHead, VelocityOutput
from ..semantics.projection import ProjectionBackend


@dataclass
class SemanticTransformerConfig:
    d: int = 1
    K: int = 8
    hidden: int = 128
    row_layers: int = 2
    heads: int = 4
    output_mode: str = "semantic_fisher_lograte"


class SemanticTransformer(nn.Module):
    def __init__(self, cfg: SemanticTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.row = RowEncoder(cfg.d, cfg.K, cfg.hidden, cfg.row_layers, cfg.heads)
        self.reg = RegisterEncoder(cfg.hidden, cfg.heads)
        self.act = ActionEncoder(cfg.hidden, cfg.heads)
        self.lambda_proj = nn.Linear(1, cfg.hidden)
        self.head = VelocityHead(cfg.hidden)

    def forward(self, x, y, B, p_lambda, lambda_value, action_feats, energies, weights,
                semantic_stats=None, gram=None,
                beta_value: float | torch.Tensor = 1.0,
                action_mask=None, active=None) -> VelocityOutput:
        x, y, B = _squash(x), _squash(y), _squash(B)
        residual = _squash(y - self._project_y(B, y))
        row_tokens, ctx = self.row(x, y, B, residual)
        reg_tokens = self.reg(B, y, residual, active, row_tokens)
        if self.cfg.output_mode in {"potential", "semantic_fisher_lograte"}:
            lam = torch.zeros_like(lambda_value).reshape(-1, 1).to(B.dtype)
        else:
            raise ValueError(f"unknown output_mode: {self.cfg.output_mode}")
        ctx = ctx + self.lambda_proj(lam)
        kpm = (~action_mask) if action_mask is not None else None
        act_tokens = self.act(
            action_feats, _squash(energies), _squash(weights), p_lambda, reg_tokens, ctx,
            semantic_stats=_squash(semantic_stats) if semantic_stats is not None else None,
            gram=gram,
            key_padding_mask=kpm,
        )
        raw_out = self.head.out(act_tokens).squeeze(-1)
        from .velocity_model import lograte_to_velocity_output, potential_to_velocity_output
        if self.cfg.output_mode == "semantic_fisher_lograte":
            return lograte_to_velocity_output(raw_out, p_lambda=p_lambda, mask=action_mask)
        if self.cfg.output_mode != "potential":
            raise ValueError(f"unknown output_mode: {self.cfg.output_mode}")
        return potential_to_velocity_output(raw_out, p_lambda=p_lambda, beta=beta_value, mask=action_mask)

    @staticmethod
    def _project_y(B, y, rho: float = 1e-3):
        # Keep model residual features on the same centered projection backend used
        # by rewards, rollout fitness and evaluation energy.
        return ProjectionBackend("ridge", rho).project_y(B, y)


def _squash(t):
    """Finite + sign-preserving log compression so huge (e.g. protected_div) values can't
    explode the encoders, while keeping order/sign/scale structure intact."""
    t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.sign(t) * torch.log1p(t.abs())
