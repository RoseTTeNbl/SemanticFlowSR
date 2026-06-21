"""Edge probability distributions on product-of-simplexes."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .template import RegisterOperatorTemplate


@dataclass
class EdgeDistribution:
    template: RegisterOperatorTemplate
    mixture_probs: torch.Tensor
    group_probs: dict[str, torch.Tensor]

    @staticmethod
    def uniform(template: RegisterOperatorTemplate, *, dtype=torch.float32) -> "EdgeDistribution":
        mixture = torch.full((template.mixture_modes,), 1.0 / template.mixture_modes, dtype=dtype)
        groups = {
            group.group_id: torch.full(
                (template.mixture_modes, group.num_candidates),
                1.0 / group.num_candidates,
                dtype=dtype,
            )
            for group in template.groups
        }
        return EdgeDistribution(template=template, mixture_probs=mixture, group_probs=groups)

    def clone(self) -> "EdgeDistribution":
        return EdgeDistribution(
            template=self.template,
            mixture_probs=self.mixture_probs.clone(),
            group_probs={key: value.clone() for key, value in self.group_probs.items()},
        )

    def normalized(self, eps: float = 1e-12) -> "EdgeDistribution":
        mixture = _normalize(self.mixture_probs, eps)
        groups = {key: _normalize(value, eps) for key, value in self.group_probs.items()}
        return EdgeDistribution(self.template, mixture, groups)

    @property
    def sqrt_mixture(self) -> torch.Tensor:
        return self.mixture_probs.clamp_min(1e-12).sqrt()

    @property
    def sqrt_groups(self) -> dict[str, torch.Tensor]:
        return {key: value.clamp_min(1e-12).sqrt() for key, value in self.group_probs.items()}


def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = torch.nan_to_num(x.float()).clamp_min(eps)
    if x.ndim == 1:
        return x / x.sum().clamp_min(eps)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)

