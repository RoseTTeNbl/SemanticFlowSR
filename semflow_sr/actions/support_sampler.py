"""Candidate support sampling for action-simplex CFM.

The model is conditioned on the sampled support S through action features and masks, so
subsampling is treated as candidate-set conditional flow matching.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class SupportSample:
    action_ids: torch.Tensor
    proposal_probs: torch.Tensor
    full_size: int
    mode: str


@dataclass
class SupportSampler:
    mode: str = "mixed_topk_random"
    max_support: int | None = 256
    topk: int | None = None
    seed: int = 0

    def sample(self, action_ids: torch.Tensor, rewards: torch.Tensor | None = None,
               gt_action_id: int | None = None, sample_index: int = 0) -> SupportSample:
        n = int(action_ids.numel())
        if self.mode == "full" or self.max_support is None or n <= self.max_support:
            return SupportSample(action_ids=action_ids, proposal_probs=torch.ones_like(action_ids, dtype=torch.float32),
                                 full_size=n, mode="full")
        if self.mode not in {"topk_reward", "mixed_topk_random", "proposal_importance"}:
            raise ValueError(f"unknown support sampler mode: {self.mode}")
        if rewards is None:
            rewards = torch.zeros(n, device=action_ids.device)

        m = min(int(self.max_support), n)
        topk = self.topk if self.topk is not None else max(1, m // 2)
        topk = min(int(topk), m, n)
        selected: list[int] = []
        proposal = torch.zeros(n, dtype=torch.float32, device=action_ids.device)

        if self.mode in {"topk_reward", "mixed_topk_random"}:
            top_idx = torch.topk(rewards, topk).indices.tolist()
            selected.extend(top_idx)
            proposal[top_idx] = 1.0

        if gt_action_id is not None:
            pos = (action_ids == int(gt_action_id)).nonzero(as_tuple=False)
            if pos.numel():
                selected.append(int(pos[0].item()))

        selected = _unique(selected)
        remaining = [i for i in range(n) if i not in set(selected)]
        slots = max(0, m - len(selected))
        if slots and remaining and self.mode != "topk_reward":
            gen = torch.Generator(device=action_ids.device)
            gen.manual_seed(int(self.seed) + int(sample_index) * 1_000_003)
            rem_tensor = torch.tensor(remaining, dtype=torch.long, device=action_ids.device)
            if self.mode == "proposal_importance":
                weights = torch.softmax(rewards[rem_tensor], dim=0)
                pick_pos = torch.multinomial(weights, min(slots, rem_tensor.numel()), replacement=False, generator=gen)
                picked = rem_tensor[pick_pos].tolist()
                proposal[rem_tensor] = (weights * min(slots, rem_tensor.numel())).clamp(max=1.0).float()
            else:
                perm = torch.randperm(rem_tensor.numel(), generator=gen, device=action_ids.device)
                picked = rem_tensor[perm[:slots]].tolist()
                prob = min(slots, len(remaining)) / max(len(remaining), 1)
                proposal[remaining] = float(prob)
            selected.extend(int(i) for i in picked)

        selected = _unique(selected)[:m]
        idx = torch.tensor(selected, dtype=torch.long, device=action_ids.device)
        probs = proposal[idx].clamp(min=1.0 / max(n, 1))
        return SupportSample(action_ids=action_ids[idx], proposal_probs=probs,
                             full_size=n, mode=self.mode)


def _unique(xs: list[int]) -> list[int]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
