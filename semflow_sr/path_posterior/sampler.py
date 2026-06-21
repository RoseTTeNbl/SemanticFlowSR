"""Complete action-trajectory sampling for path-posterior targets."""
from __future__ import annotations

import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..flow.semantic_fisher import semantic_fisher_sphere_step
from ..models.semantic_transformer import SemanticTransformer
from ..registers.executor import evaluate_register_state
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from .action_support import (
    append_stop_action,
    action_features_with_stop,
    action_semantic_effects_with_stop,
    healthy_action_ids,
    is_stop_action,
)
from .target import PathDecision, PathTrajectory


class ActionPathSampler:
    """Sample complete trajectories from the recorded local policy p0(a|s)."""

    def __init__(
        self,
        action_space: ActionSpace,
        *,
        energy_cfg: ActionEnergyConfig | None = None,
        behavior_policy_id: str = "path_posterior",
        seed: int = 0,
        enable_stop: bool = True,
        max_abs_semantic: float | None = 1e6,
        max_energy_growth: float | None = 100.0,
        max_support_size: int | None = None,
        support_mode: str = "deterministic_cap",
        support_topk: int | None = None,
        support_full_threshold: int | None = None,
    ):
        self.space = action_space
        self.executor = ActionExecutor(action_space)
        self.energy = ActionEnergy(action_space, energy_cfg)
        self.behavior_policy_id = behavior_policy_id
        self.generator = torch.Generator().manual_seed(int(seed))
        self.enable_stop = bool(enable_stop)
        self.max_abs_semantic = max_abs_semantic
        self.max_energy_growth = max_energy_growth
        self.max_support_size = None if max_support_size is None else int(max_support_size)
        self.support_mode = str(support_mode).strip().lower()
        self.support_topk = None if support_topk is None else int(support_topk)
        self.support_full_threshold = None if support_full_threshold is None else int(support_full_threshold)

    def sample(
        self,
        *,
        task_id: str,
        initial_state: RegisterState,
        x: torch.Tensor,
        y: torch.Tensor,
        model: SemanticTransformer | None,
        num_trajectories: int,
        max_steps: int,
    ) -> list[PathTrajectory]:
        out: list[PathTrajectory] = []
        for traj_idx in range(max(int(num_trajectories), 0)):
            state = initial_state.clone()
            decisions: list[PathDecision] = []
            actions: list[int] = []
            for _ in range(max(int(max_steps), 0)):
                raw_action_ids = self.space.valid_actions(state).to(device=x.device)
                B = torch.nan_to_num(evaluate_register_state(state, x))
                raw_action_ids = self._build_real_support(raw_action_ids, B, y, sample_index=len(decisions))
                action_ids = healthy_action_ids(
                    self.energy,
                    B,
                    y,
                    raw_action_ids,
                    max_abs_semantic=self.max_abs_semantic,
                    max_energy_growth=self.max_energy_growth,
                )
                action_ids = append_stop_action(action_ids, enabled=self.enable_stop)
                if action_ids.numel() == 0:
                    break
                p0 = self._local_policy(model, x, y, B, state, action_ids)
                idx = int(torch.multinomial(p0, 1, generator=self.generator).item())
                action_id = int(action_ids[idx].item())
                decisions.append(PathDecision(
                    state_id=_state_id(state),
                    action_id=action_id,
                    action_ids=action_ids.detach().cpu(),
                    p0=p0.detach().cpu(),
                    state=state.clone(),
                ))
                if is_stop_action(action_id):
                    break
                actions.append(action_id)
                state = self.executor.execute_symbolic(state, action_id)
            if decisions:
                out.append(PathTrajectory(
                    task_id=task_id,
                    decisions=decisions,
                    actions=actions,
                    metadata={
                        "trajectory_id": f"{task_id}:{traj_idx}",
                        "final_state": state,
                        "behavior_policy_id": self.behavior_policy_id,
                    },
                ))
        return out

    def _local_policy(
        self,
        model: SemanticTransformer | None,
        x: torch.Tensor,
        y: torch.Tensor,
        B: torch.Tensor,
        state: RegisterState,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        p_uniform = torch.ones(action_ids.numel(), device=B.device, dtype=B.dtype)
        p_uniform = p_uniform / p_uniform.sum().clamp(min=1e-12)
        if model is None:
            return p_uniform
        effect = action_semantic_effects_with_stop(self.energy, B, y, action_ids)
        feats = action_features_with_stop(self.space, state, action_ids).to(device=B.device, dtype=B.dtype)
        zeros = torch.zeros_like(p_uniform)
        ones = torch.ones_like(p_uniform)
        mask = torch.ones(1, action_ids.numel(), device=B.device, dtype=torch.bool)
        with torch.no_grad():
            pred = model(
                x=x.unsqueeze(0),
                y=y.unsqueeze(0),
                B=B.unsqueeze(0),
                p_lambda=p_uniform.unsqueeze(0),
                lambda_value=torch.zeros(1, device=B.device, dtype=B.dtype),
                action_feats=feats.unsqueeze(0),
                energies=zeros.unsqueeze(0),
                weights=ones.unsqueeze(0),
                semantic_stats=torch.zeros(1, action_ids.numel(), 8, device=B.device, dtype=B.dtype),
                gram=effect.gram.unsqueeze(0),
                action_mask=mask,
            )
        return semantic_fisher_sphere_step(p_uniform, pred.lograte_logits.squeeze(0), dt=1.0)

    def _build_real_support(
        self,
        action_ids: torch.Tensor,
        B: torch.Tensor,
        y: torch.Tensor,
        *,
        sample_index: int,
    ) -> torch.Tensor:
        mode = self.support_mode
        if mode in {"deterministic_cap", "id_cap"}:
            return self._cap_support(action_ids)
        if mode == "adaptive_full":
            threshold = self.support_full_threshold
            if threshold is None:
                threshold = self.max_support_size
            if threshold is None or action_ids.numel() <= int(threshold):
                return action_ids
            mode = "reward_topk_random"
        if mode in {"reward_topk_random", "mixed_topk_random", "topk_reward"}:
            return self._reward_aware_support(action_ids, B, y, sample_index=sample_index, random_fill=(mode != "topk_reward"))
        raise ValueError(f"unknown support_mode: {self.support_mode}")

    def _cap_support(self, action_ids: torch.Tensor) -> torch.Tensor:
        if self.max_support_size is None:
            return action_ids
        budget = max(int(self.max_support_size), 0)
        if action_ids.numel() <= budget:
            return action_ids
        if budget == 0:
            return action_ids[:0]
        sorted_ids = action_ids.sort().values
        idx = torch.linspace(
            0,
            sorted_ids.numel() - 1,
            steps=budget,
            device=sorted_ids.device,
        ).round().long()
        return sorted_ids[idx].unique(sorted=True)

    def _reward_aware_support(
        self,
        action_ids: torch.Tensor,
        B: torch.Tensor,
        y: torch.Tensor,
        *,
        sample_index: int,
        random_fill: bool,
    ) -> torch.Tensor:
        if self.max_support_size is None or action_ids.numel() <= self.max_support_size:
            return action_ids
        budget = max(int(self.max_support_size), 0)
        if budget == 0:
            return action_ids[:0]
        rewards = self.energy.rewards(B, y, action_ids)
        topk = self.support_topk if self.support_topk is not None else max(1, budget // 2)
        topk = min(max(int(topk), 1), budget, int(action_ids.numel()))
        top_idx = torch.topk(rewards, topk).indices
        selected = [int(i) for i in top_idx.detach().cpu().tolist()]
        if random_fill and len(selected) < budget:
            selected_set = set(selected)
            remaining = [i for i in range(int(action_ids.numel())) if i not in selected_set]
            slots = min(budget - len(selected), len(remaining))
            if slots > 0:
                gen = torch.Generator(device=action_ids.device)
                gen.manual_seed(int(self.generator.initial_seed()) + int(sample_index) * 1_000_003)
                rem = torch.tensor(remaining, dtype=torch.long, device=action_ids.device)
                perm = torch.randperm(rem.numel(), generator=gen, device=action_ids.device)
                selected.extend(int(i) for i in rem[perm[:slots]].detach().cpu().tolist())
        selected = _unique(selected)[:budget]
        idx = torch.tensor(selected, dtype=torch.long, device=action_ids.device)
        return action_ids[idx]


def _state_id(state: RegisterState) -> str:
    return "|".join(str(expr) for expr in state.exprs) + ":" + "".join(
        "1" if bool(v) else "0" for v in state.active.detach().cpu()
    )


def _unique(xs: list[int]) -> list[int]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
