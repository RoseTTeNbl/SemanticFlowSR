"""Dataset construction for block-only Semantic-Fisher RiskFlow."""
from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch.utils.data import Dataset

from ..actions.action_space import ActionSpace
from ..blocks.credit import build_table_advantages_from_trajectories
from ..blocks.evaluator import BlockTrajectoryEvaluator
from ..blocks.policy_sampler import ModelBlockTrajectorySampler
from ..blocks.semantic_effects import compute_table_semantic_effects
from ..data.synthetic_generator import GenConfig, generate_trace_task
from ..flow.semantic_fisher_table import semantic_fisher_table_lograte, semantic_fisher_table_sphere_step
from ..registers.executor import evaluate_register_state
from ..registers.state import init_register_state
from ..semantics.energy import ActionEnergyConfig
from ..sr.ops import NAME_TO_ID
from ..trajectories.risk_advantage import build_group_advantages


@dataclass
class BlockFlowBuildConfig:
    block_size: int = 3
    num_trajectories: int = 16
    max_blocks: int = 2
    block_pool_budget: int = 64
    risk_alpha: float = 0.1
    risk_mode: str = "top_alpha"
    risk_normalize: str = "rank"
    beta: float = 1.0
    gamma: float = 0.1
    gram_rank: int | None = 8
    num_time_samples: int = 1
    behavior_policy_id: str | None = None


class BlockFlowDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def build_block_flow_dataset(
    gen: GenConfig,
    *,
    num_tasks: int,
    behavior_model,
    seed: int,
    energy_cfg: ActionEnergyConfig | None = None,
    cfg: BlockFlowBuildConfig | None = None,
) -> BlockFlowDataset:
    cfg = cfg or BlockFlowBuildConfig()
    energy_cfg = energy_cfg or ActionEnergyConfig()
    rng = random.Random(seed)
    allowed = [NAME_TO_ID[o] for o in gen.ops]
    space = ActionSpace(gen.K, allowed)
    sampler = ModelBlockTrajectorySampler(
        space,
        block_size=cfg.block_size,
        block_pool_budget=cfg.block_pool_budget,
        behavior_policy_id=cfg.behavior_policy_id or f"block_flow_seed_{seed}",
        seed=seed,
    )
    evaluator = BlockTrajectoryEvaluator(space, energy_cfg)
    records: list[dict] = []
    task_idx = 0
    tries = 0
    while task_idx < int(num_tasks) and tries < int(num_tasks) * 20:
        tries += 1
        task = generate_trace_task(gen, rng)
        if task is None:
            continue
        _, _, x, y = task
        initial_state = init_register_state(gen.num_vars, gen.K, device=x.device)
        trajectories = sampler.sample(
            task_id=f"synthetic:{task_idx}",
            initial_state=initial_state,
            x=x,
            y=y,
            model=behavior_model,
            num_trajectories=cfg.num_trajectories,
            max_blocks=cfg.max_blocks,
        )
        if not trajectories:
            continue
        eval_out = evaluator.evaluate(trajectories, x, y, initial_state)
        risk = build_group_advantages(
            eval_out.rewards,
            mode=cfg.risk_mode,
            alpha=cfg.risk_alpha,
            normalize=cfg.risk_normalize,
        )
        groups = build_table_advantages_from_trajectories(
            trajectories,
            risk.trajectory_advantages,
            block_size=cfg.block_size,
            action_vocab_size=space.size,
        )
        for group in groups.values():
            if not group.candidate_blocks:
                continue
            state = group.state
            B = torch.nan_to_num(evaluate_register_state(state, x))
            sem = compute_table_semantic_effects(
                state,
                B,
                y,
                space,
                group.candidate_blocks,
                block_size=cfg.block_size,
                energy_cfg=energy_cfg,
            )
            mask = sem.mask & group.mask.to(device=sem.mask.device)
            q_start = _row_normalize(group.q_start.to(device=B.device, dtype=B.dtype), mask)
            advantages = group.advantages.to(device=B.device, dtype=B.dtype).masked_fill(~mask, 0.0)
            for time_idx in range(max(int(cfg.num_time_samples), 1)):
                lam = 0.0 if cfg.num_time_samples <= 1 else (time_idx + 1) / (cfg.num_time_samples + 1)
                w0 = semantic_fisher_table_lograte(
                    q_start,
                    advantages,
                    sem.zeta,
                    mask,
                    beta=cfg.beta,
                    gamma=cfg.gamma,
                    gram_rank=cfg.gram_rank,
                )
                q_lambda = semantic_fisher_table_sphere_step(q_start, w0, mask, dt=lam) if lam else q_start
                w_target = semantic_fisher_table_lograte(
                    q_lambda,
                    advantages,
                    sem.zeta,
                    mask,
                    beta=cfg.beta,
                    gamma=cfg.gamma,
                    gram_rank=cfg.gram_rank,
                )
                z_dot = 0.5 * q_lambda.clamp(min=1e-12).sqrt() * w_target
                records.append({
                    "x": x.float(),
                    "y": y.float(),
                    "B": B.float(),
                    "q_start": q_start.float(),
                    "q_lambda": q_lambda.float(),
                    "lambda": torch.tensor(float(lam), dtype=torch.float32),
                    "mask": mask.cpu(),
                    "advantages": advantages.float().cpu(),
                    "zeta": sem.zeta.float().cpu(),
                    "w_target": w_target.float().cpu(),
                    "zdot_target": z_dot.float().cpu(),
                    "num_trajectories": torch.tensor(len(trajectories), dtype=torch.float32),
                    "num_executable_blocks": torch.tensor(len(group.candidate_blocks), dtype=torch.float32),
                })
        task_idx += 1
    if not records:
        raise RuntimeError("failed to build any block-flow training records")
    return BlockFlowDataset(records)


def collate_block_flow(batch: list[dict]) -> dict:
    out = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        out[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return out


def _row_normalize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = torch.where(mask, values.clamp(min=0.0), torch.zeros_like(values))
    row_sums = values.sum(dim=1, keepdim=True)
    uniform = mask.float() / mask.sum(dim=1, keepdim=True).clamp(min=1).float()
    return torch.where(row_sums > 1e-12, values / row_sums.clamp(min=1e-12), uniform)
