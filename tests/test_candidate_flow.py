import torch

from semflow_sr.actions.action_executor import ActionExecutor
from semflow_sr.actions.action_space import ActionSpace
from semflow_sr.candidates.cache import CandidateTargetCache
from semflow_sr.candidates.config import (
    CandidateCacheConfig,
    CandidateTrajectoryConfig,
    build_candidate_sampler,
)
from semflow_sr.candidates.base import SemanticCandidate
from semflow_sr.candidates.evaluator import CandidateEvaluator
from semflow_sr.candidates.sampler import ActionCandidateSampler, BlockCandidateSampler, FullCandidateSampler
from semflow_sr.candidates.target import CandidateTargetBuilder, candidate_gp_log_prior
from semflow_sr.candidates.trajectory import CandidateTrajectoryTargetFactory
from semflow_sr.flow.semantic_fisher import semantic_fisher_lograte
from semflow_sr.registers.executor import evaluate_register_state
from semflow_sr.registers.state import init_register_state
from semflow_sr.semantics.energy import ActionEnergy, ActionEnergyConfig
from semflow_sr.sr.ops import NAME_TO_ID


def _toy_context():
    ops = [NAME_TO_ID["add"], NAME_TO_ID["mul"], NAME_TO_ID["square"]]
    space = ActionSpace(K=5, allowed_ops=ops)
    state = init_register_state(num_vars=1, K=5)
    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    y = x.squeeze(1) ** 2
    B = evaluate_register_state(state, x)
    cfg = ActionEnergyConfig(lambda_op=0.0)
    return space, state, x, y, B, cfg


def test_action_candidates_reproduce_action_energy_and_target():
    space, state, _, y, B, cfg = _toy_context()
    action_ids = space.valid_actions(state)[:8]
    candidates = ActionCandidateSampler(space).from_action_ids(action_ids)
    evaluator = CandidateEvaluator(space, cfg)

    cand_eval = evaluator.evaluate(state, B, y, candidates)
    action_effect = ActionEnergy(space, cfg).action_semantic_effects(B, y, action_ids)
    target = CandidateTargetBuilder(beta=0.7, gamma=0.2).build(candidates, cand_eval)
    direct_w = semantic_fisher_lograte(
        target.p_start,
        target.advantages,
        action_effect.gram,
        beta=0.7,
        gamma=0.2,
        gram_factors=action_effect.xi,
    )

    assert torch.allclose(cand_eval.rewards, action_effect.rewards)
    assert torch.allclose(cand_eval.xi, action_effect.xi)
    assert torch.allclose(cand_eval.gram, action_effect.gram)
    assert torch.allclose(target.w_target, direct_w, atol=1e-6)
    assert torch.allclose((target.p_start * target.w_target).sum(), torch.tensor(0.0), atol=1e-6)


def test_block_h3_candidates_execute_terminal_semantics():
    space, state, _, y, B, cfg = _toy_context()
    sampler = BlockCandidateSampler(space, horizon=3, first_topk=1, branch_topk=2)
    candidates = sampler.sample(state, B=B, y=y, budget=4)
    evaluator = CandidateEvaluator(space, cfg)
    out = evaluator.evaluate(state, B, y, candidates)

    assert candidates
    assert all(c.kind == "block" and len(c.actions or []) == 3 for c in candidates)
    first = candidates[0]
    manual = B
    for action_id in first.actions or []:
        manual = ActionExecutor(space).execute_semantic(manual, torch.tensor([action_id]))[0]

    assert torch.allclose(out.B_after[0], manual)
    assert out.residual_next.shape[0] == len(candidates)
    assert out.gram.shape == (len(candidates), len(candidates))


def test_full_candidates_use_precomputed_terminal_semantics():
    space, state, _, y, B, cfg = _toy_context()
    B_full = B.clone()
    B_full[:, 2] = y
    sampler = FullCandidateSampler(
        precomputed=[
            SemanticCandidate(
                candidate_id=99,
                kind="full",
                expr="x^2",
                log_prior=1.25,
                complexity=3.0,
                metadata={"B_after": B_full},
            )
        ]
    )

    candidates = sampler.sample(state, B=B, y=y, budget=4)
    out = CandidateEvaluator(space, cfg).evaluate(state, B, y, candidates)
    target = CandidateTargetBuilder(beta=1.0, gamma=0.1).build(candidates, out)

    assert len(candidates) == 1
    assert candidates[0].candidate_id == 0
    assert candidates[0].kind == "full"
    assert candidates[0].metadata["horizon"] == "full"
    assert torch.allclose(out.B_after[0], B_full)
    assert torch.allclose(target.p_start, torch.ones(1))


def test_trajectory_sampler_config_builds_h1_h3_full_groups():
    space, state, _, y, B, _ = _toy_context()
    B_full = B.clone()
    B_full[:, 2] = y
    cfg = CandidateTrajectoryConfig(
        block_sizes=("H1", "H3", "full"),
        budgets={"H1": 3, "H3": 2, "full": 1},
        block_first_topk=1,
        block_branch_topk=2,
        full_candidates=[
            SemanticCandidate(
                candidate_id=7,
                kind="full",
                expr="x^2",
                metadata={"B_after": B_full},
            )
        ],
        cache=CandidateCacheConfig(enabled=True, path="data/candidate_targets/test.jsonl"),
    )

    sampler = build_candidate_sampler(space, cfg)
    candidates = sampler.sample(state, B=B, y=y, budget=None)

    groups = [c.metadata.get("candidate_group") for c in candidates]
    assert groups.count("H1") == 3
    assert groups.count("H3") == 2
    assert groups.count("full") == 1
    assert all(c.candidate_id == i for i, c in enumerate(candidates))
    assert all(len(c.actions or []) == 1 for c in candidates if c.metadata.get("candidate_group") == "H1")
    assert all(len(c.actions or []) == 3 for c in candidates if c.metadata.get("candidate_group") == "H3")
    assert any(c.kind == "full" for c in candidates)
    assert cfg.cache.enabled is True


def test_candidate_target_cache_round_trips_jsonl_records(tmp_path):
    path = tmp_path / "candidate_targets.jsonl"
    cache = CandidateTargetCache(path)

    cache.append({
        "task_id": "toy",
        "block_sizes": ["H1", "H3", "full"],
        "candidate_count": 6,
    })

    assert cache.load() == [{
        "task_id": "toy",
        "block_sizes": ["H1", "H3", "full"],
        "candidate_count": 6,
    }]


def test_candidate_trajectory_factory_builds_terminal_target_and_optional_cache(tmp_path):
    space, state, _, y, B, cfg = _toy_context()
    B_full = B.clone()
    B_full[:, 2] = y
    target_cache = tmp_path / "candidate_targets.jsonl"
    candidate_cfg = CandidateTrajectoryConfig(
        block_sizes=("H1", "H3", "full"),
        budgets={"H1": 2, "H3": 2, "full": 1},
        block_first_topk=1,
        block_branch_topk=2,
        full_candidates=[
            SemanticCandidate(candidate_id=0, kind="full", metadata={"B_after": B_full})
        ],
        cache=CandidateCacheConfig(enabled=True, path=str(target_cache)),
    )
    factory = CandidateTrajectoryTargetFactory(
        space,
        energy_cfg=cfg,
        candidate_cfg=candidate_cfg,
        target_builder=CandidateTargetBuilder(beta=0.5, gamma=0.2, gram_rank=4),
    )

    target = factory.build(state, B, y)
    records = CandidateTargetCache(target_cache).load()

    groups = [c.metadata.get("candidate_group") for c in target.candidates]
    assert groups == ["H1", "H1", "H3", "H3", "full"]
    assert target.eval.B_after.shape[0] == 5
    assert target.w_target.shape == (5,)
    assert records[0]["candidate_count"] == 5
    assert records[0]["candidate_groups"] == groups
    assert records[0]["gram_rank"] == 4


def test_candidate_gp_prior_changes_start_distribution_not_scores():
    space, state, _, y, B, cfg = _toy_context()
    action_ids = space.valid_actions(state)
    square_ids = [int(a) for a in action_ids if space.decode(int(a)).op_id == NAME_TO_ID["square"]]
    other_ids = [int(a) for a in action_ids if space.decode(int(a)).op_id != NAME_TO_ID["square"]]
    action_ids = torch.tensor([*other_ids[:6], *square_ids[:6]], dtype=torch.long)
    candidates = ActionCandidateSampler(space).from_action_ids(action_ids)
    priors = torch.tensor([
        candidate_gp_log_prior(c, space, gp_operator_scores={"square": 3.0})
        for c in candidates
    ])
    for cand, prior in zip(candidates, priors):
        cand.log_prior = float(prior)

    target = CandidateTargetBuilder(beta=1.0, gamma=0.1).build(
        candidates,
        CandidateEvaluator(space, cfg).evaluate(state, B, y, candidates),
    )
    square_positions = [
        i for i, c in enumerate(candidates)
        if space.decode((c.actions or [])[0]).op_id == NAME_TO_ID["square"]
    ]
    non_square_positions = [
        i for i, c in enumerate(candidates)
        if space.decode((c.actions or [])[0]).op_id != NAME_TO_ID["square"]
    ]

    assert square_positions and non_square_positions
    assert target.p_start[square_positions].mean() > target.p_start[non_square_positions].mean()
    assert torch.allclose(target.scores, target.rewards)
