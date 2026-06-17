import torch

from semflow_sr.actions.support_sampler import SupportSampler


def test_mixed_topk_random_sampler_is_deterministic_and_keeps_gt_and_best():
    action_ids = torch.arange(10)
    rewards = torch.tensor([0.0, 1.0, 9.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0, 5.0])
    sampler = SupportSampler(mode="mixed_topk_random", max_support=5, topk=2, seed=123)

    s1 = sampler.sample(action_ids, rewards=rewards, gt_action_id=1, sample_index=7)
    s2 = sampler.sample(action_ids, rewards=rewards, gt_action_id=1, sample_index=7)

    assert torch.equal(s1.action_ids, s2.action_ids)
    assert 2 in s1.action_ids.tolist()  # best reward
    assert 4 in s1.action_ids.tolist()  # second-best reward
    assert 1 in s1.action_ids.tolist()  # GT forced into support
    assert s1.action_ids.numel() == 5
    assert torch.all(s1.proposal_probs > 0)


def test_full_sampler_returns_all_actions_with_unit_proposal():
    action_ids = torch.arange(6)
    sampler = SupportSampler(mode="full", max_support=3)
    sample = sampler.sample(action_ids, rewards=torch.arange(6, dtype=torch.float32))

    assert torch.equal(sample.action_ids, action_ids)
    assert torch.equal(sample.proposal_probs, torch.ones(6))
    assert sample.full_size == 6
