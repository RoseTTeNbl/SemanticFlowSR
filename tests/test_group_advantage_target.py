import torch

from semflow_sr.endpoints.target_group_advantage import GroupAdvantageTarget


def test_group_advantage_target_matches_kl_policy_improvement():
    p0 = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    rewards = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64)
    target = GroupAdvantageTarget(eta_adv=0.7, normalize=False, floor=0.0)

    p1 = target.build_p1(None, None, torch.arange(3), rewards, p0, {"rewards": rewards})

    expected = p0 * torch.exp(0.7 * rewards)
    expected = expected / expected.sum()
    assert torch.allclose(p1, expected, atol=1e-12)


def test_group_baseline_does_not_change_endpoint_without_std_normalization():
    p0 = torch.full((4,), 0.25, dtype=torch.float64)
    rewards = torch.tensor([-1.0, 2.0, 0.5, 3.0], dtype=torch.float64)
    target = GroupAdvantageTarget(eta_adv=1.3, normalize=False, center=True, floor=0.0)

    centered = target.build_p1(None, None, torch.arange(4), rewards, p0, {"rewards": rewards})
    raw = p0 * torch.exp(1.3 * rewards)
    raw = raw / raw.sum()
    assert torch.allclose(centered, raw, atol=1e-12)


def test_group_advantage_supports_importance_correction_and_floor():
    p0 = torch.full((3,), 1 / 3, dtype=torch.float64)
    rewards = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    proposal = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    target = GroupAdvantageTarget(eta_adv=1.0, normalize=False, floor=0.1)

    p1 = target.build_p1(None, None, torch.arange(3), rewards, p0,
                         {"rewards": rewards, "proposal_probs": proposal})

    expected = p0 * torch.exp(rewards) / proposal
    expected = expected / expected.sum()
    expected = 0.9 * expected + 0.1 * p0
    expected = expected / expected.sum()
    assert torch.allclose(p1, expected, atol=1e-12)

