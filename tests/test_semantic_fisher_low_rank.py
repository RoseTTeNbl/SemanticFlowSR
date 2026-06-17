import torch

from semflow_sr.flow.semantic_fisher import semantic_fisher_lograte


def test_low_rank_semantic_fisher_solver_matches_full_solver_for_exact_low_rank_gram():
    p = torch.tensor([0.15, 0.20, 0.25, 0.40])
    advantage = torch.tensor([1.2, -0.4, 0.6, -1.1])
    factors = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.2],
            [-0.3, 0.7],
            [0.1, -0.4],
        ]
    )
    gram = factors @ factors.T

    full = semantic_fisher_lograte(p, advantage, gram, beta=0.8, gamma=0.5)
    low_rank = semantic_fisher_lograte(p, advantage, gram, beta=0.8, gamma=0.5, gram_rank=2)

    assert torch.allclose(low_rank, full, atol=1e-5)
    assert torch.allclose((p * low_rank).sum(), torch.tensor(0.0), atol=1e-6)


def test_low_rank_solver_can_use_semantic_effect_factors_directly():
    p = torch.tensor([0.2, 0.3, 0.5])
    advantage = torch.tensor([0.7, -0.2, 0.1])
    xi = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.2],
            [-0.3, 0.4],
        ]
    )
    gram = xi @ xi.T

    via_gram = semantic_fisher_lograte(p, advantage, gram, beta=1.0, gamma=0.2)
    via_factors = semantic_fisher_lograte(
        p, advantage, gram, beta=1.0, gamma=0.2, gram_rank=2, gram_factors=xi
    )

    assert torch.allclose(via_factors, via_gram, atol=1e-5)
