import torch
from semflow_sr.semantics.projection import ProjectionBackend


def _explicit_ridge_proj_y(B, y, rho):
    K = B.shape[1]
    M = torch.linalg.inv(B.T @ B + rho * torch.eye(K))
    return B @ (M @ (B.T @ y))


def test_ridge_agrees_with_explicit_lstsq():
    torch.manual_seed(0)
    B = torch.randn(40, 5, dtype=torch.float64)
    y = torch.randn(40, dtype=torch.float64)
    rho = 1e-3
    pb = ProjectionBackend("ridge", rho)
    assert torch.allclose(pb.project_y(B, y), _explicit_ridge_proj_y(B, y, rho), atol=1e-8)


def test_residual_energy_correct():
    torch.manual_seed(1)
    B = torch.randn(30, 4, dtype=torch.float64)
    y = torch.randn(30, dtype=torch.float64)
    pb = ProjectionBackend("ridge", 1e-3)
    r = y - pb.project_y(B, y)
    assert torch.allclose(pb.residual_energy(B, y), 0.5 * (r * r).sum())


def test_effective_rank_matches_full_trace():
    torch.manual_seed(2)
    B = torch.randn(50, 6, dtype=torch.float64)
    rho = 1e-2
    pb = ProjectionBackend("ridge", rho)
    G = B.T @ B
    Pi = B @ torch.linalg.inv(G + rho * torch.eye(6)) @ B.T
    assert torch.allclose(pb.effective_rank(B), torch.trace(Pi), atol=1e-8)


def test_projection_distance_matches_explicit():
    torch.manual_seed(3)
    rho = 1e-3
    B1 = torch.randn(40, 5, dtype=torch.float64)
    B2 = torch.randn(40, 5, dtype=torch.float64)
    pb = ProjectionBackend("ridge", rho)

    def proj(B):
        return B @ torch.linalg.inv(B.T @ B + rho * torch.eye(5)) @ B.T
    explicit = ((proj(B1) - proj(B2)) ** 2).sum()
    assert torch.allclose(pb.projection_distance(B1, B2), explicit, atol=1e-6)


def test_projection_distance_batched():
    torch.manual_seed(4)
    rho = 1e-3
    A, m, K = 3, 30, 4
    Ba = torch.randn(A, m, K, dtype=torch.float64)
    B = torch.randn(m, K, dtype=torch.float64)
    pb = ProjectionBackend("ridge", rho)
    out = pb.projection_distance(Ba, B)
    assert out.shape == (A,)
    for i in range(A):
        assert torch.allclose(out[i], pb.projection_distance(Ba[i], B), atol=1e-6)
