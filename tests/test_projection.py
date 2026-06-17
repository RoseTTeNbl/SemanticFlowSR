import torch
from semflow_sr.semantics.projection import ProjectionBackend
from semflow_sr.models.semantic_transformer import SemanticTransformer


def _explicit_centered_ridge_proj_y(B, y, rho):
    """中心化岭投影: 列与目标去均值, 拟合后加回 y 均值(隐式截距)。"""
    K = B.shape[1]
    Bc = B - B.mean(0, keepdim=True)
    yc = y - y.mean()
    M = torch.linalg.inv(Bc.T @ Bc + rho * torch.eye(K, dtype=B.dtype))
    return Bc @ (M @ (Bc.T @ yc)) + y.mean()


def test_ridge_agrees_with_explicit_centered_lstsq():
    torch.manual_seed(0)
    B = torch.randn(40, 5, dtype=torch.float64)
    y = torch.randn(40, dtype=torch.float64)
    rho = 1e-3
    pb = ProjectionBackend("ridge", rho)
    assert torch.allclose(pb.project_y(B, y), _explicit_centered_ridge_proj_y(B, y, rho), atol=1e-8)


def test_constant_column_has_no_residual_gain():
    # 去均值投影下, 纯常数列拟合不掉任何残差(消除虚假收益陷阱)
    torch.manual_seed(2)
    y = torch.randn(64, dtype=torch.float64) + 5.0
    B = torch.ones(64, 1, dtype=torch.float64)
    pb = ProjectionBackend("ridge", 1e-3)
    base = 0.5 * ((y - y.mean()) ** 2).sum()
    assert torch.allclose(pb.residual_energy(B, y), base, atol=1e-6)


def test_residual_energy_correct():
    torch.manual_seed(1)
    B = torch.randn(30, 4, dtype=torch.float64)
    y = torch.randn(30, dtype=torch.float64)
    pb = ProjectionBackend("ridge", 1e-3)
    r = y - pb.project_y(B, y)
    assert torch.allclose(pb.residual_energy(B, y), 0.5 * (r * r).sum())


def test_residual_vector_matches_project_y():
    torch.manual_seed(11)
    B = torch.randn(24, 3, dtype=torch.float64)
    y = torch.randn(24, dtype=torch.float64) + 2.0
    pb = ProjectionBackend("ridge", 1e-3)

    residual = pb.residual_vector(B, y)

    assert torch.allclose(residual, y - pb.project_y(B, y), atol=1e-10)


def test_effective_rank_matches_full_trace():
    torch.manual_seed(2)
    B = torch.randn(50, 6, dtype=torch.float64)
    rho = 1e-2
    pb = ProjectionBackend("ridge", rho)
    Bc = B - B.mean(0, keepdim=True)
    G = Bc.T @ Bc
    Pi = Bc @ torch.linalg.inv(G + rho * torch.eye(6)) @ Bc.T
    assert torch.allclose(pb.effective_rank(B), torch.trace(Pi), atol=1e-8)


def test_projection_distance_matches_explicit():
    torch.manual_seed(3)
    rho = 1e-3
    B1 = torch.randn(40, 5, dtype=torch.float64)
    B2 = torch.randn(40, 5, dtype=torch.float64)
    pb = ProjectionBackend("ridge", rho)

    def proj(B):
        Bc = B - B.mean(0, keepdim=True)
        return Bc @ torch.linalg.inv(Bc.T @ Bc + rho * torch.eye(5)) @ Bc.T
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


def test_model_residual_feature_uses_centered_projection_backend():
    torch.manual_seed(5)
    B = torch.randn(3, 32, 4, dtype=torch.float64)
    y = torch.randn(3, 32, dtype=torch.float64) + torch.tensor([[10.0], [-4.0], [1.5]], dtype=torch.float64)
    rho = 1e-3
    pb = ProjectionBackend("ridge", rho)

    expected = pb.project_y(B, y)
    actual = SemanticTransformer._project_y(B, y, rho=rho)

    assert torch.allclose(actual, expected, atol=1e-8)
