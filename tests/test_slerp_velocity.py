import torch
from semflow_sr.geometry.slerp_path import SemanticFisherSlerpPath
from semflow_sr.utils.numerical import normalize_simplex


def _smooth(p, eps=1e-3):
    return normalize_simplex(p + eps)


def test_slerp_returns_legal_distribution():
    torch.manual_seed(0)
    A = 8
    p0 = _smooth(normalize_simplex(torch.rand(A, dtype=torch.float64)))
    p1 = _smooth(normalize_simplex(torch.rand(A, dtype=torch.float64)))
    w = torch.rand(A, dtype=torch.float64) + 0.1
    path = SemanticFisherSlerpPath()
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = path.sample(p0, p1, w, lam)
        assert torch.all(s.p_lambda >= -1e-9)
        assert torch.allclose(s.p_lambda.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-9)


def test_velocity_zero_sum():
    torch.manual_seed(1)
    A = 10
    p0 = _smooth(normalize_simplex(torch.rand(A, dtype=torch.float64)))
    p1 = _smooth(normalize_simplex(torch.rand(A, dtype=torch.float64)))
    w = torch.rand(A, dtype=torch.float64) + 0.1
    s = SemanticFisherSlerpPath().sample(p0, p1, w, 0.4)
    assert abs(s.dp_dlambda.sum().item()) < 1e-9


def test_analytic_velocity_matches_finite_difference():
    torch.manual_seed(2)
    A = 7
    p0 = _smooth(normalize_simplex(torch.rand(A, dtype=torch.float64)))
    p1 = _smooth(normalize_simplex(torch.rand(A, dtype=torch.float64)))
    w = torch.rand(A, dtype=torch.float64) + 0.3
    path = SemanticFisherSlerpPath()
    lam = 0.37
    h = 1e-6
    p_plus = path.sample(p0, p1, w, lam + h).p_lambda
    p_minus = path.sample(p0, p1, w, lam - h).p_lambda
    fd = (p_plus - p_minus) / (2 * h)
    analytic = path.sample(p0, p1, w, lam).dp_dlambda
    assert torch.allclose(analytic, fd, atol=1e-4)


def test_small_theta_fallback_stable():
    A = 5
    p0 = _smooth(normalize_simplex(torch.rand(A, dtype=torch.float64)))
    w = torch.rand(A, dtype=torch.float64) + 0.1
    s = SemanticFisherSlerpPath().sample(p0, p0.clone(), w, 0.5)   # p0==p1 -> theta=0
    assert torch.isfinite(s.p_lambda).all() and torch.isfinite(s.dp_dlambda).all()
