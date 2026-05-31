import torch
from semflow_sr.geometry.semantic_chart import semantic_chart, inverse_semantic_chart
from semflow_sr.utils.numerical import normalize_simplex


def test_chart_norm_is_one():
    torch.manual_seed(0)
    p = normalize_simplex(torch.rand(8, dtype=torch.float64))
    w = torch.rand(8, dtype=torch.float64) + 0.1
    z = semantic_chart(p, w)
    assert torch.allclose(z.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


def test_inverse_recovers_p():
    torch.manual_seed(1)
    p = normalize_simplex(torch.rand(10, dtype=torch.float64) + 1e-3)
    w = torch.rand(10, dtype=torch.float64) + 0.1
    z = semantic_chart(p, w)
    p_rec = inverse_semantic_chart(z, w)
    assert torch.allclose(p, p_rec, atol=1e-9)


def test_w_one_reduces_to_sqrt_chart():
    torch.manual_seed(2)
    p = normalize_simplex(torch.rand(6, dtype=torch.float64) + 1e-3)
    w = torch.ones(6, dtype=torch.float64)
    z = semantic_chart(p, w)
    assert torch.allclose(z, torch.sqrt(p), atol=1e-10)


def test_works_with_smoothed_one_hot():
    p = torch.tensor([0.9, 0.02, 0.02, 0.02, 0.04], dtype=torch.float64)
    w = torch.tensor([2.0, 1.0, 0.5, 1.0, 1.0], dtype=torch.float64)
    z = semantic_chart(p, w)
    p_rec = inverse_semantic_chart(z, w)
    assert torch.allclose(p, p_rec, atol=1e-9)
