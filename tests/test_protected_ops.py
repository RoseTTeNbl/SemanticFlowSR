import torch
from semflow_sr.sr import protected_ops as P


def test_protected_ops_never_nan_inf():
    x = torch.tensor([-1e9, -1.0, 0.0, 1e-9, 1.0, 1e9])
    z = torch.zeros_like(x)
    for fn, args in [
        (P.p_div, (x, z)), (P.p_log, (x,)), (P.p_sqrt, (x,)),
        (P.p_exp, (x,)), (P.p_sin, (x,)), (P.p_cos, (x,)),
        (P.p_square, (x,)), (P.p_cube, (x,)), (P.p_neg, (x,)),
        (P.p_add, (x, x)), (P.p_sub, (x, x)), (P.p_mul, (x, x)),
    ]:
        out = fn(*args)
        assert torch.isfinite(out).all(), fn.__name__


def test_div_by_zero_is_one():
    a = torch.tensor([3.0, 5.0]); b = torch.tensor([0.0, 1e-9])
    out = P.p_div(a, b)
    assert torch.allclose(out, torch.ones_like(out))
