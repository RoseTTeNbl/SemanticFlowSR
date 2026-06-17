import torch
from semflow_sr.models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig


def test_velocity_model_shapes_and_zero_sum():
    torch.manual_seed(0)
    bsz, m, K, A, d = 4, 32, 8, 12, 2
    cfg = SemanticTransformerConfig(d=d, K=K, hidden=32, row_layers=1, heads=2)
    model = SemanticTransformer(cfg)
    x = torch.randn(bsz, m, d)
    y = torch.randn(bsz, m)
    B = torch.randn(bsz, m, K)
    from semflow_sr.actions.action_features import ACTION_FEATURE_DIM
    feats = torch.randn(bsz, A, ACTION_FEATURE_DIM)
    energies = torch.rand(bsz, A); weights = torch.rand(bsz, A) + 0.1
    p_lambda = torch.softmax(torch.randn(bsz, A), -1)
    lam = torch.rand(bsz)
    mask = torch.ones(bsz, A, dtype=torch.bool)
    mask[:, -3:] = False
    semantic_stats = torch.randn(bsz, A, 8)
    gram = torch.randn(bsz, A, A)
    gram = gram @ gram.transpose(-1, -2)
    out = model(x=x, y=y, B=B, p_lambda=p_lambda, lambda_value=lam, action_feats=feats,
                energies=energies, weights=weights, semantic_stats=semantic_stats, gram=gram,
                action_mask=mask)
    assert out.v_pred.shape == (bsz, A)
    assert out.lograte_logits.shape == (bsz, A)
    assert out.z_dot_pred.shape == (bsz, A)
    masked_sum = (out.v_pred * mask).sum(-1)
    assert torch.allclose(masked_sum, torch.zeros(bsz), atol=1e-5)
    assert (out.v_pred * (~mask)).abs().sum() < 1e-6   # invalid actions stay zero
    assert (out.lograte_logits * (~mask)).abs().sum() < 1e-6
