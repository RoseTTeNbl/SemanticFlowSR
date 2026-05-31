import random
import torch
from semflow_sr.data.synthetic_generator import GenConfig
from semflow_sr.train.build_dataset import build_dataset
from semflow_sr.data.collate import collate_velocity
from semflow_sr.models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from semflow_sr.train.trainer_velocity import train_velocity, TrainConfig


def test_overfit_tiny_velocity_dataset():
    torch.manual_seed(0)
    gen = GenConfig(num_vars=1, max_depth=3, K=6, probe_size=32,
                    ops=("add", "sub", "mul", "sin", "cos", "square"))
    ds = build_dataset(gen, num_tasks=8, target="gt", seed=0, max_support=64)
    assert len(ds) > 0
    model = SemanticTransformer(SemanticTransformerConfig(d=1, K=6, hidden=32, row_layers=1, heads=2))
    cfg = TrainConfig(lr=1e-3, steps=200, batch_size=4, log_every=1000)
    stats = train_velocity(model, ds, cfg, torch.device("cpu"), collate_velocity)
    h = stats["loss_history"]
    # loss should drop substantially as the model overfits the tiny trace set
    assert h[-1] < 0.5 * (sum(h[:5]) / 5)
