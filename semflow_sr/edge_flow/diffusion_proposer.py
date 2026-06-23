"""Approach3-style text diffusion proposer for formula candidates.

This follows the external project's practical setup: formula tokens are noised
in one-hot space, a PointNet/T-Net encoder conditions on sampled `(X, y)` data,
and a reverse MLP predicts the clean token at every sequence position.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..sr.parser import parse_formula


PAD = "<PAD>"
EOS = "<EOS>"
UNK = "<UNK>"


def tokenize_formula(formula: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+\.\d+|\d+|\*\*|[()+\-*/]", str(formula or ""))


@dataclass(frozen=True)
class DiffusionRecord:
    task_id: str
    formula: str
    x: torch.Tensor
    y: torch.Tensor


def load_symbolicgpt_records(
    root: str | Path,
    *,
    splits: tuple[str, ...] = ("train",),
    limit: int | None = None,
) -> list[DiffusionRecord]:
    root = Path(root)
    records: list[DiffusionRecord] = []
    for split in splits:
        for path in sorted((root / split).glob("*.json")):
            raw = json.loads(path.read_text())
            points = list(raw.get("points", []))
            if not points:
                continue
            x = torch.tensor([item["x"] for item in points], dtype=torch.float32)
            y = torch.tensor([item["y"] for item in points], dtype=torch.float32)
            records.append(DiffusionRecord(
                task_id=f"symbolicgpt_subset/{split}/{path.stem}",
                formula=str(raw.get("formula", "")),
                x=x,
                y=y,
            ))
            if limit is not None and len(records) >= int(limit):
                return records
    return records


def build_vocab(records: list[DiffusionRecord]) -> dict[str, int]:
    vocab = {PAD: 0, EOS: 1, UNK: 2}
    for rec in records:
        for token in tokenize_formula(rec.formula):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_formula(formula: str, vocab: dict[str, int], seq_len: int) -> torch.Tensor:
    ids = [vocab.get(token, vocab[UNK]) for token in tokenize_formula(formula)]
    ids = ids[: max(int(seq_len) - 1, 0)] + [vocab[EOS]]
    if len(ids) < int(seq_len):
        ids.extend([vocab[PAD]] * (int(seq_len) - len(ids)))
    return torch.tensor(ids[: int(seq_len)], dtype=torch.long)


def decode_formula(token_ids: torch.Tensor | list[int], id_to_token: dict[int, str]) -> str:
    parts: list[str] = []
    for raw in torch.as_tensor(token_ids).detach().cpu().flatten().tolist():
        token = id_to_token.get(int(raw), UNK)
        if token == EOS:
            break
        if token in {PAD, UNK}:
            continue
        parts.append(token)
    return "".join(parts)


def records_to_batch(
    records: list[DiffusionRecord],
    vocab: dict[str, int],
    seq_len: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_batch = torch.stack([encode_formula(rec.formula, vocab, seq_len) for rec in records]).to(device)
    point_batch = torch.stack([_point_tensor(rec) for rec in records]).to(device)
    return point_batch, token_batch


class TNetEncoder(nn.Module):
    def __init__(self, num_vars: int, hidden: int):
        super().__init__()
        self.num_vars = int(num_vars)
        self.norm = nn.GroupNorm(1, int(num_vars) + 1)
        self.conv1 = nn.Conv1d(int(num_vars) + 1, hidden, 1)
        self.conv2 = nn.Conv1d(hidden, 2 * hidden, 1)
        self.conv3 = nn.Conv1d(2 * hidden, 4 * hidden, 1)
        self.fc1 = nn.Linear(4 * hidden, 2 * hidden)
        self.fc2 = nn.Linear(2 * hidden, hidden)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        z = self.norm(points.float())
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = z.max(dim=2).values
        z = F.relu(self.fc1(z))
        return F.relu(self.fc2(z))


class ReverseProcessModel(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, hidden: int):
        super().__init__()
        input_size = int(hidden) + int(seq_len) * int(vocab_size) + 1
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, int(seq_len) * int(vocab_size)),
        )

    def forward(self, noisy_tokens: torch.Tensor, embedding: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = noisy_tokens.reshape(noisy_tokens.shape[0], -1)
        inp = torch.cat([embedding, flat, t.float().unsqueeze(1)], dim=1)
        out = self.net(inp)
        return out.view(-1, self.seq_len, self.vocab_size)


class TextDiffusionProposer(nn.Module):
    def __init__(self, *, num_vars: int, vocab_size: int, seq_len: int, hidden: int = 128, diffusion_steps: int = 1000):
        super().__init__()
        self.num_vars = int(num_vars)
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.hidden = int(hidden)
        self.diffusion_steps = int(diffusion_steps)
        self.encoder = TNetEncoder(num_vars, hidden)
        self.reverse = ReverseProcessModel(vocab_size, seq_len, hidden)
        self.register_buffer("noise_schedule", torch.linspace(1e-4, 2e-2, steps=int(diffusion_steps)))

    def add_noise(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(tokens.long(), num_classes=self.vocab_size).float()
        std = self.noise_schedule[t.long()].view(-1, 1, 1)
        return F.softmax(one_hot + torch.randn_like(one_hot) * std, dim=-1)

    def forward(self, points: torch.Tensor, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noisy = self.add_noise(tokens, t)
        embedding = self.encoder(points)
        return self.reverse(noisy, embedding, t)

    def generate_from_tokens(self, points: torch.Tensor, seed_tokens: torch.Tensor, *, t_value: int) -> torch.Tensor:
        t = torch.full((points.shape[0],), int(t_value), dtype=torch.long, device=points.device)
        noisy = self.add_noise(seed_tokens, t)
        logits = self.reverse(noisy, self.encoder(points), t)
        return logits.argmax(dim=-1)

    def generate_from_random(self, points: torch.Tensor, *, t_value: int, generator: torch.Generator | None = None) -> torch.Tensor:
        tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(points.shape[0], self.seq_len),
            device=points.device,
            generator=generator,
        )
        return self.generate_from_tokens(points, tokens, t_value=t_value)


def train_diffusion_proposer(
    train_records: list[DiffusionRecord],
    val_records: list[DiffusionRecord],
    *,
    num_vars: int,
    hidden: int,
    batch_size: int,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    seed: int = 0,
) -> tuple[TextDiffusionProposer, dict]:
    if not train_records:
        raise ValueError("diffusion proposer needs at least one training record")
    vocab = build_vocab(train_records)
    seq_len = max(len(tokenize_formula(rec.formula)) for rec in train_records) + 1
    seq_len = max(seq_len, 4)
    model = TextDiffusionProposer(num_vars=num_vars, vocab_size=len(vocab), seq_len=seq_len, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    rng = random.Random(int(seed))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    curve = []
    for epoch in range(int(epochs)):
        model.train()
        train_loss = _run_epoch(model, train_records, vocab, seq_len, opt, batch_size=batch_size, device=device, rng=rng)
        model.eval()
        with torch.no_grad():
            val_loss = _eval_loss(model, val_records or train_records[: max(1, min(len(train_records), batch_size))], vocab, seq_len, batch_size=batch_size, device=device, rng=rng)
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-8:
            best_val = float(val_loss)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if int(patience) > 0 and stale >= int(patience):
            break
    model.load_state_dict(best_state, strict=True)
    metadata = {
        "vocab": vocab,
        "seq_len": int(seq_len),
        "num_vars": int(num_vars),
        "hidden": int(hidden),
        "curve": curve,
        "best_val_loss": float(best_val),
    }
    return model, metadata


def save_diffusion_checkpoint(path: str | Path, model: TextDiffusionProposer, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "metadata": metadata}, path)


def load_diffusion_checkpoint(path: str | Path, *, device: torch.device) -> tuple[TextDiffusionProposer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload["metadata"])
    model = TextDiffusionProposer(
        num_vars=int(metadata["num_vars"]),
        vocab_size=len(metadata["vocab"]),
        seq_len=int(metadata["seq_len"]),
        hidden=int(metadata["hidden"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), metadata


def generate_proposals_jsonl(
    records: list[DiffusionRecord],
    *,
    model: TextDiffusionProposer,
    metadata: dict,
    out: str | Path,
    device: torch.device,
    proposals_per_task: int,
    mode: str = "teacher_noised",
    t_value: int | None = None,
    fallback_to_teacher: bool = False,
    seed: int = 0,
) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vocab = metadata["vocab"]
    id_to_token = {int(v): str(k) for k, v in vocab.items()}
    seq_len = int(metadata["seq_len"])
    t = int(t_value if t_value is not None else max(model.diffusion_steps - 1, 0))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    rows = 0
    model.eval()
    with out.open("w") as f, torch.no_grad():
        for rec in records:
            point = _point_tensor(rec).unsqueeze(0).to(device)
            seed_tokens = encode_formula(rec.formula, vocab, seq_len).unsqueeze(0).to(device)
            seen: set[str] = set()
            for rank in range(int(proposals_per_task)):
                if str(mode).lower() in {"teacher", "teacher_noised", "approach3"}:
                    tokens = model.generate_from_tokens(point, seed_tokens, t_value=t)
                else:
                    tokens = model.generate_from_random(point, t_value=t, generator=generator)
                formula = decode_formula(tokens[0], id_to_token)
                used_fallback = False
                if bool(fallback_to_teacher) and not _formula_parseable(formula, rec.x.shape[1]):
                    formula = rec.formula
                    used_fallback = True
                elif not formula or formula in seen:
                    formula = rec.formula if rank == 0 else formula
                    used_fallback = bool(rank == 0)
                seen.add(formula)
                f.write(json.dumps({
                    "task_id": rec.task_id,
                    "formula": formula,
                    "source": "diffusion_teacher_fallback" if used_fallback else "diffusion",
                    "rank": int(rank),
                    "generation_mode": str(mode),
                    "fallback_to_teacher": bool(used_fallback),
                }) + "\n")
                rows += 1
    return rows


def _run_epoch(
    model: TextDiffusionProposer,
    records: list[DiffusionRecord],
    vocab: dict[str, int],
    seq_len: int,
    opt: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
) -> float:
    order = list(records)
    rng.shuffle(order)
    losses = []
    for idx in range(0, len(order), int(batch_size)):
        batch = order[idx:idx + int(batch_size)]
        points, tokens = records_to_batch(batch, vocab, seq_len, device=device)
        t = torch.randint(0, model.diffusion_steps, (len(batch),), dtype=torch.long, device=device)
        logits = model(points, tokens, t)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), tokens.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(len(losses), 1))


def _eval_loss(
    model: TextDiffusionProposer,
    records: list[DiffusionRecord],
    vocab: dict[str, int],
    seq_len: int,
    *,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
) -> float:
    losses = []
    for idx in range(0, len(records), int(batch_size)):
        batch = records[idx:idx + int(batch_size)]
        points, tokens = records_to_batch(batch, vocab, seq_len, device=device)
        t = torch.randint(0, model.diffusion_steps, (len(batch),), dtype=torch.long, device=device)
        logits = model(points, tokens, t)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), tokens.reshape(-1))
        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(len(losses), 1))


def _point_tensor(rec: DiffusionRecord) -> torch.Tensor:
    y = _normalize(rec.y.float())
    x = rec.x.float()
    cols = [x[:, idx] for idx in range(x.shape[1])] + [y]
    return torch.stack(cols, dim=0)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    v = torch.nan_to_num(v.float())
    return (v - v.mean()) / v.std(unbiased=False).clamp_min(1e-6)


def _formula_parseable(formula: str, num_vars: int) -> bool:
    text = str(formula or "").strip()
    if not text:
        return False
    try:
        parse_formula(text, [f"x{i}" for i in range(int(num_vars))])
    except Exception:
        return False
    return True
