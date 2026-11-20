from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.modelling.dataset import (
    TokenizedTextCache,
    WindowDataset,
    build_frozen_text_embedding_cache,
    collate_fn,
)
from dao_governance.modelling.model import TimeSeriesClassifier, copy_state_dict_to_cpu


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts, *, truncation, max_length, padding):
        assert truncation is True
        assert padding is False
        encoded = []
        for text in texts:
            self.seen.append(text)
            ids = [2, *[(ord(char) % 20) + 4 for char in text if char != " "], 3]
            encoded.append(ids[:max_length])
        return {"input_ids": encoded}


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(32, 4)

    def forward(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _window(texts: list[str], label: int = 0):
    return SimpleNamespace(
        window_texts=texts,
        window_features=[np.array([1.0, 2.0, 3.0], dtype=np.float32) for _ in texts],
        target_label=label,
        dao_cluster=0,
        voter_cluster=1,
    )


def test_cpu_checkpoint_is_independent_of_later_updates():
    model = nn.Linear(1, 1)
    checkpoint = copy_state_dict_to_cpu(model)
    expected = checkpoint["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1)
    assert torch.equal(checkpoint["weight"], expected)


def test_window_dataset_pretokenizes_unique_texts_and_pads_per_batch():
    tokenizer = DummyTokenizer()
    windows = [_window(["short", "shared"]), _window(["a longer text", "shared"], 1)]
    token_cache = TokenizedTextCache(windows, tokenizer, max_length=32)
    train_ds = WindowDataset(windows[:1], tokenizer, token_cache=token_cache)
    valid_ds = WindowDataset(windows[1:], tokenizer, token_cache=token_cache)
    token_cache.release_lookup()

    assert len(tokenizer.seen) == 3
    assert len(set(tokenizer.seen)) == 3
    batch = collate_fn([train_ds[0], valid_ds[0]])

    assert batch["input_ids"].shape == (2, 2, 13)
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["attention_mask"][0, 0].sum().item() == 7


def test_frozen_encoder_uses_cached_embeddings_and_keeps_encoder_frozen(monkeypatch, tmp_path):
    encoder = DummyEncoder()
    monkeypatch.setattr(
        "dao_governance.modelling.model.AutoModel.from_pretrained",
        lambda _: encoder,
    )
    tokenizer = DummyTokenizer()
    windows = [_window(["first", "shared"]), _window(["second", "shared"], 1)]
    token_cache = TokenizedTextCache(windows, tokenizer, max_length=32)
    embedding_cache = build_frozen_text_embedding_cache(
        token_cache,
        encoder,
        device="cpu",
        cache_path=tmp_path / "embeddings.f16",
        batch_size=2,
    )
    dataset = WindowDataset(
        windows,
        tokenizer,
        token_cache=token_cache,
        text_embeddings=embedding_cache,
    )
    token_cache.release_lookup()
    token_cache.release_tokens()
    batch = collate_fn([dataset[0], dataset[1]])

    model = TimeSeriesClassifier(
        pretrained_model_name="dummy",
        feat_dim=3,
        hidden_dim=16,
        dropout=0.0,
    )
    model.set_text_encoder_frozen()
    model.train()
    logits = model(batch)["logits"]
    loss = model.loss_fn(logits, batch["labels"])
    loss.backward()

    assert logits.shape == (2, 3)
    assert not model.text_encoder.training
    assert all(not parameter.requires_grad for parameter in model.text_encoder.parameters())
    assert model.text_encoder.embedding.weight.grad is None
    assert model.text_proj.weight.grad is not None
    assert embedding_cache.dtype == np.float16

    eval_token_cache = TokenizedTextCache(windows, tokenizer, max_length=32)
    eval_dataset = WindowDataset(windows, tokenizer, token_cache=eval_token_cache)
    eval_token_cache.release_lookup()
    eval_batch = collate_fn([eval_dataset[0], eval_dataset[1]])
    with torch.no_grad():
        eval_logits = model(eval_batch)["logits"]
    assert eval_logits.shape == (2, 3)
    assert torch.allclose(logits.detach(), eval_logits, atol=1e-3, rtol=1e-3)
    embedding_cache._mmap.close()
