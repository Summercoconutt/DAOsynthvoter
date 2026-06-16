"""Dataset and collate for behaviour modelling windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EncodedWindow:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    num_feats: torch.Tensor
    label: int
    dao_cluster: int
    voter_cluster: int


class WindowDataset(Dataset):
    def __init__(self, windows: List[Any], tokenizer, max_length: int = 128):
        self.windows = windows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> EncodedWindow:
        w = self.windows[idx]
        ids, masks = [], []
        for text in w.window_texts:
            enc = self.tok(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            ids.append(enc["input_ids"][0])
            masks.append(enc["attention_mask"][0])
        return EncodedWindow(
            input_ids=torch.stack(ids, dim=0),
            attention_mask=torch.stack(masks, dim=0),
            num_feats=torch.tensor(np.stack(w.window_features, axis=0), dtype=torch.float32),
            label=int(w.target_label),
            dao_cluster=int(w.dao_cluster),
            voter_cluster=int(w.voter_cluster),
        )


@dataclass
class NumericEncodedWindow:
    num_feats: torch.Tensor
    label: int
    dao_cluster: int
    voter_cluster: int


class NumericWindowDataset(Dataset):
    """Lightweight window dataset without text tokenization."""

    def __init__(self, windows: List[Any]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> NumericEncodedWindow:
        w = self.windows[idx]
        return NumericEncodedWindow(
            num_feats=torch.tensor(np.stack(w.window_features, axis=0), dtype=torch.float32),
            label=int(w.target_label),
            dao_cluster=int(w.dao_cluster),
            voter_cluster=int(w.voter_cluster),
        )


def numeric_collate_fn(batch: List[NumericEncodedWindow]) -> Dict[str, torch.Tensor]:
    num_feats = torch.stack([b.num_feats for b in batch], dim=0)
    if not torch.isfinite(num_feats).all():
        bad = ~torch.isfinite(num_feats)
        idx = torch.nonzero(bad, as_tuple=False)
        raise RuntimeError(
            "Non-finite values in num_feats batch. First positions (batch, t, feat): "
            f"{idx[:12].tolist()}"
        )
    return {
        "num_feats": num_feats,
        "labels": torch.tensor([b.label for b in batch], dtype=torch.long),
        "dao_clusters": torch.tensor([b.dao_cluster for b in batch], dtype=torch.long),
        "voter_clusters": torch.tensor([b.voter_cluster for b in batch], dtype=torch.long),
    }


class MemmapNumericWindowDataset(Dataset):
    """Memory-mapped numeric windows for large-scale training."""

    def __init__(self, split_meta: Dict[str, Any]):
        self.window_size = int(split_meta["window_size"])
        self.feat_dim = int(split_meta["feat_dim"])
        n = int(split_meta["n_windows"])
        self.num_feats = np.memmap(
            split_meta["num_feats"], dtype=np.float32, mode="r", shape=(n, self.window_size, self.feat_dim)
        )
        self.labels = np.memmap(split_meta["labels"], dtype=np.int64, mode="r", shape=(n,))
        self.dao_clusters = np.memmap(split_meta["dao_clusters"], dtype=np.int64, mode="r", shape=(n,))
        self.voter_clusters = np.memmap(split_meta["voter_clusters"], dtype=np.int64, mode="r", shape=(n,))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> NumericEncodedWindow:
        return NumericEncodedWindow(
            num_feats=torch.tensor(self.num_feats[idx], dtype=torch.float32),
            label=int(self.labels[idx]),
            dao_cluster=int(self.dao_clusters[idx]),
            voter_cluster=int(self.voter_clusters[idx]),
        )


def collate_fn(batch: List[EncodedWindow]) -> Dict[str, torch.Tensor]:
    num_feats = torch.stack([b.num_feats for b in batch], dim=0)
    if not torch.isfinite(num_feats).all():
        bad = ~torch.isfinite(num_feats)
        idx = torch.nonzero(bad, as_tuple=False)
        raise RuntimeError(
            "Non-finite values in num_feats batch. First positions (batch, t, feat): "
            f"{idx[:12].tolist()}"
        )
    return {
        "input_ids": torch.stack([b.input_ids for b in batch], dim=0),
        "attention_mask": torch.stack([b.attention_mask for b in batch], dim=0),
        "num_feats": num_feats,
        "labels": torch.tensor([b.label for b in batch], dtype=torch.long),
        "dao_clusters": torch.tensor([b.dao_cluster for b in batch], dtype=torch.long),
        "voter_clusters": torch.tensor([b.voter_cluster for b in batch], dtype=torch.long),
    }
