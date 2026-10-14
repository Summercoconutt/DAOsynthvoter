"""Dataset and collate for behaviour modelling windows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


@dataclass
class EncodedWindow:
    input_ids: List[torch.Tensor] | None
    text_embeddings: torch.Tensor | None
    num_feats: torch.Tensor
    label: int
    dao_cluster: int
    voter_cluster: int
    pad_token_id: int


class TokenizedTextCache:
    """Tokenize each distinct window text once and keep compact token IDs."""

    def __init__(
        self,
        windows: List[Any],
        tokenizer,
        max_length: int = 128,
        tokenization_batch_size: int = 2048,
    ):
        if max_length < 1:
            raise ValueError("max_length must be positive.")
        if tokenization_batch_size < 1:
            raise ValueError("tokenization_batch_size must be positive.")
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token for dynamic batching.")

        self.pad_token_id = int(tokenizer.pad_token_id)
        text_to_index: Dict[str, int] = {}
        unique_texts: List[str] = []
        for window in windows:
            for text in window.window_texts:
                if text not in text_to_index:
                    text_to_index[text] = len(unique_texts)
                    unique_texts.append(text)

        self.lengths = np.empty(len(unique_texts), dtype=np.int32)
        token_chunks: List[np.ndarray] = []
        for start in range(0, len(unique_texts), tokenization_batch_size):
            texts = unique_texts[start : start + tokenization_batch_size]
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            chunk_lengths = [len(ids) for ids in encoded["input_ids"]]
            if any(length == 0 for length in chunk_lengths):
                raise RuntimeError("Tokenizer returned an empty sequence.")
            self.lengths[start : start + len(texts)] = chunk_lengths
            token_chunks.append(
                np.fromiter(
                    (token_id for ids in encoded["input_ids"] for token_id in ids),
                    dtype=np.int32,
                )
            )

        self.offsets = np.empty(len(unique_texts) + 1, dtype=np.int64)
        self.offsets[0] = 0
        np.cumsum(self.lengths, dtype=np.int64, out=self.offsets[1:])
        self.token_ids = (
            np.concatenate(token_chunks) if token_chunks else np.empty(0, dtype=np.int32)
        )
        self.text_to_index = text_to_index

    def indices_for(self, windows: List[Any]) -> np.ndarray:
        if not windows:
            return np.empty((0, 0), dtype=np.int32)
        window_size = len(windows[0].window_texts)
        indices = np.empty((len(windows), window_size), dtype=np.int32)
        for row, window in enumerate(windows):
            if len(window.window_texts) != window_size:
                raise ValueError("All windows in a dataset must have the same window size.")
            for step, text in enumerate(window.window_texts):
                indices[row, step] = self.text_to_index[text]
        return indices

    def tokens_for(self, text_index: int) -> np.ndarray:
        start = int(self.offsets[text_index])
        end = int(self.offsets[text_index + 1])
        return self.token_ids[start:end]

    def release_lookup(self) -> None:
        self.text_to_index.clear()

    def release_tokens(self) -> None:
        self.token_ids = np.empty(0, dtype=np.int32)
        self.offsets = np.zeros(1, dtype=np.int64)


def build_frozen_text_embedding_cache(
    token_cache: TokenizedTextCache,
    text_encoder: torch.nn.Module,
    *,
    device: str | torch.device,
    cache_path: Path,
    batch_size: int,
) -> np.memmap:
    """Encode each unique text once into a disk-backed float16 cache."""
    if batch_size < 1:
        raise ValueError("Embedding batch_size must be positive.")
    if not token_cache.lengths.size:
        raise ValueError("Cannot cache embeddings for an empty set of texts.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_size = int(text_encoder.config.hidden_size)
    embeddings = np.memmap(
        cache_path,
        dtype=np.float16,
        mode="w+",
        shape=(len(token_cache.lengths), hidden_size),
    )
    text_encoder.eval()
    for start in range(0, len(token_cache.lengths), batch_size):
        end = min(start + batch_size, len(token_cache.lengths))
        lengths = token_cache.lengths[start:end]
        max_length = int(lengths.max())
        input_ids = torch.full(
            (end - start, max_length),
            token_cache.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, text_index in enumerate(range(start, end)):
            tokens = token_cache.tokens_for(text_index)
            length = len(tokens)
            input_ids[row, :length] = torch.from_numpy(tokens.astype(np.int64, copy=True))
            attention_mask[row, :length] = 1

        with torch.inference_mode():
            encoded = text_encoder(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
        embeddings[start:end] = encoded.last_hidden_state[:, 0, :].float().cpu().numpy().astype(
            np.float16
        )

    embeddings.flush()
    return embeddings


class WindowDataset(Dataset):
    def __init__(
        self,
        windows: List[Any],
        tokenizer,
        max_length: int = 128,
        *,
        token_cache: TokenizedTextCache | None = None,
        text_embeddings: np.ndarray | None = None,
    ):
        self.windows = windows
        cache = token_cache or TokenizedTextCache(windows, tokenizer, max_length)
        self.window_text_indices = cache.indices_for(windows)
        self.text_embeddings = text_embeddings
        self.pad_token_id = cache.pad_token_id
        if text_embeddings is not None and len(text_embeddings) != len(cache.lengths):
            raise ValueError("Text embedding cache does not match the tokenized text cache.")
        self.token_cache = None if text_embeddings is not None else cache

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> EncodedWindow:
        w = self.windows[idx]
        text_indices = self.window_text_indices[idx]
        if self.text_embeddings is None:
            if self.token_cache is None:
                raise RuntimeError("Token cache was released before tokenized data was consumed.")
            input_ids = [
                torch.tensor(self.token_cache.tokens_for(int(text_index)), dtype=torch.long)
                for text_index in text_indices
            ]
            text_embeddings = None
        else:
            input_ids = None
            text_embeddings = torch.from_numpy(
                np.asarray(self.text_embeddings[text_indices], dtype=np.float32).copy()
            )
        return EncodedWindow(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            num_feats=torch.tensor(np.stack(w.window_features, axis=0), dtype=torch.float32),
            label=int(w.target_label),
            dao_cluster=int(w.dao_cluster),
            voter_cluster=int(w.voter_cluster),
            pad_token_id=self.pad_token_id,
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
    result = {
        "num_feats": num_feats,
        "labels": torch.tensor([b.label for b in batch], dtype=torch.long),
        "dao_clusters": torch.tensor([b.dao_cluster for b in batch], dtype=torch.long),
        "voter_clusters": torch.tensor([b.voter_cluster for b in batch], dtype=torch.long),
    }
    has_embeddings = [b.text_embeddings is not None for b in batch]
    if any(has_embeddings) and not all(has_embeddings):
        raise ValueError("A batch cannot mix token IDs and cached text embeddings.")
    if all(has_embeddings):
        result["text_embeddings"] = torch.stack(
            [b.text_embeddings for b in batch], dim=0
        )
        return result

    if any(b.input_ids is None for b in batch):
        raise ValueError("Tokenized batch is missing input IDs.")
    pad_token_ids = {b.pad_token_id for b in batch}
    if len(pad_token_ids) != 1:
        raise ValueError("A batch cannot mix tokenizer pad-token IDs.")
    window_size = len(batch[0].input_ids)
    if any(len(b.input_ids) != window_size for b in batch):
        raise ValueError("All windows in a batch must have the same window size.")

    flat_ids = [ids for item in batch for ids in item.input_ids]
    lengths = torch.tensor([len(ids) for ids in flat_ids], dtype=torch.long)
    padded_ids = pad_sequence(
        flat_ids,
        batch_first=True,
        padding_value=pad_token_ids.pop(),
    )
    max_length = padded_ids.size(1)
    attention_mask = (
        torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)
    ).to(dtype=torch.long)
    batch_size = len(batch)
    result["input_ids"] = padded_ids.view(batch_size, window_size, max_length)
    result["attention_mask"] = attention_mask.view(batch_size, window_size, max_length)
    return result
