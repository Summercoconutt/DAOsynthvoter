"""Matched single-vote models and train-fitted feature encoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dao_governance.data.chronological_split import validate_chronological_manifest, validate_proposal_metadata
from dao_governance.data.eligibility import EligibleProposal
from dao_governance.features.point_in_time_clusters import UNKNOWN_CLUSTER

NUMERIC_FEATURES = frozenset(
    f"{scope}_{name}"
    for scope in ("voter", "voter_dao", "dao")
    for name in ("prior_count", "has_history", "prior_for", "prior_against", "prior_abstain")
)
CATEGORICAL_FEATURES = frozenset({"dao_cluster", "voter_cluster"})


@dataclass
class FeatureEncoder:
    numeric_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...] = ()
    imputer: Optional[SimpleImputer] = None
    scaler: Optional[RobustScaler] = None
    categories: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    fitted: bool = False

    def __post_init__(self) -> None:
        if len(set(self.numeric_columns)) != len(self.numeric_columns):
            raise ValueError("Numeric feature columns must be unique.")
        if len(set(self.categorical_columns)) != len(self.categorical_columns):
            raise ValueError("Categorical feature columns must be unique.")
        if not set(self.numeric_columns).issubset(NUMERIC_FEATURES):
            raise ValueError("Numeric model inputs must be approved point-in-time features.")
        if not set(self.categorical_columns).issubset(CATEGORICAL_FEATURES):
            raise ValueError("Only categorical DAO and voter cluster IDs are supported.")

    def _check_columns(self, frame: pd.DataFrame) -> None:
        missing = (set(self.numeric_columns) | set(self.categorical_columns)) - set(frame.columns)
        if missing:
            raise ValueError(f"Model features are missing columns: {sorted(missing)}")

    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = frame.loc[:, list(self.numeric_columns)]
        numeric = raw.apply(pd.to_numeric, errors="coerce")
        if (raw.notna() & numeric.isna()).any().any() or np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("Numeric model features contain invalid values.")
        for column in self.numeric_columns:
            values = numeric[column].dropna()
            if column.endswith("_prior_count") and (values < 0).any():
                raise ValueError("Prior vote counts must be nonnegative.")
            if column.endswith("_has_history") and not values.isin([0, 1]).all():
                raise ValueError("History indicators must be zero or one.")
            if column.endswith(("_prior_for", "_prior_against", "_prior_abstain")) and not values.between(0, 1).all():
                raise ValueError("Prior choice fractions must be between zero and one.")
        return numeric

    @staticmethod
    def _category(value: Any) -> str:
        if isinstance(value, str):
            if value:
                return value
            raise ValueError("Cluster categories must be nonempty strings or missing.")
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) or value is pd.NA:
            return UNKNOWN_CLUSTER
        raise ValueError("Cluster categories must be nonempty strings or missing.")

    def fit(self, train: pd.DataFrame) -> "FeatureEncoder":
        if self.fitted:
            raise RuntimeError("The feature encoder is already fitted and cannot be refitted.")
        self._check_columns(train)
        if train.empty:
            raise ValueError("Training features cannot be empty.")
        if self.numeric_columns:
            numeric = self._numeric(train)
            unavailable = [column for column in self.numeric_columns if numeric[column].notna().sum() == 0]
            if unavailable:
                raise ValueError(f"Training features are entirely unavailable: {unavailable}")
            self.imputer = SimpleImputer(strategy="median")
            values = self.imputer.fit_transform(numeric)
            self.scaler = RobustScaler().fit(values)
        for column in self.categorical_columns:
            known = {self._category(value) for value in train[column]}
            known.discard(UNKNOWN_CLUSTER)
            if not known:
                raise ValueError(
                    f"Training has no assigned {column} categories; use an earlier causal warm-up or disable this feature."
                )
            self.categories[column] = (UNKNOWN_CLUSTER, *sorted(known))
        self.fitted = True
        return self

    @property
    def dimension(self) -> int:
        if not self.fitted:
            raise RuntimeError("Fit the feature encoder before using its dimension.")
        return len(self.numeric_columns) + sum(len(values) for values in self.categories.values())

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Fit the feature encoder on training data before transforming.")
        self._check_columns(frame)
        blocks = []
        if self.numeric_columns:
            if self.imputer is None or self.scaler is None:
                raise RuntimeError("Fitted numeric preprocessing is missing.")
            blocks.append(self.scaler.transform(self.imputer.transform(self._numeric(frame))))
        for column in self.categorical_columns:
            vocabulary = {value: index for index, value in enumerate(self.categories[column])}
            values = np.zeros((len(frame), len(vocabulary)), dtype=np.float32)
            for row, value in enumerate(frame[column]):
                category = self._category(value)
                values[row, vocabulary.get(category, 0)] = 1.0
            blocks.append(values)
        result = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.empty((len(frame), 0), dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Model features contain non-finite values after preprocessing.")
        return result


class ProposalTextCache:
    def __init__(self, proposals: Iterable[EligibleProposal], tokenizer: Any, *, max_length: int) -> None:
        if max_length < 1 or tokenizer.pad_token_id is None:
            raise ValueError("A positive token limit and tokenizer padding ID are required.")
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.tokens: Dict[Tuple[str, str], Dict[str, list[int]]] = {}
        self.truncated = 0
        for proposal in proposals:
            key = (proposal.space, proposal.proposal_id)
            if key in self.tokens:
                raise ValueError("Proposal text keys must be unique.")
            if not isinstance(proposal.title, str) or not isinstance(proposal.body, str):
                raise ValueError("Both proposal title and body must be available.")
            full = tokenizer(proposal.title, proposal.body, truncation=False)
            encoded = tokenizer(proposal.title, proposal.body, truncation=True, max_length=max_length)
            input_ids = encoded.get("input_ids")
            attention = encoded.get("attention_mask")
            if not isinstance(input_ids, list) or not isinstance(attention, list) or not input_ids or len(input_ids) != len(attention):
                raise ValueError("Tokenizer returned invalid proposal tokens.")
            self.truncated += len(full["input_ids"]) > max_length
            self.tokens[key] = {"input_ids": input_ids, "attention_mask": attention}


class VoteDataset(Dataset):
    def __init__(self, events: pd.DataFrame, numeric_features: np.ndarray, text_cache: Optional[ProposalTextCache]) -> None:
        required = {"example_id", "space", "proposal_id", "vote_ts", "label_id"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"Vote examples are missing columns: {sorted(missing)}")
        if events["example_id"].isna().any() or events["example_id"].duplicated().any():
            raise ValueError("Vote examples require unique identifiers.")
        if not events["label_id"].isin([0, 1, 2]).all():
            raise ValueError("Vote labels must use the three canonical classes.")
        if numeric_features.ndim != 2 or len(numeric_features) != len(events) or not np.isfinite(numeric_features).all():
            raise ValueError("Numeric feature rows must be finite and aligned with vote examples.")
        self.events = events.reset_index(drop=True).copy()
        self.numeric_features = numeric_features.astype(np.float32, copy=False)
        self.text_cache = text_cache
        if text_cache is not None:
            keys = zip(self.events["space"], self.events["proposal_id"])
            if any(key not in text_cache.tokens for key in keys):
                raise ValueError("A vote example has no eligible proposal text.")

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.events.iloc[index]
        item: Dict[str, Any] = {
            "example_id": row["example_id"],
            "space": row["space"],
            "proposal_id": row["proposal_id"],
            "vote_ts": row["vote_ts"],
            "numeric_features": torch.from_numpy(self.numeric_features[index]),
            "label": int(row["label_id"]),
        }
        if self.text_cache is not None:
            tokens = self.text_cache.tokens[(row["space"], row["proposal_id"])]
            item["input_ids"] = torch.tensor(tokens["input_ids"], dtype=torch.long)
            item["attention_mask"] = torch.tensor(tokens["attention_mask"], dtype=torch.long)
        return item


def collate_vote_examples(items: Sequence[Mapping[str, Any]], *, pad_token_id: int = 0) -> Dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty vote batch.")
    batch: Dict[str, Any] = {
        "example_ids": [item["example_id"] for item in items],
        "spaces": [item["space"] for item in items],
        "proposal_ids": [item["proposal_id"] for item in items],
        "vote_timestamps": [item["vote_ts"] for item in items],
        "numeric_features": torch.stack([item["numeric_features"] for item in items]),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
    }
    if "input_ids" in items[0]:
        if any("input_ids" not in item or "attention_mask" not in item for item in items):
            raise ValueError("Text inputs must be present for every example in a batch.")
        batch["input_ids"] = pad_sequence(
            [item["input_ids"] for item in items], batch_first=True, padding_value=pad_token_id
        )
        batch["attention_mask"] = pad_sequence(
            [item["attention_mask"] for item in items], batch_first=True, padding_value=0
        )
    elif any("input_ids" in item for item in items):
        raise ValueError("Text inputs must be present for every example in a batch.")
    return batch


class VoteChoiceClassifier(nn.Module):
    def __init__(
        self,
        *,
        numeric_dim: int,
        text_encoder: Optional[nn.Module] = None,
        freeze_text_encoder: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if numeric_dim < 0 or hidden_dim < 1 or (text_encoder is None and numeric_dim == 0):
            raise ValueError("The classifier requires at least one valid input modality.")
        if freeze_text_encoder and text_encoder is None:
            raise ValueError("A missing text encoder cannot be frozen.")
        self.numeric_dim = numeric_dim
        self.text_encoder = text_encoder
        self.text_encoder_frozen = freeze_text_encoder
        text_dim = int(text_encoder.config.hidden_size) if text_encoder is not None else 0
        self.head = nn.Sequential(
            nn.Linear(text_dim + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        if text_encoder is not None and freeze_text_encoder:
            for parameter in text_encoder.parameters():
                parameter.requires_grad_(False)
            text_encoder.eval()

    def train(self, mode: bool = True) -> "VoteChoiceClassifier":
        super().train(mode)
        if self.text_encoder is not None and self.text_encoder_frozen:
            self.text_encoder.eval()
        return self

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        numeric = batch["numeric_features"]
        if not isinstance(numeric, torch.Tensor) or numeric.ndim != 2 or numeric.shape[1] != self.numeric_dim or not torch.isfinite(numeric).all():
            raise ValueError("The classifier received invalid numeric features.")
        inputs = [numeric]
        if self.text_encoder is not None:
            if self.text_encoder_frozen:
                with torch.no_grad():
                    encoded = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            else:
                encoded = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            inputs.insert(0, encoded.last_hidden_state[:, 0, :])
        return self.head(torch.cat(inputs, dim=1))

    def forward_from_embeddings(
        self,
        numeric: torch.Tensor,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.text_encoder is None:
            raise ValueError("Text attribution requires a text encoder.")
        if numeric.ndim != 2 or numeric.shape[1] != self.numeric_dim or not torch.isfinite(numeric).all():
            raise ValueError("The classifier received invalid numeric features.")
        if embeddings.ndim != 3 or attention_mask.shape != embeddings.shape[:2] or len(numeric) != len(embeddings):
            raise ValueError("Text embeddings and attention masks are not aligned.")
        encoded = self.text_encoder(inputs_embeds=embeddings, attention_mask=attention_mask)
        return self.head(torch.cat([encoded.last_hidden_state[:, 0, :], numeric], dim=1))


def validate_target_examples(events: pd.DataFrame, manifest: Mapping[str, Any], split: str) -> None:
    if split not in {"train", "validation", "test"}:
        raise ValueError("Target split must be train, validation, or test.")
    required = {"example_id", "space", "proposal_id", "vote_ts"}
    if not required.issubset(events.columns):
        raise ValueError("Target examples are missing required identity or timestamp columns.")
    if events["example_id"].isna().any() or events["example_id"].duplicated().any():
        raise ValueError("Target examples require unique identifiers.")
    validate_chronological_manifest(manifest)
    proposals = manifest.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError("The proposal split manifest is missing proposal entries.")
    membership = {}
    for row in proposals:
        key = (row["space"], row["proposal_id"])
        if key in membership:
            raise ValueError("Proposal split manifest keys must be unique.")
        membership[key] = row
    timestamps = pd.to_datetime(events["vote_ts"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Target examples require valid vote timestamps.")
    for space, proposal_id, timestamp in zip(events["space"], events["proposal_id"], timestamps):
        row = membership.get((space, proposal_id))
        if row is None or row.get("split") != split:
            raise ValueError("Target examples do not match the proposal split manifest.")
        start = pd.to_datetime(row.get("start"), utc=True, errors="coerce")
        end = pd.to_datetime(row.get("end"), utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end) or not start <= timestamp < end:
            raise ValueError("A target vote is outside its proposal window.")
        if row.get("provenance_mode", "verified") == "verified":
            available = pd.to_datetime(row.get("metadata_available_at"), utc=True, errors="coerce")
            if pd.isna(available) or available >= timestamp:
                raise ValueError("Proposal metadata is not demonstrably available before the target vote.")


def prepare_training_datasets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    proposals: Mapping[Tuple[str, str], EligibleProposal],
    manifest: Mapping[str, Any],
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str] = (),
    tokenizer: Optional[Any] = None,
    max_length: int = 128,
) -> Tuple[VoteDataset, VoteDataset, FeatureEncoder, Optional[ProposalTextCache]]:
    validate_target_examples(train, manifest, "train")
    validate_target_examples(validation, manifest, "validation")
    if train.empty or validation.empty:
        raise ValueError("Training and validation examples cannot be empty.")
    if set(train["example_id"]) & set(validation["example_id"]):
        raise ValueError("Training and validation example identifiers overlap.")
    membership = {(row["space"], row["proposal_id"]): row for row in manifest["proposals"]}
    keys = set(zip(train["space"], train["proposal_id"])) | set(zip(validation["space"], validation["proposal_id"]))
    selected_proposals = []
    for key in sorted(keys):
        proposal = proposals.get(key)
        if proposal is None:
            raise ValueError("A training or validation proposal is missing from the eligible catalog.")
        validate_proposal_metadata(proposal, membership[key])
        selected_proposals.append(proposal)
    encoder = FeatureEncoder(tuple(numeric_columns), tuple(categorical_columns)).fit(train)
    text_cache = ProposalTextCache(selected_proposals, tokenizer, max_length=max_length) if tokenizer is not None else None
    train_dataset = VoteDataset(train, encoder.transform(train), text_cache)
    validation_dataset = VoteDataset(validation, encoder.transform(validation), text_cache)
    return train_dataset, validation_dataset, encoder, text_cache


def majority_label(training_labels: Sequence[int]) -> int:
    labels = np.asarray(training_labels)
    if labels.size == 0 or labels.dtype.kind == "b" or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("The majority baseline requires valid training labels.")
    return int(np.bincount(labels.astype(int), minlength=3).argmax())


def historical_choice_baseline(features: pd.DataFrame, *, fallback_label: int) -> np.ndarray:
    if fallback_label not in (0, 1, 2):
        raise ValueError("The historical baseline fallback must be a canonical class.")
    columns = ["voter_dao_prior_for", "voter_dao_prior_against", "voter_dao_prior_abstain"]
    if not set(columns + ["voter_dao_prior_count"]).issubset(features.columns):
        raise ValueError("The historical baseline requires voter/DAO history features.")
    predictions = []
    for _, row in features.iterrows():
        count = row["voter_dao_prior_count"]
        if pd.isna(count) or count < 0:
            raise ValueError("Historical vote counts must be nonnegative.")
        if count == 0:
            predictions.append(fallback_label)
            continue
        fractions = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(fractions).all() or (fractions < 0).any() or not np.isclose(fractions.sum(), 1.0):
            raise ValueError("Historical choice fractions are invalid.")
        tied = np.flatnonzero(fractions == fractions.max())
        predictions.append(fallback_label if fallback_label in tied else int(tied[0]))
    return np.asarray(predictions, dtype=int)
