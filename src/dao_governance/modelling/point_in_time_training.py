"""Validation-selected training and aligned predictions for single-vote models."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dao_governance.modelling.metrics import macro_prf
from dao_governance.modelling.model import copy_state_dict_to_cpu
from dao_governance.modelling.point_in_time_model import (
    VoteChoiceClassifier,
    VoteDataset,
    collate_vote_examples,
    validate_target_examples,
)


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_macro_f1: float
    epochs: list[Dict[str, float]]


def _loader(dataset: VoteDataset, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    pad_id = dataset.text_cache.pad_token_id if dataset.text_cache is not None else 0
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=partial(collate_vote_examples, pad_token_id=pad_id),
    )


def _tensor_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def predict_vote_classifier(
    model: VoteChoiceClassifier,
    dataset: VoteDataset,
    *,
    batch_size: int,
    device: str = "cpu",
) -> pd.DataFrame:
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive.")
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in _loader(dataset, batch_size=batch_size, shuffle=False, seed=0):
            logits = model(_tensor_batch(batch, target_device))
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].numpy()
            if not np.isfinite(probabilities).all() or probabilities.shape != (len(labels), 3):
                raise RuntimeError("The classifier produced invalid probabilities.")
            for index, probability in enumerate(probabilities):
                rows.append(
                    {
                        "example_id": batch["example_ids"][index],
                        "space": batch["spaces"][index],
                        "proposal_id": batch["proposal_ids"][index],
                        "vote_ts": batch["vote_timestamps"][index],
                        "y_true": int(labels[index]),
                        "y_pred": int(probability.argmax()),
                        "prob_FOR": float(probability[0]),
                        "prob_AGAINST": float(probability[1]),
                        "prob_ABSTAIN": float(probability[2]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["example_id", "space", "proposal_id", "vote_ts", "y_true", "y_pred", "prob_FOR", "prob_AGAINST", "prob_ABSTAIN"],
    )


def fit_vote_classifier(
    model: VoteChoiceClassifier,
    train: VoteDataset,
    validation: VoteDataset,
    *,
    manifest: Mapping[str, Any],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int = 42,
    device: str = "cpu",
) -> TrainingResult:
    if epochs < 1 or learning_rate <= 0 or batch_size < 1:
        raise ValueError("Epochs, learning rate, and batch size must be positive.")
    if not len(train) or not len(validation):
        raise ValueError("Training and validation examples cannot be empty.")
    validate_target_examples(train.events, manifest, "train")
    validate_target_examples(validation.events, manifest, "validation")
    if set(train.events["example_id"]) & set(validation.events["example_id"]):
        raise ValueError("Training and validation example identifiers overlap.")
    if train.numeric_features.shape[1] != model.numeric_dim or validation.numeric_features.shape[1] != model.numeric_dim:
        raise ValueError("Model and dataset numeric feature dimensions differ.")
    if (train.text_cache is None) != (model.text_encoder is None) or (validation.text_cache is None) != (model.text_encoder is None):
        raise ValueError("Model and dataset text modalities differ.")
    if train.text_cache is not None and validation.text_cache is not None and train.text_cache.pad_token_id != validation.text_cache.pad_token_id:
        raise ValueError("Training and validation tokenizer padding IDs differ.")

    torch.manual_seed(seed)
    target_device = torch.device(device)
    model = model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = _loader(train, batch_size=batch_size, shuffle=True, seed=seed)
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            tensors = _tensor_batch(batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensors)
            loss = loss_fn(logits, tensors["labels"])
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(tensors["labels"])
            total_examples += len(tensors["labels"])
        predictions = predict_vote_classifier(model, validation, batch_size=batch_size, device=device)
        metrics = macro_prf(predictions["y_true"], predictions["y_pred"])
        history.append({"epoch": float(epoch), "train_loss": total_loss / total_examples, "validation_macro_f1": metrics["f1"]})
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_state = copy_state_dict_to_cpu(model)
    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch=best_epoch, best_validation_macro_f1=best_f1, epochs=history)
