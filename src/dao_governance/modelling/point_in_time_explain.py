"""Integrated Gradients for the selected single-vote classifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from dao_governance.modelling.point_in_time_model import VoteChoiceClassifier


@dataclass(frozen=True)
class IntegratedGradientsResult:
    attributions: torch.Tensor
    input_score: torch.Tensor
    baseline_score: torch.Tensor
    completeness_delta: torch.Tensor


def _validate(class_index: int, steps: int, background_split: str) -> None:
    if class_index not in (0, 1, 2) or steps < 1:
        raise ValueError("Integrated Gradients requires a canonical class and positive steps.")
    if background_split != "train":
        raise ValueError("Explanation baselines must be derived from training data.")


def integrated_gradients_numeric(
    model: VoteChoiceClassifier,
    batch: Mapping[str, torch.Tensor],
    baseline: torch.Tensor,
    *,
    class_index: int,
    steps: int,
    background_split: str,
) -> IntegratedGradientsResult:
    _validate(class_index, steps, background_split)
    original = batch["numeric_features"].detach()
    if original.ndim != 2 or original.shape[1] == 0 or baseline.shape != original.shape:
        raise ValueError("Numeric baselines must match nonempty numeric inputs.")
    if not torch.isfinite(original).all() or not torch.isfinite(baseline).all():
        raise ValueError("Numeric inputs and baselines must be finite.")
    baseline = baseline.to(device=original.device, dtype=original.dtype)
    difference = original - baseline
    total_gradient = torch.zeros_like(original)
    was_training = model.training
    model.eval()
    try:
        for index in range(steps):
            alpha = (index + 0.5) / steps
            interpolated = (baseline + alpha * difference).detach().requires_grad_(True)
            inputs = dict(batch)
            inputs["numeric_features"] = interpolated
            score = model(inputs)[:, class_index]
            total_gradient += torch.autograd.grad(score.sum(), interpolated)[0]
        attributions = difference * total_gradient / steps
        with torch.no_grad():
            input_score = model(batch)[:, class_index]
            baseline_batch = dict(batch)
            baseline_batch["numeric_features"] = baseline
            baseline_score = model(baseline_batch)[:, class_index]
    finally:
        model.train(was_training)
    delta = input_score - baseline_score - attributions.sum(dim=1)
    return IntegratedGradientsResult(
        attributions.detach().cpu(), input_score.detach().cpu(), baseline_score.detach().cpu(), delta.detach().cpu()
    )


def integrated_gradients_text(
    model: VoteChoiceClassifier,
    batch: Mapping[str, torch.Tensor],
    baseline_embeddings: torch.Tensor,
    *,
    class_index: int,
    steps: int,
    background_split: str,
) -> IntegratedGradientsResult:
    _validate(class_index, steps, background_split)
    if model.text_encoder is None:
        raise ValueError("Text attribution requires a text encoder.")
    embedding_layer = model.text_encoder.get_input_embeddings()
    original = embedding_layer(batch["input_ids"]).detach()
    if baseline_embeddings.shape != original.shape or not torch.isfinite(baseline_embeddings).all():
        raise ValueError("Text baseline embeddings must match the input embeddings and be finite.")
    baseline = baseline_embeddings.to(device=original.device, dtype=original.dtype)
    difference = original - baseline
    total_gradient = torch.zeros_like(original)
    numeric = batch["numeric_features"]
    attention = batch["attention_mask"]
    was_training = model.training
    model.eval()
    try:
        for index in range(steps):
            alpha = (index + 0.5) / steps
            interpolated = (baseline + alpha * difference).detach().requires_grad_(True)
            score = model.forward_from_embeddings(numeric, interpolated, attention)[:, class_index]
            total_gradient += torch.autograd.grad(score.sum(), interpolated)[0]
        attributions = difference * total_gradient / steps
        with torch.no_grad():
            input_score = model.forward_from_embeddings(numeric, original, attention)[:, class_index]
            baseline_score = model.forward_from_embeddings(numeric, baseline, attention)[:, class_index]
    finally:
        model.train(was_training)
    delta = input_score - baseline_score - attributions.flatten(start_dim=1).sum(dim=1)
    return IntegratedGradientsResult(
        attributions.detach().cpu(), input_score.detach().cpu(), baseline_score.detach().cpu(), delta.detach().cpu()
    )
