"""Temporal classifier with text and numeric fusion."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


def copy_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


class TimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "roberta-base",
        hidden_dim: int = 256,
        feat_dim: int = 10,
        num_classes: int = 3,
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.text_encoder_frozen = False
        text_hidden = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_hidden, hidden_dim)

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        d_model = hidden_dim + 128
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        if freeze_text_encoder:
            self.set_text_encoder_frozen(True)

    def set_text_encoder_frozen(self, frozen: bool = True) -> None:
        self.text_encoder_frozen = frozen
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.text_encoder.eval()

    def train(self, mode: bool = True) -> "TimeSeriesClassifier":
        super().train(mode)
        if self.text_encoder_frozen:
            self.text_encoder.eval()
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        f = batch["num_feats"]

        if not torch.isfinite(f).all():
            raise RuntimeError("Non-finite num_feats in forward (must be caught earlier).")

        if "text_embeddings" in batch:
            cls = batch["text_embeddings"]
        else:
            x = batch["input_ids"]
            a = batch["attention_mask"]
            bsz, steps, seqlen = x.size()
            if self.text_encoder_frozen:
                with torch.no_grad():
                    enc = self.text_encoder(
                        input_ids=x.view(bsz * steps, seqlen),
                        attention_mask=a.view(bsz * steps, seqlen),
                    )
            else:
                enc = self.text_encoder(
                    input_ids=x.view(bsz * steps, seqlen),
                    attention_mask=a.view(bsz * steps, seqlen),
                )
            cls = enc.last_hidden_state[:, 0, :].view(bsz, steps, -1)
        cls = self.text_proj(cls)
        f = self.feat_proj(f)
        fused = torch.cat([cls, f], dim=-1)
        seq = self.temporal(fused)
        logits = self.head(seq[:, -1, :])
        return {"logits": logits}

    def loss_fn(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if class_weights is None:
            return nn.CrossEntropyLoss()(logits, labels)
        return nn.CrossEntropyLoss(weight=class_weights)(logits, labels)


class NumericOnlyTimeSeriesClassifier(nn.Module):
    """Temporal classifier using numeric, time, and cluster features only (no text)."""

    def __init__(
        self,
        feat_dim: int = 10,
        num_classes: int = 3,
        dropout: float = 0.1,
        dim_feedforward: int = 1024,
    ):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        d_model = 128
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        f = batch["num_feats"]
        if not torch.isfinite(f).all():
            raise RuntimeError("Non-finite num_feats in forward (must be caught earlier).")
        seq = self.temporal(self.feat_proj(f))
        logits = self.head(seq[:, -1, :])
        return {"logits": logits}

    def loss_fn(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if class_weights is None:
            return nn.CrossEntropyLoss()(logits, labels)
        return nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
