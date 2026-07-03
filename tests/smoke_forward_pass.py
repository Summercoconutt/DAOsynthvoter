"""Optional training smoke: one forward pass on synthetic data (no GPU required)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.features.behaviour_pipeline import load_behaviour_votes, prepare_behaviour_splits
from dao_governance.modelling.dataset import WindowDataset, collate_fn
from dao_governance.modelling.model import TimeSeriesClassifier
from dao_governance.modelling.preprocess import select_numeric_columns
from dao_governance.modelling.windows import build_windows

DAO_FIXTURE = ROOT / "src" / "dao_clustering_scripts" / "data" / "processed" / "dao_feature_table.csv"


def _synthetic_votes() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2023-01-01", tz="UTC")
    for vi in range(20):
        voter = f"voter_{vi:03d}"
        for space in ("ens.eth", "uniswapgovernance.eth"):
            for j in range(8):
                rows.append(
                    {
                        "voter": voter,
                        "space": space,
                        "proposal_id": f"p_{j}",
                        "choice_norm": ["for", "against", "abstain"][j % 3],
                        "vote_timestamp": base + pd.Timedelta(days=vi, hours=j),
                        "proposal_title": f"Title {j}",
                        "voting_power": float(50 + vi + j),
                        "is_whale": False,
                    }
                )
    df = pd.DataFrame(rows)
    df["vote_ts"] = pd.to_datetime(df["vote_timestamp"], utc=True)
    df["label_id"] = df["choice_norm"].map({"for": 0, "against": 1, "abstain": 2})
    df["text"] = df["proposal_title"]
    return df


def main() -> None:
    if not DAO_FIXTURE.exists():
        raise SystemExit(f"Missing DAO fixture: {DAO_FIXTURE}")

    raw = _synthetic_votes()
    train_df, val_df, _, pre, bundle, *_ = prepare_behaviour_splits(
        raw,
        dao_feature_table_path=DAO_FIXTURE,
        train_frac=0.7,
        val_frac=0.15,
        seed=42,
        upper_quantile_cap=0.999,
        absolute_cap=1e18,
        min_votes_per_pair=5,
    )
    cols = select_numeric_columns()
    windows = build_windows(train_df, window_size=3, numeric_cols=cols)[:4]
    assert windows, "no windows"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilroberta-base")
    tok.add_special_tokens({"additional_special_tokens": ["[PREDICT]", "[LABEL_0]", "[LABEL_1]", "[LABEL_2]"]})
    ds = WindowDataset(windows, tok, max_length=48)
    batch = collate_fn([ds[i] for i in range(len(ds))])

    feat_dim = len(windows[0].window_features[0])
    model = TimeSeriesClassifier(pretrained_model_name="distilroberta-base", feat_dim=feat_dim)
    model.text_encoder.resize_token_embeddings(len(tok))
    model.eval()
    with torch.no_grad():
        logits = model(batch)["logits"]
    assert logits.shape == (len(windows), 3)
    assert bundle.meta.get("clustering_mode") == "train_only_structural"
    print("OK smoke forward pass", logits.shape, "feat_dim", feat_dim)


if __name__ == "__main__":
    main()
