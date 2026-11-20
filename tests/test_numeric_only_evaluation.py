from pathlib import Path

import pandas as pd
import pytest
import torch

from dao_governance.evaluation.evaluate import _run_inference
from dao_governance.evaluation.metrics import compute_classification_metrics
from dao_governance.modelling.metrics import macro_prf
from dao_governance.modelling.model import NumericOnlyTimeSeriesClassifier


def test_macro_f1_includes_all_three_classes_when_one_is_absent() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    assert macro_prf(y_true, y_pred)["f1"] == pytest.approx(2 / 3)
    assert compute_classification_metrics(y_true, y_pred)["macro_f1"] == pytest.approx(2 / 3)


def test_numeric_only_inference_loads_and_builds_numeric_windows(tmp_path: Path) -> None:
    rows = []
    for index in range(5):
        rows.append(
            {
                "voter": "voter-1",
                "space": "dao.eth",
                "vote_ts": f"2025-01-0{index + 1}T00:00:00Z",
                "proposal_id": f"proposal-{index}",
                "label_id": index % 3,
                "text": "proposal title",
                "voting_power": float(index + 1),
                "is_whale": False,
                "dao_cluster": 0,
                "voter_cluster": 0,
                "prior_frac_for": 0.4,
                "prior_frac_against": 0.2,
            }
        )
    dataset = tmp_path / "enriched.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)

    artifact_dir = tmp_path / "08b"
    artifact_dir.mkdir()
    model = NumericOnlyTimeSeriesClassifier(feat_dim=10, dropout=0.0, dim_feedforward=32)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    config = {
        "model_type": "numeric_only_temporal",
        "feat_dim": 10,
        "window": 3,
        "dropout": 0.0,
        "dim_feedforward": 32,
        "enriched_behaviour_csv": str(dataset),
    }
    preprocessor = {
        "voting_power": {
            "median_raw_nonneg": 1.0,
            "cap_value": 100.0,
            "log_center": 0.0,
            "log_scale": 1.0,
        },
        "extra_robust": {
            "dao_cluster": {"center": 0.0, "scale": 1.0},
            "voter_cluster": {"center": 0.0, "scale": 1.0},
            "prior_frac_for": {"center": 0.0, "scale": 1.0},
            "prior_frac_against": {"center": 0.0, "scale": 1.0},
        },
    }

    y_true, y_pred, y_prob, voters, _, _, n_windows = _run_inference(
        dataset_csv=dataset,
        artifacts_dir=artifact_dir,
        preprocessor=preprocessor,
        cfg_train=config,
        voter_ids=["voter-1"],
        seed=42,
        batch_size=2,
        max_windows=0,
    )

    assert n_windows == 3
    assert len(y_true) == len(y_pred) == len(voters) == 3
    assert y_prob.shape == (3, 3)
