from pathlib import Path

import pandas as pd
import pytest

from dao_governance.data.lean_sensitivity import build_lean_membership, load_lean_membership
from dao_governance.modelling.run_artifacts import sha256_file


def test_lean_sensitivity_maps_file_keys_to_original_proposals(tmp_path: Path):
    source = tmp_path / "raw.parquet"
    pd.DataFrame(
        {"space": ["dao", "dao", "dao"], "proposal_id": ["p1_123", "p1_123", "p2_123"],
         "Proposal ID": ["p1", "p1", "p2"]}
    ).to_parquet(source)
    audit = tmp_path / "lean.csv"
    pd.DataFrame({"space": ["dao", "dao"], "proposal_id": ["p1_123", "p2_123"],
                  "retained": [True, False]}).to_csv(audit, index=False)
    output = tmp_path / "lean.parquet"
    metadata = build_lean_membership(source, audit, output)
    assert metadata["selection"] == "legacy_outcome_selected_lean"
    assert load_lean_membership(output, raw_votes_sha256=sha256_file(source)) == {("dao", "p1")}
    with pytest.raises(ValueError, match="raw source"):
        load_lean_membership(output, raw_votes_sha256="0" * 64)
