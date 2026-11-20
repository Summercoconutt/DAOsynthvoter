from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dao_governance.data.snapshot_metadata import collect_proposal_metadata, read_proposal_keys


ID_A = "0x" + "a" * 64
ID_B = "0x" + "b" * 64


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def fetch(self, ids):
        self.calls.append(list(ids))
        return [
            {"id": ID_A, "space": {"id": "a.eth"}, "choices": ["Against", "For", "Abstain"]}
        ] if ID_A in ids else []


def _source(path: Path) -> None:
    pd.DataFrame(
        {
            "Space": ["a.eth", "a.eth", "b.eth"],
            "Proposal ID": [ID_A, ID_A, ID_B],
            "proposal_id": [f"{ID_A}_1700000000000", f"{ID_A}_1700000000000", f"{ID_B}_1700000000001"],
        }
    ).to_parquet(path, index=False)


def test_collect_metadata_uses_source_ids_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "votes.parquet"
    output = tmp_path / "proposals.jsonl"
    _source(source)
    client = FakeClient()

    assert read_proposal_keys(source) == {ID_A: "a.eth", ID_B: "b.eth"}
    counts = collect_proposal_metadata(source, output, batch_size=1, client=client)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert counts["retrieved"] == 1
    assert counts["missing"] == 1
    assert [record["status"] for record in records[1:]] == ["retrieved", "missing"]
    assert client.calls == [[ID_A], [ID_B]]

    resumed = FakeClient()
    resumed_counts = collect_proposal_metadata(source, output, resume=True, client=resumed)
    assert resumed_counts["already_recorded"] == 2
    assert resumed_counts["retrieved"] == 1 and resumed_counts["missing"] == 1
    assert not resumed.calls
    with pytest.raises(FileExistsError):
        collect_proposal_metadata(source, output, client=FakeClient())


def test_metadata_collection_rejects_filename_ids_and_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "votes.parquet"
    output = tmp_path / "proposals.jsonl"
    _source(source)
    collect_proposal_metadata(source, output, client=FakeClient())

    pd.DataFrame({"space": ["a.eth"], "proposal_id": [f"{ID_A}_1700000000000"]}).to_parquet(source)
    with pytest.raises(ValueError, match="do not infer IDs from filenames"):
        collect_proposal_metadata(source, output, resume=True, client=FakeClient())


def test_metadata_collection_rejects_conflicting_spaces(tmp_path: Path) -> None:
    source = tmp_path / "votes.parquet"
    pd.DataFrame({"space": ["a.eth", "b.eth"], "proposal_id": [ID_A, ID_A]}).to_parquet(source)
    with pytest.raises(ValueError, match="more than one space"):
        read_proposal_keys(source)
