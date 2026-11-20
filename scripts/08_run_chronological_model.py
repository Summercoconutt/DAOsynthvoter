#!/usr/bin/env python3
"""Train a provenance-gated, validation-selected chronological vote model."""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.modelling.point_in_time_runner import train_chronological_run
from dao_governance.modelling.run_artifacts import validate_run_id
from dao_governance.settings import load_config, project_root, require_local_data_path

DEFAULT_NUMERIC = (
    "voter_prior_count", "voter_has_history", "voter_dao_prior_count", "voter_dao_has_history",
    "dao_prior_count", "dao_has_history", "voter_dao_prior_for", "voter_dao_prior_against",
    "voter_dao_prior_abstain", "dao_prior_for", "dao_prior_against", "dao_prior_abstain",
)


def load_text_encoder(pretrained: str, *, local_only: bool):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=local_only)
    encoder = AutoModel.from_pretrained(pretrained, local_files_only=local_only)
    return tokenizer, encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra-config", default="")
    parser.add_argument("--historical-provenance", default="")
    parser.add_argument("--observations", default="")
    parser.add_argument("--observation-provenance", default="")
    parser.add_argument("--local-proxy", action="store_true")
    parser.add_argument("--examples-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/chronological_runs")
    parser.add_argument("--model-kind", choices=("numeric", "text", "fused"), default="fused")
    parser.add_argument("--pretrained", default="distilroberta-base")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--freeze-text-encoder", action="store_true")
    parser.add_argument("--numeric-feature", action="append", default=None)
    parser.add_argument("--dao-bundle", default="")
    parser.add_argument("--voter-bundle", default="")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confirm-full-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_full_training:
        parser.error("Full training requires explicit --confirm-full-training after resource approval.")
    if args.model_kind == "text" and (args.dao_bundle or args.voter_bundle or args.numeric_feature):
        parser.error("Text-only models cannot use numeric or cluster features.")
    if args.model_kind == "numeric" and args.freeze_text_encoder:
        parser.error("Numeric-only models have no text encoder to freeze.")
    run_id = validate_run_id(args.run_id)

    base = project_root()
    config = (base / args.config).resolve()
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=config, extra_path=extra)
    local_only = bool((cfg.get("run") or {}).get("local_data_only", False))
    if args.allow_model_download and local_only:
        parser.error("Model downloads are disabled in local-data-only mode.")
    if args.local_proxy and (not local_only or args.historical_provenance or args.observations or args.observation_provenance):
        parser.error("Local-proxy mode requires local-data-only configuration and no verified provenance files.")
    if not args.local_proxy and not (args.historical_provenance and args.observations and args.observation_provenance):
        parser.error("Verified mode requires historical provenance and observation files.")
    paths = cfg.get("paths", {})
    text_loader = (
        partial(load_text_encoder, args.pretrained, local_only=not args.allow_model_download)
        if args.model_kind in {"text", "fused"} else None
    )

    categorical = []
    if args.dao_bundle:
        categorical.append("dao_cluster")
    if args.voter_bundle:
        categorical.append("voter_cluster")
    numeric = [] if args.model_kind == "text" else (args.numeric_feature or list(DEFAULT_NUMERIC))
    source = (base / paths.get("master_votes_parquet", "data/interim/master_votes.parquet")).resolve()
    metadata = (base / paths.get("proposal_metadata_jsonl", "data/interim/snapshot_proposal_metadata.jsonl")).resolve()
    canonical = (base / paths.get("canonical_vote_events_parquet", "data/processed/canonical_vote_events.parquet")).resolve()
    historical = Path(args.historical_provenance) if args.historical_provenance else None
    observations = Path(args.observations) if args.observations else None
    observation_provenance = Path(args.observation_provenance) if args.observation_provenance else None
    examples = Path(args.examples_dir)
    dao_bundle = Path(args.dao_bundle) if args.dao_bundle else None
    voter_bundle = Path(args.voter_bundle) if args.voter_bundle else None
    if local_only:
        source = require_local_data_path(source, root=base)
        metadata = require_local_data_path(metadata, root=base)
        canonical = require_local_data_path(canonical, root=base)
        examples = require_local_data_path(examples, root=base)
        historical = require_local_data_path(historical, root=base) if historical is not None else None
        observations = require_local_data_path(observations, root=base) if observations is not None else None
        observation_provenance = require_local_data_path(observation_provenance, root=base) if observation_provenance is not None else None
        dao_bundle = require_local_data_path(dao_bundle, root=base) if dao_bundle is not None else None
        voter_bundle = require_local_data_path(voter_bundle, root=base) if voter_bundle is not None else None
    result = train_chronological_run(
        examples_dir=examples,
        raw_source=source,
        metadata_cache=metadata,
        historical_provenance=historical,
        canonical_events=canonical,
        observations=observations,
        observation_provenance=observation_provenance,
        run_root=(base / args.output_root / run_id).resolve(),
        run_id=run_id,
        model_kind=args.model_kind,
        numeric_columns=numeric,
        categorical_columns=categorical,
        text_loader=text_loader,
        text_max_length=args.max_length,
        freeze_text_encoder=args.freeze_text_encoder,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
        dao_bundle_path=dao_bundle,
        voter_bundle_path=voter_bundle,
        local_proxy=args.local_proxy,
    )
    print(f"Chronological validation run: {result}")


if __name__ == "__main__":
    main()
