#!/usr/bin/env bash
# Reproducible pipeline: stages 01-08, numeric-only ablation, then evaluation.
# Usage: ./run_all.sh [config] [data-sources-config] [run-id] [fine-tune|frozen]
set -euo pipefail

if [[ "${ALLOW_LEGACY_VOTER_SPLIT:-0}" != "1" ]]; then
  printf 'The historical runner uses a non-chronological voter split. Set ALLOW_LEGACY_VOTER_SPLIT=1 only for legacy experiments.\n' >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${1:-configs/default.yaml}"
DATA_SOURCES_CONFIG="${2:-configs/data_sources.yaml}"
RUN_ID="${3:-run_$(date -u +%Y%m%dT%H%M%SZ)}"
TEXT_ENCODER_MODE="${4:-fine-tune}"
if [[ "$TEXT_ENCODER_MODE" != "fine-tune" && "$TEXT_ENCODER_MODE" != "frozen" ]]; then
  printf 'Invalid text encoder mode: %s (expected fine-tune or frozen)\n' "$TEXT_ENCODER_MODE" >&2
  exit 2
fi
PIPELINE_ARGS=(--config "$CONFIG" --extra-config "$DATA_SOURCES_CONFIG")
STAGE08_ARGS=(--config "$CONFIG" --data-sources-config "$DATA_SOURCES_CONFIG")
printf '[run_all] Stage 08/08b run ID: %s; text encoder mode: %s\n' "$RUN_ID" "$TEXT_ENCODER_MODE"

# Optional preflight: uncomment --skip-raw-scan when raw archives are unavailable locally.
# python scripts/10_run_detection_mode.py "${PIPELINE_ARGS[@]}" --skip-raw-scan

# python scripts/01_run_data_selection.py "${PIPELINE_ARGS[@]}" # Optional: omit if selected_spaces.json already exists.
# python scripts/02_run_data_expansion.py "${PIPELINE_ARGS[@]}" # Optional: omit when a merged vote parquet is already configured.
# python scripts/03_run_global_cleaning.py "${PIPELINE_ARGS[@]}" # Enable cleaning.lean_sample only in an explicit experimental override.
# python scripts/04_run_dao_metrics.py "${PIPELINE_ARGS[@]}" # Required by Stage 08 structural DAO clustering.
# python scripts/05_run_feature_selection.py "${PIPELINE_ARGS[@]}" # Exploratory DAO analysis; not required for Stage 08.
# python scripts/06_run_dao_clustering.py "${PIPELINE_ARGS[@]}" # Exploratory DAO analysis; not required for Stage 08.
# python scripts/07_run_voter_clustering.py "${PIPELINE_ARGS[@]}" # Exploratory voter analysis; not required for Stage 08.
# Stage 08 and Stage 08b are always run in sequence for the BERT ablation study.
# 08 trains the full RoBERTa + numeric + cluster model; 08b trains the same setup without RoBERTa.
# Keep the dataset and split identical between the two runs; only the BERT inclusion changes.
# The run ID isolates every generated file. 08b consumes the exact 08 dataset and split.
python scripts/08_run_behaviour_modelling.py "${STAGE08_ARGS[@]}" --run-id "$RUN_ID" \
  --text-encoder-mode "$TEXT_ENCODER_MODE" --allow-legacy-voter-split
python scripts/08b_run_behaviour_modelling_no_roberta.py "${STAGE08_ARGS[@]}" --run-id "$RUN_ID" \
  --text-encoder-mode "$TEXT_ENCODER_MODE" --allow-legacy-voter-split
python scripts/09_run_behaviour_evaluation.py "${STAGE08_ARGS[@]}" --run-id "$RUN_ID" \
  --text-encoder-mode "$TEXT_ENCODER_MODE" --split test val
