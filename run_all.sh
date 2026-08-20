#!/usr/bin/env bash
# Reproducible pipeline: stages 01-08, numeric-only ablation, then evaluation.
# Usage: ./run_all.sh [config] [extra-config]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${1:-configs/default.yaml}"
EXTRA_CONFIG="${2:-}"
COMMON_ARGS=(--config "$CONFIG")
if [[ -n "$EXTRA_CONFIG" ]]; then
  COMMON_ARGS+=(--extra-config "$EXTRA_CONFIG") # Optional: server paths, title_body, or lean-data override.
fi

# Optional preflight: uncomment --skip-raw-scan when raw archives are unavailable locally.
# python scripts/10_run_detection_mode.py "${COMMON_ARGS[@]}" --skip-raw-scan

# python scripts/01_run_data_selection.py "${COMMON_ARGS[@]}" # Optional: omit if selected_spaces.json already exists.
# python scripts/02_run_data_expansion.py "${COMMON_ARGS[@]}" # Optional: omit when a merged vote parquet is already configured.
python scripts/03_run_global_cleaning.py "${COMMON_ARGS[@]}" # Enable cleaning.lean_sample only in an explicit experimental override.
python scripts/04_run_dao_metrics.py "${COMMON_ARGS[@]}" # Required by Stage 08 structural DAO clustering.
# python scripts/05_run_feature_selection.py "${COMMON_ARGS[@]}" # Exploratory DAO analysis; not required for Stage 08.
# python scripts/06_run_dao_clustering.py "${COMMON_ARGS[@]}" # Exploratory DAO analysis; not required for Stage 08.
# python scripts/07_run_voter_clustering.py "${COMMON_ARGS[@]}" # Exploratory voter analysis; not required for Stage 08.
python scripts/08_run_behaviour_modelling.py "${COMMON_ARGS[@]}" # Add --reuse-behaviour-csv only when its text_mode metadata matches.
python scripts/08b_run_behaviour_modelling_no_roberta.py "${COMMON_ARGS[@]}" --reuse-behaviour-csv --reuse-split-manifest # Add --reuse-window-cache only after its schema matches.
# When running lean: set paths.model_artifacts_dir_no_roberta and pass --window-cache-dir in EXTRA_CONFIG/CLI to avoid overwriting full-data 08b outputs.
python scripts/09_run_behaviour_evaluation.py "${COMMON_ARGS[@]}" --split test # Replace test with val for validation metrics.
