#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./scripts/export_stage08_run.sh TAG DEST_PATH [--include-window-cache]

Collect the relevant Stage 08 and Stage 08b outputs for a named run and zip them
into a portable archive.

Examples:
  ./scripts/export_stage08_run.sh full_ref /tmp/downloads
  ./scripts/export_stage08_run.sh full_ref /tmp/downloads/stage08_run.zip
  ./scripts/export_stage08_run.sh full_ref /tmp/downloads --include-window-cache

The TAG must match the value passed to --save-behaviour-copy when the run was
launched, for example: --save-behaviour-copy full_ref
EOF
  exit 1
}

INCLUDE_CACHE=0
if [[ $# -lt 2 ]]; then
  usage
fi
if [[ $# -gt 2 && "${3:-}" == "--include-window-cache" ]]; then
  INCLUDE_CACHE=1
fi

TAG="${1:-}"
DEST_PATH="${2:-}"
if [[ -z "$TAG" || -z "$DEST_PATH" ]]; then
  usage
fi

if [[ "$DEST_PATH" != /* ]]; then
  DEST_PATH="$ROOT/$DEST_PATH"
fi
if command -v realpath >/dev/null 2>&1; then
  DEST_PATH="$(realpath -m "$DEST_PATH")"
else
  DEST_PATH="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$DEST_PATH")"
fi

BUNDLE_NAME="stage08_08b_${TAG}"
if [[ "$DEST_PATH" == *.zip ]]; then
  OUT_DIR="$(dirname "$DEST_PATH")"
  ZIP_PATH="$DEST_PATH"
else
  OUT_DIR="$DEST_PATH"
  ZIP_PATH="$OUT_DIR/${BUNDLE_NAME}.zip"
fi
mkdir -p "$OUT_DIR"

bundle_dir="$(mktemp -d "${TMPDIR:-/tmp}/${BUNDLE_NAME}.XXXXXX")"
trap 'rm -rf "$bundle_dir"' EXIT
bundle_root="$bundle_dir/$BUNDLE_NAME"
mkdir -p "$bundle_root"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
  fi
}

# Run-level manifest and notes.
cat > "$bundle_root/README.txt" <<EOF
Stage 08 / 08b export bundle
Run tag: ${TAG}
Generated: $(date -u +'%Y-%m-%dT%H:%M:%SZ')
Project root: ${ROOT}
EOF

# Important Stage 08 outputs.
copy_if_exists "$ROOT/outputs/models/behaviour_agent2" "$bundle_root/08/behaviour_agent2"
copy_if_exists "$ROOT/outputs/models/predictive_clusters" "$bundle_root/08/predictive_clusters"
copy_if_exists "$ROOT/outputs/processed/split_manifest.json" "$bundle_root/08/split_manifest.json"
copy_if_exists "$ROOT/outputs/tables/preprocessing_report.md" "$bundle_root/08/preprocessing_report.md"
copy_if_exists "$ROOT/outputs/tables/training_report.md" "$bundle_root/08/training_report.md"
copy_if_exists "$ROOT/outputs/tables/behaviour_dataset_quality.md" "$bundle_root/08/behaviour_dataset_quality.md"

# Important Stage 08b outputs.
copy_if_exists "$ROOT/outputs/behaviour_modelling/agent2_artifacts_no_roberta" "$bundle_root/08b/agent2_artifacts_no_roberta"
if [[ "$INCLUDE_CACHE" -eq 1 ]]; then
  copy_if_exists "$ROOT/outputs/behaviour_modelling/window_cache_no_roberta" "$bundle_root/08b/window_cache_no_roberta"
else
  printf '%s\n' "[export] Skipping large cache: outputs/behaviour_modelling/window_cache_no_roberta (use --include-window-cache to include it)" >&2
fi
copy_if_exists "$ROOT/outputs/tables/preprocessing_report_no_roberta.md" "$bundle_root/08b/preprocessing_report_no_roberta.md"
copy_if_exists "$ROOT/outputs/tables/training_report_no_roberta.md" "$bundle_root/08b/training_report_no_roberta.md"
copy_if_exists "$ROOT/outputs/tables/roberta_vs_no_roberta_comparison.csv" "$bundle_root/08b/roberta_vs_no_roberta_comparison.csv"

# Tagged source CSV snapshots created by the stage scripts.
for src in \
  "$ROOT/outputs/tables/behaviour_dataset_with_clusters_08_${TAG}.csv" \
  "$ROOT/outputs/tables/behaviour_dataset_with_clusters_08b_${TAG}.csv" \
  "$ROOT/outputs/tables/behaviour_dataset_with_clusters_08_${TAG}.meta.json" \
  "$ROOT/outputs/tables/behaviour_dataset_with_clusters_08b_${TAG}.meta.json"
do
  if [[ -e "$src" ]]; then
    rel="${src#$ROOT/}"
    mkdir -p "$bundle_root/$(dirname "$rel")"
    cp -a "$src" "$bundle_root/$rel"
  fi
done

# Optional: copy any adjacent reports that were generated in the same run.
for src in \
  "$ROOT/outputs/behaviour_modelling/training_report.md" \
  "$ROOT/outputs/behaviour_modelling/training_report_no_roberta.md" \
  "$ROOT/outputs/behaviour_modelling/behaviour_dataset_quality.md" \
  "$ROOT/outputs/behaviour_modelling/behaviour_dataset_quality_no_roberta.md"; do
  if [[ -e "$src" ]]; then
    rel="${src#$ROOT/}"
    mkdir -p "$bundle_root/$(dirname "$rel")"
    cp -a "$src" "$bundle_root/$rel"
  fi
done

found=0
for candidate in \
  "$ROOT/outputs/models/behaviour_agent2" \
  "$ROOT/outputs/behaviour_modelling/agent2_artifacts_no_roberta" \
  "$ROOT/outputs/tables/behaviour_dataset_with_clusters_08_${TAG}.csv" \
  "$ROOT/outputs/tables/behaviour_dataset_with_clusters_08b_${TAG}.csv" \
  "$ROOT/outputs/tables/preprocessing_report.md" \
  "$ROOT/outputs/tables/training_report.md" \
  "$ROOT/outputs/tables/preprocessing_report_no_roberta.md" \
  "$ROOT/outputs/tables/training_report_no_roberta.md"; do
  if [[ -e "$candidate" ]]; then
    found=1
    break
  fi
done

if [[ "$found" -eq 0 ]]; then
  echo "No Stage 08 / 08b outputs found for tag '$TAG' under $ROOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$ZIP_PATH")"
(cd "$bundle_dir" && zip -rq "$ZIP_PATH" "$BUNDLE_NAME") || {
  echo "Failed to create archive at $ZIP_PATH" >&2
  exit 1
}

echo "Created bundle: $ZIP_PATH"
