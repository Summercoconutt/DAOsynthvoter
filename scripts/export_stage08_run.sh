#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf '%s\n' "Usage: ./scripts/export_stage08_run.sh RUN_DIR DEST_PATH [--allow-without-evaluation]" >&2
  printf '%s\n' "Example: ./scripts/export_stage08_run.sh outputs/stage08_runs/full/full_ref /tmp/exports" >&2
  exit 2
fi

ARGS=("$1" "$2")
if [[ $# -eq 3 ]]; then
  if [[ "$3" != "--allow-without-evaluation" ]]; then
    printf '%s\n' "Unknown option: $3" >&2
    exit 2
  fi
  ARGS+=("$3")
fi

python scripts/export_stage08_run.py "${ARGS[@]}"
