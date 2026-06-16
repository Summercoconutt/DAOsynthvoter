# No-RoBERTa behaviour modelling — supervisor runbook

**Branch: use `carlo_dev` only** (not `main`).

```bash
git clone https://github.com/Summercoconutt/DAOsynthvoter.git
cd DAOsynthvoter
git checkout carlo_dev
git pull
```

Stage 08b depends on `carlo_dev` fixes (e.g. `_eval_cluster_id` in `windows.py`, stage 09 evaluation). It was cherry-picked onto `carlo_dev` after development; smoke-tested on this branch.

This branch trains a **numeric-only** voter-choice model to compare against the existing **RoBERTa + numeric + clusters** model (stage `08`). It does **not** modify or retrain the RoBERTa pipeline.

## Prerequisites (should already exist from stage 08)

On the server, after the student’s RoBERTa run:

| Artifact | Default path (relative to `whole_pipeline/`) |
|----------|---------------------------------------------|
| Behaviour dataset CSV | `outputs/tables/behaviour_dataset.csv` |
| Train/val/test split | `outputs/processed/split_manifest.json` |
| RoBERTa model + config | `outputs/models/behaviour_agent2/` |
| RoBERTa training report | `outputs/tables/training_report.md` |

The No-RoBERTa script **reuses the same split** and **the same numeric preprocessor** as RoBERTa so the comparison is fair.

## Environment

```bash
cd whole_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt    # includes duckdb for large CSV streaming
export PYTHONPATH=src
```

GPU is recommended for training (~23M train windows). CPU works but may take days.

## Full-data training (production)

```bash
cd whole_pipeline
export PYTHONPATH=src

python scripts/08b_run_behaviour_modelling_no_roberta.py \
  --config configs/default.yaml \
  --reuse-behaviour-csv \
  --reuse-split-manifest \
  --reuse-preprocessor-from outputs/models/behaviour_agent2/config.json
```

**What this does**

1. Skips rebuilding `behaviour_dataset.csv` if it already exists.
2. Uses existing `split_manifest.json` (does not reload the full 47GB CSV into RAM).
3. Loads `numeric_preprocessor` from the RoBERTa `config.json`.
4. Builds a **memmap window cache** under `outputs/behaviour_modelling/window_cache_no_roberta/` (one-time; ~6GB on disk).
5. Trains `NumericOnlyTimeSeriesClassifier` for 4 epochs (`window=5`, `batch_size=16`, `numeric_only_lr=1e-3`).
6. Writes comparison CSV vs RoBERTa.

### If window cache is already built but training failed

```bash
python scripts/08b_run_behaviour_modelling_no_roberta.py \
  --config configs/default.yaml \
  --reuse-behaviour-csv \
  --reuse-split-manifest \
  --reuse-window-cache
```

### Quick smoke test (small subset)

```bash
python scripts/08b_run_behaviour_modelling_no_roberta.py \
  --config configs/default.yaml \
  --extra-config configs/smoke_no_roberta.yaml \
  --reuse-behaviour-csv \
  --max-train-windows 320 \
  --max-valid-windows 96
```

## Outputs (all names contain `_no_roberta`)

| Output | Path |
|--------|------|
| Model weights | `outputs/behaviour_modelling/agent2_artifacts_no_roberta/model.pt` |
| Config | `outputs/behaviour_modelling/agent2_artifacts_no_roberta/config.json` |
| Training report | `outputs/tables/training_report_no_roberta.md` |
| Preprocessing report | `outputs/tables/preprocessing_report_no_roberta.md` |
| Confusion matrices | `outputs/logs/no_roberta_confusion_valid_epoch*.csv` |
| **RoBERTa vs No-RoBERTa CSV** | `outputs/tables/roberta_vs_no_roberta_comparison.csv` |

The comparison CSV reads RoBERTa metrics from `outputs/tables/training_report.md` or `outputs/models/behaviour_agent2/config.json` — **no RoBERTa retraining required**.

## Expected scale (full dataset)

| Split | Approx. windows |
|-------|-----------------|
| Train | ~22.8M |
| Valid | ~4.9M |

Window cache build: ~1–3 hours (depends on disk). Training: GPU strongly recommended.

## Files added in this branch

- `scripts/08b_run_behaviour_modelling_no_roberta.py` — entrypoint (does not touch `08_run_behaviour_modelling.py`)
- `src/dao_governance/modelling/window_cache.py` — DuckDB streaming + memmap cache
- `src/dao_governance/modelling/model.py` — `NumericOnlyTimeSeriesClassifier`
- `src/dao_governance/modelling/dataset.py` — `NumericWindowDataset`, `MemmapNumericWindowDataset`
- `configs/smoke_no_roberta.yaml` — optional smoke config

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `split_manifest.json` not found | Run stage 08 (RoBERTa) first, or copy manifest from prior run |
| DuckDB CSV parse errors | Already handled via `strict_mode=false`; ensure CSV path in `configs/default.yaml` |
| OOM during cache build | Ensure enough disk (~10GB) for memmap; RAM can stay low thanks to streaming |
| Training very slow | Use CUDA; reduce epochs in config for a trial run |

## Send back to student

Please share:

1. `outputs/tables/roberta_vs_no_roberta_comparison.csv`
2. `outputs/tables/training_report_no_roberta.md`
3. `outputs/behaviour_modelling/agent2_artifacts_no_roberta/config.json`
