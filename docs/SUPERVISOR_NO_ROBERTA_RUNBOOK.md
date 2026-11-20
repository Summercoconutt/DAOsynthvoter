# No-RoBERTa behaviour modelling — supervisor runbook

> Historical voter-split protocol only. Append `--allow-legacy-voter-split` to
> Stage 08/08b commands; these results do not satisfy the new chronological
> protocol. See [the current audit](POINT_IN_TIME_DATA_AUDIT.md).

**Branch: use `carlo_dev` only** (not `main`).

```bash
git clone https://github.com/Summercoconutt/DAOsynthvoter.git
cd DAOsynthvoter
git checkout carlo_dev
git pull
```

Stage 08b depends on `carlo_dev` fixes (e.g. `_eval_cluster_id` in `windows.py`, stage 09 evaluation). It was cherry-picked onto `carlo_dev` after development; smoke-tested on this branch.

This branch trains a **numeric-only** voter-choice model to compare against the existing **RoBERTa + numeric + clusters** model (stage `08`). It does **not** modify or retrain the RoBERTa pipeline.

Stage 08 may use either title-only or title-plus-body text (`behaviour_model.text_mode`). Record that setting with comparison results: 08b intentionally receives no proposal text, while it receives causal prior FOR/AGAINST fractions to retain non-text vote-history information.

## Paired run inputs

The full or lean data-source YAML is authoritative for the cleaned vote parquet and DAO feature table. Stage 08 snapshots those inputs and creates the shared voter split; Stage 08b must use that same run ID and split.

Choose a distinct run ID for every experiment. Full and lean artifacts are kept under separate variant folders.

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

## Full-data paired training (production)

```bash
cd whole_pipeline
export PYTHONPATH=src

RUN_ID="full_20261008_a"
python scripts/08_run_behaviour_modelling.py \
  --config configs/default.yaml \
  --data-sources-config configs/data_sources.yaml \
  --run-id "$RUN_ID"
python scripts/08b_run_behaviour_modelling_no_roberta.py \
  --config configs/default.yaml \
  --data-sources-config configs/data_sources.yaml \
  --run-id "$RUN_ID" \
  --reuse-split-manifest
python scripts/09_run_behaviour_evaluation.py \
  --config configs/default.yaml \
  --data-sources-config configs/data_sources.yaml \
  --run-id "$RUN_ID" \
  --split test
./scripts/export_stage08_run.sh "outputs/stage08_runs/full/$RUN_ID" /tmp/exports
```

**What this does**

1. Validates and consumes the behavior dataset, DAO feature table, and split recorded by Stage 08 for this run.
2. Builds causal `prior_frac_for` and `prior_frac_against` inputs from earlier votes in the same `(voter, space)` sequence; abstentions remain in the denominator.
3. Fits the 08b-specific numeric preprocessor and stores any **memmap window cache** under this run’s `08b/window_cache/` directory.
4. Trains `NumericOnlyTimeSeriesClassifier` for 4 epochs (`window=5`, `batch_size=16`, `numeric_only_lr=1e-3`).
5. Writes run-local training/preprocessing reports and a validation comparison against the paired Stage 08 model.

The 08b tensor has `feat_dim=10`: six numeric/history inputs plus four time encodings.

### If window-cache training was interrupted

```bash
python scripts/08b_run_behaviour_modelling_no_roberta.py \
  --config configs/default.yaml \
  --data-sources-config configs/data_sources.yaml \
  --run-id full_20261008_a \
  --resume-stage \
  --reuse-split-manifest \
  --reuse-window-cache
```

The cache records hashes for its source dataset, split manifest, and fitted preprocessor as well as its numeric schema and window size. A mismatch is rejected. `--resume-stage` is only for an interrupted stage in the same run; it cannot rerun a completed 08b stage.

### Quick smoke test (small subset)

```bash
RUN_ID="smoke_no_roberta_01"
python scripts/08_run_behaviour_modelling.py \
  --config configs/default.yaml \
  --data-sources-config configs/data_sources.yaml \
  --extra-config configs/smoke_behaviour.yaml \
  --run-id "$RUN_ID"
python scripts/08b_run_behaviour_modelling_no_roberta.py \
  --config configs/default.yaml \
  --data-sources-config configs/data_sources.yaml \
  --extra-config configs/smoke_behaviour.yaml \
  --run-id "$RUN_ID"
```

## Outputs

All outputs are under `outputs/stage08_runs/<variant>/<run-id>/`. The `08b/` subdirectory contains numeric-only weights/config, enriched data, reports, logs, validation comparison, and cache; `evaluation/08b/test/` contains held-out test outputs. `evaluation/reports/within_run_model_comparison.csv` compares both models on the same test split.

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
| Stage 08 run or split not found | Run Stage 08 first with the same source YAML and run ID |
| DuckDB CSV parse errors | Already handled via `strict_mode=false`; check the run-local enriched CSV |
| OOM during cache build | Ensure enough disk (~10GB) for memmap; RAM can stay low thanks to streaming |
| Training very slow | Use CUDA; reduce epochs in config for a trial run |

## Send back to student

Please share:

1. The full or lean Stage 08 export ZIP produced by `scripts/export_stage08_run.sh`
2. `run_manifest.json` and the paired test outputs under `evaluation/`
