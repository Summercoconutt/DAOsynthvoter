# Leakage-safe behaviour modelling (Stage 8)

> Historical voter-split protocol only: this is not point-in-time or
> proposal-disjoint. Append `--allow-legacy-voter-split` to Stage 08/08b and
> detection smoke commands. See [the current audit](POINT_IN_TIME_DATA_AUDIT.md).

Stage 8 no longer uses Stage 6/7 cluster CSVs. It fits **train-only structural clusters** internally.

**Branch:** `carlo_dev` on https://github.com/Summercoconutt/DAOsynthvoter

**Supervisor guide (detailed):** `docs/SUPERVISOR_LEAKAGE_SAFE_GUIDE.md`

## Server run (after stages 3–4 minimum)

```powershell
cd D:\DAOsynthvoter
$env:PYTHONPATH = "src"

# Required inputs:
# - data/processed/votes_cleaned.parquet  (stage 03)
# - data/processed/dao_feature_table.csv  (stage 04)

python scripts/10_run_detection_mode.py --skip-raw-scan
python scripts/08_run_behaviour_modelling.py --config configs/default.yaml --data-sources-config configs/data_sources.yaml --run-id full_20261008_a
python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml --data-sources-config configs/data_sources.yaml --run-id full_20261008_a
python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --data-sources-config configs/data_sources.yaml --run-id full_20261008_a --split test
./scripts/export_stage08_run.sh outputs/stage08_runs/full/full_20261008_a /tmp/exports
```

For the lean sample, use `--data-sources-config configs/data_sources_lean.yaml` for all three commands and a distinct run ID. Add a separate `--extra-config` only for model/cleaning overrides. A run ID is immutable: every model, split, dataset, cache, report, and evaluation output is stored beneath `outputs/stage08_runs/<variant>/<run-id>/`. Stage 08b verifies and reuses the Stage 08 dataset and split; do not copy or regenerate either between the paired runs.

## What changed

| Item | Before | After |
|------|--------|-------|
| `aligned_with_majority`, `vp_share` | Model inputs | **Removed** |
| Windows | Group by `voter` | Group by `(voter, space)` |
| Clusters | Stage 6/7 global, label-heavy | **Train-only structural** fit in Stage 8 |
| Stage 7 output | Fed behaviour model | **Exploratory / EDA only** |
| Stage 08 `feat_dim` | 10 | **8** — retrain required |
| Stage 08b `feat_dim` | N/A | **10** — six numeric/history plus four time features |

## Outputs

- `outputs/stage08_runs/<variant>/<run-id>/run_manifest.json` — source hashes, config/runtime provenance, and artifact inventory
- `inputs/behaviour_dataset.csv` — common vote-level Stage 08/08b model input
- `shared/split_manifest.json` and `shared/predictive_clusters/` — paired split and fitted Stage 08 clusters
- `08/` and `08b/` — separate model weights, configs, enriched datasets, reports, and logs
- `evaluation/08/test/` and `evaluation/08b/test/` — held-out test predictions, metrics, calibration, and subgroup reports
- `evaluation/reports/within_run_model_comparison.csv` — paired held-out comparison

## Config (`configs/default.yaml`)

```yaml
behaviour_model:
  use_dao_clusters: true
  use_voter_clusters: true
```

Set either to `false` for ablation.

## Text encoder efficiency

Stage 08 pre-tokenizes each distinct window text once and dynamically pads each
batch to its longest sequence. The default `fine-tune` mode preserves the
existing behavior: the pretrained encoder is updated during training.

To freeze the pretrained encoder and cache its unique text embeddings for the
training run, use:

```powershell
python scripts/08_run_behaviour_modelling.py --config configs/default.yaml `
  --data-sources-config configs/data_sources.yaml --run-id frozen_trial_01 `
  --text-encoder-mode frozen
```

Use `--text-encoder-mode fine-tune` to explicitly select the default. Frozen
mode changes the experiment because the encoder weights are no longer trained;
the cached embeddings are stored temporarily and removed when training ends.
When selecting the mode with this command-line option, pass the same
`--text-encoder-mode` value to Stage 08b and Stage 09 so their run-identity
checks use the effective Stage 08 configuration. If the mode is set in the
configuration file instead, no additional option is needed for those stages.

## Automated checks

```powershell
python -m pytest tests/test_text_encoder_efficiency.py tests/test_leakage_safe_pipeline.py -v
```

## Large dataset (08b cache)

Run Stage 08 first to create the matching split manifest and enriched CSV. Stage 08b then fits its own causal-history preprocessor and cache. Then:

```powershell
python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml `
  --data-sources-config configs/data_sources.yaml --run-id full_20261008_a `
  --reuse-split-manifest --reuse-window-cache
```

If Stage 08 used a `--text-encoder-mode` command-line override, pass that same
option to Stage 08b and Stage 09.
The cache is under that run’s `08b/window_cache/`; reuse is accepted only when dataset, split, preprocessor, feature schema, and window-size hashes match. Start a new run ID after changing source data or modelling code.
