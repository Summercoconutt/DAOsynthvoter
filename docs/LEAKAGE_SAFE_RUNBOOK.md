# Leakage-safe behaviour modelling (Stage 8)

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
python scripts/08_run_behaviour_modelling.py --config configs/default.yaml
python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml
python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split test
```

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

- `outputs/tables/behaviour_dataset.csv` — vote-level, no clusters
- `outputs/tables/behaviour_dataset_with_clusters.csv` — train-only cluster IDs (for 08b cache)
- `outputs/models/predictive_clusters/cluster_bundle.pkl` — fitted models
- `outputs/processed/split_manifest.json` — voter split
- `outputs/models/behaviour_agent2/config.json` — includes `enriched_behaviour_csv` for Stage 09

## Config (`configs/default.yaml`)

```yaml
behaviour_model:
  use_dao_clusters: true
  use_voter_clusters: true
```

Set either to `false` for ablation.

## Automated checks

```powershell
python -m pytest tests/test_leakage_safe_pipeline.py -v
```

## Large dataset (08b cache)

Run Stage 08 first to create the matching split manifest and enriched CSV. Stage 08b then fits its own causal-history preprocessor and cache. Then:

```powershell
python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml `
  --reuse-split-manifest --reuse-window-cache
```

Delete `outputs/behaviour_modelling/window_cache_no_roberta/` after code changes to force cache rebuild.
