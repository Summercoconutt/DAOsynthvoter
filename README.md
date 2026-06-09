# Dissertation whole pipeline

Reviewer-oriented packaging of the DAO governance dissertation code: **DAO selection → data expansion → global cleaning → DAO metrics → feature selection → DAO clustering → voter clustering → behaviour modelling**.

All paths are **relative to this folder** (`whole_pipeline/`). Edit `configs/default.yaml` (and optional split YAML under `config/`) instead of hard-coding Windows paths in scripts.

## Layout

| Path | Role |
|------|------|
| `configs/default.yaml` | Merged configuration (paths, cleaning, clustering, modelling). |
| `config/*.yaml` | Split copies for readability (`paths`, `cleaning`, `clustering`, `modelling`). |
| `scripts/01_…`–`08_…` | One entrypoint per stage; each writes a short report under `outputs/reports/`. |
| `scripts/09_run_behaviour_evaluation.py` | **Comprehensive evaluation** after stage 08 (test/val splits, calibration, cluster metrics). |
| `scripts/10_run_detection_mode.py` | **Detection mode**: scan raw server data (read-only), verify behaviour inputs, optional smoke train. |
| `configs/smoke_behaviour.yaml` | Tiny training budget for GPU smoke test (merged over default). |
| `configs/example_server_raw.yaml` | Template pointing **absolute** raw parquet under `D:/111111/Data` into cleaning (does not write into Data). |
| `run_full_pipeline.py` | Runs stages in order (`--from-stage` / `--to-stage`). |
| `src/dao_governance/` | Validation, behaviour dataset, modelling (copied from parent project). |
| `src/dao_clustering_scripts/` | Snapshot merge + DAO metric tables (`01`–`03`). |
| `src/feature_selection/` | DAO feature screening pipeline. |
| `src/dao_clustering_validation/` | DAO clustering validation + exports (`cluster_assignments.csv`, etc.). |
| `src/voter_clustering/` | Voter-space features + clustering. |
| `src/behaviour_modelling/` | Legacy flat modules (some stages use `dao_governance` instead). |
| `src/data_expansion/` | Expansion / eligible-base scripts (RPC/API — see script headers). |
| `src/participation_rate/` | Participation metrics from Snapshot folders. |
| `src/representative_voter/` | Representative-voter centroids per DAO. |
| `src/pipeline/global_cleaning.py` | Stage 3 global vote cleaning + markdown report. |
| `data/` | `raw` / `interim` / `processed` / `final` placeholders — **do not overwrite immutable raw archives**. |
| `outputs/` | Reports, figures, tables, cluster exports, models. |

## Environment

```powershell
cd path\to\Dissertation\whole_pipeline
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Linux/macOS: `export PYTHONPATH=src`.

## Detection mode (behaviour model sanity check)

Use this when **raw votes live on another disk** (e.g. `D:/111111/Data` from the lab server) and you only want **local cleaning + confirmation that stage 08 runs**.

1. **Read-only scan** of uncleaned parquet under `detection.external_raw_root` (VP negatives / scale — no writes).
2. **Prerequisite check**: `master_with_dao_parquet` + `voter_cluster_assignments_csv` exist and have required columns.
3. **Smoke train** (optional): one short epoch, capped windows — verifies numerics + transformers path.

```text
python scripts/10_run_detection_mode.py
python scripts/10_run_detection_mode.py --smoke-train
python scripts/10_run_detection_mode.py --skip-raw-scan --smoke-train
```

Report: `outputs/reports/detection_mode_report.md`.

## Behaviour model evaluation (after stage 08)

Use the **same voter split** as training (`outputs/processed/split_manifest.json`):

```text
python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split test
python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split val
```

Outputs under `outputs/tables/eval/{test|val}/`:

- `metrics_summary.json`, `metrics_report.md` — macro/weighted F1, balanced accuracy, MCC, kappa, ECE, log loss
- `per_class_metrics.csv`, `classification_report.csv`
- `confusion_matrix_counts.csv` + normalized figure
- `reliability_bins.csv`, `figures/reliability_*.png`
- `metrics_by_dao_cluster.csv`, `metrics_by_voter_cluster.csv`

Do **not** use legacy `src/behaviour_modelling/evaluate.py` for models trained by stage 08.

**Cleaning server exports without touching `Data/`:** set an absolute `paths.master_votes_parquet` in `configs/example_server_raw.yaml` (copy and edit), then:

```text
python scripts/03_run_global_cleaning.py --config configs/default.yaml --extra-config configs/example_server_raw.yaml
```

Cleaned output goes only to `paths.cleaned_master_parquet` inside `whole_pipeline/`.

## Configure

1. Set `paths.snapshot_spaces_root` to your Snapshot `spaces/` directory when building from raw proposal files.
2. Point `paths.master_votes_parquet` / `paths.cleaned_master_parquet` at your merged vote parquet after expansion and cleaning (supports **absolute** paths to raw files on `D:` or other drives).
3. Keep **baseline DAO assignments** for downstream steps aligned with:

   `outputs/cluster_results/no_outliers/cluster_assignments.csv`

   (produced by stage 6; a snapshot can be seeded here for reproducibility.)

## Run

Single stage:

```text
python scripts/03_run_global_cleaning.py --config configs/default.yaml
```

Full pipeline (stops on first error):

```text
python run_full_pipeline.py --config configs/default.yaml
python run_full_pipeline.py --from-stage 4 --to-stage 8 --config configs/default.yaml
```

## Assumptions & TODOs

- **Stage 1**: Final follower/knee logic lives in `Dissertation/data_collection/fetch_spaces.py.ipynb`; this repo validates `data/raw/dao_selection/selected_spaces.json`. Copy CSV/JSON from `Dissertation/data_collection/` if missing.
- **Stage 2**: Expansion scripts differ by API (`RPC_URL`, Snapshot GraphQL). They only **document** outputs here unless you wire commands yourself.
- **Stage 4**: `dao_representative_voter_pipeline.py` defaults to an internal timestamped output folder; copy or symlink the generated `dao_representative_voters_*.csv` to `paths.representative_csv`.
- **Schema drift**: Vote tables must be normalisable to `voter`, `space`, `proposal_id`, `voting_power`, `choice_norm` / `choice` as in the original pipelines. Mixed-type columns are stringified for parquet where needed.
- **Negative voting power**: Default policy is **drop** invalid rows / mask negatives; use `cleaning.negative_voting_power_policy: clip_to_zero` only for robustness experiments (see `configs/default.yaml`).

## Outputs / validation

Each stage writes a markdown summary under `outputs/reports/stage*.md`. Structural cleaning statistics appear in `outputs/reports/stage03_global_cleaning.md` and related tables. Behaviour training writes `outputs/tables/preprocessing_report.md`, `outputs/tables/training_report.md`, and `outputs/reports/stage08_behaviour_modelling.md`.
