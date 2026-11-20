# Dissertation whole pipeline

Reviewer-oriented packaging of the DAO governance dissertation code: **DAO selection → data expansion → global cleaning → DAO metrics → feature selection → DAO clustering → voter clustering → behaviour modelling**.

> **Protocol status:** Existing Stage 08/08b models use a historical voter split,
> not the proposal-disjoint point-in-time protocol. They require explicit legacy
> opt-in and must not be reported as leakage-safe for the new task. The new
> chronological workflow is local-data-only: use `configs/local_data_only.yaml`
> and do not query external data services. Historical proposal and observation
> provenance remain unverified in the available local files; do not describe
> them as verified point-in-time facts. Full training also requires separate
> resource approval. See the [data audit](docs/POINT_IN_TIME_DATA_AUDIT.md).

All paths are **relative to this folder** (`whole_pipeline/`). Edit `configs/default.yaml` (and optional split YAML under `config/`) instead of hard-coding Windows paths in scripts.

## Layout

| Path | Role |
|------|------|
| `configs/default.yaml` | Merged configuration (paths, cleaning, clustering, modelling). |
| `config/*.yaml` | Split copies for readability (`paths`, `cleaning`, `clustering`, `modelling`). |
| `scripts/01_…`–`08_…` | One entrypoint per stage; each writes a short report under `outputs/reports/`. |
| `scripts/08b_run_behaviour_modelling_no_roberta.py` | **Numeric-only** behaviour model (no RoBERTa) for comparison — **`carlo_dev` branch only**; see `docs/SUPERVISOR_NO_ROBERTA_RUNBOOK.md`. |
| `scripts/09_run_behaviour_evaluation.py` | **Comprehensive evaluation** after stage 08 (test/val splits, calibration, cluster metrics). |
| `scripts/10_run_detection_mode.py` | **Detection mode**: scan raw server data (read-only), verify behaviour inputs, optional smoke train. |
| `docs/SUPERVISOR_LEAKAGE_SAFE_GUIDE.md` | **Leakage-safe Stage 8** — detailed supervisor runbook (`carlo_dev`). |
| `docs/Feature_Specification.md` | Full feature catalogue: Stage 08 `feat_dim=8`; Stage 08b `feat_dim=10`. |
| `docs/Leakage_Audit.md` | Leakage issue register, fixes, severity matrix, verification checklist. |
| `docs/LEAKAGE_SAFE_RUNBOOK.md` | Short English runbook for leakage-safe behaviour modelling. |
| `configs/smoke_behaviour.yaml` | Tiny training budget for GPU smoke test (merged over default). |
| `configs/example_server_raw.yaml` | Template pointing **absolute** raw parquet under `D:/111111/Data` into cleaning (does not write into Data). |
| `run_full_pipeline.py` | Runs stages 01–08 in order (`--from-stage` / `--to-stage`). |
| `run_all.sh` | Runs stages 01–08, the 08b ablation, and Stage 09 evaluation in order. |
| `src/dao_governance/` | Validation, behaviour dataset, modelling (copied from parent project). |
| `src/dao_clustering_scripts/` | Snapshot merge + DAO metric tables (`01`–`03`). |
| `src/feature_selection/` | DAO feature screening pipeline. |
| `src/dao_clustering_validation/` | DAO clustering validation + exports (`cluster_assignments.csv`, etc.). |
| `src/voter_clustering/` | Voter-space features + clustering. |
| `src/deprecated/behaviour_modelling/` | Pre-leakage-fix legacy modules, retained only for historical reference. |
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
2. **Prerequisite check**: `cleaned_master_parquet` + `dao_feature_table_csv` exist (Stage 7 voter clusters optional, EDA only).
3. **Smoke train** (optional): one short epoch, capped windows — verifies numerics + transformers path.

```text
python scripts/10_run_detection_mode.py
python scripts/10_run_detection_mode.py --smoke-train --allow-legacy-voter-split
python scripts/10_run_detection_mode.py --skip-raw-scan --smoke-train --allow-legacy-voter-split
```

Report: `outputs/reports/detection_mode_report.md`.

## Behaviour model evaluation (after stages 08 and 08b)

Stage 09 evaluates both models from the same immutable run and saved voter split. Record all desired splits in one invocation:

```text
python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --data-sources-config configs/data_sources.yaml --run-id "$RUN_ID" --split test val
```

Outputs are under `outputs/stage08_runs/<variant>/<run-id>/evaluation/{08|08b}/{test|val}/`:

- `metrics_summary.json`, `metrics_report.md` — macro/weighted F1, balanced accuracy, MCC, kappa, ECE, log loss
- `per_class_metrics.csv`, `classification_report.csv`
- `confusion_matrix_counts.csv` + normalized figure
- `reliability_bins.csv`, `figures/reliability_*.png`
- `metrics_by_dao_cluster.csv`, `metrics_by_voter_cluster.csv`

Do **not** use code under `src/deprecated/behaviour_modelling/` for models trained by Stage 08.

## Integrated Gradients / SHAP analysis

Stage 08 and Stage 08b save model weights, but they do not generate attribution
values. To reproduce a trained model and run Integrated Gradients or SHAP, keep
the following files from the same training run.

### Stage 08: RoBERTa + numeric model

Use `outputs/stage08_runs/<variant>/<run-id>/08/` for `model.pt`, `config.json`, and the exact `tokenizer/`; use the matching run's `inputs/behaviour_dataset.csv`, `shared/split_manifest.json`, and `shared/predictive_clusters/` to reconstruct training windows.
- `src/dao_governance/modelling/model.py`, `dataset.py`, `preprocess.py`, and
  `windows.py` — model definition, tokenisation/collation, preprocessing, and
  window construction.

For text explanations, use embedding-level Integrated Gradients or a custom
SHAP wrapper; integer token IDs are not differentiable. Numeric features can be
attributed directly.

### Stage 08b: numeric-only model

Use `outputs/stage08_runs/<variant>/<run-id>/08b/` for weights, config, causal-history data, and any window cache. Its input dataset and split are the same run-scoped files listed above.
- `src/dao_governance/modelling/model.py`, `dataset.py`, `preprocess.py`, and
  `windows.py` — model definition and exact numeric window construction.

Stage 08b is directly compatible with feature-level Integrated Gradients or
numeric SHAP wrappers. In both cases, use the saved preprocessing statistics,
feature ordering, temporal window size, and a documented baseline/background
dataset.

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

### Local proposal metadata

The local-only configuration uses `data/interim/snapshot_votes.parquet` and
the proposal metadata cache already under `data/interim/`. No external lookup
or verification is performed. The cache was collected earlier and is not
independent proof of historical full-text, option, or observation-time
availability.

The locally reconciled cache is
`data/interim/snapshot_proposal_metadata_local.jsonl`. If it has not already
been created, run `python scripts/bind_local_metadata.py`; the command reads
only `data/` and refuses to overwrite an existing cache. Reconciliation checks
local identifiers and stored text prefixes, not historical availability.

### Local retrospective proxy

The prepared local dataset uses the cached proposal metadata as an unverified
historical proxy and treats recorded vote time as an unverified observation-time
proxy. It is chronological under those assumptions, but it is **not verified
point-in-time evidence**. The canonical events, split, and examples already
created from local files are under `data/processed/`. To reproduce them at new
configured output paths:

```text
python scripts/03_run_global_cleaning.py --extra-config configs/local_data_only.yaml --point-in-time --local-proxy
python scripts/create_chronological_split.py --extra-config configs/local_data_only.yaml --local-proxy --output data/processed/chronological_split_local_proxy.json
python scripts/create_point_in_time_examples.py --extra-config configs/local_data_only.yaml --local-proxy --split-manifest data/processed/chronological_split_local_proxy.json --output-dir data/processed/chronological_examples_local_proxy
```

These commands refuse to overwrite existing outputs. A separate verified mode
remains available only if suitable historical proposal and observation-time
evidence is already present locally; the current cache alone does not provide
that evidence.

The default two-million-event in-memory limit fails explicitly rather than
truncating data. `--include-target-proposal` creates a separately labelled
same-proposal-history sensitivity dataset; the default excludes that history.
Optional `--dao-bundle` and `--voter-bundle` arguments use the predictive
cluster artifacts described below.

After reviewing the proxy assumptions and approving compute resources, train each
desired matched model with `--model-kind numeric`, `text`, or `fused`. The
default fused model fine-tunes `distilroberta-base` on title and body using
locally available model files; downloads are disabled in local-data-only mode:

```text
python scripts/08_run_chronological_model.py --extra-config configs/local_data_only.yaml --local-proxy --examples-dir data/processed/chronological_examples_local_proxy --run-id fused-01 --model-kind fused --confirm-full-training
```

Pass matching `--dao-bundle` and/or `--voter-bundle` paths to include those
categorical features. The run writes a validation-selected checkpoint and
matched validation predictions, never fitting on the test partition.

Compare runs on validation, freeze the selected setup and any pre-specified
comparisons, then evaluate the held-out test once:

```text
python scripts/09_run_chronological_evaluation.py compare-validation --run fused=outputs/chronological_runs/fused-01 --run numeric=outputs/chronological_runs/numeric-01 --output outputs/chronological/validation-comparison.json
python scripts/09_run_chronological_evaluation.py freeze --validation-report outputs/chronological/validation-comparison.json --selected-model fused --selected-run outputs/chronological_runs/fused-01 --additional-run numeric=outputs/chronological_runs/numeric-01 --output outputs/chronological/selection.json
python scripts/09_run_chronological_evaluation.py test --selection outputs/chronological/selection.json --validation-report outputs/chronological/validation-comparison.json --selected-run outputs/chronological_runs/fused-01 --additional-run numeric=outputs/chronological_runs/numeric-01 --output-dir outputs/chronological/test-evaluation
```

Test output includes fixed-three-class metrics, calibration, history strata,
proposal-level bootstrap intervals, and aligned predictions. Integrated
Gradients helpers for numeric inputs and text embeddings are in
`src/dao_governance/modelling/point_in_time_explain.py`; baselines must be
derived from training data and attribution completeness should be checked.
To compare a separately prepared same-proposal-history sensitivity run against
the primary history mode, add `--compare-history-sensitivity` to
`compare-validation`; mixed history modes are otherwise rejected.
Pass `--content-disjoint-sensitivity` to the test command for a separately
labelled evaluation excluding cross-split duplicate proposal text.

The historical lean sample is outcome-selected. To evaluate it only as a
retrospective sensitivity, map its audit to original proposal IDs and register
the membership **when freezing** the selection:

```text
python scripts/build_lean_sensitivity.py --extra-config configs/local_data_only.yaml --lean-audit data/processed/proposal_lean_audit.csv --output data/processed/lean-membership-local-proxy.parquet
```

Pass `--lean-membership data/processed/lean-membership-local-proxy.parquet` to the
`freeze` command and again to a separate `test` output. The same option is
required when explaining or reporting that lean evaluation. Its results are
not an unbiased estimate for the full eligible population.

After the frozen test evaluation, generate case-based explanations and a
traceable report:

```text
python scripts/11_explain_chronological_model.py --selection outputs/chronological/selection.json --validation-report outputs/chronological/validation-comparison.json --selected-run outputs/chronological_runs/fused-01 --additional-run numeric=outputs/chronological_runs/numeric-01 --test-evaluation-dir outputs/chronological/test-evaluation --output-dir outputs/chronological/explanations
python scripts/12_write_chronological_report.py --selection outputs/chronological/selection.json --validation-report outputs/chronological/validation-comparison.json --selected-run outputs/chronological_runs/fused-01 --additional-run numeric=outputs/chronological_runs/numeric-01 --test-evaluation-dir outputs/chronological/test-evaluation --explanation-dir outputs/chronological/explanations --output outputs/chronological/final-report.md
```

Explanations use training examples as backgrounds and report completeness,
high/low-attribution ablations, background variation, and classification-head
randomization. Global importance is a configurable proposal-weighted sample,
not an exact population estimate; explanations are not causal.

Export a verified chronological checkpoint with
`python scripts/export_chronological_run.py RUN_DIR DESTINATION.zip`. By default
the ZIP omits external inputs and is not standalone. `--include-sensitive-inputs`
adds the recorded inputs; `--include-test` additionally requires the frozen
`--selection`, `--validation-report`, and any `--additional-run` arguments.
Optional `--test-evaluation-dir`, `--explanation-dir`, and `--final-report`
include the corresponding verified context. Archives with inputs can contain
voter data and local provenance; share them only with authorized recipients.
Additional model runs must be exported separately. After extraction,
`exported_input_overrides` and `load_validation_checkpoint` can verify and use
the relocated inputs.

### Point-in-time clustering components

`build_cluster_fit_profiles` in `src/dao_governance/features/asof.py` builds
completed-proposal snapshots from observation-time-sorted events. For this
local experiment, use an `ObservationProvenance` explicitly marked
`verified=False, local_proxy=True`.
Persist each DAO or voter snapshot with `save_fit_profiles` in
`src/dao_governance/clustering/predictive.py`, using the same chronological
split manifest and an explicit observation-time provenance.

For the prepared local proxy, choose a warm-up cutoff before the training
cutoff and generate both profile tables without contacting external services:

```text
python scripts/create_local_cluster_profiles.py --split-manifest data/processed/chronological_split_local_proxy.json --fit-cutoff '<WARMUP_UTC>' --output-dir data/processed/cluster_profiles_run
```

For DAO-cluster voter scope, first fit the DAO bundle from an earlier warm-up,
then generate a later voter snapshot and fit its grouped voter bundle.

Stages 6 and 7 accept those persisted profiles in predictive mode:

```text
python scripts/06_run_dao_clustering.py --extra-config configs/local_data_only.yaml --predictive-profiles data/processed/cluster_profiles_run/dao.parquet --split-manifest data/processed/chronological_split_local_proxy.json --cluster-feature dao_prior_count --cluster-output data/processed/dao_clusters.pkl
python scripts/07_run_voter_clustering.py --extra-config configs/local_data_only.yaml --predictive-profiles data/processed/cluster_profiles_run/voters.parquet --split-manifest data/processed/chronological_split_local_proxy.json --cluster-feature voter_dao_prior_count --cluster-scope global --cluster-output data/processed/voter_clusters.pkl
```

Choose `--cluster-scope global`, `dao_cluster`, or `individual_dao` explicitly.
All modes retain separate voter/DAO profiles. `dao_cluster` additionally
requires `--dao-bundle data/processed/dao_clusters.pkl`, fitted on the same split at an earlier
warm-up cutoff; global and individual-DAO modes do not require DAO clustering.
Sparse, unseen, and pre-fit cases receive an explicit unknown category.
The current implementation uses one frozen warm-up model per scope, not
rolling out-of-fold refits. Global scope pools profiles across all DAOs in the
permitted fit history; choose a cutoff early enough that training examples can
receive cluster assignments. Fitting only at the end of training leaves all
training cluster categories unknown and is rejected for predictive modelling.
Cluster bundles are pickle files and must only be loaded from trusted runs.
Use `--cluster-algorithm gaussian_mixture` for the alternative model and
`--min-cluster-size` to reject undersized clusters. Bundle metadata records
training cluster sizes, sampled silhouette, seed stability, and centroids;
choose settings using validation rather than training inertia alone.

### Historical voter-split runner

Single stage:

```text
python scripts/03_run_global_cleaning.py --config configs/default.yaml
```

Full pipeline (stops on first error):

```text
python run_full_pipeline.py --config configs/default.yaml --allow-legacy-voter-split
python run_full_pipeline.py --from-stage 4 --to-stage 8 --config configs/default.yaml --allow-legacy-voter-split
ALLOW_LEGACY_VOTER_SPLIT=1 ./run_all.sh configs/default.yaml
```

## Assumptions & TODOs

- **Stage 1**: Final follower/knee logic lives in `Dissertation/data_collection/fetch_spaces.py.ipynb`; this repo validates `data/raw/dao_selection/selected_spaces.json`. Copy CSV/JSON from `Dissertation/data_collection/` if missing.
- **Stage 2**: Expansion scripts differ by API (`RPC_URL`, Snapshot GraphQL). They only **document** outputs here unless you wire commands yourself.
- **Stage 4**: `dao_representative_voter_pipeline.py` defaults to an internal timestamped output folder; copy or symlink the generated `dao_representative_voters_*.csv` to `paths.representative_csv`.
- **Schema drift**: Vote tables must be normalisable to `voter`, `space`, `proposal_id`, `voting_power`, `choice_norm` / `choice` as in the original pipelines. Mixed-type columns are stringified for parquet where needed.
- **Negative voting power**: Default policy is **drop** invalid rows / mask negatives; use `cleaning.negative_voting_power_policy: clip_to_zero` only for robustness experiments (see `configs/default.yaml`).

## Outputs / validation

Stages 01–07 retain their configured outputs. Stage 08, 08b, Stage 09, and exports are isolated under `outputs/stage08_runs/<variant>/<run-id>/`; each run contains its source/config provenance, paired model artifacts, matching split and datasets, reports/logs, and (after Stage 09) held-out evaluation for both models. The run ID is immutable; use a new ID for every experiment.

Run a paired full sample with:

```bash
ALLOW_LEGACY_VOTER_SPLIT=1 ./run_all.sh configs/default.yaml configs/data_sources.yaml full_20261008_a
./scripts/export_stage08_run.sh outputs/stage08_runs/full/full_20261008_a /tmp/exports
```

For lean, pass `configs/data_sources_lean.yaml` and a separate run ID to both commands. The exporter verifies the run manifest and refuses incomplete or overwritten bundles.
