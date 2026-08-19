# Supervisor Runbook — Leakage-Safe Behaviour Modelling (`carlo_dev`)

This guide is for supervisors and reviewers reproducing **Stage 8 behaviour prediction** on the lab server.

**Repository:** https://github.com/Summercoconutt/DAOsynthvoter/tree/carlo_dev

Use the **`carlo_dev` branch** only — not `main` (`main` lacks Stage 09 evaluation, Stage 10 detection mode, and the leakage fixes described here).

---

## 1. What changed (and why you must retrain)

The previous version had **data leakage** that inflated test metrics and undermined thesis conclusions. The fixes on `carlo_dev` are:

| Item | Before | After |
|------|--------|-------|
| `aligned_with_majority`, `vp_share` | Model numeric inputs | **Removed** from model inputs |
| Sliding-window grouping | By `voter` only | By **`(voter, space)`** |
| DAO / voter clusters | Stage 6/7 on full data + label-heavy stats | **Fit on training voters only**, then assigned to val/test |
| Stage 7 output | Fed into behaviour model | **Exploratory / EDA only** — not used in Stage 8 |
| Numeric feature dimension | 10 (included leaky columns) | **8** (4 numeric + 4 time encodings) |

**Important:** Checkpoints trained with the old code (`feat_dim=10`) are **incompatible**. You must re-run Stage 08.

---

## 2. Environment setup

```bash
git clone https://github.com/Summercoconutt/DAOsynthvoter.git
cd DAOsynthvoter
git checkout carlo_dev
git pull

python3 -m pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src"    # Windows PowerShell: $env:PYTHONPATH = "src"
```

- Python **3.10+**
- GPU optional; CUDA recommended for RoBERTa; use `configs/smoke_behaviour.yaml` for a quick smoke test
- Stage 08b (numeric-only baseline, no RoBERTa) requires **duckdb** (listed in `requirements.txt`)

---

## 3. Data and path configuration

Raw vote data usually lives on a server disk (e.g. `D:/111111/Data`). **Do not** write cleaned outputs back into the raw archive.

1. Copy and edit `configs/example_server_raw.yaml`; set `paths.master_votes_parquet` to an **absolute path** to the uncleaned parquet.
2. Run Stage 03 with the extra config merged:

```bash
python scripts/03_run_global_cleaning.py \
  --config configs/default.yaml \
  --extra-config configs/example_server_raw.yaml
```

Cleaned output is written only under this repo: `data/processed/votes_cleaned.parquet` (see `paths.cleaned_master_parquet` in `configs/default.yaml`).

For the optional lean proposal experiment, enable `cleaning.lean_sample` in a separate override. Stage 03 writes `votes_cleaned_lean.parquet` and `proposal_lean_audit.csv` without replacing canonical cleaned votes. Point a separate downstream config at the lean parquet and report results only for that outcome-selected population.

### Stage 8 minimum prerequisites (Stages 5–7 not required)

| File | Produced by | Purpose |
|------|-------------|---------|
| `data/processed/votes_cleaned.parquet` | Stage 03 | Behaviour dataset build |
| `data/processed/dao_feature_table.csv` | Stage 04 | **Structural** DAO clustering (train-only fit) |

Stage 6 (DAO clustering) and Stage 7 (voter clustering) are **optional** — for exploratory figures/tables in the paper only. They are **not** behaviour-model inputs in the leakage-safe path.

---

## 4. Recommended run order (server)

### 4.1 Detection mode (run first — verify paths and columns)

```bash
# Prerequisite check only (no raw-disk scan)
python scripts/10_run_detection_mode.py --skip-raw-scan

# Optional: 1-epoch smoke train (RoBERTa + numeric features)
python scripts/10_run_detection_mode.py --skip-raw-scan --smoke-train
```

Report: `outputs/reports/detection_mode_report.md`

### 4.2 If cleaning / DAO feature table not done yet

```bash
python scripts/03_run_global_cleaning.py --config configs/default.yaml --extra-config configs/example_server_raw.yaml
python scripts/04_run_dao_metrics.py --config configs/default.yaml
```

### 4.3 Behaviour model training (leakage-safe)

**RoBERTa main model:**

```bash
python scripts/08_run_behaviour_modelling.py --config configs/default.yaml
```

To append the existing proposal body to each title, merge an override containing:

```yaml
behaviour_model:
  text_mode: title_body
```

`title_body` requires `proposal_body` (or legacy `Proposal Body`) in the master votes parquet. It uses `[TITLE]` and `[BODY]` markers, keeps the upstream 800-character body cap, and is still truncated by `behaviour_model.max_length`. Do not pass `--reuse-behaviour-csv` when changing this setting: Stage 08 requires a rebuilt CSV with matching text-mode metadata.

**Numeric-only baseline (no RoBERTa, for comparison):**

```bash
python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml
```

See `docs/SUPERVISOR_NO_ROBERTA_RUNBOOK.md` for 08b details.

### 4.4 Post-training evaluation (Stage 09)

Uses the **same voter split** as training (`outputs/processed/split_manifest.json`):

```bash
python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split test
python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split val
```

Outputs: `outputs/tables/eval/{test|val}/` (F1, ECE, confusion matrices, stratified cluster metrics, etc.).

### 4.5 Automated tests (leakage-fix verification)

```bash
python -m pytest tests/test_leakage_safe_pipeline.py -v
```

Expected: **20 passed** (no GPU or real server data required).

---

## 5. Stage 8 internal pipeline (for review)

```
Build behaviour_dataset.csv (no cluster columns)
    → Split voters into train / val / test
    → Fit numeric preprocessor on train only (VP capping, etc.)
    → Fit structural DAO / voter KMeans on train only
    → Assign cluster IDs to all splits (transform only)
    → Build (voter, space) sliding windows
    → Train TimeSeriesClassifier (DistilRoBERTa + numeric + time)
```

Key implementation files:

- `src/dao_governance/features/behaviour_pipeline.py` — orchestration
- `src/dao_governance/features/causal_clusters.py` — train-only clustering
- `src/dao_governance/modelling/preprocess.py` — numeric columns: `voting_power`, `is_whale`, `dao_cluster`, `voter_cluster`
- `src/dao_governance/modelling/windows.py` — window build and label history

---

## 6. Main output files

| Path | Description |
|------|-------------|
| `outputs/tables/behaviour_dataset.csv` | Vote-level table, **no** clusters |
| `outputs/tables/behaviour_dataset_with_clusters.csv` | Train-fitted cluster IDs (for 08b cache) |
| `outputs/processed/split_manifest.json` | Three-way voter split |
| `outputs/models/predictive_clusters/cluster_bundle.pkl` | Cluster models (train-fit only) |
| `outputs/models/behaviour_agent2/` | RoBERTa model + `config.json` (`feat_dim=8`) |
| `outputs/behaviour_modelling/agent2_artifacts_no_roberta/` | Numeric-only ablation + `config.json` (`feat_dim=10`) |
| `outputs/reports/stage08_behaviour_modelling.md` | Stage 8 summary |
| `outputs/tables/eval/test/metrics_report.md` | Stage 9 test metrics |

---

## 7. Configuration (`configs/default.yaml`)

```yaml
behaviour_model:
  use_dao_clusters: true    # set false for ablation
  use_voter_clusters: true
  text_mode: title  # title | title_body
  group_windows_by: [voter, space]
```

Default `predictive_cluster_artifacts_dir`: `outputs/models/predictive_clusters`

---

## 8. Large datasets and caching (08b)

The first run of 08 or 08b creates the split manifest and enriched CSV. Subsequent runs can reuse them:

```bash
python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml \
  --reuse-split-manifest --reuse-window-cache
```

**Note:** The 08b cache validates numeric columns and window size. Delete or rematerialize it after causal-history schema changes; the current cache has 10 features per step.

---

## 9. Known limitations (not leakage, but disclose in the thesis)

1. **`is_whale`:** If the upstream export uses a global q99 threshold, slight future information is possible; kept as a structural feature for now.
2. **DAO structural metrics (Stage 4):** Full-history aggregates — temporal limitation, not label leakage.
3. **Cluster IDs:** Static per `(voter, space)` within a split (structural features only; no vote-label statistics).
4. **`[LABEL_k]` text prefixes:** Autoregressive history for past votes in the same sequence; current-step label is not a numeric feature.

---

## 10. Troubleshooting

| Symptom | Action |
|---------|--------|
| `feat_dim` mismatch / `model.pt` load failure | Delete `outputs/models/behaviour_agent2/` and re-run Stage 08 |
| Detection reports missing `votes_cleaned.parquet` | Run Stage 03 first |
| Missing `dao_feature_table.csv` | Run Stage 04 first |
| Zero windows | Check cleaned data has enough `(voter, space)` votes; lower `min_votes_per_pair` in clustering module |
| pytest failures | Run from repo root with `PYTHONPATH=src` |

---

## 11. Related documentation

- `docs/Feature_Specification.md` — full feature catalogue and Stage 8 input spec (`feat_dim=8`)
- `docs/Leakage_Audit.md` — leakage issue register, fix status, severity ranking, verification checklist
- `docs/LEAKAGE_SAFE_RUNBOOK.md` — short English runbook
- `docs/SUPERVISOR_NO_ROBERTA_RUNBOOK.md` — 08b numeric-only baseline
- `SERVER_CHECKLIST.txt` — quick server checklist

For debugging, cross-check markdown reports under `outputs/reports/` and assertions in `tests/test_leakage_safe_pipeline.py`.
