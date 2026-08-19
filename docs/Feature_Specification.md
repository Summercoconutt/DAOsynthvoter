# Feature Specification — DAO Behaviour Modelling Pipeline

**Branch:** `carlo_dev`  
**Scope:** All features used across Stages 1–9, with emphasis on **Stage 8 model inputs** (leakage-safe, `feat_dim = 8`).  
**Related:** `docs/Leakage_Audit.md`, `docs/SUPERVISOR_LEAKAGE_SAFE_GUIDE.md`

---

## 1. Taxonomy

| Category | Used in Stage 8 model? | Fit scope | Notes |
|----------|-------------------------|-----------|-------|
| **A. Vote-level numeric (preprocessed)** | Yes | Preprocessor fit on **train voters only** | `voting_power`, `is_whale` |
| **B. Cluster IDs (numeric)** | Yes (optional ablation) | KMeans fit on **train voters only** | `dao_cluster`, `voter_cluster` |
| **C. Time encodings** | Yes | Per-row, no fit | 4 dims appended per window step |
| **D. Text (proposal title; optional body)** | Yes (08 only; not 08b) | RoBERTa pretrained | `text_mode: title_body` appends the available body; history steps prefixed `[LABEL_k]`; current step `[PREDICT]` |
| **E. Structural DAO table features** | Indirect | Stage 4 full history; Stage 8 cluster fit on train spaces | See §4 |
| **F. Structural voter-pair features** | Indirect | Computed from train votes for cluster fit | See §5 |
| **G. Label-derived / exploratory** | **No** | Stage 6/7 EDA only | `z_rep_*`, `pct_*`, `aligned_with_majority` |
| **H. Removed from model** | **No** | — | `aligned_with_majority`, `vp_share`, `vp_ratio_pct` |

---

## 2. Stage 8 model input vector

### 2.1 Definition

Each **window step** (history or current) produces a numeric vector of length **`feat_dim = 8`**:

```
feat_dim = len(select_numeric_columns()) + 4 time features
         = 4 + 4 = 8
```

**Source of truth:** `src/dao_governance/modelling/preprocess.py` → `select_numeric_columns()`  
**Window assembly:** `src/dao_governance/modelling/windows.py` → `build_windows()`

### 2.2 Numeric columns (4)

| # | Column | Type | Preprocessing (train-fit) | Semantics |
|---|--------|------|---------------------------|-----------|
| 1 | `voting_power` | float | Negatives → NaN → median impute → clip at train q0.999 cap → `log1p` → robust scale (median / IQR) | Voter's VP on this proposal |
| 2 | `is_whale` | bool → 0/1 | Cast to bool, no scaling | Whale flag from upstream export |
| 3 | `dao_cluster` | int → float | Robust scale (median / IQR) on train; `-1` if unseen DAO | Train-only structural DAO cluster ID |
| 4 | `voter_cluster` | int → float | Robust scale (median / IQR) on train; `-1` if low activity | Train-only structural (voter, space) cluster ID |

**Config toggles** (`configs/default.yaml`):

```yaml
behaviour_model:
  use_dao_clusters: true    # false → all dao_cluster = -1
  use_voter_clusters: true  # false → all voter_cluster = -1
```

### 2.3 Time features (4)

Appended after numeric columns for **every** window step (`windows._time_feats`):

| # | Feature | Formula | Range |
|---|---------|---------|-------|
| 5 | hour | `ts.hour / 23.0` | [0, 1] |
| 6 | weekday | `ts.weekday() / 6.0` | [0, 1] |
| 7 | month | `(ts.month - 1) / 11.0` | [0, 1] |
| 8 | day | `(ts.day - 1) / 30.0` | [0, 1] |

Source column: `vote_ts` (UTC). Missing timestamps → all zeros.

### 2.4 Text channel (Stage 08 only)

| Field | Source | Window encoding |
|-------|--------|-----------------|
| `text` (`text_mode: title`) | `proposal_title` from behaviour dataset | History: `"[LABEL_{0|1|2}] " + text`; current: `"[PREDICT] " + text` |
| `text` (`text_mode: title_body`) | `proposal_title` plus `proposal_body` / `Proposal Body` | `"[TITLE] " + title + " [BODY] " + body`, then the history/current prefix |

Tokenizer: `distilroberta-base` (configurable via `behaviour_model.pretrained`).  
Stage **08b** (`NumericOnlyTimeSeriesClassifier`) uses **numeric + time only** — no text.

`text_mode` defaults to `title`. `title_body` requires the body column in the master votes parquet. The current expansion caps body text at 800 characters, and RoBERTa still applies `behaviour_model.max_length` token truncation. Switching modes requires rebuilding `behaviour_dataset.csv`; Stage 08 records the selected mode in its adjacent CSV metadata and in the model configuration.

### 2.5 Stage 08b causal history features

Stage 08b additionally uses `prior_frac_for` and `prior_frac_against`, producing `feat_dim = 10` (six numeric features plus four time features). For each row, both fractions are calculated from earlier valid votes by the same `(voter, space)` pair; all prior votes, including `ABSTAIN`, are included in the denominator. The current vote is excluded. Stage 08 remains at `feat_dim = 8` and does not receive these two fields.

### 2.6 Target (not an input)

| Column | Values | Mapping |
|--------|--------|---------|
| `label_id` | 0, 1, 2 | `for` → 0, `against` → 1, `abstain` → 2 |

Derived from `choice_norm` in `behaviour_dataset.py`. Rows with other choices are dropped.

---

## 3. Windowing and grouping

| Parameter | Value | Config / code |
|-----------|-------|---------------|
| Group key | `(voter, space)` | `WINDOW_GROUP_COLS` in `windows.py`; `behaviour_model.group_windows_by` |
| Window size | 5 (default) | `behaviour_model.window` |
| Min sequence length | `window_size` votes in group | Shorter groups skipped |
| Sort key | `vote_ts` ascending | Within each `(voter, space)` |
| Prediction index | Last step in window | Target = `label_id` at current step |

**Tensor shape per sample:** `(window_size, feat_dim)` → default `(5, 8)`.

---

## 4. DAO structural features (cluster fit only)

Used to fit **train-only** DAO KMeans in `causal_clusters.fit_dao_clusters()`.  
**Not** passed directly as model inputs; only the resulting `dao_cluster` ID enters the model.

**Constant list:** `DAO_CLUSTER_FEATURES` in `src/dao_governance/features/causal_clusters.py`

| Feature | Description |
|---------|-------------|
| `log_n_unique_voters` | log1p of unique voters (fallback: derived from `n_unique_voters`) |
| `mean_robust_participation_vp` | Mean robust participation VP per proposal |
| `std_robust_participation_vp` | Std robust participation VP |
| `gini_voting_power` | VP inequality within DAO |
| `whale_ratio_top1pct` | Share of VP held by top 1% voters |
| `proposal_frequency_per_30d` | Proposal rate |
| `repeat_voter_rate` | Fraction of repeat voters |

**Input table:** `data/processed/dao_feature_table.csv` (Stage 04).  
**Fit scope:** DAO spaces that appear in **training-split votes** only.  
**K selection:** KMeans with k ∈ [2, 6] by minimum inertia (`_pick_kmeans_k`).  
**Unseen spaces at assign time:** `dao_cluster = -1`.

---

## 5. Voter structural features (cluster fit only)

Used to fit **train-only** voter KMeans in `causal_clusters.fit_voter_clusters()`.  
Computed by `build_voter_structural_features()` from vote rows — **no label proportions**.

| Feature | Definition |
|---------|------------|
| `log_total_votes` | log1p vote count per (voter, space) |
| `avg_voting_power` | Mean VP |
| `std_voting_power` | Std VP (0 if single vote) |
| `active_span_days` | Days between first and last vote + 1 |
| `vote_frequency` | `total_votes / active_span_days` |
| `participation_rate` | `total_votes / n_proposals_in_space` |

**Fit scope:** `(voter, space)` pairs with ≥ `min_votes_per_pair` (default 5) in **train split**.  
**Assign scope:** All splits; low-activity pairs get `voter_cluster = -1`.

---

## 6. Behaviour dataset columns (Stage 8 CSV)

Built by `build_behaviour_dataset()` — **no cluster merge by default**.

| Column | Required | In default CSV |
|--------|----------|----------------|
| `voter` | Yes | Yes |
| `space` | Yes | Yes |
| `proposal_id` | Yes | Yes |
| `choice_norm` | Yes | Yes |
| `vote_ts` / `vote_timestamp` | Yes | Yes (`vote_ts` normalised) |
| `proposal_title` → `text` | Yes | Yes |
| `proposal_body` / `Proposal Body` | Only `text_mode: title_body` | Appended to text with `[BODY]`; Stage 03 canonicalizes raw `Proposal Body` to `proposal_body` |
| `voting_power` | Yes | Yes |
| `is_whale` | Yes | Yes |
| `label_id` | Yes (derived) | Yes |
| `dao_cluster` | No | **Stripped** (assigned in Stage 8) |
| `voter_cluster` | No | **Stripped** |
| `aligned_with_majority` | No | **Not built** in leakage-safe path |

**Outputs:**

- `outputs/tables/behaviour_dataset.csv` — base table  
- `outputs/tables/behaviour_dataset_with_clusters.csv` — after train-only assign (for 08b cache / Stage 09 eval)

### 6.1 Optional lean proposal sample

Stage 03 can optionally write `votes_cleaned_lean.parquet` without replacing the canonical cleaned votes. This experimental variant removes an entire `(space, proposal_id)` when FOR or AGAINST reaches its configured consensus threshold, or when title/body character counts are below configured minima. Vote fractions include ABSTAIN in their denominator. The companion `proposal_lean_audit.csv` records each proposal's counts, fractions, text lengths, retention flag, and exclusion reasons.

This is outcome-based sample selection, not leakage remediation. A model trained on the lean sample must be evaluated on the same sample definition and described as applying only to retained, non-landslide proposals.

---

## 7. Exploratory features (NOT model inputs)

These appear in earlier pipeline stages for **clustering validation / EDA** only.

### 7.1 Stage 6 — DAO clustering validation

`clustering_validation_pipeline.py` → `FEATURE_COLUMNS` includes label-heavy `z_rep_*`:

- `z_rep_pct_for_votes`, `z_rep_pct_against_votes`, `z_rep_pct_abstain_votes`
- `z_rep_choice_entropy`, `z_rep_pct_aligned_with_majority`

Separate constant `PREDICTIVE_DAO_CLUSTER_FEATURES` mirrors §4 (structural only) for documentation; Stage 8 uses `causal_clusters`, not Stage 6 assignments.

### 7.2 Stage 7 — Voter clustering

`run_voter_clustering.py` → `FEATURE_COLS` (structural, exploratory):

- `log_total_votes`, `avg_voting_power`, `std_voting_power`, `active_span_days`, `vote_frequency`, `participation_rate`, `n_daos_participated`

Legacy label-heavy list `EXPLORATORY_LABEL_FEATURE_COLS` (not used for predictive clustering):

- `pct_for_votes`, `pct_against_votes`, `pct_abstain_votes`, `pct_aligned_with_majority`, `vote_entropy`, `is_whale_ratio`

**Stage 7 `voter_cluster_assignments.csv` is not consumed by Stage 8** in the leakage-safe path.

### 7.3 Stage 4 — Representative voter / DAO table extras

`dao_feature_table.csv` contains many columns (participation stats, `raw_rep_*`, `z_rep_*`). Only the §4 structural subset is used for predictive DAO clustering. Label-derived `z_rep_*` columns exist for feature-selection and exploratory DAO clustering.

---

## 8. Removed / excluded features

| Feature | Former role | Reason for removal |
|---------|-------------|-------------------|
| `aligned_with_majority` | Numeric model input | Equals current or historical label vs majority — direct leakage |
| `vp_share` | Numeric model input | Often co-computed with outcome; removed from `select_numeric_columns()` |
| `vp_ratio_pct` | Numeric | Stripped by `load_behaviour_votes()` |
| Stage 6 `dao_cluster` assignments | Merged into behaviour CSV | Global fit + label-heavy features |
| Stage 7 `voter_cluster` assignments | Merged into behaviour CSV | Global fit + label-heavy aggregates |

**Defence in depth:** If `aligned_with_majority` appears in a dataframe, `windows.LABEL_DERIVED_AT_PREDICT_TIME` forces **0.0** at the current prediction step (history may still carry it if explicitly passed — column is not in `select_numeric_columns()` by default).

---

## 9. Train / validation / test protocol

| Step | What is fit | Data scope |
|------|-------------|------------|
| 1. Voter split | — | `split_by_voter_three_way`: disjoint voters |
| 2. Numeric preprocessor | VP cap, log scale, cluster robust scale | Train rows only |
| 3. DAO KMeans | Scaler + centroids | Train spaces + structural DAO table cols |
| 4. Voter KMeans | Scaler + centroids | Train votes → structural pair features |
| 5. Assign clusters | Transform only | All splits |
| 6. Build windows | — | Per-split dataframes |
| 7. Train model | RoBERTa + temporal head | Train windows only |

Manifest: `outputs/processed/split_manifest.json`  
Cluster artifact: `outputs/models/predictive_clusters/cluster_bundle.pkl`  
Model config: `outputs/models/behaviour_agent2/config.json` (`feat_dim`, `enriched_behaviour_csv`)

---

## 10. Version compatibility

| Version | `feat_dim` | Numeric columns |
|---------|------------|-----------------|
| Pre-leakage-fix | 10 | Included `aligned_with_majority`, `vp_share` (+ 4 time) |
| **carlo_dev (current)** | **8** | `voting_power`, `is_whale`, `dao_cluster`, `voter_cluster` (+ 4 time) |

Old checkpoints are **incompatible** — retrain Stage 08 after pulling leakage-safe `carlo_dev`.

---

## 11. Automated validation

```bash
export PYTHONPATH=src
python -m pytest tests/test_leakage_safe_pipeline.py -v
```

Key assertions:

- `select_numeric_columns()` exact set of 4 columns  
- `(voter, space)` separate window sequences  
- Cluster assignments invariant to label corruption on val/test  
- `feat_dim = len(cols) + 4` in end-to-end `prepare_behaviour_splits`

---

## 12. File reference map

| Concern | Primary module |
|---------|----------------|
| Model numeric column list | `src/dao_governance/modelling/preprocess.py` |
| Window + time features | `src/dao_governance/modelling/windows.py` |
| Memmap cache (08b) | `src/dao_governance/modelling/window_cache.py` |
| Train-only clustering | `src/dao_governance/features/causal_clusters.py` |
| Split → preprocess → cluster orchestration | `src/dao_governance/features/behaviour_pipeline.py` |
| Vote-level CSV build | `src/dao_governance/features/behaviour_dataset.py` |
| Fusion model | `src/dao_governance/modelling/model.py` |
| Stage 08 entry | `scripts/08_run_behaviour_modelling.py` |
