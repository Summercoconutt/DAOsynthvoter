# Leakage Audit — Behaviour Modelling Pipeline

**Branch:** `carlo_dev` (leakage-safe implementation)  
**Audit date:** 2026-07  
**Related:** `docs/Feature_Specification.md`, `tests/test_leakage_safe_pipeline.py`

This document records **label leakage and split contamination** risks identified in the dissertation behaviour-modelling path, their severity, remediation status, and residual limitations.

---

## 1. Executive summary

| Status | Finding |
|--------|---------|
| **Fixed (Critical)** | `aligned_with_majority` and `vp_share` removed from model inputs |
| **Fixed (Critical)** | Global Stage 6/7 cluster CSVs no longer fed into Stage 8 |
| **Fixed (High)** | Clustering moved **after** voter split; fit on train only |
| **Fixed (High)** | Windows grouped by `(voter, space)` instead of `voter` alone |
| **Mitigated (Medium)** | Label-derived columns zeroed at prediction step if present |
| **Documented (Low)** | `is_whale` global threshold; DAO metrics use full history |
| **Verified** | 9 automated pytest cases pass without GPU or server data |

**Bottom line:** Stage 8 metrics on held-out **voters** are now structurally defensible for thesis reporting, provided the server run follows `docs/SUPERVISOR_LEAKAGE_SAFE_GUIDE.md` and old `feat_dim=10` checkpoints are not reused.

---

## 2. Leakage taxonomy

| Type | Definition | Example in this project |
|------|------------|---------------------------|
| **L1 — Direct label leakage** | Feature at prediction time encodes current or future label | `aligned_with_majority` at current step |
| **L2 — Indirect label leakage** | Feature aggregates labels over votes used at test time | `z_rep_pct_for_votes` in global DAO clusters |
| **L3 — Split leakage** | Test-unit statistics influence train features | KMeans on all voters before split |
| **L4 — Unit leakage** | Same voter in train and test | Mitigated by voter-level split |
| **L5 — Temporal leakage** | Future votes inform past prediction | Partial risk in full-history DAO metrics |
| **L6 — Autoregressive (allowed)** | Past **known** labels in history steps | `[LABEL_k]` text prefixes |

---

## 3. Issue register (pre-fix → post-fix)

### 3.1 Critical — resolved

#### ISSUE-001: `aligned_with_majority` as numeric input

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Status** | **Fixed** |
| **Files** | `preprocess.py`, `windows.py`, `behaviour_pipeline.py` |
| **Mechanism** | Column indicates whether vote matches proposal majority choice — a deterministic function of `choice_norm` / `label_id`. |
| **Why it leaks** | At the current prediction step, knowing alignment reveals or strongly constrains the target class. |
| **Fix** | Removed from `select_numeric_columns()`. Listed in `LABEL_DERIVED_AT_PREDICT_TIME` — forced to `0.0` at current step if column smuggled in. Stripped by `load_behaviour_votes()`. |
| **Test** | `test_select_numeric_columns_excludes_label_derived`, `test_current_window_step_zeros_smuggled_alignment` |

#### ISSUE-002: `vp_share` as numeric input

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Status** | **Fixed** |
| **Files** | `preprocess.py`, `behaviour_pipeline.py` |
| **Mechanism** | Share of total proposal VP; often correlated with outcome and majority structure. |
| **Fix** | Removed from `select_numeric_columns()`; stripped on load. |
| **Test** | `test_select_numeric_columns_excludes_label_derived` |

#### ISSUE-003: Global DAO / voter clusters with label-heavy features

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Status** | **Fixed** |
| **Files** | `behaviour_dataset.py`, `scripts/08_*.py`, `causal_clusters.py`, `clustering_validation_pipeline.py`, `run_voter_clustering.py` |
| **Mechanism (before)** | Stage 6 used `z_rep_*` (label proportions). Stage 7 used `pct_for_votes`, `pct_aligned_with_majority`, etc. Assignments merged into behaviour CSV **before** train/test split. |
| **Why it leaks (L2+L3)** | Test voters' labels influenced cluster centroids and IDs seen at train time; cluster ID becomes a proxy for label distribution. |
| **Fix** | Stage 8 fits **structural-only** KMeans on train voters (`causal_clusters.py`). Stage 6/7 marked exploratory only. `include_legacy_clusters=False` by default. |
| **Test** | `test_cluster_fit_uses_train_voters_only`, `test_changing_test_labels_does_not_change_cluster_assign`, `test_dao_features_are_structural_only` |

---

### 3.2 High — resolved

#### ISSUE-004: Clustering before train/test split

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Status** | **Fixed** |
| **Files** | `behaviour_pipeline.py`, `scripts/08_run_behaviour_modelling.py` |
| **Mechanism** | Any unsupervised fit on val/test units leaks partition structure into features. |
| **Fix** | Order: `split_by_voter_three_way` → `fit_numeric_preprocessor(train)` → `fit_cluster_bundle(train)` → `assign_clusters_to_votes(all splits)`. |
| **Test** | `test_end_to_end_prepare` |

#### ISSUE-005: Window grouping by `voter` only

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Status** | **Fixed** |
| **Files** | `windows.py`, `window_cache.py`, `configs/default.yaml` |
| **Mechanism** | Mixing votes across DAOs in one sequence lets DAO-specific label history from one space inform another. |
| **Fix** | `WINDOW_GROUP_COLS = ("voter", "space")`. |
| **Test** | `test_voter_space_separate_sequences` |

#### ISSUE-006: Numeric preprocessor fit on full dataset

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Status** | **Fixed** |
| **Files** | `behaviour_pipeline.py`, `preprocess.py` |
| **Mechanism** | VP quantile cap and robust scales computed on test voters. |
| **Fix** | `fit_numeric_preprocessor(train_raw)` only; val/test transformed. |

---

### 3.3 Medium — mitigated or accepted

#### ISSUE-007: `[LABEL_k]` text prefixes in history steps

| Field | Value |
|-------|-------|
| **Severity** | Medium (intentional autoregression) |
| **Status** | **Accepted by design** |
| **Files** | `windows.py` |
| **Mechanism** | Historical steps include true past labels in text channel. |
| **Rationale** | Simulates known vote history; current step uses `[PREDICT]` only. Standard for sequential vote modelling. **Not** duplicated in numeric features at current step. |
| **Reviewer note** | Report as autoregressive setting; metrics are not cold-start per voter. |

#### ISSUE-008: `is_whale` from global q99 threshold

| Field | Value |
|-------|-------|
| **Severity** | Medium (possible L5) |
| **Status** | **Documented — not fixed** |
| **Files** | Upstream export / `behaviour_dataset.py` |
| **Mechanism** | If whale flag uses population-wide VP quantile including test-period votes, slight future information. |
| **Mitigation** | Flag is structural; consider train-only threshold in future work. |

#### ISSUE-009: DAO feature table from full history (Stage 4)

| Field | Value |
|-------|-------|
| **Severity** | Medium (temporal, not label L1) |
| **Status** | **Documented — accepted** |
| **Files** | Stage 04 metrics scripts |
| **Mechanism** | `gini_voting_power`, participation stats aggregate all votes through export cutoff. |
| **Rationale** | Structural DAO descriptors; cluster fit still train-only. Not equivalent to injecting test labels. |

#### ISSUE-010: Static cluster ID within `(voter, space)` across time

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Status** | **Accepted** |
| **Mechanism** | Voter cluster ID fixed from structural aggregates over train-period votes for that pair; applied to all timesteps in val/test. |
| **Rationale** | No label proportions in cluster features; ID does not change when test labels are permuted (`test_changing_test_labels_does_not_change_cluster_assign`). |

---

### 3.4 Low / false alarms

#### ISSUE-011: `dao_cluster` / `voter_cluster` as model inputs after fix

| Field | Value |
|-------|-------|
| **Severity** | Was High pre-fix; **Low post-fix** |
| **Status** | **Fixed protocol** |
| **Note** | Clusters remain as inputs but are **train-fit structural** IDs. Ablation: `use_dao_clusters: false`, `use_voter_clusters: false`. |

#### ISSUE-012: Evaluation stratification by cluster

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Status** | **Not leakage** |
| **Files** | `evaluation/evaluate.py`, `scripts/09_run_behaviour_evaluation.py` |
| **Note** | Post-hoc grouping for reporting only; clusters not used as inputs during inference beyond model's existing numeric columns. |

---

## 4. Remediation architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 03–04: votes_cleaned.parquet + dao_feature_table.csv      │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ build_behaviour_dataset (NO cluster columns)                    │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ split_by_voter_three_way (disjoint voters)                      │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ fit_numeric_preprocessor(TRAIN)                                 │
│ fit_cluster_bundle(TRAIN) — structural features only            │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ assign_clusters_to_votes(train | val | test)                    │
│ normalise_columns(preprocessor)                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ build_windows(group_by=voter+space) → feat_dim=8                │
│ Train TimeSeriesClassifier on TRAIN windows only                │
└─────────────────────────────────────────────────────────────────┘
```

**Orchestration:** `src/dao_governance/features/behaviour_pipeline.py`  
**Entrypoint:** `scripts/08_run_behaviour_modelling.py`

---

## 5. What still uses label-heavy features (safe contexts)

| Stage | Features | Safe because |
|-------|----------|--------------|
| 4 | `z_rep_*` in DAO table | Not passed to Stage 8 model; only subset of structural cols used for predictive clustering |
| 5 | Feature selection on `z_rep_*` | Screening for exploratory DAO clustering paper figures |
| 6 | `FEATURE_COLUMNS` with `z_rep_*` | Validation / heatmaps only; `cluster_assignments.csv` not used by Stage 8 |
| 7 | `EXPLORATORY_LABEL_FEATURE_COLS` | Documented legacy; not in `FEATURE_COLS` for clustering |

---

## 6. Split and evaluation protocol

| Rule | Implementation |
|------|----------------|
| Split unit | **Voter** (not vote, not pair) |
| Default fractions | 70% train / 15% val / 15% test (`configs/default.yaml`) |
| Seed | `project.seed` (default 42) |
| Evaluation voters | Stage 09 reads `split_manifest.json` — same lists as training |
| Detection prerequisites | `pipeline/detection.py` — cleaned votes + DAO feature table (Stage 7 optional) |

**Important:** Metrics measure generalisation to **unseen voters**, not unseen proposals within seen voters. Document this limitation in the thesis.

---

## 7. Verification checklist

### 7.1 Automated (CI / local)

```bash
export PYTHONPATH=src
python -m pytest tests/test_leakage_safe_pipeline.py -v
```

| Test class | Guarantees |
|------------|------------|
| `TestNoLeakyModelInputs` | Numeric column whitelist; alignment zeroed at predict step |
| `TestClusterSafety` | Structural DAO features; train-only fit; label-invariant assign |
| `TestWindowGrouping` | Separate sequences per `(voter, space)` |
| `TestBehaviourDataset` | Legacy clusters / leaky cols stripped |
| `TestIntegrationPrepareSplits` | Full pipeline produces finite `feat_dim=8` windows |

### 7.2 Manual (server)

1. `python scripts/10_run_detection_mode.py --skip-raw-scan` → prerequisites OK  
2. After Stage 08: `config.json` shows `"feat_dim": 8`  
3. `behaviour_dataset.csv` has **no** `dao_cluster` / `voter_cluster` columns  
4. `behaviour_dataset_with_clusters.csv` exists after Stage 08  
5. Stage 09 eval uses `enriched_behaviour_csv` from `config.json`

---

## 8. Severity matrix (before fix)

| Rank | Issue | Type | Status |
|------|-------|------|--------|
| 1 | `aligned_with_majority` in numeric input | L1 | Fixed |
| 2 | Global label-heavy clusters → model | L2+L3 | Fixed |
| 3 | Cluster fit before split | L3 | Fixed |
| 4 | `vp_share` in numeric input | L1/L2 | Fixed |
| 5 | Voter-only window grouping | L2 | Fixed |
| 6 | Preprocessor on full data | L3 | Fixed |
| 7 | `is_whale` global threshold | L5 | Documented |
| 8 | Full-history DAO metrics | L5 | Documented |
| 9 | Autoregressive text labels | L6 | By design |

---

## 9. Incompatibility notice

| Artifact | Pre-fix | Post-fix (`carlo_dev`) |
|----------|---------|------------------------|
| `feat_dim` | 10 | **8** |
| `config.json` / `model.pt` | Old numeric layout | **Retrain required** |
| Window cache (`08b`) | May contain leaky columns | Delete cache dir after upgrade |
| Stage 7 assignments | Required input | **Optional EDA** |

---

## 10. References

| Document | Purpose |
|----------|---------|
| `docs/Feature_Specification.md` | Full feature catalogue and `feat_dim` layout |
| `docs/SUPERVISOR_LEAKAGE_SAFE_GUIDE.md` | Detailed server runbook for supervisor |
| `docs/LEAKAGE_SAFE_RUNBOOK.md` | Short English runbook |
| `SERVER_CHECKLIST.txt` | Quick server checklist |

**Code anchors:**

- `select_numeric_columns()` — `src/dao_governance/modelling/preprocess.py`
- `DAO_CLUSTER_FEATURES`, `VOTER_CLUSTER_FEATURES` — `src/dao_governance/features/causal_clusters.py`
- `prepare_behaviour_splits()` — `src/dao_governance/features/behaviour_pipeline.py`
- `LABEL_DERIVED_AT_PREDICT_TIME` — `src/dao_governance/modelling/windows.py`
