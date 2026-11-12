# Handover: Stage 08 model analysis and final report

## Purpose

The repository currently contains four run roots under `outputs/stage08_runs/`: fine-tuned and frozen-encoder runs for both the **full** and **lean** sample variants. Each run root has paired Stage 08/08b models, run-matched evaluation, provenance, and data for model analysis. Analyse the trained behaviour models, diagnose them with explainable-AI (XAI) methods, and use the evidence to write the final report.

The prediction target is a voter's next recorded choice: `FOR`, `AGAINST`, or `ABSTAIN`. This is predictive modelling, not a causal analysis of why a voter chose an option.

## What is in the current `stage08_runs/` folders

The layout is `outputs/stage08_runs/<variant>/<run-directory>/`, where `<variant>` is `full` or `lean`. These are the current run directories and recorded encoder modes:

| Run directory | Variant | Stage 08 text encoder |
|---|---|---|
| `full/full_20261013/` | Full | Fine-tuned |
| `full/full_20261028_frozen/` | Full | Frozen |
| `lean/lean_20261030/` | Lean | Fine-tuned |
| `lean/lean_20261103_frozen/` | Lean | Frozen |

Each run follows the same general layout:

| Path | Contents and use |
|---|---|
| `run_manifest.json` | Run ID and variant, input/config provenance, source revision/runtime, and recorded artifact hashes. Use this as the provenance index and verify it before analysis. |
| `config/resolved.yaml`, `config/data_sources.yaml` | Resolved modelling/cleaning settings and data-source selection for this run. These distinguish the full and lean source datasets. |
| `inputs/behaviour_dataset.csv` | Common vote-level dataset used by both models; includes voter/space/proposal, vote timestamp, title/body/text, and `label_id`. `behaviour_dataset.meta.json` records the text mode; `dao_feature_table.csv` is the run's DAO feature snapshot. |
| `shared/split_manifest.json` | Train/validation/test voter membership and split settings. The split is by voter; preserve this assignment for evaluation and explanations. |
| `shared/predictive_clusters/`, `shared/predictive_clusters_08b/` | Run-scoped fitted cluster artifacts used by the stages. Keep them with their run; do not substitute artifacts from another run. |
| `08/` | Text-plus-numeric model weights/config, exact tokenizer, enriched vote-level data with cluster assignments, and training/preprocessing reports and logs. |
| `08b/` | Numeric-only model weights/config, enriched data with causal-history features, reports/logs, and `window_cache/`. The cache contains large training/validation arrays and metadata; it is not an explanation output or a substitute for the held-out test inputs. |
| `evaluation/08/{test,val}/`, `evaluation/08b/{test,val}/` | Paired held-out test and validation predictions, metrics, class reports, confusion matrices, calibration tables/figures, and cluster-stratified metrics. Use `test/` for final performance and explanation case studies. |
| `evaluation/reports/` | Run-level evaluation summary and within-run 08-versus-08b comparison. |

The vote-level inputs, enriched CSVs, and some caches are very large (multi-GB to tens of GB). Avoid loading them wholesale for case selection; use the compact metrics/predictions first, then restrict or stream the relevant data when reconstructing selected windows. The saved prediction CSVs contain `y_true`, `y_pred`, voter and cluster IDs, and `prob_FOR`, `prob_AGAINST`, `prob_ABSTAIN`; they do **not** contain a proposal ID or stable window index.

**Run-ID issue to resolve before exporting or citing the lean fine-tuned run:** its directory is `lean/lean_20261030/`, while both `run_manifest.json` and `08/config.json` record `run_id: lean_30102026`. The exporter requires the directory name to equal the manifest run ID, so this run will fail that directory-identity check unless the discrepancy is resolved. Do not silently rename it or edit its manifest: preserve the evidence and agree on the canonical run identity first. The other three run directories match their recorded run IDs.

## First: verify that each ZIP is self-consistent

For an exported ZIP, expect a top-level `stage08_08b_<VARIANT>_<RUN_ID>/` directory. It should be copied from one immutable run root, not assembled from shared “latest” paths:

| Directory | Required contents |
|---|---|
| Run root | `run_manifest.json`, `config/resolved.yaml`, `config/data_sources.yaml`, and `export_manifest.json` |
| `inputs/` | The behavior dataset used by both models and the DAO feature-table snapshot; the manifest records the authoritative cleaned-parquet path and hash |
| `shared/` | The voter `split_manifest.json` and Stage 08 train-fitted cluster artifacts |
| `08/` | `model.pt`, `config.json`, exact `tokenizer/`, Stage 08 enriched dataset, reports, and epoch logs |
| `08b/` | Numeric-only `model.pt`, `config.json`, its causal-history enriched dataset, reports, logs, within-run training comparison, and run-keyed window cache when materialized |
| `evaluation/08/test/` and `evaluation/08b/test/` | Matched held-out test predictions, metrics, per-class tables, confusion matrices, calibration outputs, and cluster-stratified metrics |
| `evaluation/reports/` | Evaluation summary and within-run 08-versus-08b comparison |

The manifest records the run ID, sample variant, resolved-config hash, configured input hashes, Git revision/worktree state, runtime versions, and hashes/sizes for recorded artifacts. Model configs use paths relative to the run root. The exporter rejects missing, changed, or incomplete recorded artifacts and refuses to overwrite an existing ZIP. Check the run ID, variant, source commit, and manifest before analysis.

Export only after both models and paired Stage 09 evaluation have completed. The default exporter requires all three stages (`08`, `08b`, `evaluation`) and verifies recorded hashes. For an intentional model-only handoff, `--allow-without-evaluation` is available, but that bundle is not sufficient for the primary held-out performance analysis below.

Reproduce the paired full and lean runs with separate IDs, for example:

```bash
./run_all.sh configs/default.yaml configs/data_sources.yaml full_20261008_a
./run_all.sh configs/default.yaml configs/data_sources_lean.yaml lean_20261008_a
./scripts/export_stage08_run.sh outputs/stage08_runs/full/full_20261008_a /tmp/exports
./scripts/export_stage08_run.sh outputs/stage08_runs/lean/lean_20261008_a /tmp/exports
```

Stage 09 evaluates both architectures against the same run’s saved test voters and writes outputs under `evaluation/08/test/` and `evaluation/08b/test/`. Training reports remain validation evidence, not held-out test results.

## What “full”, “lean”, “frozen”, “fine-tuned”, and “08b” mean

- **Full vs lean** describes the data/sample variant, not the model architecture. In these runs, full resolves to `votes_cleaned_from_server.parquet` and lean to `votes_cleaned_lean.parquet`. Lean is an experimental, outcome-selected sample: the documented filter may exclude whole proposals based on consensus and title/body-length rules. Verify the run's `config/resolved.yaml` and, where available, the proposal audit; do not infer exact thresholds from the directory name. Lean conclusions apply only to retained proposals and are not an unbiased estimate for the full population.
- **Fine-tuned vs frozen** describes only the Stage 08 pretrained text encoder. Fine-tuned mode updates the pretrained `distilroberta-base` text-encoder weights during training. Frozen mode holds those encoder weights fixed and caches text embeddings for efficiency; the projection, numeric, temporal, and classification layers are still trained. The exact mode is in `08/config.json` and `config/resolved.yaml`, not just in the run-directory suffix.
- **Stage 08 vs Stage 08b** describes the model comparison: Stage 08 uses proposal text (these runs use `text_mode: title`) plus numeric/time inputs; Stage 08b is numeric-only, with causal historical-choice features and no proposal text. 08b is an ablation, not another name for the lean run. The frozen/fine-tuned distinction does not apply to 08b because it has no text encoder.
- The documented leakage-safe setup splits voters into train/validation/test, fits preprocessing and structural clusters on training data, and forms temporal windows within `(voter, space)`. Stage 08b should use the Stage 08 split from the same data variant so the within-run comparison is fair.
- The current configs use a 70% train / 15% validation / remaining test voter split. The saved manifest/config remains authoritative for each run. Full-versus-lean metric differences may reflect population selection, class mix, and data volume as well as any model effect. Fine-tuned-versus-frozen comparisons should likewise be made within variant and checked against the exact run provenance.

## Analysis workflow

### 1. Establish run identity and data coverage

For each ZIP, record the archive/tag, run type (full or lean), model type (08 or 08b), source commit, seed, text mode, split fractions, window size, label map, feature names/order, and any available cleaning/filter configuration. Check that the Stage 08 and 08b split manifests match within a data variant.

Summarize training, validation, and test sample/window counts; unique voters, spaces, and proposals where recoverable; and the class counts for `FOR`, `AGAINST`, and `ABSTAIN`. Document any missing artifacts or deviations. Do not compare metrics until sample identity and split alignment are established.

### 2. Evaluate held-out predictive performance

Use held-out **test** metrics for the primary results. Use validation results for model selection/context only. Report at least:

- Macro-F1 and per-class precision, recall, F1, and support. Include balanced accuracy.
- Confusion matrix (counts and a clearly labelled normalization), plus accuracy as a secondary metric.
- Log loss and calibration evidence such as ECE/reliability plots, if available.
- Variation across DAO/voter clusters or other supported subgroups, with subgroup sizes and appropriate caution for small groups.

Discuss class imbalance explicitly. Do not let aggregate accuracy or weighted F1 conceal weak `AGAINST`/`ABSTAIN` performance. Distinguish train/validation training curves from the independently held-out test results.

Compare 08 versus 08b **within each run variant** using the same test voters and outcome definition. A full-versus-lean comparison is not a like-for-like test of architecture: show each population's coverage and class mix, and qualify any metric differences accordingly.

### 3. Diagnose the models with XAI

Use the saved model config and training-time preprocessing as the source of truth. Explain examples from the held-out test set after the analysis design is fixed. Use a background/reference set drawn only from training data. Explain class-specific predictions and include correct, incorrect, and low-confidence cases from all three classes where available.

The current folders contain model/evaluation artifacts, **not precomputed SHAP or Integrated Gradients results**. A reproducible explanation workflow is:

1. Pick a run and model, verify its manifest/config, and read the matching `evaluation/<08-or-08b>/test/predictions.csv`. Select examples by true/predicted class and probabilities (for example, correct, misclassified, and low-confidence cases); use test labels only to describe/evaluate the cases, not as model inputs.
2. Reconstruct the selected test windows from that run's enriched CSV and `shared/split_manifest.json`, using the repository's exact preprocessing and windowing code. The evaluator filters test voters, groups by `(voter, space)`, orders by `vote_ts`, builds windows, and writes predictions in that order. Since predictions lack a window key, reproduce that exact ordering and validate every aligned record against `y_true`, voter, cluster IDs, and probabilities before attributing anything. Do not join on voter alone: one voter can have multiple windows and spaces.
3. Use the same `model.pt`, saved `config.json` (including `numeric_preprocessor`, feature order, window size, and text encoder mode), and exact tokenizer/model definition from that run. Use only training windows to form SHAP backgrounds or reference distributions. Save an explicit mapping from each explanation back to its run, model, split, and reconstructed window.
4. For Stage 08, attribute text through token embeddings (not integer token IDs), and numeric/time inputs separately. The current four numeric columns are `voting_power`, `is_whale`, `dao_cluster`, and `voter_cluster`; window construction adds four time encodings per step, for `feat_dim: 8`. Preserve `[LABEL_n]` history markers and the current `[PREDICT]` marker and use the exact archived tokenizer.
5. For Stage 08b, explain the sequence tensor with the six configured numeric/history columns (`voting_power`, `is_whale`, `dao_cluster`, `voter_cluster`, `prior_frac_for`, `prior_frac_against`) plus four time encodings per step (`feat_dim: 10`). Summarize attributions both by feature and by window position (history versus prediction step); document any aggregation.
6. Validate explanations with multiple checks: IG completeness/convergence or SHAP additivity where applicable; deletion/ablation of highly attributed inputs versus matched low-attribution inputs; stability across reasonable training-only backgrounds/seeds; and a model-randomization sanity check. Re-evaluate the target-class probability for each perturbation and report the change. These are model-faithfulness checks, not tests of causation.

**Stage 08 (text + numeric):**

- Token IDs are discrete and are not directly differentiable. For text attribution, use embedding-level Integrated Gradients (IG) or a carefully validated SHAP wrapper around the embedding/model path; use the exact tokenizer from the archive, including its added special tokens.
- Attribute numeric inputs separately from text, using the saved feature order and training-fitted preprocessing. If combining modalities, explain how their attributions are normalized and compared.
- Aggregate token attributions to readable words/spans, handle padding and special markers consistently, and report the baseline/reference choice. Do not interpret truncated or empty text as meaningful evidence.

**Stage 08b (numeric-only):**

- Use feature-level IG or a numeric SHAP method with a documented training-only background. A custom model wrapper may be needed for SHAP.
- Preserve the temporal window and its feature ordering. Report both per-feature and, where useful, per-time-step/grouped-feature summaries; do not flatten away the sequence structure without explaining the aggregation.
- Check the saved config before assuming the input schema: the current documented model has six numeric/history inputs plus four time encodings per step, but exported artifacts may be from another run or code version.

For both models, include a global summary and a few local case studies with the true label, predicted class, confidence, and attribution visualization/table. Check attribution stability across reasonable backgrounds/seeds and run a sanity check (for example, compare with shuffled labels or a feature/token ablation). Correlated features can share or distort attribution; attributions describe model behaviour, not causal effects or human intent. Avoid presenting a feature as important solely because one example has a large attribution.

### 4. Write the final report

Suggested report structure:

1. **Question and scope:** target, populations, and intended comparison.
2. **Data and protocol:** full/lean definitions, filtering, class balance, temporal unit/window, voter split, leakage safeguards, and model configurations.
3. **Results:** test metrics with class-wise tables and confusion matrices; calibration and subgroup results where supported.
4. **Full-versus-lean and 08-versus-08b comparisons:** separate population effects from text-modality/model effects; state where comparisons are not equivalent.
5. **XAI diagnosis:** method, background/baseline, sample selection, global patterns, representative local cases, and stability/sanity checks.
6. **Limitations and conclusions:** selection effects, class imbalance, generalization limits, calibration, missing evidence, and conclusions proportionate to the results.

Every table/figure should identify the run, split, model, population, and sample count. Include software/commit and XAI settings for reproducibility. Preserve scripts/notebooks used for analysis and generated XAI outputs with the final report.

## Reproduction references

The repository's [README](../README.md) has Stage 08/08b artifact and IG/SHAP notes. The [leakage-safe runbook](LEAKAGE_SAFE_RUNBOOK.md) and [supervisor guide](SUPERVISOR_LEAKAGE_SAFE_GUIDE.md) describe the split, clustering, and temporal-window safeguards. The [no-RoBERTa runbook](SUPERVISOR_NO_ROBERTA_RUNBOOK.md) describes the 08b baseline and its feature schema. The [feature specification](Feature_Specification.md) documents the lean sampling caveat.
