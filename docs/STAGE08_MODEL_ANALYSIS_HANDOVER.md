# Handover: Stage 08 model analysis and final report

## Purpose

You will receive two Stage 08 export ZIPs: a **full run** and a **lean run**. Please analyse the trained behaviour models, diagnose them with explainable-AI (XAI) methods, and use the evidence to write the final report. This handover describes the intended models, the checks needed before analysis, and the expected report contents.

The prediction target is a voter's next recorded choice: `FOR`, `AGAINST`, or `ABSTAIN`. This is predictive modelling, not a causal analysis of why a voter chose an option.

## First: verify that each ZIP is self-consistent

The ZIPs should contain a top-level `stage08_08b_<TAG>/` directory, with the following run-matched artifacts:

| Contents | Stage 08: RoBERTa model | Stage 08b: numeric-only model |
|---|---|---|
| Model files | `08/behaviour_agent2/model.pt`, `config.json`, and `tokenizer/` | `08b/agent2_artifacts_no_roberta/model.pt` and `config.json` |
| Data/split | `08/split_manifest.json` and `outputs/tables/behaviour_dataset_with_clusters_08_<TAG>.csv` | `outputs/tables/behaviour_dataset_with_clusters_08b_<TAG>.csv`; matching split manifest |
| Training/preprocessing evidence | `08/training_report.md`, `08/preprocessing_report.md` | `08b/training_report_no_roberta.md`, `08b/preprocessing_report_no_roberta.md` |
| Helpful supporting files | `08/predictive_clusters/` | `08b/roberta_vs_no_roberta_comparison.csv`; optional window cache if explicitly included |

Both model configs should agree with the intended run's sample, feature schema, label mapping, window size, preprocessing, and split. The 08b run should reuse the Stage 08 split for that same full or lean sample. Check file sizes and timestamps as well as names; a matching filename alone does not prove that files came from the same run. Record each archive's tag, source commit, and any recorded run/config metadata.

**Important packaging limitation:** `scripts/export_stage08_run.sh` uses `TAG` to name the bundle and locate tagged CSV snapshots, but its model, cluster, split, report, and comparison-file source paths are fixed to the default (full-run) locations. It does not select the `_lean` artifact paths based on the tag. It also does not include Stage 09 test/validation evaluation reports. Therefore, do not assume the `lean` ZIP contains lean model artifacts, or treat training reports as held-out test results. If an archive lacks a model/config/data/split set that agrees internally, pause analysis and request a corrected export with matching run-specific artifacts.

For final performance analysis, request the Stage 09 **test-split** outputs for each run if they are not supplied separately. Ideally include `metrics_summary.json`, `metrics_report.md`, `per_class_metrics.csv` or `classification_report.csv`, confusion-matrix counts, calibration/reliability outputs, and cluster-stratified metrics. Preserve the matching split manifest and source dataset with each run.

## What “full”, “lean”, and “08b” mean

- **Full vs lean** describes the data/sample variant, not the model architecture. The lean data path is an experimental filtered sample. In the documented setup it can exclude entire proposals based on consensus and title/body-length rules. Verify the actual cleaning config and proposal audit for this run; do not infer the filter settings from the word “lean”.
- **Stage 08 vs Stage 08b** describes the model comparison: Stage 08 uses proposal text (normally title) plus numeric/time inputs; Stage 08b is a numeric-only model with causal historical-choice features and no proposal text. 08b is an ablation, not another name for the lean run.
- The documented leakage-safe setup splits voters into train/validation/test, fits preprocessing and structural clusters on training data, and forms temporal windows within `(voter, space)`. Stage 08b should use the Stage 08 split from the same data variant so the within-run comparison is fair.
- Lean results apply only to the retained, filtered population. Because the sampling/filtering is outcome-related, present them as a sensitivity/subpopulation analysis, not as an unbiased estimate for the full population. Full-versus-lean metric differences may reflect population selection as well as data volume.

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
