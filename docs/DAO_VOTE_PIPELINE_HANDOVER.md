# DAO vote prediction: local agent handover

## Purpose

Inspect the actual Snapshot data, then use the workplan below to repair the pipeline. The earlier review was limited to source code and a small set of repository artifacts; it did **not** inspect the actual vote or proposal dataset or measure model performance. Verify all findings against the local checkout and data before relying on them.

The raw dataset and its location were not available in the review environment. Locate it from the local configuration or ask the user for the path. Do not scan unrelated directories, expose credentials, or overwrite raw data.

## Agreed prediction task

- Predict the recorded **for / against / abstain** choice, conditional on the voter casting a vote. This is not a prediction of whether or when a person will vote.
- Include only **single-choice proposals with exactly three options** clearly equivalent to for, against, and abstain. Map vote positions using that proposal's actual ordered options; exclude and report ambiguous or unsupported cases.
- Make each prediction **immediately before the vote**, using only information observable at that time.
- Use a **chronological, proposal-disjoint** train/validation/test split. The same voter may occur in multiple splits. Validation is for model and clustering choices; reserve test for the final selected setup.
- Voter and DAO history may update during validation and test using earlier observable votes. Do not fit model, preprocessing, or clustering parameters on test data.
- The primary model should fine-tune a RoBERTa-family encoder on proposal title and body, combine its representation with voter and DAO features, and use a simple three-class classification head.
- DAO clusters and voter clusters are optional model features. Voter clustering should be **separate within each DAO**. Keep sparse/new DAOs and voters with an explicit unknown or no-cluster category.
- Keep the additional history-sequence Transformer only if a fair comparison demonstrates an advantage.

## First: inspect the local data, read-only

1. Check the checkout and working tree, and inspect local configuration before running any pipeline:

   ```sh
   git rev-parse --show-toplevel
   git status --short
   ```

   Review `configs/default.yaml`, any local data-source overrides, and the launcher actually present in the local checkout. The paths in `configs/example_server_raw.yaml` are examples, not evidence that the data exists there.

2. Inventory the authorized vote and proposal files. Record file formats, sizes, schemas, row counts, date ranges, and how the tables join. Inspect Parquet metadata and use bounded or streaming aggregation where appropriate; do not load a potentially large corpus into memory just to discover its schema. Do not print voter addresses or credentials in the report.

3. Determine whether proposal records actually contain ordered options, voting type, title, complete body, creation/open/close times, and a stable `(space, proposal_id)` key. Determine which vote fields contain voter, choice, vote timestamp, voting power, and any revision identifiers. Check missingness and whether bodies were already truncated. If essential option/type metadata is absent, locate an authoritative source or report the blocker; do not infer labels from option numbers.

4. Report counts of eligible and excluded proposals/votes, with reasons; label proportions; voter and DAO coverage; duplicate/revote and timestamp issues; and history depth. Check whether each class has enough examples in the proposed time periods. If vote revisions exist and their intended target is unclear, ask the user rather than silently choosing the first or last record.

5. If an existing split manifest and training dataset are available, measure proposal-ID overlap and exact normalized title/body overlap between target splits. Check whether voter/DAO statistics and clusters were computed using the target vote or later events. Verify the provenance of majority alignment, vote-share, whale flags, and voting power against the actual data rather than assuming every export was built by the bundled collector.

6. Summarize findings and limitations before training. Use aggregate results, not raw records. If the required dataset or metadata is unavailable, state what is missing and stop rather than producing a misleading model.

## Findings from the reviewed checkout

These are source-based findings; the local checkout may differ.

### Data and labels

- Stages 1 and 2 largely report on existing artifacts rather than collecting the data (`scripts/01_run_data_selection.py`, `scripts/02_run_data_expansion.py`).
- The bundled collector requests proposal title and body, but not the option list or voting type; it truncates bodies to 800 characters (`src/data_expansion/441_platforms_parquet_UNIX.py`).
- Cleaning filters normalized vote labels but does not establish that each proposal has the required options (`src/pipeline/global_cleaning.py`). The voter standardizer assumes 1/2/3 mean for/against/abstain and reduces weighted votes to a highest-weight choice (`src/voter_clustering/config.py`, `src/voter_clustering/standardize_votes.py`).
- The checked-in `selected_spaces.json` declared and contained 1,739 distinct IDs under `selected_space_ids`. Stage 1 looked for a list or a `spaces` key, so it would report zero for that file. The only checked-in DAO feature table contained two rows; it is not the vote corpus.

### Features and leakage

- The active behaviour dataset uses the proposal title but not its body (`src/dao_governance/features/behaviour_dataset.py`). The default tokenizer limit is 128 tokens.
- The model receives voting power, vote-share, whale/alignment flags, and cluster IDs, but not the computed voter-history or DAO feature tables as direct numeric inputs (`src/dao_governance/modelling/preprocess.py`).
- In the bundled collector, alignment compares the target vote with the final majority, while vote-share and whale status depend on total proposal voting power. These are passed to the target row and are unavailable immediately before the vote. Current-vote voting power is usable only if independently reconstructable at prediction time.
- Stage 8 splits by voter **after** upstream aggregates and clusters are built (`scripts/08_run_behaviour_modelling.py`, `src/dao_governance/modelling/preprocess.py`). The same proposal can occur in multiple splits, and voter/DAO aggregates may include the current or future vote. The numeric scaler itself is fitted on train only, which does not repair earlier leakage.
- The current five-vote windows exclude each voter's first four votes (`src/dao_governance/modelling/windows.py`).

### Clustering and model

- In this checkout, stages 6 and 7 already feed stage 8; they are not replaced by stages 8/8b. Stage 7 requires DAO assignments and stage 8 requires the merged votes and voter assignments, so clustering is not optional.
- Voter-space profiles are clustered by **DAO cluster**, not independently within each DAO. K-means is chosen by silhouette on the supplied data; assignments, rather than a reusable train-only prediction pipeline, are exported. Cluster IDs are scaled as if their numerical order were meaningful.
- DAO clustering already has useful silhouette, null, and seed-stability diagnostics. Preserve useful diagnostics, but avoid fitting its final assignments on held-out or future data.
- Stage 8 is a classifier: fine-tuned `distilroberta-base` encodes titles; a second two-layer Transformer combines the text and numeric representations across the vote window; a neural head emits three logits (`src/dao_governance/modelling/model.py`). The second Transformer could model history interactions, but its advantage has not been demonstrated and it has no explicit sequence positional encoding.
- Stage 8b is not a clean text-only ablation: it also loses the history-label text tokens and changes model capacity and optimization. It can rebuild/overwrite the split and read a previous RoBERTa result without confirming a matching dataset and manifest.

### Runner and evaluation

- Stage 8 selects a checkpoint by validation macro-F1 but does not run test evaluation. `run_full_pipeline.py` ends at stage 8; an evaluation function exists without a tracked entrypoint calling it.
- The supplied launcher used `configs/data_sources.yaml`, `scripts/09_run_behaviour_evaluation.py`, and the flags `--data-sources-config`, `--run-id`, and `--text-encoder-mode`. Those files or flags did not match the reviewed checkout. Its claims that stages 6/7 are unnecessary, that the encoder can be frozen, and that runs are isolated must not be assumed for another checkout. Inspect the local launcher and scripts before executing them.

## Implementation workplan

### 1. Canonical source and eligibility

Wire stages 1–3 to the actual Snapshot exports or a documented collector without changing immutable raw files. Maintain a proposal table with identifiers, title, full body, timestamps, voting type, and ordered options, plus a vote-event table with provenance.

Map each vote using its proposal's actual options. Require exactly three unambiguous canonical options and an eligible single-choice voting type. Explicitly report missing, ambiguous, weighted, ranked, multi-choice, or otherwise unsupported cases; do not silently relabel or default them. Define duplicate and revote handling from the actual source semantics. Distinguish genuinely empty body text from unavailable text.

### 2. Split before fitted features

Create a versioned, reproducible manifest containing dataset fingerprint, proposal IDs, time cutoffs, and split membership. Keep each proposal wholly in one **target** split. Ensure training labels are known before validation predictions, and validation outcomes used for selection are known before test predictions. Purge or explicitly handle proposals spanning cutoffs.

Check proposal-ID and chronological disjointness. Audit identical title/body text on different proposal IDs and report a content-disjoint sensitivity evaluation, excluding cross-cutoff duplicates as necessary without breaking chronology. Report split-level class, DAO, and voter coverage.

Earlier observed votes from other proposals may be used as history; this does not put those earlier proposals in the target split and does not permit refitting on test.

### 3. Point-in-time voter and DAO features

Build direct per-voter, per-voter/DAO, and DAO features from events observable strictly before each target vote. Handle timestamp ties and insufficient history explicitly. Use final-outcome statistics only after the relevant proposal has closed, and never use the target choice or the current proposal's final totals. Exclude target voting power unless its pre-vote availability is verified.

Allow historical state to update from earlier observable validation/test events, as agreed, but fit imputation, scaling, and feature selection on training data only. Keep cold-start examples and mark unavailable history rather than silently fabricating values.

### 4. Optional DAO and per-DAO voter clustering

Decouple feature creation from stages 6–7 so models can run with or without either cluster feature. Fit candidate DAO clusterers and separate within-DAO voter clusterers using training history only. Use an initial training warm-up period, or a time-safe out-of-fold equivalent, so a training example's cluster cannot encode its own or future outcomes.

Persist fitted transformations and assignment models for later observations; do not refit them on test. Compare k-means with a suitable alternative such as a Gaussian mixture where data supports it. Refine feature sets and hyperparameters using validation, considering stability, minimum cluster sizes, interpretability, and predictive contribution, not silhouette alone.

Encode DAO cluster and `(DAO, local voter cluster)` categorically, with an explicit unknown/no-cluster fallback. Report coverage and compare against a no-cluster model on identical rows.

### 5. Classifier and matched baselines

Create one example per eligible vote rather than requiring five prior votes. Encode title and body with the RoBERTa-family model, concatenate the text representation with direct as-of features, and fine-tune a three-class classifier. Preserve full source text; report tokenizer truncation and use chunking only if warranted.

Compare majority and historical-choice baselines, numeric-only, text-only, fused, and fused-with-cluster models on the same examples and split manifest. Represent historical vote information consistently across ablations. A frozen text encoder may be an additional comparison, not a substitute for the agreed fine-tuned primary model. Ensure the best-validation checkpoint is actually preserved on CPU as well as GPU.

### 6. Evaluation and reproducibility

Select supervised and clustering settings on validation only, then freeze the selected setup for final test evaluation. Report macro-F1 over the fixed three classes, per-class precision/recall and support, confusion matrices, and useful accuracy/calibration measures. Report known/new-voter and sparse-history/DAO strata; account for correlated votes by aggregating or bootstrapping at proposal level.

Update the runner, configs, entrypoints, and documentation so the split precedes fitted feature selection/clustering, stages 6–7 are optional, stages 8/8b are supervised runs, and a real evaluation entrypoint follows them. Persist run-specific data/split fingerprints, configuration, model, preprocessing and cluster artifacts, and metrics. Do not run full training merely as a data inspection step.

## Acceptance checks

- Reversed option ordering maps correctly; missing or ambiguous metadata and weighted/multi-choice votes are reported and excluded.
- No target proposal appears in more than one split; prediction times respect the chosen cutoffs. The same voter may appear across splits.
- Changing the target vote or future votes cannot change an earlier example's features or cluster assignment. Earlier observable history may change later features.
- Direct voter/DAO features and both title and body reach the classifier. Cold-start records remain eligible.
- Cluster IDs are categorical, sparse cases have an explicit fallback, and inference uses persisted train-fitted clusterers. Running without clusters requires no cluster files.
- All compared models use the same target rows and split manifest; test data is not used to select hyperparameters or cluster settings.
- Synthetic integration tests exercise the pipeline and launcher without requiring a production-sized dataset or an external model download. A local-data audit documents actual coverage and any remaining blockers.
