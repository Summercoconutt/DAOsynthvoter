# DAO vote prediction: local agent handover

**Branch:** `carlo_dev`. This handover is based on source at `6e6fd77` and supersedes the earlier handover based on `main`. Recheck the local revision and data before acting.

## Purpose and agreed task

Inspect the actual Snapshot data, then use the workplan below to align the pipeline with the intended prediction task. The source review did **not** inspect the external vote/proposal data or trained run artifacts, so actual data coverage, leakage rates, and model performance remain unverified.

- Predict the recorded **for / against / abstain** choice, conditional on the voter casting a vote. This does not predict whether or when someone votes.
- Include only **single-choice proposals with exactly three options** clearly equivalent to for, against, and abstain. Map vote positions using that proposal's actual ordered options; exclude and report ambiguous or unsupported cases.
- Make each prediction **immediately before the vote**, using only information observable at that time.
- Use a **chronological, proposal-disjoint** train/validation/test split. Voters may recur across splits. Use validation for model and clustering choices; reserve test for the final selected setup.
- Voter and DAO history may update during validation and test using earlier observable votes. Do not fit model, preprocessing, or clustering parameters on test data.
- The primary model should fine-tune a RoBERTa-family encoder on proposal title and body, combine its representation with voter and DAO features, and use a simple three-class classification head.
- DAO clusters and voter clusters are optional model features. Voter clustering should be **separate within each DAO**. Keep sparse/new DAOs and voters with an explicit unknown or no-cluster category.
- Keep the additional history-sequence Transformer only if a fair comparison demonstrates an advantage.

## First: inspect the local data, read-only

1. Confirm the branch, revision, working tree, and local configuration before running any pipeline:

   ```sh
   git branch --show-current
   git rev-parse HEAD
   git status --short
   ```

   Inspect `configs/default.yaml`, the local data-source override, and `run_all.sh`. The default `configs/data_sources.yaml` is intentionally Git-ignored and may exist only locally; `configs/data_sources_lean.yaml` is a separate example/override. Do not commit credentials or local-only configuration.

2. Locate only the authorized vote and proposal files. Record file formats, sizes, schemas, row counts, date ranges, and join keys. Use Parquet metadata and bounded or streaming aggregation where appropriate; do not load a potentially large corpus into memory just to discover its schema. Do not print voter addresses or credentials in the report.

3. Determine whether proposal records actually contain ordered options, voting type, title, complete body, creation/open/close times, and a stable `(space, proposal_id)` key. Determine which vote fields contain voter, choice, vote timestamp, voting power, and any revision identifiers. Check missingness and whether bodies were already truncated. If essential option/type metadata is absent, locate an authoritative source or report the blocker; do not infer labels from option numbers.

4. Report eligible and excluded proposals/votes, with reasons; label proportions; voter and DAO coverage; duplicate/revote and timestamp issues; and history depth. Check whether each class has enough examples in the proposed time periods. If vote revisions exist and their intended target is unclear, ask the user rather than silently choosing the first or last record.

5. If existing run artifacts are available under `outputs/stage08_runs/<variant>/<run-id>/`, verify their manifest, hashes, configuration, and input paths before using them. `docs/STAGE08_MODEL_ANALYSIS_HANDOVER.md` describes expected artifacts and a possible lean run-ID mismatch; verify it locally and do not silently rename or edit old runs.

6. For an existing voter split, measure proposal-ID overlap and exact normalized title/body overlap between target splits. Audit whether DAO and voter features or cluster assignments use a target or later vote. Verify the provenance of `is_whale`, voting power, and proposal outcomes against the actual exports. Report aggregate counts and proportions, not raw records.

7. Quantify how the full and lean datasets differ in retained proposals, votes, classes, and DAOs. The lean sample is outcome-selected, so do not present its results as unbiased performance over all eligible proposals.

Summarize findings and limitations before training. If the required dataset or metadata is unavailable, state what is missing and stop rather than producing a misleading model.

## Current implementation on `carlo_dev`

### Data and eligibility

- Stages 1 and 2 largely report on existing artifacts rather than collecting the data (`scripts/01_run_data_selection.py`, `scripts/02_run_data_expansion.py`).
- The bundled collector requests proposal title and body, but not the option list or voting type; it truncates bodies to 800 characters (`src/data_expansion/441_platforms_parquet_UNIX.py:287`, `:563`). Actual external exports may differ.
- Cleaning filters normalized vote labels but does not establish that each proposal has the required options (`src/pipeline/global_cleaning.py:200`). A voter standardizer assumes 1/2/3 mean for/against/abstain and reduces weighted votes to a highest-weight choice (`src/voter_clustering/config.py:74`, `src/voter_clustering/standardize_votes.py:29`). Verify which source produced the actual `choice_norm` values.
- The checked-in `selected_spaces.json` declared and contained 1,739 distinct IDs under `selected_space_ids`. Stage 1 looked for a list or a `spaces` key, so it would report zero for that file (`scripts/01_run_data_selection.py:44`). This metadata is not the vote corpus.
- Stage 3 can write a separate lean sample using final for/against vote fractions and text-length thresholds (`src/dao_governance/data/proposal_sampling.py:16`). `cleaning.lean_sample.enabled` is true in the reviewed default config, but the canonical cleaned output remains separate; the lean source override selects the lean output. Treat the outcome-selected sample as a separate, retrospective experiment.

### Text, model, and inputs

- Stage 08 supports `text_mode: title_body`, but the reviewed default is `title`; `max_length` defaults to 128 tokens (`configs/default.yaml:99`, `src/dao_governance/features/behaviour_dataset.py:40`). Complete body availability must still be verified.
- The model is a **three-class classifier**: `distilroberta-base` encodes each window text; a second two-layer Transformer combines projected text and numeric inputs across the history window; a neural head emits three logits (`src/dao_governance/modelling/model.py`). Fine-tuned and frozen encoder modes are both implemented. The second Transformer could model history interactions, but its advantage has not been demonstrated and it has no explicit sequence positional encoding.
- Stage 08 receives `voting_power`, `is_whale`, `dao_cluster`, `voter_cluster`, and four calendar features per step (`src/dao_governance/modelling/preprocess.py:154`, `src/dao_governance/modelling/windows.py:37`). The computed DAO and voter structural statistics are not passed directly to the classifier; they are used indirectly to form clusters.
- Direct `aligned_with_majority` and `vp_share` inputs have been removed. However, `is_whale` remains: the bundled collector calculates it using each vote's share of the **final** proposal total (`src/data_expansion/441_platforms_parquet_UNIX.py:551`, `:568`, `:589`). Check actual export provenance and exclude or reconstruct it from genuinely pre-vote information. Current-vote voting power is usable only if independently available before the vote.
- Windows are grouped by `(voter, space)` and default to five votes, so the first four votes of each voter/DAO pair are not prediction examples (`src/dao_governance/modelling/windows.py:81`).

### Split and clusters

- Stage 08 splits randomly by voter, not by proposal or time (`scripts/08_run_behaviour_modelling.py:243`, `src/dao_governance/modelling/preprocess.py:168`). The same proposal may therefore appear in multiple splits; measure the overlap locally.
- Stages 6–7 are exploratory in this branch. Stage 08 instead fits structural K-means models after the voter split, saves a cluster bundle, and exposes `use_dao_clusters` and `use_voter_clusters` flags. This is an improvement over pre-split label-heavy assignments, but it does not by itself establish point-in-time safety.
- DAO clustering filters a precomputed DAO feature table to spaces seen in training (`src/dao_governance/features/causal_clusters.py:110`). Stage 4 normally builds that table from the supplied vote history before the model split (`scripts/04_run_dao_metrics.py`, `src/dao_clustering_scripts/02_compute_metrics.py`), so its descriptors can contain held-out or future information. Inspect the actual table's provenance.
- Voter cluster fitting uses structural, not label-proportion, features; however, assignment recomputes each `(voter, space)` profile over the entire supplied split, including its final vote count and last timestamp (`src/dao_governance/features/causal_clusters.py:158`, `:257`, `:320`). For earlier target rows this can expose current or later activity even though test labels are not used in fitting.
- The predictive voter K-means is fitted globally over voter/DAO pairs, not as a separate model per DAO. Its k-selection minimizes training inertia alone, which tends to favor larger k; cluster IDs enter the model as numerical values (`src/dao_governance/features/causal_clusters.py:87`, `:197`; `src/dao_governance/modelling/preprocess.py:154`).
- The flags can disable cluster use, but `resolve_stage_inputs` still requires a DAO feature table even if DAO clustering is disabled (`src/dao_governance/modelling/run_artifacts.py:50`). Remove that unnecessary dependency in the redesigned no-cluster path.

### Paired runs and evaluation

- Stage 08b now reuses the run-scoped dataset and voter split, and Stage 09 checks run identity, source hashes, and the shared split before paired evaluation (`scripts/08b_run_behaviour_modelling_no_roberta.py`, `scripts/09_run_behaviour_evaluation.py:83`). Stage 09 exists and reports held-out classification and calibration metrics; these metrics still answer the current voter-split question.
- The 08/08b comparison is not solely a text ablation: Stage 08 represents prior labels using text markers, while 08b adds `prior_frac_for` and `prior_frac_against` numeric features; model capacity and optimization also differ. Prior fractions sort timestamp ties by proposal/source order, so strict pre-vote availability for equal timestamps needs a test (`src/dao_governance/features/behaviour_pipeline.py:25`).
- `run_all.sh` runs 08, 08b, and 09 with run IDs and optional frozen mode; its earlier stage commands are commented out. Confirm prerequisites and local data-source configuration rather than assuming it collects raw data. Existing run manifests, evaluation, and export tooling should be reused where compatible with the new protocol.

## Implementation workplan

### 1. Canonical source and eligibility

Wire stages 1–3 to the actual Snapshot exports or a documented collector without changing immutable raw files. Maintain a proposal table with identifiers, title, full body, timestamps, voting type, and ordered options, plus a vote-event table with provenance.

Map each vote using its proposal's actual options. Require exactly three unambiguous canonical options and an eligible single-choice voting type. Explicitly report missing, ambiguous, weighted, ranked, multi-choice, or otherwise unsupported cases; do not silently relabel or default them. Define duplicate and revote handling from actual source semantics. Distinguish genuinely empty body text from unavailable text. Correct Stage 1's selected-space parsing.

Use the full eligible population for the primary experiment. Preserve the lean variant only as a clearly labelled, outcome-selected sensitivity analysis.

### 2. Split before fitted features

Create a versioned, reproducible manifest containing dataset fingerprint, proposal IDs, time cutoffs, and split membership. Keep each proposal wholly in one **target** split. Ensure training labels are known before validation predictions, and validation outcomes used for selection are known before test predictions. Purge or explicitly handle proposals spanning cutoffs.

Check proposal-ID and chronological disjointness. Audit identical title/body text on different proposal IDs and report a content-disjoint sensitivity evaluation, excluding cross-cutoff duplicates as necessary without breaking chronology. Report split-level class, DAO, and voter coverage.

Earlier observed votes from other proposals may be used as history; this does not put those earlier proposals in the target split and does not permit refitting on test.

### 3. Point-in-time voter and DAO features

Build direct per-voter, per-voter/DAO, and DAO features from events observable strictly before each target vote. Handle timestamp ties and insufficient history explicitly. Use final-outcome statistics only after the relevant proposal has closed, and never use the target choice or the current proposal's final totals. Exclude target voting power and whale flags unless their pre-vote availability is verified.

Allow historical state to update from earlier observable validation/test events, as agreed, but fit imputation, scaling, and feature selection on training data only. Keep cold-start examples and mark unavailable history rather than silently fabricating values.

### 4. Optional DAO and per-DAO voter clustering

Move predictive DAO clustering to stage 6 and per-DAO voter clustering to stage 7, using the shared split manifest; stages 08/08b should consume their persisted outputs rather than fit separate cluster models. Keep exploratory clustering distinct. Decouple feature creation from clustering so either cluster feature can be disabled without unused prerequisites. Fit candidate DAO clusterers and separate within-DAO voter clusterers using training history only. Use an initial training warm-up period, or a time-safe out-of-fold equivalent, so a training example's cluster cannot encode its own or future outcomes.

Persist fitted transformations and assignment models for later observations; do not refit them on test. Compare k-means with a suitable alternative such as a Gaussian mixture where data supports it. Refine feature sets and hyperparameters using validation, considering stability, minimum cluster sizes, interpretability, and predictive contribution, not inertia alone. Preserve useful existing diagnostics.

Encode DAO cluster and `(DAO, local voter cluster)` categorically, with an explicit unknown/no-cluster fallback. Report coverage and compare against a no-cluster model on identical rows.

### 5. Classifier and matched baselines

Create one example per eligible vote rather than requiring five prior votes. Encode title and body with the RoBERTa-family model, concatenate the text representation with direct as-of features, and fine-tune a three-class classifier. Preserve full source text; report tokenizer truncation and use chunking only if warranted.

Compare majority and historical-choice baselines, numeric-only, text-only, fused, and fused-with-cluster models on the same examples and split manifest. Represent historical vote information consistently across ablations. A frozen text encoder may be an additional comparison, not a substitute for the agreed fine-tuned primary model. Verify the best-validation checkpoint is actually preserved on CPU as well as GPU.

### 6. Evaluation and reproducibility

Select supervised and clustering settings on validation only, then freeze the selected setup for final test evaluation. Report macro-F1 over the fixed three classes, per-class precision/recall and support, confusion matrices, and useful accuracy/calibration measures. Report known/new-voter and sparse-history/DAO strata; account for correlated votes by aggregating or bootstrapping at proposal level.

Extend the existing run-scoped manifest, identity checks, evaluation, and export mechanisms for proposal/time splits. Include stable proposal and example identifiers in prediction outputs so paired models and explanations can be aligned. Update the runner, configs, entrypoints, and documentation so the split precedes fitted feature selection/clustering, stages 6–7 are optional, and test evaluation follows model selection. Preserve old run artifacts as historical evidence; do not overwrite or silently relabel them.

## Acceptance checks

- Reversed option ordering maps correctly; missing or ambiguous metadata and weighted/multi-choice votes are reported and excluded.
- No target proposal appears in more than one split; prediction times respect the chosen cutoffs. The same voter may appear across splits.
- Changing the target vote or future votes cannot change an earlier example's features or cluster assignment. Earlier observable history may change later features; tied timestamps are handled conservatively.
- Direct voter/DAO features and both title and body reach the classifier. Cold-start records remain eligible.
- Cluster IDs are categorical, sparse cases have an explicit fallback, and inference uses persisted train-fitted clusterers. Running without clusters requires no cluster files or unused DAO feature table.
- All compared models use the same target rows and split manifest; test data is not used to select hyperparameters or cluster settings.
- Extend the existing synthetic tests in `tests/`, including `test_leakage_safe_pipeline.py`, to cover proposal-disjoint splits, future **structural** information, tied timestamps, optional clusters, and matched ablations. Integration checks should not require production data or an external model download.
- A local-data audit documents actual coverage, split overlap, feature provenance, and any remaining blockers before a full training run.
