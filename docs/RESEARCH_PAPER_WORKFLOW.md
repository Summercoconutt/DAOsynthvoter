# DAO Synthetic Voters — Codex academic publication workflow

## Tools and responsibilities
- Codex + this GitHub repository: inspect pipeline, plan experiments, audit code/results, prepare reproducible tables and figures.
- Private GitHub repository or private local folder: unpublished manuscript, confidential outcomes, author notes.
- Zotero: verified DOI/citation library, exported to BibTeX.
- Overleaf: private LNCS project; compile LaTeX PDF.
- EasyChair: manual final submission after human review.

This repository is PUBLIC. Do not place real unpublished paper text, figures that identify authors, or non-public lab results here. Keep model artifacts and large datasets outside Git.

## Initial setup
1. Open the repository Summercoconutt/DAOsynthvoter in Codex; Codex loads AGENTS.md.
2. Ask Codex to use the dao-paper skill in .agents/skills/dao-paper/SKILL.md.
3. Create a separate private paper repository and copy paper-template/ into it. Alternatively use a private Overleaf project from the official Springer LNCS template.
4. Use the official Springer LNCS template and its llncs.cls, not an improvised substitute.
5. Import reference records from Zotero as references.bib. Every citation must be verified.

## Work loop
A. Research question → hypotheses, scope, desired comparisons and defensible baselines.
B. Data audit → Snapshot/DAO coverage, target construction, negative/extreme VP, leakage, overlap, time windows.
C. Experiment audit → exact config/seed/code commit, valid train/validation/test process, class balance, majority baseline, numeric-only vs text models.
D. Evidence package → figures and tables generated from source outputs, including input path/hash, run ID, metric definition, split and limitations.
E. Private LaTeX draft → introduction, related work, data, method, experiments, results, limitations, conclusion.
F. Technical and submission review → provenance check, references, anonymity, layout, 15-page cap for CAAW Full Paper (including references and appendices), PDF rendering.

Do not equate accuracy with usefulness when the For class is dominant. Discuss baseline comparisons and per-class uncertainty. Governance simulation under changed rules requires separate behavioural assumptions; it is not a proven causal effect.

## Ready-to-use Codex prompts
1. Use dao-paper. Inventory the actual DAO repository and list missing experimental artifacts. Do not modify files.
2. Use dao-paper. Build an evidence ledger of all accessible numeric-only and RoBERTa evaluations: source path, train/val/test split, model config, Macro-F1, calibration, and unresolved comparability questions. Never invent values.
3. Use dao-paper. Prepare publication-quality visualisations only from verified model outputs and a provenance table; explain every figure's scientific claim.
4. Use dao-paper. Draft the Methods section in my PRIVATE LNCS manuscript. Add clear TODO markers where evidence is incomplete.
5. Use dao-paper. Review my PRIVATE manuscript for CAAW double-blind compliance, unverified numerical claims, citations and formatting.

## Publication rules
Official CAAW 2027: https://caaw.io/2027/
Submission deadline: 2026-11-20 23:59 AoE. Double blind; PDF in Springer LNCS; Full <= 15 pages, Short <= 8, including references and appendices. Confirm updated rules before submitting. No automatic conference submission.
