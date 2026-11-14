---
name: dao-paper
description: Create and review reproducible DAO Synthetic Voters academic-paper materials, including experiment audits, evidence-backed tables and charts, literature and Springer LNCS manuscript preparation. Use when drafting or evaluating the user's research paper.
---

# DAO paper workflow

Read AGENTS.md and docs/RESEARCH_PAPER_WORKFLOW.md.

1. Define one research question and one claim; never assume prediction implies causation.
2. Inventory the actual available experiment reports. Report which server artifacts are missing.
3. Create an evidence ledger: claim, source artifact, configuration, data split, model, seed, code revision, limitations.
4. Check baseline (majority-class and numeric-only), Macro-F1, per-class results, calibration, and split leakage. Check whether train, validation and test were used correctly.
5. Generate LaTeX tables and figure files from real experiment output, with an auditable source path. Do not type in or guess numerical findings.
6. Draft in a PRIVATE manuscript derived from paper-template; add TODOs for unverified facts. Keep bibliographic citations grounded in verified records.
7. Audit anonymous submission: no names, affiliations, identifiable acknowledgements, personal repository links, PDF metadata, or unapproved private data.
8. Check the current CAAW author instructions and Springer LNCS format, page count and rendering before any manual submission.

Never submit to EasyChair, publish a preprint, share non-public results, or email supervisors without the user's explicit request.
On completion state what you actually changed, verified and could not verify.
