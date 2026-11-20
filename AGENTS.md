# Codex instructions for DAOsynthvoter

Research goal: reproducible analysis of DAO voting behaviour and prediction.

Before every task:
- Read README.md, docs/RESEARCH_PAPER_WORKFLOW.md, and the real file tree. README references to stages may be stale.
- Use .agents/skills/dao-paper/SKILL.md for paper writing and research evaluation.
- Identify data source, experiment run, split, metric definition, and evidence paths before writing a result.
- Do NOT invent citations, results, data, confidence intervals, or model configurations.
- Check voting-class imbalance, leakage, temporal availability, voter/proposal overlap, and comparability of baselines.
- Distinguish predictive association from causal conclusions and governance simulations.
- Outputs under outputs/ are ignored. Do not assume the full lab-server results are in this repository.
- Never publish private results, personal identifiers, secrets or unpublished draft text in this public repository.
- Do not overwrite raw data or run full training without the user's request.
- For any code changes, explain changed files, tests, assumptions and limitations. Prefer a small branch and PR.

Paper manuscript must be held in a PRIVATE repo or private Overleaf project; paper-template/ here is an intentionally anonymous starter only.
