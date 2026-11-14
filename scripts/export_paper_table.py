#!/usr/bin/env python3
"""Export an existing CSV to a LaTeX table and record source provenance.

This formats metrics but does not validate or calculate them.
Use private destination directories for non-public results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
SPECIAL = re.compile(r"[\\&%$#_{}~^]")


def escape_latex(value: str) -> str:
    return SPECIAL.sub(lambda m: ESCAPE_MAP[m.group(0)], value)


def make_table(rows: list[list[str]], caption: str, label: str) -> str:
    if not rows or not rows[0]:
        raise ValueError("CSV needs a header row.")
    n_cols = len(rows[0])
    if n_cols > 12:
        raise ValueError("Too many columns for a paper table; select columns first.")
    if any(len(row) != n_cols for row in rows):
        raise ValueError("CSV has uneven row lengths.")
    if not re.fullmatch(r"[A-Za-z0-9:._-]+", label):
        raise ValueError("Invalid LaTeX label.")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + escape_latex(caption) + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + "l" * n_cols + "}",
        r"\toprule",
        " & ".join(escape_latex(v) for v in rows[0]) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(escape_latex(v) for v in row) + r" \\" for row in rows[1:])
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    target = args.output.expanduser().resolve()
    if source == target or target.suffix.lower() != ".tex":
        parser.error("Output must be a distinct .tex path.")
    source_bytes = source.read_bytes()
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    table = make_table(rows, args.caption, args.label)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(table, encoding="utf-8")
    manifest = {
        "input": str(source),
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output": str(target),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "columns": rows[0],
        "rows": len(rows) - 1,
        "note": "Formatting only; metric accuracy has NOT been validated.",
    }
    target.with_suffix(".provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {target} and provenance manifest.")


if __name__ == "__main__":
    main()
