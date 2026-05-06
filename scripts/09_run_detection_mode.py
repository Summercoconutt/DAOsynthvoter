#!/usr/bin/env python3
"""
Detection mode: scan uncleaned raw data (read-only), verify behaviour-model inputs,
optionally run a tiny smoke training run.

Typical workflow (server raw on D:\\111111\\Data, cleaning only locally):

1. Point cleaning input at an absolute parquet path (never writes into Data\\):

     python scripts/03_run_global_cleaning.py --config configs/default.yaml ^
       --extra-config configs/example_server_raw.yaml

   (Edit example_server_raw.yaml with the real filename.)

2. Run clustering / voter stages as usual until master_with_dao + voter assignments exist.

3. Dry-run checks:

     python scripts/09_run_detection_mode.py

4. Confirm smoke train completes:

     python scripts/09_run_detection_mode.py --smoke-train
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.settings import load_config, project_root
from pipeline.detection import (
    ScanResult,
    check_behaviour_prerequisites,
    iter_data_files,
    render_scan_results,
    scan_csv_quick,
    scan_parquet_quick,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Behaviour pipeline detection / smoke train.")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument(
        "--extra-config",
        type=str,
        default="",
        help="Optional YAML merged over --config (paths, detection overrides).",
    )
    ap.add_argument(
        "--raw-root",
        type=str,
        default="",
        help="Override detection.external_raw_root (e.g. if raw data not under default D: path).",
    )
    ap.add_argument("--skip-raw-scan", action="store_true", help="Skip scanning external_raw_root.")
    ap.add_argument(
        "--smoke-train",
        action="store_true",
        help="After checks, run stage 08 with configs/smoke_behaviour.yaml (1 epoch, capped windows).",
    )
    ap.add_argument(
        "--force-smoke-train",
        action="store_true",
        help="Run smoke train even if prerequisite checks fail (debug only).",
    )
    args = ap.parse_args()

    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=cfg_path, extra_path=extra)

    det = cfg.get("detection") or {}
    raw_root = Path(args.raw_root.strip() or det.get("external_raw_root") or "").expanduser()
    glob_pat = str(det.get("parquet_glob") or "**/*.parquet")
    max_files = int(det.get("max_files_to_scan") or 12)
    sample_rows = int(det.get("sample_rows_per_file") or 60_000)

    lines = [
        "# Detection mode report",
        "",
        f"- Generated: **{datetime.now(timezone.utc).isoformat()}**",
        f"- Config: `{cfg_path}`",
        f"- Extra: `{extra}`" if extra else "- Extra: _(none)_",
        "",
        "## A. Behaviour-model prerequisites",
        "",
    ]

    ok_pre, pre_lines = check_behaviour_prerequisites(cfg, base)
    lines.extend(pre_lines)
    lines.append("")
    if ok_pre:
        lines.append("**Prerequisite summary: PASS**")
    else:
        lines.append("**Prerequisite summary: FAIL** — fix paths above before relying on smoke/full train.")

    lines.extend(["", "## B. Raw data scan (read-only)", ""])

    if args.skip_raw_scan:
        lines.append("_Skipped (`--skip-raw-scan`)._")
    elif not raw_root or not raw_root.is_dir():
        lines.append(f"_External raw root missing or not a directory: `{raw_root}` — set `detection.external_raw_root` or `--raw-root`._")
    else:
        lines.append(f"- Scanning: `{raw_root}` glob `{glob_pat}` (max **{max_files}** files)")
        files = iter_data_files(raw_root, glob_pat, limit=max_files)
        if not files:
            lines.append("- No `.parquet`/`.csv` files matched (adjust glob or path).")
        else:
            scans = []
            for fp in files:
                try:
                    if fp.suffix.lower() == ".parquet":
                        scans.append(scan_parquet_quick(fp, max_rows=sample_rows))
                    else:
                        scans.append(scan_csv_quick(fp, max_rows=sample_rows))
                except Exception as exc:
                    scans.append(
                        ScanResult(
                            path=str(fp.resolve()),
                            rows=-1,
                            cols=[],
                            vp_column=None,
                            notes=[repr(exc)],
                        )
                    )
            lines.extend(render_scan_results(scans))

    lines.extend(
        [
            "",
            "## C. Cleaning reminder",
            "",
            "- Raw stays under `D:/111111/Data` (or your server mirror); **never** overwrite it.",
            "- Set `paths.master_votes_parquet` to an **absolute** parquet path in a small merge YAML, then:",
            "",
            "  `python scripts/03_run_global_cleaning.py --config configs/default.yaml --extra-config configs/example_server_raw.yaml`",
            "",
            "- Writes cleaned votes **only** under this repo (`paths.cleaned_master_parquet`).",
            "",
        ]
    )

    report_path = base / "outputs" / "reports" / "detection_mode_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[09] Wrote: {report_path}")

    if args.smoke_train:
        if not ok_pre and not args.force_smoke_train:
            print("[09] Smoke train skipped (prerequisites failed). Use --force-smoke-train to override.")
            sys.exit(1 if not ok_pre else 0)

        smoke_yaml = ROOT / "configs" / "smoke_behaviour.yaml"
        if not smoke_yaml.exists():
            raise FileNotFoundError(smoke_yaml)

        env = os.environ.copy()
        sep = os.pathsep
        env["PYTHONPATH"] = str(SRC) + sep + env.get("PYTHONPATH", "")

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "08_run_behaviour_modelling.py"),
            "--config",
            args.config,
            "--extra-config",
            "configs/smoke_behaviour.yaml",
        ]
        print("[09] Running smoke train:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(base), env=env, check=True)
        print("[09] Smoke train finished OK.")

    if not ok_pre and not args.smoke_train:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
