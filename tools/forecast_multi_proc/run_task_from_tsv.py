#!/usr/bin/env python3
"""Run one workflow task from a TSV row by index."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORECAST_CGOA_DIR = REPO_ROOT / "tools" / "forecast_cgoa"


def _load_row(task_file: Path, task_id: int) -> dict[str, str]:
    with task_file.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if task_id < 0 or task_id >= len(rows):
        raise IndexError(f"task_id={task_id} out of range for {task_file} (n={len(rows)})")
    return rows[task_id]


def _run(cmd: list[str]) -> None:
    print("[TASK]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_stage(stage: str, row: dict[str, str], output_root: Path, force: bool) -> None:
    py = "python"
    if stage == "ic":
        cmd = [
            py,
            str(FORECAST_CGOA_DIR / "run_ic.py"),
            "--ic-phy-config",
            row["ic_phy_config"],
            "--ic-bgc-config",
            row["ic_bgc_config"],
            "--year",
            row["year"],
            "--month",
            row["month"],
            "--output-root",
            str(output_root),
        ]
    elif stage == "phy":
        cmd = [
            py,
            str(FORECAST_CGOA_DIR / "run_phy_obc.py"),
            "--config",
            row["obc_phy_config"],
            "--year",
            row["year"],
            "--month",
            row["month"],
            "--ensemble",
            row["ensemble"],
            "--output-root",
            str(output_root),
        ]
    elif stage == "bgc":
        cmd = [
            py,
            str(FORECAST_CGOA_DIR / "run_bgc_obc.py"),
            "--config",
            row["obc_bgc_config"],
            "--year",
            row["year"],
            "--month",
            row["month"],
            "--ensemble",
            row["ensemble"],
            "--output-root",
            str(output_root),
        ]
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    if force:
        cmd.append("--force")

    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one task from a stage TSV file")
    parser.add_argument("--stage", choices=["ic", "phy", "bgc"], required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    row = _load_row(Path(args.task_file), args.task_id)
    run_stage(args.stage, row, Path(args.output_root), force=args.force)


if __name__ == "__main__":
    main()
