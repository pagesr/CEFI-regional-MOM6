#!/usr/bin/env python3
"""Run one OBC PHY daily workflow task from a TSV row by index."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
FORECAST_CGOA_DIR = THIS_DIR / "standalone" / "forecast_cgoa"


def _load_row(task_file: Path, task_id: int) -> dict[str, str]:
    with task_file.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if task_id < 0 or task_id >= len(rows):
        raise IndexError(f"task_id={task_id} out of range for {task_file} (n={len(rows)})")
    return rows[task_id]


def _run(cmd: list[str]) -> None:
    print("[TASK]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _python_executable() -> str:
    """
    Resolve python executable for child stage scripts.
    Prefer explicit PYTHON_BIN, then current interpreter, then PATH fallback.
    """
    if os.environ.get("PYTHON_BIN"):
        return os.environ["PYTHON_BIN"]
    if sys.executable:
        return sys.executable
    return shutil.which("python") or shutil.which("python3") or "python"


def run_stage(stage: str, row: dict[str, str], output_root: Path, force: bool) -> None:
    py = _python_executable()
    if stage == "phy":
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
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    if force:
        cmd.append("--force")

    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one task from a stage TSV file")
    parser.add_argument("--stage", choices=["phy"], required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Keep stage logs with workflow outputs (instead of defaulting to forecast_cgoa/logs).
    os.environ["FORECAST_LOG_ROOT"] = str(Path(args.output_root).resolve() / "logs")

    row = _load_row(Path(args.task_file), args.task_id)
    run_stage(args.stage, row, Path(args.output_root), force=args.force)


if __name__ == "__main__":
    main()
