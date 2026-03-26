#!/usr/bin/env python3
"""Submit OBC PHY daily workflow as a Slurm array with configurable concurrency."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from build_task_lists import build_task_lists

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_ROOT = THIS_DIR / "tasks"


def _count_rows(tsv_path: Path) -> int:
    with tsv_path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream, delimiter="\t"))


def _sbatch(cmd: list[str]) -> str:
    print("[SBATCH]", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True).strip()
    # Expected: "Submitted batch job 12345"
    job_id = out.split()[-1]
    print(f"[SBATCH] submitted job_id={job_id}")
    return job_id


def _resolve_slurm_logs_dir(output_root: Path, requested_log_dir: str | None) -> Path:
    """
    Pick a writable directory for Slurm stdout/stderr files.

    Priority:
      1) --slurm-log-dir (if provided and writable)
      2) <output-root>/logs (legacy default; if writable)
      3) tools/forecast_multi_proc/logs (always inside workflow tree)
    """
    candidates: list[Path] = []
    if requested_log_dir:
        candidates.append(Path(requested_log_dir).expanduser())
    else:
        candidates.append(output_root / "logs")
    candidates.append(THIS_DIR / "logs")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
            return candidate.resolve()
        except OSError:
            continue

    raise RuntimeError(
        "Could not find a writable Slurm log directory. "
        "Pass --slurm-log-dir to a writable location."
    )


def _build_export(task_file: Path, output_root: Path, force: bool, conda_env_path: str) -> str:
    force_flag = "--force" if force else ""
    return (
        f"TASK_FILE={task_file},"
        f"OUTPUT_ROOT={output_root},"
        f"REPO_ROOT={REPO_ROOT},"
        f"FORCE_FLAG={force_flag},"
        f"CONDA_ENV_PATH={conda_env_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit PHY OBC daily array")
    parser.add_argument("--years", nargs="*", default=["2012"])
    parser.add_argument("--months", nargs="*", default=["01"])
    parser.add_argument("--ensembles", nargs="*", default=["01"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT))
    parser.add_argument(
        "--slurm-log-dir",
        default=None,
        help=(
            "Directory for Slurm stdout/stderr logs. "
            "If omitted, submit_workflow tries <output-root>/logs first "
            "and falls back to tools/forecast_multi_proc/logs if needed."
        ),
    )
    parser.add_argument("--max-parallel", type=int, default=20, help="Max concurrent array tasks")
    parser.add_argument(
        "--conda-env-path",
        default="/nbhome/role.medgrp/.conda/envs/medpy311",
        help="Conda environment path activated inside each Slurm array task",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    logs_dir = _resolve_slurm_logs_dir(output_root, args.slurm_log_dir)
    print(f"[LOGS] Using Slurm log directory: {logs_dir}")

    phy_task_file = build_task_lists(
        years=args.years,
        months=args.months,
        ensembles=args.ensembles,
        output_root=output_root,
        config_root=Path(args.config_root),
        task_root=Path(args.task_root),
    )

    nphy = _count_rows(phy_task_file)
    print(f"[TASKS] PHY={nphy}")

    if nphy == 0:
        raise RuntimeError("No PHY tasks found; check years/months/ensembles inputs")

    if args.dry_run:
        print(
            "[DRY-RUN] sbatch "
            f"--array=0-{nphy-1}%{args.max_parallel} "
            f"--output={logs_dir}/%x_%A_%a.out "
            f"--error={logs_dir}/%x_%A_%a.err "
            f"--export={_build_export(phy_task_file, output_root, args.force, args.conda_env_path)} "
            f"{THIS_DIR / 'submit_phy_array.slurm'}"
        )
        return

    phy_cmd = [
        "sbatch",
        f"--array=0-{nphy-1}%{args.max_parallel}",
        f"--output={logs_dir}/%x_%A_%a.out",
        f"--error={logs_dir}/%x_%A_%a.err",
        f"--export={_build_export(phy_task_file, output_root, args.force, args.conda_env_path)}",
        str(THIS_DIR / "submit_phy_array.slurm"),
    ]

    phy_job = _sbatch(phy_cmd)
    print(f"[SUBMITTED] phy_job={phy_job}")


if __name__ == "__main__":
    main()
