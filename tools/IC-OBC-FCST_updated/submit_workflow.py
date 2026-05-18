#!/usr/bin/env python3
"""Submit multi-process CGOA workflow as Slurm arrays with configurable concurrency."""

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
    parser = argparse.ArgumentParser(description="Submit IC/PHY/BGC arrays with dependencies")
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
    parser.add_argument(
        "--no-ic-dependency",
        action="store_true",
        help="Submit PHY/BGC arrays immediately (do not wait for IC array afterok).",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    logs_dir = _resolve_slurm_logs_dir(output_root, args.slurm_log_dir)
    print(f"[LOGS] Using Slurm log directory: {logs_dir}")

    files = build_task_lists(
        years=args.years,
        months=args.months,
        ensembles=args.ensembles,
        output_root=output_root,
        config_root=Path(args.config_root),
        task_root=Path(args.task_root),
    )

    nic = _count_rows(files["ic"])
    nphy = _count_rows(files["phy"])
    nbgc = _count_rows(files["bgc"])
    print(f"[TASKS] IC={nic} PHY={nphy} BGC={nbgc}")

    if nic == 0 or nphy == 0 or nbgc == 0:
        raise RuntimeError("No tasks found for one or more stages; check years/months/ensembles inputs")

    ic_cmd = [
        "sbatch",
        f"--array=0-{nic-1}%{args.max_parallel}",
        f"--output={logs_dir}/%x_%A_%a.out",
        f"--error={logs_dir}/%x_%A_%a.err",
        f"--export={_build_export(files['ic'], output_root, args.force, args.conda_env_path)}",
        str(THIS_DIR / "submit_ic_array.slurm"),
    ]

    if args.dry_run:
        print("[DRY-RUN]", " ".join(ic_cmd))
        if args.no_ic_dependency:
            print("[DRY-RUN] Would submit PHY/BGC immediately (no dependency on IC)")
        else:
            print("[DRY-RUN] Would submit PHY/BGC with afterok dependency on IC job")
        return

    ic_job = _sbatch(ic_cmd)

    phy_cmd = [
        "sbatch",
        f"--array=0-{nphy-1}%{args.max_parallel}",
        f"--output={logs_dir}/%x_%A_%a.out",
        f"--error={logs_dir}/%x_%A_%a.err",
        f"--export={_build_export(files['phy'], output_root, args.force, args.conda_env_path)}",
        str(THIS_DIR / "submit_phy_array.slurm"),
    ]
    bgc_cmd = [
        "sbatch",
        f"--array=0-{nbgc-1}%{args.max_parallel}",
        f"--output={logs_dir}/%x_%A_%a.out",
        f"--error={logs_dir}/%x_%A_%a.err",
        f"--export={_build_export(files['bgc'], output_root, args.force, args.conda_env_path)}",
        str(THIS_DIR / "submit_bgc_array.slurm"),
    ]
    if not args.no_ic_dependency:
        phy_cmd.insert(1, f"--dependency=afterok:{ic_job}")
        bgc_cmd.insert(1, f"--dependency=afterok:{ic_job}")

    phy_job = _sbatch(phy_cmd)
    bgc_job = _sbatch(bgc_cmd)

    print(f"[SUBMITTED] ic_job={ic_job} phy_job={phy_job} bgc_job={bgc_job}")


if __name__ == "__main__":
    main()
