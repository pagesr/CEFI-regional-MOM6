#!/usr/bin/env python3
"""Build PHY-only task TSV files for OBC daily workflow orchestration."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

FORECAST_CGOA_DIR = Path(__file__).resolve().parent / "standalone" / "forecast_cgoa"
sys.path.insert(0, str(FORECAST_CGOA_DIR))

from generate_configs import generate_case_configs  # type: ignore  # noqa: E402
from utils.helpers import ENSEMBLES, RESTART_MONTHS, YEARS, ensure_dir  # type: ignore  # noqa: E402
from utils.paths import DEFAULT_CONFIG_ROOT, DEFAULT_OUTPUT_ROOT  # type: ignore  # noqa: E402


PHY_FIELDS = ["year", "month", "ensemble", "obc_phy_config"]


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_task_lists(
    years: list[str],
    months: list[str],
    ensembles: list[str],
    output_root: Path,
    config_root: Path,
    task_root: Path,
) -> Path:
    output_root = output_root.resolve()
    config_root = config_root.resolve()
    task_root = task_root.resolve()

    phy_rows: list[dict[str, str]] = []

    for year in years:
        for month in months:
            for ens in ensembles:
                cfgs = generate_case_configs(
                    year,
                    month,
                    ens,
                    output_root=output_root,
                    config_root=config_root,
                )
                phy_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "ensemble": ens,
                        "obc_phy_config": str(cfgs["obc_phy"]),
                    }
                )
    phy_file = task_root / "phy_tasks.tsv"
    _write_rows(phy_file, PHY_FIELDS, phy_rows)
    return phy_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task TSV file for OBC PHY daily workflow")
    parser.add_argument("--years", nargs="*", default=YEARS)
    parser.add_argument("--months", nargs="*", default=RESTART_MONTHS)
    parser.add_argument("--ensembles", nargs="*", default=ENSEMBLES)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--task-root", default=str(Path(__file__).resolve().parent / "tasks"))
    args = parser.parse_args()

    phy_file = build_task_lists(
        years=args.years,
        months=args.months,
        ensembles=args.ensembles,
        output_root=Path(args.output_root),
        config_root=Path(args.config_root),
        task_root=Path(args.task_root),
    )
    print(f"phy: {phy_file}")


if __name__ == "__main__":
    main()
