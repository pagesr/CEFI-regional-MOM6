#!/usr/bin/env python3
"""Build task TSV files for multi-process Slurm workflow orchestration."""

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


IC_FIELDS = ["year", "month", "ic_phy_config", "ic_bgc_config"]
PHY_FIELDS = ["year", "month", "ensemble", "obc_phy_config"]
BGC_FIELDS = ["year", "month", "ensemble", "obc_bgc_config"]


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
) -> dict[str, Path]:
    output_root = output_root.resolve()
    config_root = config_root.resolve()
    task_root = task_root.resolve()

    ic_rows: list[dict[str, str]] = []
    phy_rows: list[dict[str, str]] = []
    bgc_rows: list[dict[str, str]] = []

    for year in years:
        for month in months:
            ic_cfgs = generate_case_configs(
                year,
                month,
                ensembles[0],
                output_root=output_root,
                config_root=config_root,
            )
            ic_rows.append(
                {
                    "year": year,
                    "month": month,
                    "ic_phy_config": str(ic_cfgs["ic_phy"]),
                    "ic_bgc_config": str(ic_cfgs["ic_bgc"]),
                }
            )

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
                bgc_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "ensemble": ens,
                        "obc_bgc_config": str(cfgs["obc_bgc"]),
                    }
                )

    files = {
        "ic": task_root / "ic_tasks.tsv",
        "phy": task_root / "phy_tasks.tsv",
        "bgc": task_root / "bgc_tasks.tsv",
    }
    _write_rows(files["ic"], IC_FIELDS, ic_rows)
    _write_rows(files["phy"], PHY_FIELDS, phy_rows)
    _write_rows(files["bgc"], BGC_FIELDS, bgc_rows)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task TSV files for forecast multi-proc workflow")
    parser.add_argument("--years", nargs="*", default=YEARS)
    parser.add_argument("--months", nargs="*", default=RESTART_MONTHS)
    parser.add_argument("--ensembles", nargs="*", default=ENSEMBLES)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--task-root", default=str(Path(__file__).resolve().parent / "tasks"))
    args = parser.parse_args()

    files = build_task_lists(
        years=args.years,
        months=args.months,
        ensembles=args.ensembles,
        output_root=Path(args.output_root),
        config_root=Path(args.config_root),
        task_root=Path(args.task_root),
    )

    for stage, path in files.items():
        print(f"{stage}: {path}")


if __name__ == "__main__":
    main()
