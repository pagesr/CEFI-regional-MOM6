#!/usr/bin/env python3
"""Main IC/OBC automation driver for CGOA regional forecast workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_configs import generate_case_configs
from run_bgc_obc import run_bgc_obc
from run_ic import run_ic
from run_phy_obc import run_phy_obc
from utils.helpers import ENSEMBLES, RESTART_MONTHS, YEARS, ensure_dir
from utils.paths import DEFAULT_CONFIG_ROOT, DEFAULT_LOG_ROOT, DEFAULT_OUTPUT_ROOT


def run_workflow(
    years: list[str],
    months: list[str],
    ensembles: list[str],
    output_root: Path,
    config_root: Path,
    force: bool,
) -> None:
    ensure_dir(output_root)
    ensure_dir(config_root)
    ensure_dir(DEFAULT_LOG_ROOT)

    for year in years:
        for month in months:
            # generate IC configs from the first ensemble; IC is ensemble-independent
            ic_configs = generate_case_configs(year, month, ensembles[0], output_root=output_root, config_root=config_root)
            run_ic(ic_configs["ic_phy"], ic_configs["ic_bgc"], year, month, output_root=output_root, force=force)

            for ensemble in ensembles:
                cfgs = generate_case_configs(year, month, ensemble, output_root=output_root, config_root=config_root)
                run_phy_obc(cfgs["obc_phy"], year, month, ensemble, output_root=output_root, force=force)
                run_bgc_obc(cfgs["obc_bgc"], year, month, ensemble, output_root=output_root, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated IC/OBC workflow for CGOA forecast")
    parser.add_argument("--years", nargs="*", default=YEARS)
    parser.add_argument("--months", nargs="*", default=RESTART_MONTHS)
    parser.add_argument("--ensembles", nargs="*", default=ENSEMBLES)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_workflow(
        years=args.years,
        months=args.months,
        ensembles=args.ensembles,
        output_root=Path(args.output_root),
        config_root=Path(args.config_root),
        force=args.force,
    )


if __name__ == "__main__":
    main()
