#!/usr/bin/env python3
"""Run physics and BGC IC generation for one year/month."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.helpers import ensure_dir, expected_marker_file, write_marker
from utils.logging_utils import run_command
from utils.paths import DEFAULT_LOG_ROOT, IC_BGC_SCRIPT, IC_PHY_SCRIPT, INITIAL_DIR


def run_ic(ic_phy_cfg: Path, ic_bgc_cfg: Path, year: str, month: str, output_root: Path, force: bool = False) -> None:
    ic_out_dir = ensure_dir(output_root / year / month / "IC")
    phy_marker = expected_marker_file("ic_phy", ic_out_dir)
    bgc_marker = expected_marker_file("ic_bgc", ic_out_dir)

    if force or not phy_marker.exists():
        run_command(
            ["python", str(IC_PHY_SCRIPT), "--config", str(ic_phy_cfg)],
            cwd=INITIAL_DIR,
            log_file=DEFAULT_LOG_ROOT / f"{year}_{month}_ic_phy.log",
        )
        write_marker("ic_phy", ic_out_dir)

    if force or not bgc_marker.exists():
        run_command(
            ["python", str(IC_BGC_SCRIPT), "--config", str(ic_bgc_cfg)],
            cwd=INITIAL_DIR,
            log_file=DEFAULT_LOG_ROOT / f"{year}_{month}_ic_bgc.log",
        )
        write_marker("ic_bgc", ic_out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-phy-config", required=True)
    parser.add_argument("--ic-bgc-config", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_ic(
        Path(args.ic_phy_config),
        Path(args.ic_bgc_config),
        args.year,
        args.month,
        Path(args.output_root),
        force=args.force,
    )


if __name__ == "__main__":
    main()
