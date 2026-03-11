#!/usr/bin/env python3
"""Run padded physics OBC generation for one case (year/month/ensemble)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from utils.helpers import ensure_dir, expected_marker_file, write_marker
from utils.logging_utils import run_command
from utils.paths import DEFAULT_LOG_ROOT, PHY_OBC_DIR, PHY_OBC_SCRIPT


def run_phy_obc(config: Path, year: str, month: str, ensemble: str, output_root: Path, force: bool = False) -> None:
    out_dir = ensure_dir(output_root / year / month / "OBC" / "PHY" / f"e{ensemble}")
    marker = expected_marker_file(f"phy_obc_e{ensemble}", out_dir)
    if (not force) and marker.exists():
        return


    with config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    fcst_hist = Path(cfg["fct_dir"]) / f"{year}-{month}-e{ensemble}" / "history"
    if not fcst_hist.exists():
        raise FileNotFoundError(
            "PHY OBC forecast history directory not found: "
            f"{fcst_hist}. Check fct_dir/ensemble mapping in generated obc_phy config."
        )

    run_command(
        ["python", PHY_OBC_SCRIPT.name, "--config", str(config)],
        cwd=PHY_OBC_DIR,
        log_file=DEFAULT_LOG_ROOT / f"{year}_{month}_e{ensemble}_phy_obc.log",
    )
    write_marker(f"phy_obc_e{ensemble}", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_phy_obc(Path(args.config), args.year, args.month, args.ensemble, Path(args.output_root), force=args.force)


if __name__ == "__main__":
    main()
