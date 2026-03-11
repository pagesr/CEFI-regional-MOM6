#!/usr/bin/env python3
"""Run BGC OBC generation for one case (year/month/ensemble)."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.helpers import ensure_dir, expected_marker_file, write_marker
from utils.logging_utils import run_command
from utils.paths import BGC_OBC_DIR, BGC_OBC_SCRIPT, DEFAULT_LOG_ROOT


def run_bgc_obc(config: Path, year: str, month: str, ensemble: str, output_root: Path, force: bool = False) -> None:
    out_dir = ensure_dir(output_root / year / month / "OBC" / "BGC" / f"e{ensemble}")
    marker = expected_marker_file(f"bgc_obc_e{ensemble}", out_dir)
    if (not force) and marker.exists():
        return

    run_command(
        ["python", BGC_OBC_SCRIPT.name, "--config", str(config)],
        cwd=BGC_OBC_DIR,
        log_file=DEFAULT_LOG_ROOT / f"{year}_{month}_e{ensemble}_bgc_obc.log",
    )
    write_marker(f"bgc_obc_e{ensemble}", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_bgc_obc(Path(args.config), args.year, args.month, args.ensemble, Path(args.output_root), force=args.force)


if __name__ == "__main__":
    main()
