#!/usr/bin/env python3
"""Run BGC OBC generation for one case (year/month/ensemble)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from utils.helpers import ensure_dir, expected_marker_file, write_marker
from utils.logging_utils import run_command
from utils.paths import BGC_OBC_DIR, BGC_OBC_POSTPROCESS_SCRIPT, BGC_OBC_SCRIPT, DEFAULT_LOG_ROOT


def run_bgc_obc(config: Path, year: str, month: str, ensemble: str, output_root: Path, force: bool = False) -> None:
    config = config.resolve()
    output_root = output_root.resolve()
    out_dir = ensure_dir(output_root / year / month / "OBC" / "BGC" / f"e{ensemble}")
    marker = expected_marker_file(f"bgc_obc_e{ensemble}", out_dir)
    if (not force) and marker.exists():
        print(f"[OBC-BGC] skipped {year}-{month} e{ensemble} (marker exists: {marker})")
        return

    print(f"[OBC-BGC] running {year}-{month} e{ensemble} (force={force})")

    with config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    # Self-heal config typing/format for compatibility with OBC_BGC.py
    cfg["year"] = int(cfg["year"])
    cfg["month"] = str(cfg.get("month", month)).zfill(2)
    cfg["ensemble"] = str(cfg.get("ensemble", ensemble)).zfill(2)
    with config.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)

    fcst_hist = Path(cfg["fct_dir"]) / f"{cfg['year']}-{cfg['month']}-e{cfg['ensemble']}" / "history"
    if not fcst_hist.exists():
        raise FileNotFoundError(
            "BGC OBC forecast history directory not found: "
            f"{fcst_hist}. Check fct_dir/month/ensemble in generated obc_bgc config."
        )

    run_command(
        ["python", BGC_OBC_SCRIPT.name, "--config", str(config)],
        cwd=BGC_OBC_DIR,
        log_file=DEFAULT_LOG_ROOT / f"{year}_{month}_e{ensemble}_bgc_obc.log",
    )

    if bool(cfg.get("merge_to_single_file", True)):
        run_command(
            [
                "bash",
                str(BGC_OBC_POSTPROCESS_SCRIPT),
                str(out_dir),
                year,
                month,
                ensemble,
            ],
            cwd=BGC_OBC_DIR,
            log_file=DEFAULT_LOG_ROOT / f"{year}_{month}_e{ensemble}_bgc_postprocess.log",
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
