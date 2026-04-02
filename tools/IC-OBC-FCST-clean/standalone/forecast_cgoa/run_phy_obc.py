#!/usr/bin/env python3
"""Run padded physics OBC generation for one case (year/month/ensemble)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from utils.helpers import ensure_dir, expected_marker_file, write_marker
from utils.logging_utils import run_command
from utils.paths import (
    DEFAULT_LOG_ROOT,
    PHY_OBC_DIR,
    PHY_OBC_SCRIPT,
    PHY_OBC_TRACER_DIR,
    PHY_OBC_TRACER_SCRIPT,
)


def _expected_phy_outputs(cfg: dict, year: str) -> list[Path]:
    """Return expected per-year OBC output files for the configured variables/segments."""
    output_dir = Path(cfg["output_dir"])
    variables = cfg.get("variables", [])
    segment_ids = [int(seg["id"]) for seg in cfg.get("segments", [])]
    return [output_dir / f"{var}_{seg_id:03d}_{year}.nc" for var in variables for seg_id in segment_ids]


def run_phy_obc(config: Path, year: str, month: str, ensemble: str, output_root: Path, force: bool = False) -> None:
    config = config.resolve()
    output_root = output_root.resolve()
    out_dir = ensure_dir(output_root / year / month / "OBC" / "PHY" / f"e{ensemble}")
    print(f"[OBC-PHY] config={config}")
    print(f"[OBC-PHY] output_root={output_root}")
    print(f"[OBC-PHY] case_out_dir={out_dir}")

    with config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    # Self-heal config typing for compatibility with scientific PHY OBC script.
    # This prevents failures if an older/stale generated config contains string years.
    cfg["first_year"] = int(cfg["first_year"])
    cfg["last_year"] = int(cfg["last_year"])
    cfg["month"] = str(cfg.get("month", month)).zfill(2)
    cfg["ensemble"] = str(cfg.get("ensemble", ensemble)).zfill(2)
    with config.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)

    marker = expected_marker_file(f"phy_obc_e{ensemble}", out_dir)
    expected_files = _expected_phy_outputs(cfg, year)
    missing_files = [f for f in expected_files if not f.exists()]
    print(f"[OBC-PHY] expected_files={len(expected_files)} missing_files={len(missing_files)}")
    if missing_files:
        preview = ", ".join(str(p.name) for p in missing_files[:4])
        suffix = " ..." if len(missing_files) > 4 else ""
        print(f"[OBC-PHY] missing preview: {preview}{suffix}")
    if (not force) and marker.exists() and not missing_files:
        print(f"[OBC-PHY] skipped {year}-{month} e{ensemble} (marker and outputs exist)")
        return

    if marker.exists() and missing_files:
        print(
            f"[OBC-PHY] rerunning {year}-{month} e{ensemble}: "
            f"{len(missing_files)} expected output file(s) missing despite marker"
        )
    else:
        print(f"[OBC-PHY] running {year}-{month} e{ensemble} (force={force})")

    fcst_hist = Path(cfg["fct_dir"]) / f"{year}-{month}-e{ensemble}" / "history"
    if not fcst_hist.exists():
        raise FileNotFoundError(
            "PHY OBC forecast history directory not found: "
            f"{fcst_hist}. Check fct_dir/ensemble mapping in generated obc_phy config."
        )

    base_cfg = dict(cfg)
    uv_zos_vars = [v for v in base_cfg.get("variables", []) if v in {"uv", "zos"}]
    tracer_vars = [v for v in base_cfg.get("variables", []) if v in {"thetao", "so"}]

    if tracer_vars:
        tracer_cfg = dict(base_cfg)
        tracer_cfg["variables"] = tracer_vars
        tracer_cfg["ncrcat_names"] = [v for v in base_cfg.get("ncrcat_names", []) if v in tracer_vars]
        tracer_tmp = out_dir / f"obc_phy_tracers_{year}{month}_e{ensemble}.yaml"
        with tracer_tmp.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(tracer_cfg, stream, sort_keys=False)
        run_command(
            [sys.executable, PHY_OBC_TRACER_SCRIPT.name, "--config", str(tracer_tmp)],
            cwd=PHY_OBC_TRACER_DIR,
            log_file=DEFAULT_LOG_ROOT / f"{year}_{month}_e{ensemble}_phy_obc_tracers.log",
        )

    if uv_zos_vars:
        uv_cfg = dict(base_cfg)
        uv_cfg["variables"] = uv_zos_vars
        uv_cfg["ncrcat_names"] = [v for v in base_cfg.get("ncrcat_names", []) if v in uv_zos_vars]
        uv_tmp = out_dir / f"obc_phy_uvzos_{year}{month}_e{ensemble}.yaml"
        with uv_tmp.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(uv_cfg, stream, sort_keys=False)
        run_command(
            [sys.executable, PHY_OBC_SCRIPT.name, "--config", str(uv_tmp)],
            cwd=PHY_OBC_DIR,
            log_file=DEFAULT_LOG_ROOT / f"{year}_{month}_e{ensemble}_phy_obc_uvzos.log",
        )

    print(f"[OBC-PHY] finished script run for {year}-{month} e{ensemble}")
    write_marker(f"phy_obc_e{ensemble}", out_dir)
    print(f"[OBC-PHY] wrote marker: {marker}")


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
