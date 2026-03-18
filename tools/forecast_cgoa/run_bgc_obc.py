#!/usr/bin/env python3
"""Run BGC OBC generation for one case (year/month/ensemble)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import xarray as xr
import yaml

from utils.helpers import ensure_dir, expected_marker_file, write_marker
from utils.logging_utils import run_command
from utils.paths import BGC_OBC_DIR, BGC_OBC_SCRIPT, DEFAULT_LOG_ROOT


def _pad_bgc_file_one_month(src: Path) -> Path:
    """Append one extra time step by duplicating the final state and shifting time by +1 month-equivalent."""
    if src.name.endswith("_pad.nc"):
        return src

    dst = src.with_name(f"{src.stem}_pad.nc")
    ds = xr.open_dataset(src)
    if "time" not in ds.dims or ds.sizes.get("time", 0) == 0:
        raise ValueError(f"Cannot pad file without non-empty time dimension: {src}")

    if ds.sizes["time"] > 1:
        dt = ds["time"].values[-1] - ds["time"].values[-2]
    else:
        dt = 30.0

    last = ds.isel(time=[-1]).copy(deep=True)
    last["time"] = ds["time"].isel(time=-1) + dt

    if "time_bnds" in ds.variables and "time" in ds["time_bnds"].dims:
        last_tb = ds["time_bnds"].isel(time=[-1]).copy(deep=True)
        last_tb = last_tb + dt
    else:
        last_tb = None

    out = xr.concat([ds, last], dim="time")
    if last_tb is not None:
        out["time_bnds"] = xr.concat([ds["time_bnds"], last_tb], dim="time")

    out.to_netcdf(dst, format="NETCDF3_64BIT", engine="netcdf4", unlimited_dims="time")
    ds.close()
    out.close()
    return dst


def _ncks_append(src: Path, dst: Path) -> None:
    subprocess.run(["ncks", "-A", str(src), str(dst)], check=True)


def _merge_bgc_outputs(out_dir: Path, cfg: dict, year: int, month: str, ensemble: str) -> Path:
    tracers = cfg.get("tracers", [])
    segment_ids = [int(seg["id"]) for seg in cfg.get("segments", [])]

    merged_var_files: list[Path] = []
    for tracer in tracers:
        padded_files = []
        for seg_id in segment_ids:
            raw = out_dir / f"{tracer}_{seg_id:03d}_{year}.nc"
            if raw.exists():
                padded_files.append(_pad_bgc_file_one_month(raw))

        if not padded_files:
            continue

        merged_var = out_dir / f"{tracer}_{year}_pad.nc"
        shutil.copy2(padded_files[0], merged_var)
        for pf in padded_files[1:]:
            _ncks_append(pf, merged_var)
        merged_var_files.append(merged_var)

    if not merged_var_files:
        raise FileNotFoundError(f"No merged BGC variable files were produced in {out_dir}")

    final_file = out_dir / f"bgc_obc_{year}_{month}_e{ensemble}.nc"
    shutil.copy2(merged_var_files[0], final_file)
    for vf in merged_var_files[1:]:
        _ncks_append(vf, final_file)

    return final_file


def run_bgc_obc(config: Path, year: str, month: str, ensemble: str, output_root: Path, force: bool = False) -> None:
    out_dir = ensure_dir(output_root / year / month / "OBC" / "BGC" / f"e{ensemble}")
    marker = expected_marker_file(f"bgc_obc_e{ensemble}", out_dir)
    if (not force) and marker.exists():
        return

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

    _merge_bgc_outputs(out_dir, cfg, cfg["year"], cfg["month"], cfg["ensemble"])
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
