#!/usr/bin/env python3
"""Run padded physics OBC generation for one case (year/month/ensemble)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import xarray as xr
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



def _segment_002_output_is_valid(ncfile: Path) -> tuple[bool, str]:
    """Return whether an existing segment-002 output matches MOM6 J=N:0 order."""
    if "_002" not in ncfile.name:
        return True, ""

    try:
        with xr.open_dataset(ncfile, decode_times=False) as ds:
            ny = "ny_segment_002"
            lat = "lat_segment_002"
            lon = "lon_segment_002"

            if ny not in ds.dims or lat not in ds.variables or lon not in ds.variables:
                return False, "missing segment-002 ny/lon/lat metadata"

            ny_coord = np.asarray(ds[ny].values)
            expected_ny = np.arange(ds.sizes[ny])
            if ny_coord.shape != expected_ny.shape or not np.array_equal(ny_coord, expected_ny):
                return False, "ny_segment_002 is not a clean 0..N-1 index"

            lat_vals = np.asarray(ds[lat].values, dtype="float64")
            if lat_vals.size >= 2 and not lat_vals[0] > lat_vals[-1]:
                return False, (
                    "segment 002 is south-to-north; expected north-to-south "
                    f"(first lat={lat_vals[0]:.6g}, last lat={lat_vals[-1]:.6g})"
                )

            nz = "nz_segment_002"
            if nz in ds.dims and nz in ds.coords:
                nz_coord = np.asarray(ds[nz].values)
                expected_nz = np.arange(ds.sizes[nz])
                if nz_coord.shape != expected_nz.shape or not np.array_equal(nz_coord, expected_nz):
                    return False, "nz_segment_002 is not a clean 0..N-1 index"

            if "zos_segment_002" in ds.variables:
                zos = np.asarray(ds["zos_segment_002"].values)
                if not np.isfinite(zos).all():
                    bad = int((~np.isfinite(zos)).sum())
                    return False, f"zos_segment_002 contains {bad} NaN/Inf values"
    except Exception as exc:  # noqa: BLE001
        return False, f"could not validate existing output: {exc}"

    return True, ""


def _invalid_segment_002_outputs(files: list[Path]) -> list[tuple[Path, str]]:
    invalid = []
    for ncfile in files:
        ok, reason = _segment_002_output_is_valid(ncfile)
        if not ok:
            invalid.append((ncfile, reason))
    return invalid

def _expected_phy_outputs(cfg: dict, year: str) -> list[Path]:
    """Return expected per-year OBC output files for the configured variables/segments."""
    output_dir = Path(cfg["output_dir"])
    variables = cfg.get("variables", [])
    segment_ids = [int(seg["id"]) for seg in cfg.get("segments", [])]
    return [output_dir / f"{var}_{seg_id:03d}_{year}.nc" for var in variables for seg_id in segment_ids]


def _matplotlib_available() -> bool:
    """Return whether diagnostic plotting can render PNGs in this environment."""
    return importlib.util.find_spec("matplotlib") is not None


def _expected_phy_diagnostic_outputs(cfg: dict, year: str) -> list[Path]:
    """Return expected boundary-profile PNGs when PHY diagnostics are enabled."""
    if not bool(cfg.get("diagnostic_plots", False)):
        return []
    if not _matplotlib_available():
        print("[OBC-PHY] diagnostic_plots=true but matplotlib is unavailable; PNGs are not expected")
        return []

    output_dir = Path(cfg["output_dir"])
    diag_dir = output_dir / "diagnostics" / "boundary_profiles"
    variables = cfg.get("variables", [])
    segment_ids = [int(seg["id"]) for seg in cfg.get("segments", [])]
    diag_vars: list[str] = []
    for var in variables:
        if var == "uv":
            diag_vars.extend(["u", "v"])
        else:
            diag_vars.append(var)

    month = str(cfg.get("month", "01")).zfill(2)
    ensemble = str(cfg.get("ensemble", "01")).zfill(2)
    return [
        diag_dir / f"{var}_{seg_id:03d}_{year}{month}_e{ensemble}_profile.png"
        for var in diag_vars
        for seg_id in segment_ids
    ]


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
    expected_diag_files = _expected_phy_diagnostic_outputs(cfg, year)
    missing_files = [f for f in [*expected_files, *expected_diag_files] if not f.exists()]
    invalid_files = _invalid_segment_002_outputs([f for f in expected_files if f.exists()])
    print(
        f"[OBC-PHY] expected_files={len(expected_files)} "
        f"expected_diag_files={len(expected_diag_files)} missing_files={len(missing_files)} "
        f"invalid_segment_002_files={len(invalid_files)}"
    )
    if missing_files:
        preview = ", ".join(str(p.name) for p in missing_files[:4])
        suffix = " ..." if len(missing_files) > 4 else ""
        print(f"[OBC-PHY] missing preview: {preview}{suffix}")
    if invalid_files:
        preview = "; ".join(f"{p.name}: {reason}" for p, reason in invalid_files[:4])
        suffix = " ..." if len(invalid_files) > 4 else ""
        print(f"[OBC-PHY] invalid segment-002 preview: {preview}{suffix}")
    if (not force) and marker.exists() and not missing_files and not invalid_files:
        print(f"[OBC-PHY] skipped {year}-{month} e{ensemble} (marker and outputs exist and validate)")
        return

    if marker.exists() and (missing_files or invalid_files):
        print(
            f"[OBC-PHY] rerunning {year}-{month} e{ensemble}: "
            f"{len(missing_files)} expected output file(s) missing and "
            f"{len(invalid_files)} segment-002 output file(s) invalid despite marker"
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
