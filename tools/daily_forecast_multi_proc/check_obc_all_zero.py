#!/usr/bin/env python3
"""Check PHY/BGC OBC NetCDF outputs and report files containing only zeros."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass
class FileCheckResult:
    file_path: Path
    all_numeric_vars_zero: bool
    zero_vars: list[str]
    checked_vars: list[str]
    note: str = ""


def _is_numeric(da: xr.DataArray) -> bool:
    return np.issubdtype(da.dtype, np.number)


def _data_array_all_zero(da: xr.DataArray) -> bool:
    arr = da.values
    if arr.size == 0:
        return False
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return False
    return np.all(arr[finite_mask] == 0)


def check_netcdf_file(nc_path: Path) -> FileCheckResult:
    try:
        with xr.open_dataset(nc_path, decode_times=False) as ds:
            checked_vars: list[str] = []
            zero_vars: list[str] = []
            for name, da in ds.data_vars.items():
                if not _is_numeric(da):
                    continue
                checked_vars.append(name)
                if _data_array_all_zero(da):
                    zero_vars.append(name)

            if not checked_vars:
                return FileCheckResult(
                    file_path=nc_path,
                    all_numeric_vars_zero=False,
                    zero_vars=[],
                    checked_vars=[],
                    note="No numeric data variables found.",
                )

            return FileCheckResult(
                file_path=nc_path,
                all_numeric_vars_zero=(len(zero_vars) == len(checked_vars)),
                zero_vars=zero_vars,
                checked_vars=checked_vars,
            )
    except Exception as exc:  # noqa: BLE001
        return FileCheckResult(
            file_path=nc_path,
            all_numeric_vars_zero=False,
            zero_vars=[],
            checked_vars=[],
            note=f"ERROR opening file: {exc}",
        )


def _iter_netcdf_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.nc") if p.is_file())


def _default_dirs(output_root: Path, year: str, month: str, ensemble: str) -> tuple[Path, Path]:
    ens = f"e{int(ensemble):02d}"
    mm = f"{int(month):02d}"
    phy = output_root / year / mm / "OBC" / "PHY" / ens
    bgc = output_root / year / mm / "OBC" / "BGC" / ens
    return phy, bgc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan OBC PHY/BGC NetCDF outputs and write WARNNING.txt "
            "for files whose numeric variables are all zeros."
        )
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Root output directory (default: tools/forecast_multi_proc/outputs).",
    )
    parser.add_argument("--year", required=True, help="Year, e.g. 2012.")
    parser.add_argument("--month", required=True, help="Month, e.g. 01.")
    parser.add_argument("--ensemble", required=True, help="Ensemble number, e.g. 03.")
    parser.add_argument(
        "--phy-dir",
        default=None,
        help="Optional explicit PHY OBC directory. Overrides --output-root/year/month/ensemble.",
    )
    parser.add_argument(
        "--bgc-dir",
        default=None,
        help="Optional explicit BGC OBC directory. Overrides --output-root/year/month/ensemble.",
    )
    parser.add_argument(
        "--warning-file",
        default=str(Path(__file__).resolve().parent / "WARNNING.txt"),
        help="Path for warning report file (default: tools/forecast_multi_proc/WARNNING.txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()

    if args.phy_dir and args.bgc_dir:
        phy_dir = Path(args.phy_dir).expanduser().resolve()
        bgc_dir = Path(args.bgc_dir).expanduser().resolve()
    else:
        phy_dir, bgc_dir = _default_dirs(output_root, args.year, args.month, args.ensemble)

    warning_file = Path(args.warning_file).expanduser().resolve()
    warning_file.parent.mkdir(parents=True, exist_ok=True)

    directories = [("PHY", phy_dir), ("BGC", bgc_dir)]
    report_lines: list[str] = []

    report_lines.append("OBC zero-value scan report")
    report_lines.append(f"PHY directory: {phy_dir}")
    report_lines.append(f"BGC directory: {bgc_dir}")
    report_lines.append("")

    warnings_found = 0

    for stage_name, directory in directories:
        report_lines.append(f"[{stage_name}]")
        if not directory.is_dir():
            report_lines.append(f"WARNING: directory not found: {directory}")
            report_lines.append("")
            warnings_found += 1
            continue

        files = _iter_netcdf_files(directory)
        if not files:
            report_lines.append("WARNING: no .nc files found.")
            report_lines.append("")
            warnings_found += 1
            continue

        for nc_file in files:
            result = check_netcdf_file(nc_file)
            if result.note:
                report_lines.append(f"WARNING: {nc_file.name}: {result.note}")
                warnings_found += 1
                continue

            if result.all_numeric_vars_zero:
                report_lines.append(
                    "WARNING: "
                    f"{nc_file.name} -> ALL numeric vars are zero: {', '.join(result.zero_vars)}"
                )
                warnings_found += 1
            elif result.zero_vars:
                report_lines.append(
                    "WARNING: "
                    f"{nc_file.name} -> zero vars: {', '.join(result.zero_vars)} "
                    f"(checked: {', '.join(result.checked_vars)})"
                )
                warnings_found += 1
            else:
                report_lines.append(f"OK: {nc_file.name}")
        report_lines.append("")

    report_lines.append(f"Total warnings: {warnings_found}")
    warning_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote report: {warning_file}")
    print(f"Warnings found: {warnings_found}")


if __name__ == "__main__":
    main()
