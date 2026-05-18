#!/usr/bin/env python3
"""Verify that GOA MOM6 OBC segment 002 files are written north-to-south.

The CGOA MOM_input/XML defines segment 002 as ``I=0,J=N:0``.  This script
checks generated ``*_002*.nc`` files for that expected ordering and for basic
coordinate/data integrity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_PREFIXES = ("uv", "zos", "thetao", "so", "tu", "tz")
CRITICAL_NAMES = (
    "zos_segment_002",
    "u_segment_002",
    "v_segment_002",
    "thetao_segment_002",
    "so_segment_002",
    "zamp_segment_002",
    "zphase_segment_002",
    "uamp_segment_002",
    "vamp_segment_002",
    "uphase_segment_002",
    "vphase_segment_002",
)


def _finite_values(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values)
    if not np.issubdtype(arr.dtype, np.number):
        return np.asarray([], dtype="float64")
    return arr.ravel()


def _check_clean_index(ds: xr.Dataset, dim: str) -> list[str]:
    failures: list[str] = []
    if dim not in ds.dims:
        return [f"missing dimension {dim}"]
    if dim not in ds.coords:
        return [f"missing coordinate variable {dim}"]
    coord = np.asarray(ds[dim].values)
    expected = np.arange(ds.sizes[dim])
    if coord.shape != expected.shape or not np.array_equal(coord, expected):
        failures.append(f"{dim} coordinate is not 0..N-1: {coord[:5]} ... {coord[-5:]}")
    return failures


def _check_segment_002_file(ncfile: Path) -> list[str]:
    failures: list[str] = []
    with xr.open_dataset(ncfile, decode_times=False) as ds:
        ny = "ny_segment_002"
        nz = "nz_segment_002"
        lon = "lon_segment_002"
        lat = "lat_segment_002"

        failures.extend(_check_clean_index(ds, ny))
        if nz in ds.dims:
            failures.extend(_check_clean_index(ds, nz))

        if lon not in ds.variables or lat not in ds.variables:
            failures.append("missing lon_segment_002 and/or lat_segment_002")
        else:
            lon_vals = np.asarray(ds[lon].values, dtype="float64")
            lat_vals = np.asarray(ds[lat].values, dtype="float64")
            if lon_vals.ndim != 1 or lat_vals.ndim != 1:
                failures.append("lon_segment_002/lat_segment_002 must be 1-D")
            elif lon_vals.size != ds.sizes.get(ny, -1) or lat_vals.size != ds.sizes.get(ny, -1):
                failures.append("lon_segment_002/lat_segment_002 length does not match ny_segment_002")
            elif lon_vals.size >= 2:
                if not lat_vals[0] > lat_vals[-1]:
                    failures.append(
                        "segment 002 is not north-to-south: "
                        f"first lat={lat_vals[0]:.6g}, last lat={lat_vals[-1]:.6g}"
                    )
                if np.nanmean(np.diff(lat_vals)) >= 0.0:
                    failures.append("lat_segment_002 does not generally decrease from north to south")

                first = (lon_vals[0], lat_vals[0])
                last = (lon_vals[-1], lat_vals[-1])
                if not (-160.5 <= first[0] <= -156.5 and 55.0 <= first[1] <= 58.5):
                    failures.append(
                        "first segment-002 point is not near expected northern end "
                        f"(-158.7, 56.7): got lon={first[0]:.6g}, lat={first[1]:.6g}"
                    )
                if not (-146.5 <= last[0] <= -142.0 and 47.0 <= last[1] <= 50.0):
                    failures.append(
                        "last segment-002 point is not near expected southern/western corner "
                        f"(-144.3, 48.4): got lon={last[0]:.6g}, lat={last[1]:.6g}"
                    )

        for name, da in ds.data_vars.items():
            if name not in CRITICAL_NAMES and not name.endswith("_segment_002"):
                continue
            vals = _finite_values(da)
            if vals.size and not np.isfinite(vals).all():
                failures.append(f"{name} contains NaN or Inf values")

        if "zos_segment_002" in ds.variables:
            vals = _finite_values(ds["zos_segment_002"])
            if vals.size == 0:
                failures.append("zos_segment_002 has no numeric values")
            elif not np.isfinite(vals).all():
                failures.append("zos_segment_002 contains NaN or Inf values")

    return failures


def _check_corner_consistency(directory: Path, seg002_file: Path) -> list[str]:
    failures: list[str] = []
    seg001_file = directory / seg002_file.name.replace("_002", "_001", 1)
    if not seg001_file.exists():
        return [f"matching segment 001 file not found for corner check: {seg001_file.name}"]

    with xr.open_dataset(seg002_file, decode_times=False) as ds2, xr.open_dataset(seg001_file, decode_times=False) as ds1:
        if "lon_segment_002" not in ds2 or "lat_segment_002" not in ds2:
            return failures
        if "lon_segment_001" not in ds1 or "lat_segment_001" not in ds1:
            return failures

        seg2_corner = np.array([float(ds2["lon_segment_002"].values[-1]), float(ds2["lat_segment_002"].values[-1])])
        seg1_lons = np.asarray(ds1["lon_segment_001"].values, dtype="float64")
        seg1_lats = np.asarray(ds1["lat_segment_001"].values, dtype="float64")
        seg1_endpoints = np.array([[seg1_lons[0], seg1_lats[0]], [seg1_lons[-1], seg1_lats[-1]]])
        distances = np.sqrt(np.sum((seg1_endpoints - seg2_corner) ** 2, axis=1))
        if float(np.min(distances)) > 1.0e-6:
            failures.append(
                "segment 002 southern corner does not match either segment 001 endpoint: "
                f"seg2={tuple(seg2_corner)}, seg1 endpoints={seg1_endpoints.tolist()}"
            )
    return failures


def _iter_files(directory: Path, prefixes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for prefix in prefixes:
        files.extend(sorted(directory.glob(f"{prefix}_002*.nc")))
    return sorted(set(files))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing generated OBC NetCDF files.")
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=list(DEFAULT_PREFIXES),
        help=f"File prefixes to verify (default: {' '.join(DEFAULT_PREFIXES)}).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail if a requested prefix has no segment 002 files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"ERROR: directory not found: {directory}")

    prefixes = tuple(args.prefixes)
    files = _iter_files(directory, prefixes)
    failures: list[str] = []

    found_prefixes = {p.name.split("_002", 1)[0] for p in files}
    for prefix in prefixes:
        if prefix not in found_prefixes and not args.allow_missing:
            failures.append(f"missing required segment 002 file prefix: {prefix}")

    for ncfile in files:
        file_failures = _check_segment_002_file(ncfile)
        file_failures.extend(_check_corner_consistency(directory, ncfile))
        if file_failures:
            failures.extend(f"{ncfile.name}: {msg}" for msg in file_failures)
        else:
            print(f"OK: {ncfile.name}")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"Verified {len(files)} segment 002 file(s) in {directory}")


if __name__ == "__main__":
    main()
