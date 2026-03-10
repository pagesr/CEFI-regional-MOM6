#!/usr/bin/env python3
# Other way to interpolate by Remi (BGC)

import sys
import os
import argparse
import yaml
import glob

import numpy as np

import xesmf
import xarray as xr
import xarray


def parse_args():
    parser = argparse.ArgumentParser(description="NEP -> GOA BGC IC (Remi method)")
    parser.add_argument(
        "--config_file",
        type=str,
        default="nep_to_goa_bgc_ic.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


args = parse_args()

with open(args.config_file, "r") as f:
    config = yaml.safe_load(f)

DEBUG = config.get("debug", False)

GOA_STATIC = config["GOA_STATIC"]
NEP_STATIC = config["NEP_STATIC"]
NEP_RESTART_DIR = config["NEP_RESTART_DIR"]


def regrid_tracer(fld, method="bilinear"):
    coords = xr.open_dataset(GOA_STATIC)
    coords = coords.rename({"geolon": "lon", "geolat": "lat"})  # interp on this
    gsource = xr.open_dataset(NEP_STATIC, decode_times=False)   # Source grid
    gsource = gsource.rename({"geolon": "lon", "geolat": "lat"})  # interp on this

    regrid = xesmf.Regridder(
        gsource,
        coords,
        method=method,
        periodic=False,
        filename=config.get("regrid_tracer_weights", "regrid_bilin_bgc.nc"),
        reuse_weights=config.get("reuse_weights", True),
    )
    tdest = regrid(fld)
    return tdest


# ----------------------------------------------------------
# Inputs
# ----------------------------------------------------------
BGC_VARS = config["BGC_VARS"]

# Known-missing variables (do NOT scan for these; fill with zeros later)
missing_to_zero = list(config.get("missing_to_zero", []))

# Only scan for variables that are NOT in missing_to_zero
BGC_SCAN_VARS = [v for v in BGC_VARS if v not in set(missing_to_zero)]

# Methods
TRACER_METHOD = config.get("tracer_method", "nearest_s2d")

# Output + time metadata
OUT_IC = config.get("OUT_IC", "ic_nep_to_goa_bgc.nc")
IC_DATE = np.datetime64(config.get("IC_DATE", "1996-01-01T00:00:00"))

TIME_REF = np.datetime64(config.get("TIME_REF", "1993-01-01T00:00:00"))
TIME_UNITS = config.get("TIME_UNITS", "days since 1993-01-01 00:00:00")
TIME_CAL = config.get("TIME_CAL", "proleptic_gregorian")


def rename_only_existing(da, mapping):
    """Rename only dims/coords that actually exist in this DataArray."""
    m = {k: v for k, v in mapping.items() if (k in da.dims) or (k in da.coords)}
    return da.rename(m)


dim_map = {
    "Time": "time",
    "Layer": "zl",
    "jh": "yh",
    "ih": "xh",
}


# ----------------------------------------------------------
# 1) Find variables across *res*.nc in the directory
#    Stop early once all required (excluding missing_to_zero) are found
# ----------------------------------------------------------
files = sorted(glob.glob(os.path.join(NEP_RESTART_DIR, "*res*.nc")))
if len(files) == 0:
    raise FileNotFoundError(f"No *res*.nc files found in: {NEP_RESTART_DIR}")

if DEBUG:
    print("--------------------------------------------------")
    print("DEBUG=True")
    print("NEP restart dir:", NEP_RESTART_DIR)
    print("Known missing_to_zero (won't scan, will fill with 0):", len(missing_to_zero))
    if len(missing_to_zero) > 0:
        print(missing_to_zero)
    print("Will scan for variables:", len(BGC_SCAN_VARS), "/", len(BGC_VARS))
    print("Found", len(files), "restart files:")
    for ffp in files:
        print("  ", ffp)
    print("--------------------------------------------------")

found = {}  # varname -> filepath (or None if missing_to_zero)
need = set(BGC_SCAN_VARS)

for fp in files:
    if not need:
        if DEBUG:
            print("All required variables found (excluding missing_to_zero). Stop scanning.")
        break

    if DEBUG:
        print("Scanning file:", fp, "| remaining vars:", len(need))

    ds = xr.open_dataset(fp, decode_times=False)
    present = set(ds.variables)
    hits = need.intersection(present)

    if hits and DEBUG:
        print("  hits:", sorted(hits))

    for v in hits:
        found[v] = fp
        if DEBUG:
            print(f"  Found variable {v} in {fp}")

    need -= hits
    ds.close()

# Add placeholders for known-missing variables so downstream logic is uniform
for v in missing_to_zero:
    found[v] = None

# Report missing (unexpected)
still_missing = sorted(list(need))

print("--------------------------------------------------")
print("NEP restart dir:", NEP_RESTART_DIR)
print("Found variables (excluding missing_to_zero):", len([v for v in BGC_SCAN_VARS if v in found and found[v] is not None]), "/", len(BGC_SCAN_VARS))
print("Known missing_to_zero:", len(missing_to_zero))
print("Still missing (NOT in missing_to_zero):", len(still_missing))
if len(still_missing) > 0:
    print(still_missing)

if DEBUG:
    print("--------------------------------------------------")
    print("Variable -> file mapping (found):")
    for v in BGC_VARS:
        fp = found.get(v, None)
        if fp is None:
            print(f"  {v:24s} -> (missing -> will set to 0)")
        else:
            print(f"  {v:24s} -> {fp}")
    print("--------------------------------------------------")


# ----------------------------------------------------------
# 2) Regrid variables one-by-one (same as Temp approach)
# ----------------------------------------------------------
regridded_vars = {}
template = None  # will hold a regridded DataArray for shape/template

for i, v in enumerate(BGC_VARS):
    fp = found.get(v, None)

    if fp is not None:
        if DEBUG:
            print(f"[{i+1}/{len(BGC_VARS)}] Regridding variable: {v} from file {fp}")

        ds = xr.open_dataset(fp, decode_times=False)
        fld = ds[v]

        rg = regrid_tracer(fld, method=TRACER_METHOD)
        regridded_vars[v] = rg
        ds.close()

        if DEBUG:
            print(f"  Done {v}, result dims={rg.dims}, shape={rg.shape}")

        if template is None:
            template = rg
    else:
        if DEBUG:
            print(f"[{i+1}/{len(BGC_VARS)}] Variable {v} missing -> will fill with zeros later")
        regridded_vars[v] = None

if template is None:
    raise RuntimeError("None of the requested BGC variables were found in the restart directory.")

# ----------------------------------------------------------
# 3) Fill missing variables with zeros on target grid
# ----------------------------------------------------------
for v in BGC_VARS:
    if regridded_vars[v] is None:
        if DEBUG:
            print(f"Filling missing variable with zeros: {v}")
        z = xr.zeros_like(template)
        z = z.rename(v)
        regridded_vars[v] = z

# ----------------------------------------------------------
# 4) Convert to MOM6 naming + construct output dataset
# ----------------------------------------------------------
time_days = float((IC_DATE - TIME_REF) / np.timedelta64(1, "D"))
time = xr.DataArray([time_days], dims=("time",), name="time")
time.attrs = {"units": TIME_UNITS, "calendar": TIME_CAL, "cartesian_axis": "T"}

out_data_vars = {}

for v in BGC_VARS:
    da = regridded_vars[v]
    da = rename_only_existing(da, dim_map)
    da = da.transpose("time", "zl", "yh", "xh")
    da = da.assign_coords(time=time)
    da.name = v
    out_data_vars[v] = da

first_var = BGC_VARS[0]
zl_vals = (
    out_data_vars[first_var]["zl"].values
    if "zl" in out_data_vars[first_var].coords
    else np.arange(out_data_vars[first_var].sizes["zl"], dtype="float64")
)

zl = xr.DataArray(zl_vals.astype("float64"), dims=("zl",), name="zl")
zl.attrs = {
    "long_name": "Layer pseudo-depth, -z*",
    "units": "meter",
    "cartesian_axis": "Z",
    "positive": "down",
}

ic = xr.Dataset(
    data_vars=out_data_vars,
    coords={"time": time, "zl": zl},
    attrs={"regrid_method": TRACER_METHOD},
)

if DEBUG:
    print("--------------------------------------------------")
    print("About to write IC file:", OUT_IC)
    print("Variables in output:")
    for vv in ic.data_vars:
        print(f"  {vv:24s} shape={ic[vv].shape}")
    print("--------------------------------------------------")

# ----------------------------------------------------------
# 5) Write netcdf (time unlimited, no _FillValue)
# ----------------------------------------------------------
all_vars = list(ic.data_vars) + list(ic.coords)
encoding = {vn: {"_FillValue": None} for vn in all_vars}
encoding["time"].update({"dtype": "float64"})

outdir = os.path.dirname(OUT_IC)
if outdir and not os.path.exists(outdir):
    os.makedirs(outdir)
import time
t0 = time.time()
print("--------------------------------------------------")
print("Writing:", OUT_IC)
print("nvars:", len(ic.data_vars), "dims:", dict(ic.dims))
print("start write:", time.ctime())
print("--------------------------------------------------", flush=True)



ic.to_netcdf(
    OUT_IC,
    format="NETCDF3_64BIT",
    engine="netcdf4",
    encoding=encoding,
    unlimited_dims=("time",),
)

print("--------------------------------------------------")
print("Done write:", time.ctime(), "elapsed:", f"{time.time()-t0:0.1f}s")
print("--------------------------------------------------", flush=True)
print("Wrote:", OUT_IC)
print(ic)