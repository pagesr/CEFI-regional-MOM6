#!/usr/bin/env python3
"""
Regrid NEP COBALT tracer monthly file to GOA2p5k open boundary segments (TRACER-ONLY).

Input
-----
A single NEP file containing ALL tracers and monthly time steps, e.g.
  /archive/Dmitry.Dukhovskoy/fre/NEP/forecast_bgc/NEPbgc_fcst_dailyOB01/2012-01-e01/history/ocean_cobalt_tracers_month_z.nc

We DO NOT use restart files here.
We DO NOT reconstruct an intermediate dataset.

Method
------
For each tracer in `tracers`:
  - load the DataArray from the file
  - optionally select the first 12 months (time=0..11)
  - attach 2D lon/lat from NEP static grid (geolon/geolat)
  - seg.regrid_tracer(...)

Run
---
  python write_CGOA_boundary_tracers_from_single_file.py --config write_CGOA_boundary_south_2D.yaml
"""

from subprocess import run
from os import path
import os
import argparse
import warnings
import yaml
import xarray as xr
import numpy as np

from boundary import Segment

warnings.filterwarnings("ignore")


# ----------------------------
# Config helpers
# ----------------------------
def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def _require(cfg, key):
    if key not in cfg or cfg[key] in [None, ""]:
        raise ValueError(f"Missing required config key: {key}")
    return cfg[key]


# ----------------------------
# Utilities
# ----------------------------
def _attach_2d_lonlat(da, lon2d, lat2d, name="var"):
    """
    Attach 2D lon/lat coordinates to a DataArray.
    Assumes last two dims are horizontal.
    """
    hdims = da.dims[-2:]

    if lon2d.shape != tuple(da.sizes[d] for d in hdims):
        raise ValueError(
            f"{name}: lon2d shape {lon2d.shape} != da horizontal shape "
            f"{tuple(da.sizes[d] for d in hdims)} for dims {hdims}"
        )
    if lat2d.shape != tuple(da.sizes[d] for d in hdims):
        raise ValueError(
            f"{name}: lat2d shape {lat2d.shape} != da horizontal shape "
            f"{tuple(da.sizes[d] for d in hdims)} for dims {hdims}"
        )

    da = da.assign_coords(lon=(hdims, lon2d), lat=(hdims, lat2d))
    da["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    da["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    return da


def _safe_rename_vars(ds, rename_map):
    """Rename variables that exist in ds, ignore the rest."""
    if not rename_map:
        return ds
    existing = {k: v for k, v in rename_map.items() if k in ds.variables}
    if existing:
        ds = ds.rename(existing)
    return ds


# ----------------------------
# Core routine
# ----------------------------
def regrid_tracers_from_file(
    year,
    month,
    ensemble,
    nep_static,
    input_file,
    segments,
    tracers,
    time_sel="first12",
):

    # --- load NEP static grid (T points) ---
    st = xr.open_dataset(nep_static, decode_times=False)
    lonT = st["geolon"].values
    latT = st["geolat"].values

    # --- open the tracer file ---
    ds_in = xr.open_dataset(input_file)


    # Optional renames if the file uses different names than you want in output
    # (edit only if needed)
    rename_map = {
        "DIC": "dic",
        "ALK": "alk",
        "talk": "alk",
        "dissic": "dic",
        "si": "sio4",
        
    }
    ds_in = _safe_rename_vars(ds_in, rename_map)

    # --- choose time subset ---
    if "time" not in ds_in.dims:
        raise ValueError(f"No 'time' dimension found in input file: {input_file}")

    if time_sel == "first12":
        ds_in = ds_in.isel(time=slice(0, 12))
    elif time_sel == "all":
        pass
    else:
        raise ValueError("time_sel must be 'first12' or 'all'")

    # --- time metadata to force into Segment writer ---
    # If the file already has good CF time metadata, Segment will typically preserve it,
    # but we keep your pattern to be explicit/consistent.
    # ---- time metadata (avoid attr/encoding conflict in xarray) ----
    time_units = ds_in["time"].attrs.get(
        "units", f"days since {int(year):04d}-{int(month):02d}-01 00:00:00"
    )
    
    time_attrs = {
        "long_name": ds_in["time"].attrs.get("long_name", "time"),
        "axis": "T",
        "bounds": ds_in["time"].attrs.get("bounds", None),
    }
    time_attrs = {k: v for k, v in time_attrs.items() if v is not None}
    
    time_encoding = {
        "_FillValue": None,
        "dtype": "float64",
        "units": time_units,  # <- move units here
        "calendar": ds_in["time"].attrs.get("calendar", "gregorian"),  # <- move calendar here
    }

    rho0 = 1026.0  # kg m-3 reference density
    
    for v in tracers:
    
        if v not in ds_in.variables:
            raise ValueError(f"Tracer '{v}' not found in input file: {input_file}")
    
        tracer = ds_in[v]
    
        # ---------------------------------------------------
        # UNIT CORRECTION FOR DIC & ALK
        # ---------------------------------------------------
        if v.lower() in ["dic", "alk","sio4"]:
            print(f"Converting {v} from mol m-3 to mol kg-1 (divide by {rho0})")
            tracer = tracer / rho0
            tracer.attrs["units"] = "mol kg-1"
            tracer.attrs["conversion_note"] = "Converted from mol m-3 assuming rho=1026 kg m-3"
    
        # Expect 4D: (time, z, yh, xh)
        if tracer.ndim != 4:
            raise ValueError(
                f"Tracer '{v}' has unexpected ndim={tracer.ndim}. "
                f"Expected 4D (time,z,yh,xh). dims={tracer.dims}"
            )
    
        tracer = _attach_2d_lonlat(tracer, lonT, latT, name=v)
        print(v, tracer.dims, tracer.shape)
        if "z_l" in tracer.dims:
            tracer = tracer.rename({"z_l": "z"})
            print(v, tracer.dims, tracer.shape)
    
        for seg in segments:
            print(f"{seg.border} {v}")
            seg.regrid_tracer(
                tracer,
                suffix=str(year),
                flood=False,
                weight_save=True,
                time_attrs=time_attrs,
                time_encoding=time_encoding,
            )
    ds_in.close()
    st.close()


def ncrcat_years(nsegments, output_dir, variables, ncrcat_names):
    if not ncrcat_names:
        ncrcat_names = variables[:]
    for var, var_name in zip(variables, ncrcat_names):
        for seg in range(1, nsegments + 1):
            run([f"ncrcat -O {var}_{seg:03d}_* {var_name}_{seg:03d}.nc"], cwd=output_dir, shell=True)


def main(config_file):
    cfg = load_config(config_file)

    year = int(cfg.get("year", 2012))          # single-year driver (simple)
    month = str(cfg.get("month", "01"))        # used only for default time_units fallback
    ensemble = str(cfg.get("ensemble", "01"))  # used for path templating if you want

    output_dir = cfg.get("output_dir", "./outputs_CGOA_tracers")
    hgrid_file = cfg.get("hgrid", "/work/Remi.Pages/GOA2p5k/GRID/CGOA_2.5k/ocean_hgrid.nc")
    nep_static = _require(cfg, "NEP_STATIC")

    # Either provide the full input file path, or provide fct_dir and we build it.
    input_file = cfg.get("input_file", None)
    if input_file is None:
        fct_dir = _require(cfg, "fct_dir")
        fcst_hist = path.join(fct_dir, f"{year}-{month}-e{ensemble}", "history")
        input_file = path.join(fcst_hist, "ocean_cobalt_tracers_month_z.nc")

    time_sel = cfg.get("time_sel", "first12")  # 'first12' or 'all'

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # Load GOA hgrid + segments
    hgrid = xr.open_dataset(hgrid_file)

    segments = []
    for seg_cfg in cfg.get("segments", []):
        segments.append(Segment(seg_cfg["id"], seg_cfg["border"], hgrid, output_dir=output_dir))

    # tracers list (YAML override allowed)
    cobalt_vars = [
        "alk","dic","po4","sio4","o2","no3","nh4","fed","fedet","ndet","nbact",
        "nsmz","nmdz","nlgz","chl","chl_Lg","chl_Md","chl_Sm","chl_Di","simd",
        "silg","ndi","nlg","nsm","nmd","pdi","plg","pmd","psm"
    ]
    tracers = cfg.get("tracers", cobalt_vars)

    print(f"Input: {input_file}")
    print(f"Time selection: {time_sel}")
    print(f"Tracers: {len(tracers)}")
    print(f"Segments: {len(segments)}")

    regrid_tracers_from_file(
        year=year,
        month=month,
        ensemble=ensemble,
        nep_static=nep_static,
        input_file=input_file,
        segments=segments,
        tracers=tracers,
        time_sel=time_sel,
    )

    # Optional ncrcat
    if cfg.get("ncrcat_years", False):
        ncrcat_names = cfg.get("ncrcat_names", [])
        assert len(ncrcat_names) in [0, len(tracers)], "ncrcat_names must be empty or match tracers length"
        ncrcat_years(len(segments), output_dir, tracers, ncrcat_names)

    hgrid.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regrid NEP COBALT tracers monthly file to GOA OBC segments")
    parser.add_argument("--config", type=str, required=True, help="YAML configuration file")
    args = parser.parse_args()
    main(args.config)