#!/usr/bin/env python3
"""
Regrid NEP COBALT tracer monthly file to GOA2p5k open boundary segments (TRACER-ONLY),
then pad each output file by duplicating the last time slice and assigning it to the
first day of the next month.

Example
-------
If output time is:
  2012-04-16 ... 2013-03-16

the script appends:
  2013-04-01

using the same data values as 2013-03-16.
"""

from subprocess import run
from os import path
import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import yaml
import xarray as xr
import pandas as pd

from boundary import Segment

warnings.filterwarnings("ignore")


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def _require(cfg, key):
    if key not in cfg or cfg[key] in [None, ""]:
        raise ValueError(f"Missing required config key: {key}")
    return cfg[key]


def lon_to_360(lon):
    """Return longitude values in the 0-360 convention."""
    return lon % 360.0


def lon_to_180(lon):
    """Return longitude values in the -180 to 180 convention."""
    return ((lon + 180.0) % 360.0) - 180.0


def _lon_range(label, lon):
    arr = np.asarray(lon, dtype="float64")
    print(f"[BGC-OBC-v2] {label} lon range: {np.nanmin(arr):.6f} to {np.nanmax(arr):.6f}")


def _attach_2d_lonlat(da, lon2d, lat2d, name="var"):
    """
    Attach 2D lon/lat coordinates to a DataArray.
    Assumes the last two dimensions are horizontal.
    """
    hdims = da.dims[-2:]

    if lon2d.shape != tuple(da.sizes[d] for d in hdims):
        raise ValueError(
            f"{name}: lon2d shape {lon2d.shape} != horizontal shape "
            f"{tuple(da.sizes[d] for d in hdims)} for dims {hdims}"
        )
    if lat2d.shape != tuple(da.sizes[d] for d in hdims):
        raise ValueError(
            f"{name}: lat2d shape {lat2d.shape} != horizontal shape "
            f"{tuple(da.sizes[d] for d in hdims)} for dims {hdims}"
        )

    da = da.assign_coords(lon=(hdims, lon2d), lat=(hdims, lat2d))
    da["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    da["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    return da


def _safe_rename_vars(ds, rename_map):
    """Rename variables that exist in ds and ignore the rest."""
    if not rename_map:
        return ds
    existing = {k: v for k, v in rename_map.items() if k in ds.variables}
    if existing:
        ds = ds.rename(existing)
    return ds


def _next_month_first_day(ts):
    """
    Return the first day of the next month at 00:00:00.

    Example:
        2013-03-16 -> 2013-04-01
    """
    ts = pd.Timestamp(ts)
    return (ts + pd.offsets.MonthBegin(1)).normalize()


def pad_last_timestep_same_file(ncfile):
    """
    Duplicate the last time record and append it as the first day
    of the following month.

    Example:
        2013-03-16 -> 2013-04-01
    """
    ncfile = Path(ncfile)

    ds = xr.open_dataset(ncfile, decode_times=True)

    if "time" not in ds.dims:
        ds.close()
        raise ValueError(f"No time dimension found in {ncfile}")

    if ds.sizes["time"] < 1:
        ds.close()
        raise ValueError(f"Empty time dimension in {ncfile}")

    last_time = ds["time"].values[-1]

    try:
        last_ts = pd.Timestamp(last_time)
    except Exception:
        last_ts = pd.Timestamp(str(last_time))

    new_time = _next_month_first_day(last_ts)

    # Copy only the last record
    last_rec = ds.isel(time=-1).copy(deep=True)
    last_rec = last_rec.expand_dims(time=[new_time])

    # Append to dataset
    ds_pad = xr.concat([ds, last_rec], dim="time")

    # Preserve encoding where possible
    encoding = {}
    for v in ds_pad.variables:
        if v in ds.variables and hasattr(ds[v], "encoding"):
            enc = ds[v].encoding.copy()
            enc.pop("source", None)
            enc.pop("original_shape", None)
            encoding[v] = enc

    ds.close()

    tmpfile = ncfile.with_suffix(".tmp.nc")
    ds_pad.to_netcdf(tmpfile, encoding=encoding)
    ds_pad.close()

    tmpfile.replace(ncfile)

    print(
        f"Padded {ncfile.name}: "
        f"{last_ts.strftime('%Y-%m-%d')} -> {new_time.strftime('%Y-%m-%d')}"
    )


def regrid_tracers_from_file(
    year,
    month,
    ensemble,
    nep_static,
    input_file,
    segments,
    tracers,
    output_dir,
    time_sel="first12",
):
    # Load NEP static grid at T points
    st = xr.open_dataset(nep_static, decode_times=False)
    lonT = lon_to_360(st["geolon"]).values
    latT = st["geolat"].values
    _lon_range("NEP geolon source after 0-360 conversion", lonT)

    # Open input tracer file
    ds_in = xr.open_dataset(input_file)

    # Optional renames if needed
    rename_map = {
        "DIC": "dic",
        "ALK": "alk",
        "talk": "alk",
        "dissic": "dic",
        "si": "sio4",
    }
    ds_in = _safe_rename_vars(ds_in, rename_map)

    if "time" not in ds_in.dims:
        ds_in.close()
        st.close()
        raise ValueError(f"No 'time' dimension found in input file: {input_file}")

    # Select first 12 times or all
    if time_sel == "first12":
        ds_in = ds_in.isel(time=slice(0, 12))
        # --- FORCE TIME TO START OF MONTH ---
        time_vals = pd.to_datetime(ds_in["time"].values)

        new_time = [pd.Timestamp(t).to_period("M").to_timestamp() for t in time_vals]

        ds_in = ds_in.assign_coords(time=("time", new_time))
    elif time_sel == "all":
        pass
    else:
        ds_in.close()
        st.close()
        raise ValueError("time_sel must be 'first12' or 'all'")

    time_units = ds_in["time"].attrs.get(
        "units",
        f"days since {int(year):04d}-{int(month):02d}-01 00:00:00"
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
        "units": time_units,
        "calendar": ds_in["time"].attrs.get("calendar", "gregorian"),
    }

    rho0 = 1026.0  # kg m-3

    for v in tracers:
        if v not in ds_in.variables:
            ds_in.close()
            st.close()
            raise ValueError(f"Tracer '{v}' not found in input file: {input_file}")

        tracer = ds_in[v]

        # Optional unit conversion
        if v.lower() in ["dic", "alk", "sio4"]:
            print(f"Converting {v} from mol m-3 to mol kg-1 (divide by {rho0})")
            tracer = tracer / rho0
            tracer.attrs["units"] = "mol kg-1"
            tracer.attrs["conversion_note"] = (
                "Converted from mol m-3 assuming rho=1026 kg m-3"
            )

        # Expect 4D: (time, z, yh, xh)
        if tracer.ndim != 4:
            ds_in.close()
            st.close()
            raise ValueError(
                f"Tracer '{v}' has unexpected ndim={tracer.ndim}. "
                f"Expected 4D (time,z,yh,xh). dims={tracer.dims}"
            )

        tracer = _attach_2d_lonlat(tracer, lonT, latT, name=v)

        if "z_l" in tracer.dims:
            tracer = tracer.rename({"z_l": "z"})

        print(f"{v}: input time length = {tracer.sizes['time']}")

        for seg_id, seg in segments:
            print(f"Regridding {v} on segment {seg.border} (id={seg_id})")

            seg.regrid_tracer(
                tracer,
                suffix=str(year),
                flood=False,
                weight_save=True,
                time_attrs=time_attrs,
                time_encoding=time_encoding,
            )

            out_file = path.join(output_dir, f"{v}_{seg_id:03d}_{year}.nc")
            if not path.exists(out_file):
                ds_in.close()
                st.close()
                raise FileNotFoundError(f"Expected output file not found: {out_file}")

            print(f"Padding file: {out_file}")
            pad_last_timestep_same_file(out_file)

            # Quick verification
            ds_chk = xr.open_dataset(out_file, decode_times=True)
            print(f"Final time length for {path.basename(out_file)} = {ds_chk.sizes['time']}")
            print(f"Last 2 dates = {ds_chk['time'].values[-2:]}")
            ds_chk.close()

    ds_in.close()
    st.close()


def ncrcat_years(nsegments, output_dir, variables, ncrcat_names):
    if not ncrcat_names:
        ncrcat_names = variables[:]

    for var, var_name in zip(variables, ncrcat_names):
        for seg in range(1, nsegments + 1):
            run(
                [f"ncrcat -O {var}_{seg:03d}_* {var_name}_{seg:03d}.nc"],
                cwd=output_dir,
                shell=True
            )


def main(config_file):
    cfg = load_config(config_file)

    year = int(cfg.get("year", 2012))
    month = str(cfg.get("month", "01"))
    ensemble = str(cfg.get("ensemble", "01"))

    output_dir = cfg.get("output_dir", "./outputs_CGOA_tracers")
    hgrid_file = cfg.get(
        "hgrid",
        "/work/Remi.Pages/GOA2p5k/GRID/CGOA_2.5k/ocean_hgrid.nc"
    )
    nep_static = _require(cfg, "NEP_STATIC")

    input_file = cfg.get("input_file", None)
    if input_file is None:
        fct_dir = _require(cfg, "fct_dir")
        fcst_hist = path.join(fct_dir, f"{year}-{month}-e{ensemble}", "history")
        input_file = path.join(fcst_hist, "ocean_cobalt_tracers_month_z.nc")

    time_sel = cfg.get("time_sel", "first12")

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    regrid_dir = cfg.get("regrid_dir", output_dir)
    if not path.exists(regrid_dir):
        os.makedirs(regrid_dir)

    hgrid = xr.open_dataset(hgrid_file)
    hgrid = hgrid.copy(deep=True)
    hgrid["x"] = lon_to_360(hgrid["x"])
    _lon_range("GoA hgrid x target after 0-360 conversion", hgrid["x"].values)

    # Store both segment id and segment object so the output filename is unambiguous
    segments = []
    for seg_cfg in cfg.get("segments", []):
        seg_id = int(seg_cfg["id"])
        seg_obj = Segment(
            seg_id,
            seg_cfg["border"],
            hgrid,
            output_dir=output_dir,
            regrid_dir=regrid_dir,
        )
        _lon_range(f"segment {seg_id:03d} {seg_cfg['border']} target", seg_obj.coords["lon"].values)
        segments.append((seg_id, seg_obj))

    cobalt_vars = [
        "alk", "dic", "po4", "sio4", "o2", "no3", "nh4", "fed", "fedet", "ndet", "nbact",
        "nsmz", "nmdz", "nlgz", "chl", "chl_Lg", "chl_Md", "chl_Sm", "chl_Di", "simd",
        "silg", "ndi", "nlg", "nsm", "nmd", "pdi", "plg", "pmd", "psm"
    ]
    tracers = cfg.get("tracers", cobalt_vars)

    print(f"Input: {input_file}")
    print(f"Time selection: {time_sel}")
    print(f"Output dir: {output_dir}")
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
        output_dir=output_dir,
        time_sel=time_sel,
    )

    if cfg.get("ncrcat_years", False):
        ncrcat_names = cfg.get("ncrcat_names", [])
        assert len(ncrcat_names) in [0, len(tracers)], \
            "ncrcat_names must be empty or match tracers length"
        ncrcat_years(len(segments), output_dir, tracers, ncrcat_names)

    hgrid.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regrid NEP COBALT tracers monthly file to GOA OBC segments"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML configuration file")
    args = parser.parse_args()
    main(args.config)