#!/usr/bin/env python3
"""
Generate GOA2p5k Open Boundary Conditions (OBC) from NEP10k (hindcast + forecast).

Context / goal
--------------
We want a single OBC file per year where:
  - t = 0 matches the Initial Condition (IC) used to start the GOA run
  - t = 1..11 comes from the NEP forecast for the following 11 months
  - t = 12 is an extra padded step that repeats the final available month

Why the extra time step?
------------------------
The model needs one additional OBC time step to run all the way through.
So we build 13 time steps total:
  - 12 "real" monthly states
  - 1 extra padded state copied from the last available month

Data sources used here
----------------------
1) Hindcast / Restart (t = 0 ONLY)
   - 3D fields (T, S, U, V) are taken from the NEP restart:
       MOM_YYYYMM01.res.nc
     This ensures vertical levels match the forecast (75 levels).
   - Surface SSH (zos) for t = 0 is taken from the NEP hindcast monthly output:
       ocean_month.nc

2) Forecast (t = 1..11)
   - SSH (zos) comes from:
       forecast history/ocean_month.nc  (we drop the first time because t=0 is hindcast)
   - 3D fields (T, S, U, V) come from monthly forecast files:
       forecast history/oceanm_YYYY_02.nc ... oceanm_YYYY_12.nc
     (each file contains a single monthly time)

3) Padding (t = 12)
   - Last time step is duplicated from t = 11 for all variables

Run
---
  python write_CGOA_boundary_2Dfrc.py --config write_CGOA_boundary_south_2D.yaml
  python write_CGOA_boundary_2Dfrc.py --config write_CGOA_boundary_west_2D.yaml
"""

from subprocess import run
from os import path
import os
import argparse
import warnings
import yaml
import xarray as xr
import glob
import pandas as pd
import numpy as np

from boundary import Segment

warnings.filterwarnings("ignore")


# ----------------------------
# Config helpers
# ----------------------------
def load_config(config_file):
    """Load YAML config file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def _require(cfg, key):
    """Require a YAML key to exist and be non-empty."""
    if key not in cfg or cfg[key] in [None, ""]:
        raise ValueError(f"Missing required config key: {key}")
    return cfg[key]


# ----------------------------
# Utilities
# ----------------------------
def _attach_2d_lonlat(da, lon2d, lat2d, dims_expected=None, name="var"):
    """
    Attach 2D lon/lat coordinates to a DataArray, with dimension/shape checks.

    Parameters
    ----------
    da : xarray.DataArray
        A 2D or 3D (time/z + horizontal) DataArray.
        The *last two dims* are treated as the horizontal dims.
    lon2d, lat2d : np.ndarray
        2D longitude/latitude arrays matching the horizontal shape of `da`.
    dims_expected : tuple, optional
        If provided, enforce that `da.dims[-2:]` matches this tuple.
    name : str
        Variable name for clearer error messages.

    Returns
    -------
    xarray.DataArray
        Same data with 2D lon/lat attached as coordinates.
    """
    hdims = da.dims[-2:]
    if dims_expected is not None and tuple(hdims) != tuple(dims_expected):
        raise ValueError(
            f"{name}: horizontal dims mismatch. da last2 dims={hdims} expected={dims_expected}"
        )

    if lon2d.shape != tuple(da.sizes[d] for d in hdims):
        raise ValueError(
            f"{name}: lon2d shape {lon2d.shape} does not match da horizontal shape "
            f"{tuple(da.sizes[d] for d in hdims)} for dims {hdims}"
        )
    if lat2d.shape != tuple(da.sizes[d] for d in hdims):
        raise ValueError(
            f"{name}: lat2d shape {lat2d.shape} does not match da horizontal shape "
            f"{tuple(da.sizes[d] for d in hdims)} for dims {hdims}"
        )

    da = da.assign_coords(
        lon=(hdims, lon2d),
        lat=(hdims, lat2d),
    )
    da["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    da["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    return da


# ----------------------------
# Core routine
# ----------------------------
def write_year(year, glorys_dir, nep_static, segments, variables, month, ensemble, fct_dir, rst_dir,
               is_first_year=False, is_last_year=False):

    nt = 13
    nt_src = 12  # real source months
    nz = 75
    ny = 816
    nx = 342
    nxq = 343
    nyq = 817
    nzi = 76
    nnv = 2

    # -------------------------
    # BUILD CF TIME + BOUNDS
    # -------------------------
    ref = pd.Timestamp(year=int(year), month=int(month), day=1)

    # 13 time steps total: 12 real + 1 padded extra month
    month_starts = pd.date_range(ref, periods=nt, freq="MS")
    next_starts  = pd.date_range(ref + pd.offsets.MonthBegin(1), periods=nt, freq="MS")

    time_days = ((month_starts - ref) / np.timedelta64(1, "D")).to_numpy(dtype="float64")

    # Keep same behavior as before: place last timestamp at end of its month
    last_month_end = month_starts[-1] + pd.offsets.MonthEnd(0)
    time_days[-1] = float((last_month_end - ref) / np.timedelta64(1, "D"))

    b0 = ((month_starts - ref) / np.timedelta64(1, "D")).to_numpy(dtype="float64")
    b1 = ((next_starts  - ref) / np.timedelta64(1, "D")).to_numpy(dtype="float64")
    time_bnds = np.stack([b0, b1], axis=1)

    # -------------------------
    # COORDINATES
    # -------------------------
    coords = dict(
        time=("time", time_days),
        nv=("nv", np.arange(nnv)),
        z=("z", np.arange(nz)),
        zi=("zi", np.arange(nzi)),
        yh=("yh", np.arange(ny)),
        xh=("xh", np.arange(nx)),
        yq=("yq", np.arange(nyq)),
        xq=("xq", np.arange(nxq)),
    )

    # -------------------------
    # EMPTY DATASET (fill later)
    # -------------------------
    ds = xr.Dataset(
        data_vars=dict(
            time_bnds=(("time", "nv"), time_bnds),

            zos    =(("time", "yh", "xh"), np.zeros((nt, ny, nx))),
            so     =(("time", "z", "yh", "xh"), np.zeros((nt, nz, ny, nx))),
            thetao =(("time", "z", "yh", "xh"), np.zeros((nt, nz, ny, nx))),
            uo     =(("time", "z", "yh", "xq"), np.zeros((nt, nz, ny, nxq))),
            vo     =(("time", "z", "yq", "xh"), np.zeros((nt, nz, nyq, nx))),
        ),
        coords=coords
    )

    # -------------------------
    # TIME ATTRIBUTES (CF-style)
    # -------------------------
    ds["time"].attrs = {
        "units": f"days since {int(year):04d}-{int(month):02d}-01 00:00:00",
        "long_name": "time",
        "axis": "T",
        "calendar_type": "GREGORIAN",
        "calendar": "gregorian",
        "bounds": "time_bnds",
    }

    ds["time_bnds"].attrs = {"long_name": "time bounds"}

    # ==========================================
    # Step 2: FILL ds(time=0) FROM HINDCAST DATA
    # ==========================================
    ds_z_hind = xr.open_dataset(
        path.join(rst_dir, f"restdate_{year}{month}01/MOM_{year}{month}01.res.nc"),
        decode_cf=False
    )

    ds_z_hind = ds_z_hind.rename({'Salt': 'so', 'Temp': 'thetao', 'u': 'uo', 'v': 'vo'})

    ds['so'][0, :, :, :] = np.array(ds_z_hind['so'][0, :, :, :])
    ds['thetao'][0, :, :, :] = np.array(ds_z_hind['thetao'][0, :, :, :])
    ds['uo'][0, :, :, :] = np.array(ds_z_hind['uo'][0, :, :, :])
    ds['vo'][0, :, :, :] = np.array(ds_z_hind['vo'][0, :, :, :])

    ds_sfc_hind = xr.open_dataset(path.join(glorys_dir, f"{year}0101/ocean_month.nc"))
    ds['zos'][0, :, :] = np.array(ds_sfc_hind['zos'][0, :, :])

    ds["uo"][0] = ds["uo"][0].where(ds["uo"][0] <= 1e10, np.nan)
    ds["vo"][0] = ds["vo"][0].where(ds["vo"][0] <= 1e10, np.nan)

    # ==========================================
    # Step 3: FILL ds(time=1..11) FROM FORECAST
    # ==========================================
    fcst_hist = path.join(fct_dir, f"{year}-{month}-e{ensemble}/history")

    liste_files = [
        f"oceanm_{year}_02.nc", f"oceanm_{year}_03.nc", f"oceanm_{year}_04.nc", f"oceanm_{year}_05.nc",
        f"oceanm_{year}_06.nc", f"oceanm_{year}_07.nc", f"oceanm_{year}_08.nc", f"oceanm_{year}_09.nc",
        f"oceanm_{year}_10.nc", f"oceanm_{year}_11.nc", f"oceanm_{year}_12.nc"
    ]

    c = 1
    for file in liste_files:
        print(file)
        tmp_z = xr.open_dataset(path.join(fcst_hist, file))
        tmp_z = tmp_z.rename_vars({'salt': 'so', 'potT': 'thetao', 'u': 'uo', 'v': 'vo'})

        ds['so'][c, :, :, :] = np.array(tmp_z['so'][0, :, :, :])
        ds['thetao'][c, :, :, :] = np.array(tmp_z['thetao'][0, :, :, :])
        ds['uo'][c, :, :, :] = np.array(tmp_z['uo'][0, :, :, :])
        ds['vo'][c, :, :, :] = np.array(tmp_z['vo'][0, :, :, :])
        c = c + 1

    ds["uo"] = ds["uo"].where(ds["uo"] <= 1e10, np.nan)
    ds["vo"] = ds["vo"].where(ds["vo"] <= 1e10, np.nan)

    ds_sfc_fcst_full = xr.open_dataset(path.join(fcst_hist, "ocean_month.nc"))
    ds_sfc_fcst = ds_sfc_fcst_full[["zos"]].isel(time=slice(1, None))
    ds['zos'][1:12, :, :] = np.array(ds_sfc_fcst['zos'][:])

    # Apply NaN mask from a reference forecast month onto t=0
    ds["vo"][0] = ds["vo"][0].where(~ds["vo"].isel(time=8).isnull())
    ds["uo"][0] = ds["uo"][0].where(~ds["uo"].isel(time=8).isnull())
    ds["zos"][0] = ds["zos"][0].where(~ds["zos"].isel(time=8).isnull())
    ds["thetao"][0] = ds["thetao"][0].where(~ds["thetao"].isel(time=8).isnull())
    ds["so"][0] = ds["so"][0].where(~ds["so"].isel(time=8).isnull())

    # ==========================================
    # Step 3b: PAD EXTRA LAST TIME STEP
    # ==========================================
    # Duplicate the final available month into the new extra slot
    ds["zos"][12, :, :] = ds["zos"][11, :, :]
    ds["so"][12, :, :, :] = ds["so"][11, :, :, :]
    ds["thetao"][12, :, :, :] = ds["thetao"][11, :, :, :]
    ds["uo"][12, :, :, :] = ds["uo"][11, :, :, :]
    ds["vo"][12, :, :, :] = ds["vo"][11, :, :, :]

    # ==========================================
    # Step 4: Load NEP static grid (2D lon/lat)
    # ==========================================
    st = xr.open_dataset(nep_static, decode_times=False)

    lonT = st["geolon"].values
    latT = st["geolat"].values

    lonU = st["geolon_u"].values
    latU = st["geolat_u"].values

    lonV = st["geolon_v"].values
    latV = st["geolat_v"].values

    time_attrs = {
        "units": f"days since {int(year):04d}-{int(month):02d}-01 00:00:00",
        "long_name": "time",
        "axis": "T",
        "calendar": "gregorian",
        "bounds": "time_bnds",
    }

    time_encoding = {
        "_FillValue": None,
        "dtype": "float64",
    }

    # ==========================================
    # Step 5: Regrid and write OBC per segment
    # ==========================================

    if "zos" in variables and "zos" in ds:
        for seg in segments:
            print(f"{seg.border} zos")
            tracer = ds["zos"]
            print(tracer.shape)
            tracer = _attach_2d_lonlat(tracer, lonT, latT, name="zos")
            seg.regrid_tracer(
                tracer, suffix=year, flood=False, weight_save=True,
                time_attrs=time_attrs, time_encoding=time_encoding
            )

    if "uv" in variables and ("uo" in ds) and ("vo" in ds):
        for seg in segments:
            print(f"{seg.border} uv")
            uo = ds["uo"]
            vo = ds["vo"]

            uo = _attach_2d_lonlat(uo, lonU, latU, name="uo")
            vo = _attach_2d_lonlat(vo, lonV, latV, name="vo")

            seg.regrid_velocity(
                uo, vo, suffix=year, flood=False, rotate=False, weight_save=True,
                time_attrs=time_attrs, time_encoding=time_encoding
            )

    for var in variables:
        if var in ["zos", "uv"]:
            continue

        if var in ds:
            for seg in segments:
                print(ds[var].shape)
                print(var)
                print("~~~~~~~~~~~~~~~~~")
                print("~~~~~~~~~~~~~~~~~")
                print(f"{seg.border} {var}")
                tracer = ds[var]
                tracer = _attach_2d_lonlat(tracer, lonT, latT, name=var)
                print(tracer)
                seg.regrid_tracer(
                    tracer, suffix=year, flood=False, weight_save=True,
                    time_attrs=time_attrs, time_encoding=time_encoding
                )
        elif var in ds:
            for seg in segments:
                print(f"{seg.border} {var} (from ocean_month.nc)")
                tracer = ds_sfc[var]
                tracer = _attach_2d_lonlat(tracer, lonT, latT, name=var)
                seg.regrid_tracer(
                    tracer, suffix=year, flood=False, weight_save=True,
                    time_attrs=time_attrs, time_encoding=time_encoding
                )
        else:
            raise ValueError(f"{var} not found in datasets for year={year}")

    ds.close()
    st.close()


def ncrcat_years(nsegments, output_dir, variables, ncrcat_names):
    """Concatenate per-year segment files into a single file per variable/segment."""
    if not ncrcat_names:
        ncrcat_names = variables[:]

    for var, var_name in zip(variables, ncrcat_names):
        for seg in range(1, nsegments + 1):
            run([f"ncrcat -O {var}_{seg:03d}_* {var_name}_{seg:03d}.nc"], cwd=output_dir, shell=True)


def main(config_file):
    cfg = load_config(config_file)

    first_year = int(cfg.get("first_year", 2012))
    last_year = int(cfg.get("last_year", 2012))

    glorys_dir = cfg.get(
        "glorys_dir",
        "/archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/history/",
    )
    fct_dir = cfg.get('fct_dir', '/archive/Remi.Pages/forecast_goa/NEPbgc_fcst_dailyOB01/')
    month = str(cfg.get('month', '01')).zfill(2)
    ensemble = str(cfg.get('ensemble', '01')).zfill(2)

    output_dir = cfg.get("output_dir", "./outputs_CGOA_feb26")
    rst_dir = cfg.get("rst_dir", "/archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/restart/")

    hgrid_file = cfg.get("hgrid", "/work/Remi.Pages/GOA2p5k/GRID/CGOA_2.5k/ocean_hgrid.nc")
    ncrcat_years_flag = cfg.get("ncrcat_years", False)
    ncrcat_names = cfg.get("ncrcat_names", [])

    nep_static = _require(cfg, "NEP_STATIC")
    _ = cfg.get("GOA_STATIC", None)

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    hgrid = xr.open_dataset(hgrid_file)

    variables = cfg.get("variables", [])

    segments = []
    for seg_cfg in cfg.get("segments", []):
        segment = Segment(seg_cfg["id"], seg_cfg["border"], hgrid, output_dir=output_dir)
        segments.append(segment)

    for y in range(first_year, last_year + 1):
        print(y)
        write_year(
            y,
            glorys_dir=glorys_dir,
            nep_static=nep_static,
            segments=segments,
            variables=variables,
            month=month,
            ensemble=ensemble,
            fct_dir=fct_dir,
            rst_dir=rst_dir,
            is_first_year=(y == first_year),
            is_last_year=(y == last_year),
        )

    if ncrcat_years_flag:
        assert len(ncrcat_names) == len(variables), (
            "Could not concatenate annual files because the number of file output names "
            "did not match the number of variables provided."
        )
        ncrcat_years(len(segments), output_dir, variables, ncrcat_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OBC from NEP (using static 2D lon/lat)")
    parser.add_argument(
        "--config",
        type=str,
        default="glorys_obc_CGOA.yaml",
        help="YAML configuration file"
    )
    args = parser.parse_args()
    main(args.config)