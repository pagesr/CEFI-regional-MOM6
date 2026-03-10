#!/usr/bin/env python3
"""
Generate GOA2p5k OBC (T, S, SSH, U, V) from NEP10k.
The t=0 come fron the restart for T S U V and fron the outputs hindcast for zos
The t1--> 12 come from the frocasts. 
SO in the end the OBC t= hindcast that match the IC and after one month, sithch to forecast tfor the nexy 11 month. 
I we ever want to go directly everthoing from the forecast that could be done here easily 



Run:
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
# Config
# ----------------------------
def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def _require(cfg, key):
    if key not in cfg or cfg[key] in [None, ""]:
        raise ValueError(f"Missing required config key: {key}")
    return cfg[key]


def _attach_2d_lonlat(da, lon2d, lat2d, dims_expected=None, name="var"):
    """
    Attach 2D lon/lat coordinates to a DataArray, checking dims match.

    da: DataArray (2D or 3D with time/z + horizontal)
    lon2d/lat2d: numpy arrays 2D
    dims_expected: tuple of 2 dim names for horizontal dims in da (optional)
    """
    # Identify horizontal dims on da: take last 2 dims
    hdims = da.dims[-2:]
    if dims_expected is not None and tuple(hdims) != tuple(dims_expected):
        raise ValueError(
            f"{name}: horizontal dims mismatch. da last2 dims={hdims} expected={dims_expected}"
        )

    # shape check
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

    # Assign as 2D coordinates on the *same horizontal dims*
    da = da.assign_coords(
        lon=(hdims, lon2d),
        lat=(hdims, lat2d),
    )
    # Helpful attrs for xesmf / conventions
    da["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    da["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    return da




def write_year(year, glorys_dir, nep_static, segments, variables, month, ensemble, fct_dir,rst_dir,
               is_first_year=False, is_last_year=False):
    rst_dir    = "/archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/restart/" # need to be added to the yaml
    nt     = 12 
    #Step 1 creat a new dataset  
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
    
    month_starts = pd.date_range(ref, periods=nt, freq="MS")
    next_starts  = pd.date_range(ref + pd.offsets.MonthBegin(1), periods=nt, freq="MS")
    
    # make it a mutable numpy array
    time_days = ((month_starts - ref) / np.timedelta64(1, "D")).to_numpy(dtype="float64")
    
    # ONLY change the last time stamp
    last_month_end = month_starts[-1] + pd.offsets.MonthEnd(0)   # e.g. 2012-12-31 00:00
    time_days[-1] = float((last_month_end - ref) / np.timedelta64(1, "D"))
    
    # bounds (also make numpy arrays to be safe)
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
    
            # examples of empty vars you can fill later
            zos    =(("time","yh","xh"), np.zeros((nt, ny, nx))),
            so     =(("time","z","yh","xh"), np.zeros((nt, nz, ny, nx))),
            thetao =(("time","z","yh","xh"), np.zeros((nt, nz, ny, nx))),
            uo     =(("time","z","yh","xq"), np.zeros((nt, nz, ny, nxq))),
            vo     =(("time","z","yq","xh"), np.zeros((nt, nz, nyq, nx))),
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

    # Step 2 fill the new dataset
    # ==========================
    # Hindcast (t=0 only)  --- RESTART
    # ==========================
    ds_z_hind = xr.open_dataset(
        path.join(rst_dir, f"restdate_{year}{month}01/MOM_{year}{month}01.res.nc"),
        decode_cf=False)
    
    
    # IMPORTANT: rename vars (ASSIGN BACK!)
    ds_z_hind = ds_z_hind.rename({'Salt': 'so', 'Temp': 'thetao', 'u': 'uo', 'v': 'vo'})
    
    # FIll the new dataset 
    ds['so'][0,:,:,:] = np.array(ds_z_hind['so'][0,:,:,:])
    ds['thetao'][0,:,:,:] = np.array(ds_z_hind['thetao'][0,:,:,:])
    ds['uo'][0,:,:,:] = np.array(ds_z_hind['uo'][0,:,:,:])
    ds['vo'][0,:,:,:] = np.array(ds_z_hind['vo'][0,:,:,:])
    
    
    
    # ==========================
    # Hindcast zos (t=0)
    # ==========================
    ds_sfc_hind = xr.open_dataset(path.join(glorys_dir, f"{year}0101/ocean_month.nc"))
    ds['zos'][0,:,:] = np.array(ds_sfc_hind['zos'][0,:,:])
    # Remove bad value 
    ds["uo"][0] = ds["uo"][0].where(ds["uo"][0] <= 1e10, np.nan)
    ds["vo"][0] = ds["vo"][0].where(ds["vo"][0] <= 1e10, np.nan)
    
    
    
    # Now go for the fcst 
    
    # ==========================================================
    # --- Forecast directories
    # ==========================================================
    fcst_hist = path.join(fct_dir, f"{year}-{month}-e{ensemble}/history")
    # ==========================================================
    # --- Forecast SURFACE (zos) from ocean_month.nc (12 months)
    # Drop first forecast time because hindcast already provides t=0
    # ==========================================================
    ds_sfc_fcst = xr.open_dataset(path.join(fcst_hist, "ocean_month.nc"))
    ds_sfc_fcst = ds_sfc_fcst.isel(time=slice(1, None))
    
    liste_files = [
    f"oceanm_{year}_02.nc", f"oceanm_{year}_03.nc", f"oceanm_{year}_04.nc", f"oceanm_{year}_05.nc",
    f"oceanm_{year}_06.nc", f"oceanm_{year}_07.nc", f"oceanm_{year}_08.nc", f"oceanm_{year}_09.nc",
    f"oceanm_{year}_10.nc", f"oceanm_{year}_11.nc", f"oceanm_{year}_12.nc"]
    c=1
    for file in liste_files:
        print(file)
        tmp_z = xr.open_dataset(path.join(fcst_hist, file))
        tmp_z = tmp_z.rename_vars({'salt': 'so','potT': 'thetao','u': 'uo','v': 'vo'})
        ds['so'][c,:,:,:] = np.array(tmp_z['so'][0,:,:,:])
        ds['thetao'][c,:,:,:] = np.array(tmp_z['thetao'][0,:,:,:])
        ds['uo'][c,:,:,:] = np.array(tmp_z['uo'][0,:,:,:])
        ds['vo'][c,:,:,:] = np.array(tmp_z['vo'][0,:,:,:])
        c=c+1
        
    
    ds["uo"] = ds["uo"].where(ds["uo"] <= 1e10, np.nan)
    ds["vo"] = ds["vo"].where(ds["vo"] <= 1e10, np.nan)
    ds_sfc_fcst_full = xr.open_dataset(path.join(fcst_hist, "ocean_month.nc"))
    ds_sfc_fcst = ds_sfc_fcst_full[["zos"]].isel(time=slice(1, None))  # <-- drop first time
    ds['zos'][1:12,:,:] = np.array(ds_sfc_fcst['zos'][:])
    
    
    ds["vo"][0] = ds["vo"][0].where(~ds["vo"].isel(time=8).isnull())
    ds["uo"][0] = ds["uo"][0].where(~ds["uo"].isel(time=8).isnull())
    ds["zos"][0] = ds["zos"][0].where(~ds["zos"].isel(time=8).isnull())
    ds["thetao"][0] = ds["thetao"][0].where(~ds["thetao"].isel(time=8).isnull())
    ds["so"][0] = ds["so"][0].where(~ds["so"].isel(time=8).isnull())
    # Now interpolation 


    
    
    
    # --- open NEP static (2D lon/lat) ---
    st = xr.open_dataset(nep_static, decode_times=False)
    # Pull 2D coords (names are standard in your NEP static; adjust here if needed)
    # T points
    lonT = st["geolon"].values
    latT = st["geolat"].values
    # U points
    lonU = st["geolon_u"].values
    latU = st["geolat_u"].values
    # V points
    lonV = st["geolon_v"].values
    latV = st["geolat_v"].values
    
    # --- open NEP static (2D lon/lat) ---
    st = xr.open_dataset(nep_static, decode_times=False)

    # Pull 2D coords (names are standard in your NEP static; adjust here if needed)
    # T points
    lonT = st["geolon"].values
    latT = st["geolat"].values
    # U points
    lonU = st["geolon_u"].values
    latU = st["geolat_u"].values
    # V points
    lonV = st["geolon_v"].values
    latV = st["geolat_v"].values
        
    time_attrs = {
        "units": f"days since {int(year):04d}-{int(month):02d}-01 00:00:00",
        "long_name": "time",
        "axis": "T",
        "calendar": "gregorian",
        "bounds": "time_bnds",   # only if you actually write time_bnds somewhere
    }
    
    time_encoding = {
        "_FillValue": None,
        "dtype": "float64",
    }

    # -----------------------
    # SSH (zos) first
    # -----------------------
    if "zos" in variables and "zos" in ds:
        for seg in segments:
            print(f"{seg.border} zos")
            tracer = ds["zos"]#.transpose("time", "yh", "xh")
            #ds_sfc_fcst["zos"] = ds_sfc_fcst["zos"]
            print(tracer.shape)
            # zos dims are usually (time, yh, xh)
            tracer = _attach_2d_lonlat(tracer, lonT, latT, name="zos")
            seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True, time_attrs=time_attrs,time_encoding=time_encoding)

    # -----------------------
    # U/V
    # -----------------------
    if "uv" in variables and ("uo" in ds) and ("vo" in ds):
        for seg in segments:
            print(f"{seg.border} uv")
            uo = ds["uo"]
            vo = ds["vo"]

            # uo dims typically (time, z, yh, xq) OR (time, z, yh, xh-like)
            # vo dims typically (time, z, yq, xh)
            # We attach 2D coords from static on their native horizontal dims:
            uo = _attach_2d_lonlat(uo, lonU, latU, name="uo")
            vo = _attach_2d_lonlat(vo, lonV, latV, name="vo")

            seg.regrid_velocity(uo, vo, suffix=year, flood=False, rotate=False, weight_save=True, time_attrs=time_attrs,time_encoding=time_encoding)

    # -----------------------
    # Other tracers (thetao, so, etc.)
    # -----------------------
    for var in variables:
        if var in ["zos", "uv"]:
            continue

        # most tracers live in ocean_month_z.nc (ds_z)
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
                seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True, time_attrs=time_attrs,time_encoding=time_encoding)
        # sometimes a tracer could be in surface file
        elif var in ds:
            for seg in segments:
                print(f"{seg.border} {var} (from ocean_month.nc)")
                tracer = ds_sfc[var]
                tracer = _attach_2d_lonlat(tracer, lonT, latT, name=var)
                seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True, time_attrs=time_attrs,time_encoding=time_encoding)
        else:
            raise ValueError(f"{var} not found in datasets for year={year}")

    # close to reduce open handles
    ds.close()
    st.close()


def ncrcat_years(nsegments, output_dir, variables, ncrcat_names):
    if not ncrcat_names:
        ncrcat_names = variables[:]

    for var, var_name in zip(variables, ncrcat_names):
        for seg in range(1, nsegments + 1):
            run([f"ncrcat -O {var}_{seg:03d}_* {var_name}_{seg:03d}.nc"], cwd=output_dir, shell=True)


def main(config_file):
    cfg = load_config(config_file)

    first_year = cfg.get("first_year", 2012)
    last_year = cfg.get("last_year", 2012)
    glorys_dir = cfg.get(
        "glorys_dir",
        "/archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/history/",
    )
    fct_dir = cfg.get('fct_dir','/archive/Remi.Pages/forecast_goa/NEPbgc_fcst_dailyOB01/')
    month   = cfg.get('month','01')
    ensemble   = cfg.get('ensemble','01')
    
    output_dir = cfg.get("output_dir", "./outputs_CGOA_feb26")

    hgrid_file = cfg.get("hgrid", "/work/Remi.Pages/GOA2p5k/GRID/CGOA_2.5k/ocean_hgrid.nc")
    ncrcat_years_flag = cfg.get("ncrcat_years", False)
    ncrcat_names = cfg.get("ncrcat_names", [])

    # new YAML keys
    nep_static = _require(cfg, "NEP_STATIC")
    # GOA_STATIC not used inside this script (Segment uses GOA hgrid); keep in YAML for your other workflows
    _ = cfg.get("GOA_STATIC", None)

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # Load GOA hgrid
    hgrid = xr.open_dataset(hgrid_file)

    # Variables + segments
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
    parser.add_argument("--config", type=str, default="glorys_obc_CGOA.yaml",
                        help="YAML configuration file")
    args = parser.parse_args()
    main(args.config)