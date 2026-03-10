#!/usr/bin/env python3
"""
FAST production OBC generator (NEP10k -> GOA2.5k) for segment_002 (west boundary)
using 1-D nominal axes (xh/yh/xq/yq).

- Reads NEP monthly files:
    ocean_month_z.nc (uo, vo, thetao, so, z_l)
    ocean_month.nc   (zos)
- Attaches 1-D lon/lat coords:
    tracers/zos : lon=xh, lat=yh
    uo          : lon=xq, lat=yh
    vo          : lon=xh, lat=yq
- Regrids to GOA boundary points defined by Segment(border=...) using xESMF LocStream.

Run:
  module load nco/5.0.1   (optional, only if you use ncrcat_years)
  ./write_nep_boundary_fast_1d_west_002.py --config nep_obc_CGOA_west.yaml
"""

from subprocess import run
from os import path
import os
import argparse
import warnings
import yaml
import xarray as xr

from boundary import Segment

warnings.filterwarnings("ignore")


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def write_year(year, nep_dir, segments, variables, is_first_year=False, is_last_year=False):
    # NEP monthly sources
    ds_z = xr.open_dataset(path.join(nep_dir, f"{year}0101crpt_forc/ocean_month_z.nc"))
    ds_z = ds_z.rename({"z_l": "z"})  # keep your convention

    ds_sfc = xr.open_dataset(path.join(nep_dir, f"{year}0101crpt_forc/ocean_month.nc"))

    # Adjust time for matching initial conditions (same logic as your 001 script)
    if "time" in ds_z:
        if is_first_year:
            tnew = xr.concat((ds_z["time"][0].dt.floor("1d"), ds_z["time"][1:]), dim="time")
            ds_z["time"] = ("time", tnew.data)
        elif is_last_year:
            tnew = xr.concat((ds_z["time"][0:-1], ds_z["time"][-1].dt.ceil("1d")), dim="time")
            ds_z["time"] = ("time", tnew.data)

    # -----------------------
    # SSH (zos) first
    # -----------------------
    if "zos" in variables and "zos" in ds_sfc:
        for seg in segments:
            print(f"{seg.border} zos")
            tracer = ds_sfc["zos"]
            tracer = tracer.assign_coords(lon=ds_sfc["xh"], lat=ds_sfc["yh"])
            tracer["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
            tracer["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
            seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True)

    # -----------------------
    # U/V
    # -----------------------
    if "uv" in variables and ("uo" in ds_z) and ("vo" in ds_z):
        for seg in segments:
            print(f"{seg.border} uv")
            uo = ds_z["uo"].assign_coords(lon=ds_z["xq"], lat=ds_z["yh"])
            vo = ds_z["vo"].assign_coords(lon=ds_z["xh"], lat=ds_z["yq"])

            uo["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
            uo["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
            vo["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
            vo["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})

            # rotate=False to keep earth-frame as you want (same as your 002 script)
            seg.regrid_velocity(uo, vo, suffix=year, flood=False, rotate=False, weight_save=True)

    # -----------------------
    # Other tracers (thetao, so, etc.)
    # -----------------------
    for var in variables:
        if var in ["zos", "uv"]:
            continue

        if var in ds_z:
            for seg in segments:
                print(f"{seg.border} {var}")
                tracer = ds_z[var].assign_coords(lon=ds_z["xh"], lat=ds_z["yh"])
                tracer["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
                tracer["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
                seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True)

        elif var in ds_sfc:
            for seg in segments:
                print(f"{seg.border} {var} (from ocean_month.nc)")
                tracer = ds_sfc[var].assign_coords(lon=ds_sfc["xh"], lat=ds_sfc["yh"])
                tracer["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
                tracer["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
                seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True)

        else:
            raise ValueError(f"{var} not found in datasets for year={year}")

    ds_z.close()
    ds_sfc.close()


def ncrcat_years(nsegments, output_dir, variables, ncrcat_names):
    if not ncrcat_names:
        ncrcat_names = variables[:]
    for var, var_name in zip(variables, ncrcat_names):
        for seg in range(1, nsegments + 1):
            run([f"ncrcat -O {var}_{seg:03d}_* {var_name}_{seg:03d}.nc"], cwd=output_dir, shell=True)


def main(config_file):
    cfg = load_config(config_file)

    first_year = cfg.get("first_year", 1993)
    last_year = cfg.get("last_year", 1993)

    nep_dir = cfg.get(
        "glorys_dir",  # keep old key for compatibility
        "/archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/history/",
    )
    output_dir = cfg.get("output_dir", "./outputs_CGOA_fast1d_west")
    hgrid_file = cfg.get("hgrid", "/work/Remi.Pages/GOA2p5k/GRID/CGOA_2.5k/ocean_hgrid.nc")

    ncrcat_years_flag = cfg.get("ncrcat_years", False)
    ncrcat_names = cfg.get("ncrcat_names", [])

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    hgrid = xr.open_dataset(hgrid_file)

    variables = cfg.get("variables", [])
    segments = []
    for seg_cfg in cfg.get("segments", []):
        segments.append(Segment(seg_cfg["id"], seg_cfg["border"], hgrid, output_dir=output_dir))

    # Helpful: print the GOA boundary points being used
    for seg in segments:
        c = seg.coords
        print(seg.border, seg.segstr, "GOA boundary points:", c["lon"].shape)

    for y in range(first_year, last_year + 1):
        print("YEAR:", y)
        write_year(
            y,
            nep_dir=nep_dir,
            segments=segments,
            variables=variables,
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
    parser = argparse.ArgumentParser(description="FAST NEP->GOA OBC using 1-D xh/yh (seg002 west)")
    parser.add_argument("--config", type=str, default="nep_obc_CGOA_west.yaml",
                        help="YAML configuration file")
    args = parser.parse_args()
    main(args.config)