#!/usr/bin/env python3
"""
Generate GOA2p5k OBC (T, S, SSH, U, V) from NEP10k (a.k.a. "glorys" in your paths).

WEST boundary version (seg 002) — kept as close as possible to your seg001 script.

UPDATE in this version:
- Same as 001: uses 2D lon/lat from NEP_STATIC (ocean_daily.static.nc).
- Segment config should point to id=2, border='west' in YAML.
- Velocity rotation remains OFF (rotate=False), per your request.

Run:
  ./write_glorys_boundary_west.py --config glorys_obc_CGOA_west.yaml
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


def write_year(year, glorys_dir, nep_static, segments, variables,
               is_first_year=False, is_last_year=False):
    """
    glorys_dir: NEP history directory
    nep_static: path to NEP ocean_daily.static.nc (2D geolon/geolat)
    """

    # --- open source files (NEP) ---
    ds_z = xr.open_dataset(path.join(glorys_dir, f"{year}0101crpt_forc/ocean_month_z.nc"))
    ds_z = ds_z.rename({"z_l": "z"})  # keep your existing z rename

    ds_sfc = xr.open_dataset(path.join(glorys_dir, f"{year}0101crpt_forc/ocean_month.nc"))

    # --- open NEP static (2D lon/lat) ---
    st = xr.open_dataset(nep_static, decode_times=False)

    # T points
    lonT = st["geolon"].values
    latT = st["geolat"].values
    # U points
    lonU = st["geolon_u"].values
    latU = st["geolat_u"].values
    # V points
    lonV = st["geolon_v"].values
    latV = st["geolat_v"].values

    # --- adjust time for matching initial conditions (keep your logic) ---
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
            tracer = _attach_2d_lonlat(tracer, lonT, latT, name="zos")
            seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True)

    # -----------------------
    # U/V
    # -----------------------
    if "uv" in variables and ("uo" in ds_z) and ("vo" in ds_z):
        for seg in segments:
            print(f"{seg.border} uv")
            uo = ds_z["uo"]
            vo = ds_z["vo"]

            uo = _attach_2d_lonlat(uo, lonU, latU, name="uo")
            vo = _attach_2d_lonlat(vo, lonV, latV, name="vo")

            # keep rotate=False exactly as in 001
            seg.regrid_velocity(uo, vo, suffix=year, flood=False, rotate=False, weight_save=True)

    # -----------------------
    # Other tracers (thetao, so, etc.)
    # -----------------------
    for var in variables:
        if var in ["zos", "uv"]:
            continue

        # most tracers live in ocean_month_z.nc (ds_z)
        if var in ds_z:
            for seg in segments:
                print(f"{seg.border} {var}")
                tracer = ds_z[var]
                tracer = _attach_2d_lonlat(tracer, lonT, latT, name=var)
                seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True)

        # sometimes a tracer could be in surface file
        elif var in ds_sfc:
            for seg in segments:
                print(f"{seg.border} {var} (from ocean_month.nc)")
                tracer = ds_sfc[var]
                tracer = _attach_2d_lonlat(tracer, lonT, latT, name=var)
                seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True)
        else:
            raise ValueError(f"{var} not found in datasets for year={year}")

    ds_z.close()
    ds_sfc.close()
    st.close()


def ncrcat_years(nsegments, output_dir, variables, ncrcat_names):
    if not ncrcat_names:
        ncrcat_names = variables[:]

    for var, var_name in zip(variables, ncrcat_names):
        for seg in range(1, nsegments + 1):
            run([f"ncrcat -O {var}_{seg:03d}_* {var_name}_{seg:03d}.nc"], cwd=output_dir, shell=True)


def main(config_file):
    cfg = load_config(config_file)

    first_year = cfg.get("first_year", 1997)
    last_year = cfg.get("last_year", 1997)

    glorys_dir = cfg.get(
        "glorys_dir",
        "/archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/history/",
    )
    output_dir = cfg.get("output_dir", "./outputs_CGOA_feb26")

    hgrid_file = cfg.get("hgrid", "/work/Remi.Pages/GOA2p5k/GRID/CGOA_2.5k/ocean_hgrid.nc")
    ncrcat_years_flag = cfg.get("ncrcat_years", False)
    ncrcat_names = cfg.get("ncrcat_names", [])

    # YAML keys
    nep_static = _require(cfg, "NEP_STATIC")
    _ = cfg.get("GOA_STATIC", None)  # not used here; kept for consistency

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
    parser.add_argument("--config", type=str, default="glorys_obc_CGOA_west.yaml",
                        help="YAML configuration file")
    args = parser.parse_args()
    main(args.config)