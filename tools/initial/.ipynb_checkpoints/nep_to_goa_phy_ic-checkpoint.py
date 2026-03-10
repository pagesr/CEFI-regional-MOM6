#!/usr/bin/env python3
# Other way to interpolate by Remi

import sys
import os
import argparse
import yaml

import numpy as np

import xesmf
import xarray as xr
import xarray


def parse_args():
    parser = argparse.ArgumentParser(description="NEP -> GOA PHY IC (Remi method)")
    parser.add_argument(
        "--config_file",
        type=str,
        default="nep_to_goa_phy_ic.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


args = parse_args()

with open(args.config_file, "r") as f:
    config = yaml.safe_load(f)

GOA_STATIC = config.get(
    "GOA_STATIC",
    "/archive/Remi.Pages/fre/Arc_12/2026_02.01/CGOA_BGC_2025_07_base_nep_phy_feb26/gfdl.ncrc6-intel23-prod/pp/ocean_daily/ocean_daily.static.nc",
)
NEP_STATIC = config.get(
    "NEP_STATIC",
    "/archive/Liz.Drenkard/fre/cefi/NEP/2025_07/NEP10k_202507_physics_bgc/gfdl.ncrc6-intel23-repro/pp/ocean_daily/ocean_daily.static.nc",
)

def regrid_tracer(fld, method='bilinear'):
    coords = xr.open_dataset(GOA_STATIC)
    coords = coords.rename({'geolon': 'lon', 'geolat': 'lat'})  # interp on this 
    gsource = xr.open_dataset(NEP_STATIC,decode_times=False) # Source
    gsource = gsource.rename({'geolon': 'lon', 'geolat': 'lat'})  # interp on this 
    print(gsource)
    regrid = xesmf.Regridder(
        gsource,
        coords,
        method=method,
        periodic=False,
        filename=config.get("regrid_tracer_weights", 'regrid_bilin.nc'),
        reuse_weights=config.get("reuse_weights", True)
    )
    tdest = regrid(fld)
    return tdest

def regrid_u(fld, method='bilinear'):
    coords = xr.open_dataset(GOA_STATIC)
    coords = coords.rename({'geolon_u': 'lon', 'geolat_u': 'lat'})  # interp on this 
    gsource = xr.open_dataset(NEP_STATIC,decode_times=False) # Source
    gsource = gsource.rename({'geolon_u': 'lon', 'geolat_u': 'lat'})  # interp on this 
    
    regrid = xesmf.Regridder(
        gsource,
        coords,
        method=method,
        periodic=False,
        filename=config.get("regrid_u_weights", 'regrid_bilin_uu.nc'),
        reuse_weights=config.get("reuse_weights", True)
    )
    tdest = regrid(fld)
    return tdest

def regrid_v(fld, method="bilinear"):
    coords = xr.open_dataset(GOA_STATIC)
    coords = coords.rename({"geolon_v": "lon", "geolat_v": "lat"})
    gsource = xr.open_dataset(NEP_STATIC, decode_times=False)
    gsource = gsource.rename({"geolon_v": "lon", "geolat_v": "lat"})

    regrid = xesmf.Regridder(
        gsource, coords,
        method=method,
        periodic=False,
        filename=config.get("regrid_v_weights", "regrid_bilin_vv.nc"),
        reuse_weights=config.get("reuse_weights", True),
    )
    return regrid(fld)


# Use the functions
variable_names = config.get('variable_names', {})
temp_var = variable_names.get('temperature', 'Temp')
sal_var = variable_names.get('salinity', 'Salt')
ssh_var = variable_names.get('sea_surface_height', 'ave_ssh')
u_var = variable_names.get('zonal_velocity', 'u')
v_var = variable_names.get('meridional_velocity', 'v')

nep_restart = config.get(
    "nep_restart",
    "/archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_spinup/restart/restdate_19960101/MOM_19960101.res.nc",
)

nep = (
    xarray.open_dataset(nep_restart,decode_times=False)
    [[temp_var, sal_var, ssh_var, u_var, v_var]]
)

# === Regrid to MOM6 Grid ===
regridded = regrid_tracer(nep[temp_var], method=config.get("tracer_method", 'nearest_s2d'))  # Should return (time, depth, J, I)
regridded_salt = regrid_tracer(nep[sal_var], method=config.get("tracer_method", 'nearest_s2d'))  # Should return (time, depth, J, I)
print(np.nanmax(regridded_salt))
regridded_shh = regrid_tracer(nep[ssh_var], method=config.get("tracer_method", 'nearest_s2d'))  # Should return (time, depth, J, I)
regridded_u = regrid_u(nep[u_var], method=config.get("uv_method", 'nearest_s2d'))  # Should return (time, depth, J, I)
regridded_v = regrid_v(nep[v_var], method=config.get("uv_method", "nearest_s2d"))
print("regridded_v dims:", regridded_v.dims)

print("regridded_u dims:", regridded_u.dims)

print("regridded_t dims:", regridded.dims)
print("regridded_shh dims:", regridded_shh.dims)

import os
import numpy as np
import xarray as xr

OUT_IC  = config.get("OUT_IC", "ic_nep_to_goa.nc")
IC_DATE = np.datetime64(config.get("IC_DATE", "1993-01-01T00:00:00"))

TIME_REF   = np.datetime64(config.get("TIME_REF", "1993-01-01T00:00:00"))
TIME_UNITS = config.get("TIME_UNITS", "days since 1993-01-01 00:00:00")
TIME_CAL   = config.get("TIME_CAL", "gregorian")

def rename_only_existing(da, mapping):
    """Rename only dims/coords that actually exist in this DataArray."""
    m = {k: v for k, v in mapping.items() if (k in da.dims) or (k in da.coords)}
    return da.rename(m)

# global mapping (we'll apply selectively)
dim_map = {
    "Time":  "time",
    "Layer": "zl",
    "jh":    "yh",
    "ih":    "xh",
    "iq":    "xq",
    "jq":    "yq",
}

# ---- rename + order ----
temp = rename_only_existing(regridded, dim_map).rename("temp").transpose("time","zl","yh","xh")
salt = rename_only_existing(regridded_salt, dim_map).rename("salt").transpose("time","zl","yh","xh")
ssh  = rename_only_existing(regridded_shh, dim_map).rename("ssh").transpose("time","yh","xh")
u    = rename_only_existing(regridded_u, dim_map).rename("u").transpose("time","zl","yh","xq")
v    = rename_only_existing(regridded_v, dim_map).rename("v").transpose("time","zl","yq","xh")

# ---- MOM6 numeric time ----
time_days = float((IC_DATE - TIME_REF) / np.timedelta64(1, "D"))
time = xr.DataArray([time_days], dims=("time",), name="time")
time.attrs = {"units": TIME_UNITS, "calendar": TIME_CAL}

temp = temp.assign_coords(time=time)
salt = salt.assign_coords(time=time)
ssh  = ssh.assign_coords(time=time)
u    = u.assign_coords(time=time)
v    = v.assign_coords(time=time)

# ---- zl coord + attrs ----
zl_vals = temp["zl"].values if "zl" in temp.coords else np.arange(temp.sizes["zl"], dtype="float64")
zl = xr.DataArray(zl_vals.astype("float64"), dims=("zl",), name="zl")
zl.attrs = {
    "long_name": "Layer pseudo-depth, -z*",
    "units": "meter",
    "cartesian_axis": "Z",
    "positive": "down",
}

# ---- build dataset ----
ic = xr.Dataset(
    data_vars={"temp": temp, "salt": salt, "ssh": ssh, "u": u, "v": v},
    coords={"time": time, "zl": zl},
    attrs={"regrid_method": config.get("regrid_method_attr", "nearest_s2d")},
)

# ---- write ----
all_vars = list(ic.data_vars) + list(ic.coords)
encoding = {vn: {"_FillValue": None} for vn in all_vars}
encoding["time"].update({"dtype": "float64"})

outdir = os.path.dirname(OUT_IC)
if outdir and not os.path.exists(outdir):
    os.makedirs(outdir)

ic.to_netcdf(
    OUT_IC,
    format="NETCDF3_64BIT",
    engine="netcdf4",
    encoding=encoding,
    unlimited_dims=("time",),
)

print("Wrote:", OUT_IC)
print(ic)