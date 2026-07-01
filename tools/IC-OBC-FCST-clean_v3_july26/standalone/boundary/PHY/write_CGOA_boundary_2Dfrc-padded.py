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
   - Surface SSH (zos) for t = 0 is taken from the NEP restart variable:
       ave_ssh

2) Forecast (t = 1..11)
   - SSH (zos) comes from:
       forecast history/ocean_daily.nc  (we drop the first time because t=0 is restart-derived)
   - 3D fields (T, S, U, V) come from monthly forecast files beginning at
     the month after the configured start month (with year rollover), for
     11 files total.
     Example for month=04:
       oceanm_YYYY_05.nc ... oceanm_YYYY_12.nc, oceanm_(YYYY+1)_01.nc ... _03.nc

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
import importlib.util
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
    print(f"[PHY-OBC] Loading config: {config_file}")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def _require(cfg, key):
    """Require a YAML key to exist and be non-empty."""
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
    print(f"[PHY-OBC-v2] {label} lon range: {np.nanmin(arr):.6f} to {np.nanmax(arr):.6f}")


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



def _select_time_z_slice(da: xr.DataArray, time_index=0, z_index=0) -> xr.DataArray:
    """Return a 2D horizontal or 1D boundary slice for diagnostics."""
    out = da
    if "time" in out.dims:
        idx = min(max(int(time_index), 0), out.sizes["time"] - 1)
        out = out.isel(time=idx)
    z_dims = [d for d in out.dims if d == "z" or d.startswith("nz_")]
    if z_dims:
        zdim = z_dims[0]
        idx = min(max(int(z_index), 0), out.sizes[zdim] - 1)
        out = out.isel({zdim: idx})
    return out.squeeze(drop=True)


def _align_longitude_convention(source_lon, target_lon):
    """Return source longitudes shifted to the target longitude convention."""
    src = np.asarray(source_lon, dtype="float64").copy()
    tgt = np.asarray(target_lon, dtype="float64")
    src_med = np.nanmedian(src)
    tgt_med = np.nanmedian(tgt)
    if np.isfinite(src_med) and np.isfinite(tgt_med):
        if tgt_med < 0.0 and src_med > 180.0:
            src = np.where(src > 180.0, src - 360.0, src)
        elif tgt_med > 180.0 and src_med < 0.0:
            src = np.where(src < 0.0, src + 360.0, src)
    return src


def _nearest_source_profile(source: xr.DataArray, target_lon, target_lat, time_index=0, z_index=0):
    """Sample a source field at nearest source-grid points to a target boundary."""
    src = _select_time_z_slice(source, time_index=time_index, z_index=z_index)
    if "lon" not in src.coords or "lat" not in src.coords:
        raise ValueError(f"{source.name}: diagnostic source field has no lon/lat coordinates")

    src_vals = np.asarray(src.values, dtype="float64")
    tgt_lon = np.asarray(target_lon, dtype="float64").ravel()
    tgt_lat = np.asarray(target_lat, dtype="float64").ravel()
    lon = _align_longitude_convention(src["lon"].values, tgt_lon)
    lat = np.asarray(src["lat"].values, dtype="float64")

    finite = np.isfinite(src_vals) & np.isfinite(lon) & np.isfinite(lat)
    # MOM restart/history fields often carry exact zeros over land rather than NaNs.
    # Prefer nonzero finite samples so the diagnostic line follows nearby wet NEP data.
    valid = finite & (np.abs(src_vals) > 1.0e-14)
    if not valid.any():
        valid = finite
    if not valid.any():
        return np.full_like(tgt_lon, np.nan, dtype="float64"), np.full_like(tgt_lon, np.nan, dtype="float64")

    src_vals_flat = src_vals[valid]
    lon_flat = lon[valid]
    lat_flat = lat[valid]
    profile = np.full(tgt_lon.shape, np.nan, dtype="float64")
    dist = np.full(tgt_lon.shape, np.nan, dtype="float64")

    # Avoid constructing a full target x source distance matrix for large grids.
    for start in range(0, tgt_lon.size, 32):
        stop = min(start + 32, tgt_lon.size)
        lon_chunk = tgt_lon[start:stop, None]
        lat_chunk = tgt_lat[start:stop, None]
        scale = np.cos(np.deg2rad(lat_chunk))
        d2 = ((lon_flat[None, :] - lon_chunk) * scale) ** 2 + (lat_flat[None, :] - lat_chunk) ** 2
        idx = np.nanargmin(d2, axis=1)
        profile[start:stop] = src_vals_flat[idx]
        dist[start:stop] = np.sqrt(d2[np.arange(stop - start), idx])

    return profile, dist


def _boundary_profile(da: xr.DataArray, seg: Segment, time_index=0, z_index=0):
    """Return a 1D GOA OBC profile from a regridded segment DataArray."""
    out = _select_time_z_slice(da, time_index=time_index, z_index=z_index)
    for dim in list(out.dims):
        if out.sizes[dim] == 1:
            out = out.isel({dim: 0}, drop=True)
    seg_dim = f"nx_{seg.segstr}" if seg.border in ["south", "north"] else f"ny_{seg.segstr}"
    if seg_dim in out.dims:
        return np.asarray(out.values, dtype="float64")
    if out.ndim == 1:
        return np.asarray(out.values, dtype="float64")
    raise ValueError(f"Could not find 1D segment dimension {seg_dim} in {da.name}; dims={out.dims}")


def _plot_boundary_diagnostic(
        source: xr.DataArray,
        regridded: xr.DataArray,
        seg: Segment,
        var_name: str,
        year,
        month,
        ensemble,
        output_dir,
        time_index=0,
        z_index=0):
    """Write a NEP-nearest-source vs GOA-OBC profile diagnostic figure."""
    if importlib.util.find_spec("matplotlib") is None:
        print("[PHY-OBC][DIAG] matplotlib is not available; skipping boundary diagnostic plots", flush=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag_dir = path.join(output_dir, "diagnostics", "boundary_profiles")
    os.makedirs(diag_dir, exist_ok=True)

    target_lon = np.asarray(seg.coords["lon"].values, dtype="float64")
    target_lat = np.asarray(seg.coords["lat"].values, dtype="float64")
    nep_profile, dist = _nearest_source_profile(
        source, target_lon, target_lat, time_index=time_index, z_index=z_index
    )
    goa_profile = _boundary_profile(
        regridded, seg, time_index=time_index, z_index=z_index
    )
    n = min(nep_profile.size, goa_profile.size)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, nep_profile[:n], label="NEP nearest nonzero source", linewidth=1.4)
    axes[0].plot(x, goa_profile[:n], label="GOA regridded OBC", linewidth=1.1, alpha=0.85)
    axes[0].set_ylabel(var_name)
    axes[0].set_title(
        f"{var_name} {seg.segstr} ({seg.border}) {year}-{month} e{ensemble} "
        f"time_idx={time_index} z_idx={z_index}"
    )
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, goa_profile[:n] - nep_profile[:n], label="GOA - NEP nearest", color="tab:red")
    axes[1].set_xlabel("Boundary point index as written to OBC file")
    axes[1].set_ylabel("difference")
    axes[1].grid(True, alpha=0.3)
    ax2 = axes[1].twinx()
    ax2.plot(x, dist[:n], label="nearest distance (deg)", color="0.5", alpha=0.55)
    ax2.set_ylabel("nearest distance (deg)")

    # Include endpoint coordinates in the figure as a quick orientation check.
    endpoint_text = (
        f"first lon/lat=({target_lon[0]:.4f}, {target_lat[0]:.4f})\n"
        f"last lon/lat=({target_lon[-1]:.4f}, {target_lat[-1]:.4f})"
    )
    axes[0].text(
        0.01, 0.02, endpoint_text, transform=axes[0].transAxes,
        ha="left", va="bottom", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
    )

    outfile = path.join(
        diag_dir,
        f"{var_name}_{seg.num:03d}_{year}{str(month).zfill(2)}_e{str(ensemble).zfill(2)}_profile.png",
    )
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[PHY-OBC][DIAG] wrote {outfile}", flush=True)

def _all_finite_values_zero(da, atol=1e-14):
    vals = np.asarray(da.values)
    finite = np.isfinite(vals)
    if not finite.any():
        return False
    return np.all(np.abs(vals[finite]) <= atol)


def _remove_if_exists(filepath):
    if path.exists(filepath):
        os.remove(filepath)
        print(f"[PHY-OBC] Removed stale weight file: {filepath}")


def _interp_time_to_target(src_values, src_time, target_time, dims):
    da_src = xr.DataArray(src_values, coords={"time": src_time}, dims=dims)
    da_out = da_src.interp(time=target_time, kwargs={"fill_value": "extrapolate"})
    return np.asarray(da_out.values)


def _hold_monthly_values_on_daily_grid(src_values, src_time, target_time):
    """
    Map monthly source snapshots to daily target times using step-wise holds.

    Each daily timestamp gets the most recent monthly source value at or before
    that time (i.e., piecewise-constant in time).
    """
    src_dt = np.asarray(pd.DatetimeIndex(src_time).values, dtype="datetime64[ns]")
    tgt_dt = np.asarray(pd.DatetimeIndex(target_time).values, dtype="datetime64[ns]")
    src_idx = np.searchsorted(src_dt, tgt_dt, side="right") - 1
    src_idx = np.clip(src_idx, 0, len(src_dt) - 1)
    return src_values[src_idx]


def _pad_last_month_for_interp(da: xr.DataArray) -> xr.DataArray:
    """Append one extra monthly sample (copy of last value) one month later."""
    next_month = pd.Timestamp(da["time"].values[-1]) + pd.DateOffset(months=1)
    last = da.isel(time=-1).expand_dims(time=[next_month])
    return xr.concat([da, last], dim="time")


def _find_ssh_var_name(ds: xr.Dataset) -> str:
    """Return SSH variable name from daily source files."""
    for name in ("zos", "ssh", "ave_ssh", "sea_surface_height"):
        if name in ds.data_vars:
            return name
    raise KeyError(
        "No SSH variable found in daily source dataset. "
        "Expected one of: zos, ssh, ave_ssh, sea_surface_height."
    )


def _safe_rename_vars(ds: xr.Dataset, rename_map: dict[str, str]) -> xr.Dataset:
    """Rename only variables that exist in dataset."""
    existing = {k: v for k, v in rename_map.items() if k in ds.variables and k != v}
    if not existing:
        return ds
    return ds.rename_vars(existing)



def _is_index_like(values: np.ndarray, atol: float = 1.0e-6) -> bool:
    """Return True when values are just 0, 1, 2, ... or 1, 2, 3, ... indices."""
    vals = np.asarray(values, dtype="float64").ravel()
    if vals.size == 0 or not np.all(np.isfinite(vals)):
        return False
    zero_based = np.arange(vals.size, dtype="float64")
    one_based = np.arange(1, vals.size + 1, dtype="float64")
    return (
        np.allclose(vals, zero_based, atol=atol, rtol=0.0)
        or np.allclose(vals, one_based, atol=atol, rtol=0.0)
    )


def _valid_vertical_centers(values, nz: int) -> np.ndarray | None:
    """Return usable positive-down physical vertical centers, rejecting indices."""
    vals = np.asarray(values, dtype="float64").ravel()
    if vals.size != nz or not np.all(np.isfinite(vals)):
        return None
    if np.nanmedian(vals) < 0.0:
        vals = np.abs(vals)
    if _is_index_like(vals):
        return None
    if not np.all(np.diff(vals) > 0.0):
        return None
    return vals


def _vertical_center_values(ds: xr.Dataset, var_name: str, nz: int) -> np.ndarray:
    """Extract physical vertical center depths for a source variable.

    The OBC writer uses z_to_dz(), which computes layer thickness from the
    destination 'z' coordinate.  Therefore 'z' must contain real depth centers,
    not level indices.  Prefer coordinates attached to the source variable, then
    common dataset-level vertical coordinates.  Raise if only index-like values
    are available, because silently using 0..N creates 1 m layers and one huge
    bottom layer.
    """
    var = ds[var_name]
    candidates = []

    # Coordinates attached to the variable/dimensions are the most reliable.
    for dim in var.dims:
        if var.sizes.get(dim) == nz:
            if dim in var.coords:
                candidates.append((f"{var_name}.{dim}", var.coords[dim].values))
            if dim in ds.coords:
                candidates.append((dim, ds.coords[dim].values))
            if dim in ds.variables:
                candidates.append((dim, ds[dim].values))

    # Common MOM/FMS vertical coordinate names seen in restart/history files.
    for name in ("z", "zl", "z_l", "Layer", "layer", "lev", "depth", "st_ocean"):
        if name in ds.variables:
            candidates.append((name, ds[name].values))

    for name, values in candidates:
        centers = _valid_vertical_centers(values, nz)
        if centers is not None:
            _progress("GRID", f"Using vertical centers from {name}: first={centers[0]:g} last={centers[-1]:g} nz={nz}")
            return centers

    checked = ", ".join(name for name, _ in candidates) or "none"
    raise ValueError(
        f"Could not find physical vertical center depths for {var_name} (nz={nz}). "
        f"Checked: {checked}. Refusing to use 0..{nz - 1} index values because that "
        "would create invalid dz_* OBC variables."
    )

def _progress(tag, message):
    print(f"[PHY-OBC][{tag}] {message}", flush=True)


def _time_index_for_date(ds: xr.Dataset, target_date: pd.Timestamp, time_name: str = "time") -> int:
    """Return the index for an exact calendar date in a dataset time coordinate."""
    if time_name not in ds:
        raise KeyError(f"Dataset has no '{time_name}' coordinate.")
    time_vals = pd.to_datetime(ds[time_name].values).normalize()
    matches = np.where(time_vals == target_date.normalize())[0]
    if matches.size == 0:
        raise ValueError(
            f"Requested date {target_date.strftime('%Y-%m-%d')} was not found in "
            f"dataset '{time_name}' coordinate (range: {time_vals.min()} .. {time_vals.max()})."
        )
    return int(matches[0])


# ----------------------------
# Core routine
# ----------------------------
def write_year(year, glorys_dir, nep_static, segments, variables, month, ensemble, fct_dir, rst_dir,
               is_first_year=False, is_last_year=False, weight_save=True, interp_tracer_daily=True,
               diagnostic_plots=False, diagnostic_time_index=0, diagnostic_z_index=0):
    _progress(
        "START",
        f"[PHY-OBC] ==== write_year year={year} month={month} ensemble={ensemble} "
        f"vars={variables} segments={[s.border for s in segments]} weight_save={weight_save} ===="
    )

    # Build daily time source from hindcast + forecast daily SSH.
    _progress("TIME", "Opening hindcast/forecast daily SSH files")
    hind_daily_file = path.join(glorys_dir, f"{int(year):04d}{int(month):02d}01/ocean_daily.nc")
    ds_sfc_hind_daily = xr.open_dataset(hind_daily_file)
    print(f"[PHY-OBC] Loaded hindcast daily file: {hind_daily_file}")

    fcst_hist = path.join(fct_dir, f"{year}-{month}-e{ensemble}/history")
    print(f"[PHY-OBC] Forecast history dir: {fcst_hist}")
    fcst_daily_file = path.join(fcst_hist, "ocean_daily.nc")
    ds_sfc_fcst_daily = xr.open_dataset(fcst_daily_file)
    print(f"[PHY-OBC] Loaded forecast daily file: {fcst_daily_file}")
    hind_ssh_var = _find_ssh_var_name(ds_sfc_hind_daily)
    fcst_ssh_var = _find_ssh_var_name(ds_sfc_fcst_daily)
    _progress("TIME", f"Using daily SSH vars hindcast={hind_ssh_var} forecast={fcst_ssh_var}")

    ref = pd.Timestamp(year=int(year), month=int(month), day=1)
    hind_ref_idx = _time_index_for_date(ds_sfc_hind_daily, ref)
    fcst_ref_idx = _time_index_for_date(ds_sfc_fcst_daily, ref)
    _progress(
        "TIME",
        f"Using reference date {ref.strftime('%Y-%m-%d')} "
        f"(hind_idx={hind_ref_idx}, fcst_idx={fcst_ref_idx})",
    )

    target_time = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.to_datetime(ds_sfc_hind_daily["time"].isel(time=slice(hind_ref_idx, hind_ref_idx + 1)).values),
                pd.to_datetime(ds_sfc_fcst_daily["time"].isel(time=slice(fcst_ref_idx + 1, None)).values),
            ]
        )
    ).normalize()
    _progress("TIME", f"Daily target steps (no pad) = {len(target_time)}")

    nt = len(target_time) + 1  # +1 for padded extra step
    nz = 75
    ny = 816
    nx = 342
    nxq = 343
    nyq = 817
    nzi = 76
    nnv = 2

    _progress("TIME", f"Building CF time axes with padded nt={nt}")
    # -------------------------
    # BUILD CF TIME + BOUNDS
    # -------------------------
    pad_time = target_time[-1] + pd.Timedelta(days=1)
    all_times = target_time.append(pd.DatetimeIndex([pad_time]))
    time_days = ((all_times - ref) / np.timedelta64(1, "D")).to_numpy(dtype="float64")
    time_bnds = np.zeros((nt, nnv), dtype="float64")
    time_bnds[:, 0] = time_days
    time_bnds[:-1, 1] = time_days[1:]
    time_bnds[-1, 1] = time_days[-1] + 1.0

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

    _progress("SRC", "Loading restart and monthly forecast source fields")
    # ==========================================
    # Step 2: Build monthly 3D source + daily SSH source
    # ==========================================
    restart_file = path.join(rst_dir, f"restdate_{year}{month}01/MOM_{year}{month}01.res.nc")
    ds_z_hind = xr.open_dataset(restart_file, decode_cf=False)
    print(f"[PHY-OBC] Loaded restart: {restart_file}")

    if "ave_ssh" not in ds_z_hind.data_vars:
        available = ", ".join(sorted(ds_z_hind.data_vars))
        raise KeyError(
            f"Restart file {restart_file} is missing required variable 'ave_ssh' "
            f"for v3_july26 OBC zos time=0. Available variables: {available}"
        )
    restart_ave_ssh = np.asarray(ds_z_hind["ave_ssh"][0], dtype=np.float32)

    ds_z_hind = _safe_rename_vars(ds_z_hind, {'Salt': 'so', 'Temp': 'thetao', 'u': 'uo', 'v': 'vo'})
    z_centers = _vertical_center_values(ds_z_hind, "thetao", nz)

    src_time = [pd.Timestamp(int(year), int(month), 1)]
    so_src = np.zeros((12, nz, ny, nx), dtype=np.float32)
    thetao_src = np.zeros((12, nz, ny, nx), dtype=np.float32)
    uo_src = np.zeros((12, nz, ny, nxq), dtype=np.float32)
    vo_src = np.zeros((12, nz, nyq, nx), dtype=np.float32)

    so_src[0] = np.asarray(ds_z_hind["so"][0], dtype=np.float32)
    thetao_src[0] = np.asarray(ds_z_hind["thetao"][0], dtype=np.float32)
    uo_src[0] = np.asarray(ds_z_hind["uo"][0], dtype=np.float32)
    vo_src[0] = np.asarray(ds_z_hind["vo"][0], dtype=np.float32)

    start_ts = pd.Timestamp(int(year), int(month), 1)
    for idx, offset in enumerate(range(1, 12), start=1):
        tgt = start_ts + pd.DateOffset(months=offset)
        file = f"oceanm_{tgt.year}_{tgt.month:02d}.nc"
        _progress("SRC", f"Loading monthly file {idx}/11: {file}")
        tmp_z = xr.open_dataset(path.join(fcst_hist, file))
        tmp_z = _safe_rename_vars(tmp_z, {'salt': 'so', 'potT': 'thetao', 'temp': 'thetao', 'u': 'uo', 'v': 'vo'})
        src_time.append(pd.Timestamp(tgt.year, tgt.month, 1))
        so_src[idx] = np.asarray(tmp_z["so"][0], dtype=np.float32)
        thetao_src[idx] = np.asarray(tmp_z["thetao"][0], dtype=np.float32)
        uo_src[idx] = np.asarray(tmp_z["uo"][0], dtype=np.float32)
        vo_src[idx] = np.asarray(tmp_z["vo"][0], dtype=np.float32)
        tmp_z.close()

    src_time_index = pd.DatetimeIndex(src_time)
    tracer_time_index: pd.DatetimeIndex
    tracer_sources: dict[str, np.ndarray] = {}
    if interp_tracer_daily:
        _progress("INTERP", "Preparing thetao/so on daily OBC time grid")
        tracer_time_index = all_times
        tracer_sources["so"] = _hold_monthly_values_on_daily_grid(so_src, src_time_index, all_times)
        tracer_sources["thetao"] = _hold_monthly_values_on_daily_grid(thetao_src, src_time_index, all_times)
    else:
        _progress("INTERP", "Preparing thetao/so on monthly OBC time grid (with padded last month)")
        tracer_time_index = src_time_index.append(pd.DatetimeIndex([src_time_index[-1] + pd.DateOffset(months=1)]))
        tracer_sources["so"] = np.concatenate([so_src, so_src[-1:]], axis=0)
        tracer_sources["thetao"] = np.concatenate([thetao_src, thetao_src[-1:]], axis=0)

    ds["zos"][0, :, :] = restart_ave_ssh
    ds["zos"][1:nt - 1] = np.asarray(
        ds_sfc_fcst_daily[fcst_ssh_var].isel(time=slice(fcst_ref_idx + 1, None)).values,
        dtype=np.float32,
    )
    _progress(
        "SRC",
        "Loaded restart ave_ssh for zos time=0 + forecast daily SSH/zos continuation",
    )

    _progress("MASK", "Applying t=0 NaN mask from reference daily index")
    # Apply NaN mask from a reference forecast month onto t=0
    mask_idx = min(8, nt - 2)
    ds["zos"][0] = ds["zos"][0].where(~ds["zos"].isel(time=mask_idx).isnull())

    _progress("PAD", "Padding final extra time step")
    # ==========================================
    # Step 3b: PAD EXTRA LAST TIME STEP
    _progress("GRID", f"Opening static grid file: {nep_static}")
    # ==========================================
    # Duplicate the final available daily state into the extra slot
    ds["zos"][nt - 1, :, :] = ds["zos"][nt - 2, :, :]

    # ==========================================
    # Step 4: Load NEP static grid (2D lon/lat)
    # ==========================================
    st = xr.open_dataset(nep_static, decode_times=False)

    lonT = lon_to_360(st["geolon"]).values
    latT = st["geolat"].values

    lonU = lon_to_360(st["geolon_u"]).values
    latU = st["geolat_u"].values

    lonV = lon_to_360(st["geolon_v"]).values
    latV = st["geolat_v"].values

    _lon_range("NEP geolon source after 0-360 conversion", lonT)
    _lon_range("NEP geolon_u source after 0-360 conversion", lonU)
    _lon_range("NEP geolon_v source after 0-360 conversion", lonV)

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
        _progress("REGRID", "Regridding variable group: zos")
        for seg in segments:
            _progress("REGRID", f"segment={seg.border} var=zos")
            tracer = ds["zos"]
            print(tracer.shape)
            tracer = _attach_2d_lonlat(tracer, lonT, latT, name="zos")
            out = seg.regrid_tracer(
                tracer, suffix=year, flood=False, weight_save=weight_save,
                time_attrs=time_attrs, time_encoding=time_encoding
            )
            zkey = f"zos_{seg.segstr}"
            if zkey in out and _all_finite_values_zero(out[zkey]):
                print(f"[PHY-OBC] WARNING: {zkey} is all zeros. Regenerating tracer weights and retrying once.")
                _remove_if_exists(path.join(seg.regrid_dir, f"regrid_{seg.segstr}_t.nc"))
                seg.regrid_tracer(
                    tracer, suffix=year, flood=False, weight_save=weight_save,
                    time_attrs=time_attrs, time_encoding=time_encoding
                )
            if diagnostic_plots and zkey in out:
                _plot_boundary_diagnostic(
                    tracer, out[zkey], seg, "zos", year, month, ensemble, seg.output_dir,
                    time_index=diagnostic_time_index, z_index=diagnostic_z_index,
                )

    if "uv" in variables:
        _progress("REGRID", "Regridding variable group: uv")
        uv_monthly_time = src_time_index
        uo_monthly = xr.DataArray(
            uo_src,
            dims=("time", "z", "yh", "xq"),
            coords={"time": uv_monthly_time, "z": z_centers, "yh": np.arange(ny), "xq": np.arange(nxq)},
        )
        vo_monthly = xr.DataArray(
            vo_src,
            dims=("time", "z", "yq", "xh"),
            coords={"time": uv_monthly_time, "z": z_centers, "yq": np.arange(nyq), "xh": np.arange(nx)},
        )
        for seg in segments:
            _progress("REGRID", f"segment={seg.border} var=uv")
            _progress("REGRID", f"segment={seg.border} var=uv (monthly source -> OBC monthly)")
            uo_m = _attach_2d_lonlat(uo_monthly, lonU, latU, name="uo")
            vo_m = _attach_2d_lonlat(vo_monthly, lonV, latV, name="vo")
            out_uv_monthly = seg.regrid_velocity(
                uo_m, vo_m, write=False, flood=False, rotate=False, weight_save=weight_save,
                time_attrs=time_attrs, time_encoding=time_encoding
            )
            out_uv_monthly = out_uv_monthly.assign_coords(time=uv_monthly_time)

            _progress("INTERP", f"segment={seg.border} var=uv (OBC monthly -> daily)")
            out_uv_monthly = _pad_last_month_for_interp(out_uv_monthly)
            out_uv = out_uv_monthly.interp(time=all_times, kwargs={"fill_value": "extrapolate"})
            out_uv = out_uv.assign_coords(time=("time", time_days))
            out_uv["time"].attrs = time_attrs
            out_uv["time"].encoding = time_encoding
            seg.to_netcdf(out_uv, "uv", suffix=year)
            if diagnostic_plots:
                ukey_diag = f"u_{seg.segstr}"
                vkey_diag = f"v_{seg.segstr}"
                if ukey_diag in out_uv:
                    _plot_boundary_diagnostic(
                        uo_m, out_uv[ukey_diag], seg, "u", year, month, ensemble, seg.output_dir,
                        time_index=diagnostic_time_index, z_index=diagnostic_z_index,
                    )
                if vkey_diag in out_uv:
                    _plot_boundary_diagnostic(
                        vo_m, out_uv[vkey_diag], seg, "v", year, month, ensemble, seg.output_dir,
                        time_index=diagnostic_time_index, z_index=diagnostic_z_index,
                    )

            ukey = f"u_{seg.segstr}"
            vkey = f"v_{seg.segstr}"
            bad_u = ukey in out_uv and _all_finite_values_zero(out_uv[ukey])
            bad_v = vkey in out_uv and _all_finite_values_zero(out_uv[vkey])
            if bad_u or bad_v:
                print(
                    f"[PHY-OBC] WARNING: {seg.border} uv output has all-zero component(s): "
                    f"{'u' if bad_u else ''}{' and ' if (bad_u and bad_v) else ''}{'v' if bad_v else ''}. "
                    "Regenerating UV weights and retrying once."
                )
                _remove_if_exists(path.join(seg.regrid_dir, f"regrid_{seg.segstr}_u.nc"))
                _remove_if_exists(path.join(seg.regrid_dir, f"regrid_{seg.segstr}_v.nc"))
                out_uv_monthly = seg.regrid_velocity(
                    uo_m, vo_m, write=False, flood=False, rotate=False, weight_save=weight_save,
                    time_attrs=time_attrs, time_encoding=time_encoding
                )
                out_uv_monthly = out_uv_monthly.assign_coords(time=uv_monthly_time)
                out_uv_monthly = _pad_last_month_for_interp(out_uv_monthly)
                out_uv = out_uv_monthly.interp(time=all_times, kwargs={"fill_value": "extrapolate"})
                out_uv = out_uv.assign_coords(time=("time", time_days))
                out_uv["time"].attrs = time_attrs
                out_uv["time"].encoding = time_encoding
                seg.to_netcdf(out_uv, "uv", suffix=year)
                if diagnostic_plots:
                    if ukey in out_uv:
                        _plot_boundary_diagnostic(
                            uo_m, out_uv[ukey], seg, "u", year, month, ensemble, seg.output_dir,
                            time_index=diagnostic_time_index, z_index=diagnostic_z_index,
                        )
                    if vkey in out_uv:
                        _plot_boundary_diagnostic(
                            vo_m, out_uv[vkey], seg, "v", year, month, ensemble, seg.output_dir,
                            time_index=diagnostic_time_index, z_index=diagnostic_z_index,
                        )

    for var in variables:
        if var in ["zos", "uv"]:
            continue

        if var in tracer_sources:
            tracer_time = tracer_time_index
            tracer_time_days = ((tracer_time - ref) / np.timedelta64(1, "D")).to_numpy(dtype="float64")
            tracer_vals = tracer_sources[var]
            tracer = xr.DataArray(
                tracer_vals,
                dims=("time", "z", "yh", "xh"),
                coords={"time": tracer_time_days, "z": z_centers, "yh": np.arange(ny), "xh": np.arange(nx)},
                name=var,
            )
            _progress(
                "REGRID",
                f"Regridding tracer variable: {var} "
                f"({'daily-held' if interp_tracer_daily else 'monthly'})",
            )
            tracer = _attach_2d_lonlat(tracer, lonT, latT, name=var)
            for seg in segments:
                _progress("REGRID", f"segment={seg.border} var={var}")
                out = seg.regrid_tracer(
                    tracer, suffix=year, flood=False, weight_save=weight_save,
                    time_attrs=time_attrs, time_encoding=time_encoding
                )
                tkey = f"{var}_{seg.segstr}"
                if tkey in out and _all_finite_values_zero(out[tkey]):
                    print(f"[PHY-OBC] WARNING: {tkey} is all zeros. Regenerating tracer weights and retrying once.")
                    _remove_if_exists(path.join(seg.regrid_dir, f"regrid_{seg.segstr}_t.nc"))
                    seg.regrid_tracer(
                        tracer, suffix=year, flood=False, weight_save=weight_save,
                        time_attrs=time_attrs, time_encoding=time_encoding
                    )
                if diagnostic_plots and tkey in out:
                    _plot_boundary_diagnostic(
                        tracer, out[tkey], seg, var, year, month, ensemble, seg.output_dir,
                        time_index=diagnostic_time_index, z_index=diagnostic_z_index,
                    )
        elif var in ds:
            _progress("REGRID", f"Regridding tracer variable: {var}")
            for seg in segments:
                print(ds[var].shape)
                print(var)
                print("~~~~~~~~~~~~~~~~~")
                print("~~~~~~~~~~~~~~~~~")
                _progress("REGRID", f"segment={seg.border} var={var}")
                tracer = ds[var]
                tracer = _attach_2d_lonlat(tracer, lonT, latT, name=var)
                print(tracer)
                out = seg.regrid_tracer(
                    tracer, suffix=year, flood=False, weight_save=weight_save,
                    time_attrs=time_attrs, time_encoding=time_encoding
                )
                tkey = f"{var}_{seg.segstr}"
                if tkey in out and _all_finite_values_zero(out[tkey]):
                    print(f"[PHY-OBC] WARNING: {tkey} is all zeros. Regenerating tracer weights and retrying once.")
                    _remove_if_exists(path.join(seg.regrid_dir, f"regrid_{seg.segstr}_t.nc"))
                    seg.regrid_tracer(
                        tracer, suffix=year, flood=False, weight_save=weight_save,
                        time_attrs=time_attrs, time_encoding=time_encoding
                    )
        elif var in ds:
            for seg in segments:
                print(f"{seg.border} {var} (from ocean_month.nc)")
                tracer = ds_sfc[var]
                tracer = _attach_2d_lonlat(tracer, lonT, latT, name=var)
                out = seg.regrid_tracer(
                    tracer, suffix=year, flood=False, weight_save=weight_save,
                    time_attrs=time_attrs, time_encoding=time_encoding
                )
                tkey = f"{var}_{seg.segstr}"
                if tkey in out and _all_finite_values_zero(out[tkey]):
                    print(f"[PHY-OBC] WARNING: {tkey} is all zeros. Regenerating tracer weights and retrying once.")
                    _remove_if_exists(path.join(seg.regrid_dir, f"regrid_{seg.segstr}_t.nc"))
                    seg.regrid_tracer(
                        tracer, suffix=year, flood=False, weight_save=weight_save,
                        time_attrs=time_attrs, time_encoding=time_encoding
                    )
        else:
            raise ValueError(f"{var} not found in datasets for year={year}")

    ds_sfc_hind_daily.close()
    ds_sfc_fcst_daily.close()
    ds_z_hind.close()
    ds.close()
    st.close()
    _progress("DONE", f"Completed year={year} month={month} ensemble={ensemble}")


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
    weight_save = bool(cfg.get("weight_save", True))
    interp_tracer_daily = bool(cfg.get("interp_tracer_daily", False))
    diagnostic_plots = bool(cfg.get("diagnostic_plots", False))
    diagnostic_time_index = int(cfg.get("diagnostic_time_index", 0))
    diagnostic_z_index = int(cfg.get("diagnostic_z_index", 0))
    regrid_dir = cfg.get("regrid_dir", output_dir)

    nep_static = _require(cfg, "NEP_STATIC")
    _ = cfg.get("GOA_STATIC", None)

    if not path.exists(output_dir):
        os.makedirs(output_dir)
    if not path.exists(regrid_dir):
        os.makedirs(regrid_dir)
    print(f"[PHY-OBC] Output dir: {output_dir}")
    print(f"[PHY-OBC] Regrid dir: {regrid_dir}")

    hgrid = xr.open_dataset(hgrid_file)
    hgrid = hgrid.copy(deep=True)
    hgrid["x"] = lon_to_360(hgrid["x"])
    _lon_range("GoA hgrid x target after 0-360 conversion", hgrid["x"].values)
    print(f"[PHY-OBC] Loaded hgrid: {hgrid_file}")

    variables = cfg.get("variables", [])

    segments = []
    for seg_cfg in cfg.get("segments", []):
        segment = Segment(seg_cfg["id"], seg_cfg["border"], hgrid, output_dir=output_dir, regrid_dir=regrid_dir)
        _lon_range(f"segment {int(seg_cfg['id']):03d} {seg_cfg['border']} target", segment.coords["lon"].values)
        segments.append(segment)
    print(f"[PHY-OBC] Segments configured: {[f'{s.num}:{s.border}' for s in segments]}")

    for y in range(first_year, last_year + 1):
        print(f"[PHY-OBC] Starting processing year={y}")
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
            weight_save=weight_save,
            interp_tracer_daily=interp_tracer_daily,
            diagnostic_plots=diagnostic_plots,
            diagnostic_time_index=diagnostic_time_index,
            diagnostic_z_index=diagnostic_z_index,
        )

    if ncrcat_years_flag:
        assert len(ncrcat_names) == len(variables), (
            "Could not concatenate annual files because the number of file output names "
            "did not match the number of variables provided."
        )
        ncrcat_years(len(segments), output_dir, variables, ncrcat_names)
        print("[PHY-OBC] Completed ncrcat yearly concatenation")


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
