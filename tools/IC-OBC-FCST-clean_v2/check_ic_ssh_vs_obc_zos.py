from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================
# Files
# ============================================================

IC_FILE = Path(
    "/work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/IC-OBC-FCST-clean_v2/output/2012/01/IC/ic_phy_20120101.nc"
)

OBC_FILES = {
    "001": Path(
        "/work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/IC-OBC-FCST-clean_v2/output/2012/01/OBC/PHY/e01/zos_001_2012.nc"
    ),
    "002": Path(
        "/work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/IC-OBC-FCST-clean_v2/output/2012/01/OBC/PHY/e01/zos_002_2012.nc"
    ),
}

OUTDIR = Path("./ssh_ic_vs_obc_check")
OUTDIR.mkdir(exist_ok=True)

# ============================================================
# Helpers
# ============================================================

def find_var(ds, candidates):
    for v in candidates:
        if v in ds:
            return v
    raise KeyError(
        f"Could not find any of {candidates}. Available variables:\n{list(ds.data_vars)}"
    )


def find_obc_zos_var(ds, seg):
    candidates = [
        f"zos_segment_{seg}",
        f"ssh_segment_{seg}",
        f"ave_ssh_segment_{seg}",
        f"sea_surface_height_segment_{seg}",
        "zos",
        "ssh",
        "ave_ssh",
        "sea_surface_height",
        "eta_t",
    ]

    for v in candidates:
        if v in ds:
            return v

    raise KeyError(
        f"Could not find OBC zos/ssh variable for segment {seg}. "
        f"Available variables:\n{list(ds.data_vars)}"
    )


def first_time_or_squeeze(da):
    indexers = {}
    for d in da.dims:
        if d.lower() in ["time", "time_counter", "ocean_time", "t"]:
            indexers[d] = 0

    if indexers:
        da = da.isel(indexers)

    return da.squeeze(drop=True)


def clean_array(a):
    a = np.asarray(a, dtype=float)
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a) > 1e10] = np.nan
    return a


def to_2d_ic_field(da):
    da = first_time_or_squeeze(da)
    arr = clean_array(da.values)

    if arr.ndim != 2:
        raise ValueError(f"Expected IC ssh to be 2D after squeezing, got shape {arr.shape}")

    return arr


def to_1d_obc_field(da):
    da = first_time_or_squeeze(da)
    arr = clean_array(da.values)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        if 1 in arr.shape:
            return arr.reshape(-1)
        raise ValueError(f"OBC field is 2D but neither dimension is singleton: {arr.shape}")

    raise ValueError(f"Expected OBC zos to be 1D after squeezing, got shape {arr.shape}")


def get_ic_edges(ssh2d):
    return {
        "south_j0": ssh2d[0, :],
        "north_jend": ssh2d[-1, :],
        "west_i0": ssh2d[:, 0],
        "east_iend": ssh2d[:, -1],
    }


def build_obc_samples(obc, target_len):
    """
    Build possible samples from an OBC supergrid-like vector.

    For your files:
        segment 001: len = 961  = 2*480 + 1
        segment 002: len = 1057 = 2*528 + 1

    The tracer-center-like points are usually obc[1::2].
    """
    samples = {}

    n = len(obc)

    if n == target_len:
        samples["full"] = obc

    if n == 2 * target_len + 1:
        samples["odd_1::2"] = obc[1::2]          # length target_len
        samples["even_0:-1:2"] = obc[0:-1:2]     # length target_len
        samples["even_2::2"] = obc[2::2]         # length target_len
        samples["mean_even_bounds"] = 0.5 * (obc[0:-1:2] + obc[2::2])

    if n == 2 * target_len:
        samples["odd_1::2"] = obc[1::2]
        samples["even_0::2"] = obc[0::2]
        samples["pair_mean"] = 0.5 * (obc[0::2] + obc[1::2])

    # Keep only samples with the right length
    samples = {k: v for k, v in samples.items() if len(v) == target_len}

    return samples


def compare_1d(obc_sample, ic_edge):
    results = []

    for orient, edge in {
        "normal": ic_edge,
        "reversed": ic_edge[::-1],
    }.items():

        diff = obc_sample - edge
        ok = np.isfinite(diff)

        if ok.sum() == 0:
            results.append({
                "orientation": orient,
                "n_common": 0,
                "mean": np.nan,
                "std": np.nan,
                "rmse": np.nan,
                "min": np.nan,
                "max": np.nan,
            })
            continue

        dd = diff[ok]

        results.append({
            "orientation": orient,
            "n_common": int(ok.sum()),
            "mean": float(np.nanmean(dd)),
            "std": float(np.nanstd(dd)),
            "rmse": float(np.sqrt(np.nanmean(dd**2))),
            "min": float(np.nanmin(dd)),
            "max": float(np.nanmax(dd)),
        })

    return results


def print_result(seg, edge_name, sample_name, r):
    print(
        f"{seg:>3s} | {edge_name:>10s} | {sample_name:>16s} | {r['orientation']:>8s} | "
        f"n={r['n_common']:5d} | "
        f"mean={r['mean']: .4f} m | "
        f"std={r['std']: .4f} m | "
        f"rmse={r['rmse']: .4f} m | "
        f"min/max={r['min']: .4f}/{r['max']: .4f} m"
    )


def plot_best(seg, obc_sample, ic_edge, edge_name, sample_name, orientation, outdir):
    if orientation == "reversed":
        ic_edge = ic_edge[::-1]

    diff = obc_sample - ic_edge
    x = np.arange(len(obc_sample))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, ic_edge, label=f"IC ssh, {edge_name}")
    ax.plot(x, obc_sample, label=f"OBC zos t0, {sample_name}, seg {seg}", alpha=0.8)
    ax.set_xlabel("Tracer-like boundary index")
    ax.set_ylabel("SSH / ZOS [m]")
    ax.set_title(f"Segment {seg}: IC ssh vs OBC zos at time=0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"seg{seg}_ic_ssh_vs_obc_zos.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(x, diff)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Tracer-like boundary index")
    ax.set_ylabel("OBC zos - IC ssh [m]")
    ax.set_title(f"Segment {seg}: difference, {edge_name}, {sample_name}, {orientation}")
    fig.tight_layout()
    fig.savefig(outdir / f"seg{seg}_obc_minus_ic_diff.png", dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

print("Opening IC:")
print(IC_FILE)

ic = xr.open_dataset(IC_FILE, decode_times=False)

ic_ssh_var = find_var(ic, ["ssh", "ave_ssh", "zos", "eta_t", "sea_surface_height"])
print(f"\nIC SSH variable used: {ic_ssh_var}")
print(ic[ic_ssh_var])

ssh_ic = to_2d_ic_field(ic[ic_ssh_var])

print(f"\nIC ssh shape: {ssh_ic.shape}")
print(
    "IC ssh range:",
    float(np.nanmin(ssh_ic)),
    float(np.nanmax(ssh_ic)),
    "mean:",
    float(np.nanmean(ssh_ic)),
)

ic_edges = get_ic_edges(ssh_ic)

print("\n============================================================")
print("Comparing IC ssh edges with OBC zos(time=0)")
print("============================================================")
print(
    "seg |    IC edge |       OBC sample |   orient |      n |       mean |        std |       rmse | min/max"
)

best_matches = {}

for seg, obc_file in OBC_FILES.items():
    print("\n------------------------------------------------------------")
    print(f"Segment {seg}")
    print(obc_file)

    obc_ds = xr.open_dataset(obc_file, decode_times=False)

    obc_zos_var = find_obc_zos_var(obc_ds, seg)
    print(f"OBC ZOS variable used: {obc_zos_var}")
    print(obc_ds[obc_zos_var])

    zos_obc_full = to_1d_obc_field(obc_ds[obc_zos_var])

    print(
        f"OBC zos full shape: {zos_obc_full.shape}; "
        f"range={np.nanmin(zos_obc_full):.4f}/{np.nanmax(zos_obc_full):.4f} m; "
        f"mean={np.nanmean(zos_obc_full):.4f} m"
    )

    all_results = []

    for edge_name, ic_edge in ic_edges.items():
        samples = build_obc_samples(zos_obc_full, target_len=len(ic_edge))

        if len(samples) == 0:
            continue

        for sample_name, obc_sample in samples.items():
            results = compare_1d(obc_sample, ic_edge)

            for r in results:
                print_result(seg, edge_name, sample_name, r)
                all_results.append((edge_name, sample_name, obc_sample, r))

    valid = [
        item for item in all_results
        if np.isfinite(item[3]["rmse"]) and item[3]["n_common"] > 0
    ]

    if not valid:
        print(f"WARNING: No valid comparison found for segment {seg}")
        continue

    best_edge, best_sample_name, best_obc_sample, best_r = min(
        valid, key=lambda item: item[3]["rmse"]
    )

    best_matches[seg] = (best_edge, best_sample_name, best_obc_sample, best_r)

    print("\nBest match:")
    print_result(seg, best_edge, best_sample_name, best_r)

    plot_best(
        seg=seg,
        obc_sample=best_obc_sample,
        ic_edge=ic_edges[best_edge],
        edge_name=best_edge,
        sample_name=best_sample_name,
        orientation=best_r["orientation"],
        outdir=OUTDIR,
    )

print("\n============================================================")
print("Summary")
print("============================================================")

for seg, item in best_matches.items():
    best_edge, best_sample_name, best_obc_sample, best_r = item
    print_result(seg, best_edge, best_sample_name, best_r)

print(f"\nPlots saved in: {OUTDIR.resolve()}")