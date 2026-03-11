#!/usr/bin/env python3
"""Generate per-run YAML configs from templates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from utils.helpers import ENSEMBLES, RESTART_MONTHS, YEARS, ensure_dir
from utils.paths import DEFAULT_CONFIG_ROOT, DEFAULT_TEMPLATE_ROOT


DEFAULTS = {
    "GOA_STATIC": "/archive/Remi.Pages/fre/Arc_12/2026_02.01/CGOA_BGC_2025_07_base_nep_phy_feb26/gfdl.ncrc6-intel23-prod/pp/ocean_daily/ocean_daily.static.nc",
    "NEP_STATIC": "/archive/Liz.Drenkard/fre/cefi/NEP/2025_07/NEP10k_202507_physics_bgc/gfdl.ncrc6-intel23-repro/pp/ocean_daily/ocean_daily.static.nc",
    "HINDCAST_RESTART_ROOT": "/archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/restart",
    "HINDCAST_HISTORY_DIR": "/archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/history",
    "FORECAST_PHY_ROOT": "/archive/Dmitry.Dukhovskoy/fre/NEP/forecast_bgc/NEPbgc_fcst_dailyOB01",
    "FORECAST_BGC_DIR": "/archive/Dmitry.Dukhovskoy/fre/NEP/forecast_bgc/NEPbgc_fcst_dailyOB01",
    "GOA_HGRID": "/work/Remi.Pages/GOA2p5k/GRID/CGOA_2.5k/ocean_hgrid.nc",
}


def _render(obj: Any, values: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _render(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_render(v, values) for v in obj]
    if isinstance(obj, str):
        return obj.format(**values)
    return obj


def render_template(template_path: Path, values: dict[str, Any]) -> dict[str, Any]:
    with template_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return _render(raw, values)


def generate_case_configs(
    year: str,
    month: str,
    ensemble: str,
    output_root: Path,
    config_root: Path = DEFAULT_CONFIG_ROOT,
    template_root: Path = DEFAULT_TEMPLATE_ROOT,
) -> dict[str, Path]:
    case_values = {
        **DEFAULTS,
        "YEAR": year,
        "MONTH": month,
        "ENSEMBLE": ensemble,
        "ENSEMBLE_LABEL": f"e{ensemble}",
        "CASE_ROOT": str(output_root / year / month),
        "IC_ROOT": str(output_root / year / month / "IC"),
        "OBC_PHY_ROOT": str(output_root / year / month / "OBC" / "PHY" / f"e{ensemble}"),
        "OBC_BGC_ROOT": str(output_root / year / month / "OBC" / "BGC" / f"e{ensemble}"),
        "NEP_RESTART_FILE": f"{DEFAULTS['HINDCAST_RESTART_ROOT']}/restdate_{year}{month}01/MOM_{year}{month}01.res.nc",
        "NEP_RESTART_DIR": f"{DEFAULTS['HINDCAST_RESTART_ROOT']}/restdate_{year}{month}01",
        "IC_DATE_ISO": f"{year}-{month}-01T00:00:00",
        "TIME_REF_ISO": f"{year}-{month}-01T00:00:00",
        "TIME_UNITS": f"days since {year}-{month}-01 00:00:00",
    }

    run_cfg_dir = ensure_dir(config_root / year / month / f"e{ensemble}")
    mapping = {
        "ic_phy": "ic_phy_template.yaml",
        "ic_bgc": "ic_bgc_template.yaml",
        "obc_phy": "obc_phy_template.yaml",
        "obc_bgc": "obc_bgc_template.yaml",
    }

    out = {}
    for key, tpl_name in mapping.items():
        rendered = render_template(template_root / tpl_name, case_values)
        out_path = run_cfg_dir / f"{key}.yaml"
        with out_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(rendered, stream, sort_keys=False)
        out[key] = out_path

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--years", nargs="*", default=YEARS)
    parser.add_argument("--months", nargs="*", default=RESTART_MONTHS)
    parser.add_argument("--ensembles", nargs="*", default=ENSEMBLES)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    config_root = Path(args.config_root)

    for year in args.years:
        for month in args.months:
            for ens in args.ensembles:
                generate_case_configs(year, month, ens, output_root=output_root, config_root=config_root)


if __name__ == "__main__":
    main()
