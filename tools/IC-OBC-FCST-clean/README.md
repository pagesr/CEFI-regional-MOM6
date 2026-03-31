# IC-OBC-FCST-clean

Standalone workflow copied from `tools/forecast_multi_proc` and merged with selected PHY-OBC behavior from `tools/OBC_PHY_DAILY`.

What this directory runs:
- IC stage
- OBC BGC stage
- OBC PHY stage with:
  - daily SSH (`zos`) from `ocean_daily.nc` (hindcast t=0 + forecast t=1..)
  - `uv` monthly-to-daily interpolation for OBC output
  - PHY OBC daily `zos`/`uv` behavior copied exactly from `tools/OBC_PHY_DAILY`

Original directories were not modified:
- `tools/forecast_multi_proc/`
- `tools/OBC_PHY_DAILY/`
