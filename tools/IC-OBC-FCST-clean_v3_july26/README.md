# IC-OBC-FCST-clean

Standalone workflow copied from `tools/forecast_multi_proc` and merged with selected PHY-OBC behavior from `tools/OBC_PHY_DAILY`.

What this directory runs:
- IC stage
- OBC BGC stage
- OBC PHY stage with:
  - SSH (`zos`) with restart `ave_ssh` at t=0 and forecast `ocean_daily.nc` for later records
  - `uv` monthly-to-daily interpolation for OBC output
  - PHY OBC daily `zos`/`uv` behavior copied exactly from `tools/OBC_PHY_DAILY`

Original directories were not modified:
- `tools/forecast_multi_proc/`
- `tools/OBC_PHY_DAILY/`

## v3_july26 zos day-0 change

This `tools/IC-OBC-FCST-clean_v3_july26` copy preserves the v2 workflow except for PHY OBC `zos` construction.  The first `zos` OBC time record (`time=0`) is now sourced from the NEP restart file variable `ave_ssh` (`MOM_YYYYMM01.res.nc`) and then written with the same OBC segment variable names as v2, such as `zos_segment_001` and `zos_segment_002`.  Later `zos` time records continue to use the v2 daily-history workflow from forecast `ocean_daily.nc` files.
