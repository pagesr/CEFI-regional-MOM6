# OBC_PHY_DAILY

PHY-only Slurm-array workflow for daily OBC generation.

This directory is a copy of `tools/forecast_multi_proc` adapted to:
- run **only PHY OBC** generation (no IC stage, no BGC stage),
- keep `t=0` from hindcast,
- use `ocean_daily.nc` for SSH (`zos`) from forecast,
- use forecast monthly `oceanm_YYYY_MM.nc` for `u/v` (and 3D monthly source fields),
  then linearly interpolate in time to the daily SSH timeline.

## Run

```bash
python tools/OBC_PHY_DAILY/submit_workflow.py \
  --years 2012 \
  --months 01 \
  --ensembles 01 02 03 04 \
  --output-root /path/to/outputs \
  --config-root /path/to/generated_configs \
  --max-parallel 20
```

## Dry-run

```bash
python tools/OBC_PHY_DAILY/submit_workflow.py \
  --years 2012 --months 01 --ensembles 01 02 \
  --output-root ./tools/OBC_PHY_DAILY/outputs \
  --config-root ./tools/OBC_PHY_DAILY/generated_configs \
  --dry-run
```
