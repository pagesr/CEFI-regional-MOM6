# forecast_multi_proc

Parallel Slurm-array orchestration for the existing `tools/forecast_cgoa` workflow.

This directory does **not** modify `forecast_cgoa` logic; it schedules the existing stage runners in parallel:
- IC stage (`run_ic.py`) once per year/month
- PHY OBC stage (`run_phy_obc.py`) per year/month/ensemble
- BGC OBC stage (`run_bgc_obc.py`) per year/month/ensemble

## What this adds

- Task TSV generation (`build_task_lists.py`)
- Per-task runner (`run_task_from_tsv.py`)
- Three array job scripts:
  - `submit_ic_array.slurm`
  - `submit_phy_array.slurm`
  - `submit_bgc_array.slurm`
- Top-level submitter (`submit_workflow.py`) with:
  - Slurm dependencies (`PHY/BGC` wait for `IC`)
  - array concurrency limit (default `%20`)

## Example

```bash
python tools/forecast_multi_proc/submit_workflow.py \
  --years 2012 2013 \
  --months 01 02 \
  --ensembles 01 02 03 04 \
  --output-root /work5/rnp/IC-BC-GOA/CEFI-regional-MOM6/tools/forecast_multi_proc/outputs \
  --config-root /work5/rnp/IC-BC-GOA/CEFI-regional-MOM6/tools/forecast_multi_proc/generated_configs \
  --max-parallel 20
```

## Dry-run

```bash
python tools/forecast_multi_proc/submit_workflow.py \
  --years 2012 --months 01 --ensembles 01 02 \
  --output-root ./tools/forecast_multi_proc/outputs \
  --config-root ./tools/forecast_multi_proc/generated_configs \
  --dry-run
```

## Notes

- The array scripts activate Conda env `/nbhome/role.medgrp/.conda/envs/medpy311` by default.
- Override env path per submission with:
  `--conda-env-path /path/to/your/env` on `submit_workflow.py`
  (or edit the default in `submit_*_array.slurm`).
- If `conda.sh` is unavailable on compute nodes, scripts fall back to
  `CONDA_ENV_PATH/bin/python` (or `python`/`python3` in `PATH`).
- By default, PHY/BGC arrays wait for IC completion (`afterok` dependency).
  To run stages immediately (more concurrent jobs), add `--no-ic-dependency`.
- Slurm stdout/stderr are written to `<output-root>/logs` (set by `submit_workflow.py`)
  so jobs do not depend on write permissions inside the repository tree.
- `run_bgc_obc.py` may invoke NCO post-processing depending on generated config (`merge_to_single_file`).
