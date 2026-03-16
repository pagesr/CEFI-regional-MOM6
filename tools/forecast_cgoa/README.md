# CGOA Forecast IC/OBC Automation

This directory provides a production-style automation wrapper around the existing working scientific scripts.

## What it automates

For each `(year, restart_month)` it runs:
1. Physics IC from hindcast.
2. BGC IC from hindcast.
3. Physics OBC for each ensemble (`e01..e05`) with hindcast at `t=0` and forecast for `t=1..11` (using the existing padded PHY script).
4. BGC OBC for each ensemble (`e01..e05`) with forecast for `t=0..11`, with flooding/padding enabled through config.

Supported default loops:
- Years: `2012..2019`
- Restart months: `01 04 07 10`
- Ensembles: `01 02 03 04 05`

## Layout

- `run_workflow.py`: main driver.
- `generate_configs.py`: YAML config generator from templates.
- `run_ic.py`: IC step runner.
- `run_phy_obc.py`: physics OBC step runner.
- `run_bgc_obc.py`: BGC OBC step runner.
- `submit_workflow.slurm`: Slurm submission script.
- `config_templates/`: template YAMLs for IC/OBC.
- `utils/`: shared paths/logging/helpers.
- `logs/`: per-step logs and Slurm stdout/stderr.
- `generated_configs/`: auto-generated run configs.
- `outputs/`: organized outputs (`year/month/IC|OBC/PHY|BGC/ensemble`).

## Run locally

```bash
cd /work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/forecast_cgoa
python run_workflow.py
```

### Common options

```bash
python run_workflow.py \
  --years 2012 2013 \
  --months 01 04 \
  --ensembles 01 02 \
  --output-root /path/to/output_root \
  --config-root /path/to/generated_configs \
  --force
```

## Run with Slurm

```bash
cd /work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/forecast_cgoa
sbatch submit_workflow.slurm
```

## Restart / skip behavior

Each step writes a marker file in the target output folder:
- `.ic_phy.done`
- `.ic_bgc.done`
- `.phy_obc_eXX.done`
- `.bgc_obc_eXX.done`

If marker exists, the step is skipped unless `--force` is set.

## Logging

Per-step logs are written to `logs/`, for example:
- `logs/2012_01_ic_phy.log`
- `logs/2012_01_ic_bgc.log`
- `logs/2012_01_e01_phy_obc.log`
- `logs/2012_01_e01_bgc_obc.log`

Each log includes start/end time, command, cwd, and return code.

## Editable settings

Update static paths and defaults in:
- `generate_configs.py` (`DEFAULTS` dictionary)
- `config_templates/*.yaml`
- `submit_workflow.slurm` (partition, env, runtime)

## Troubleshooting

If you see an activation error like:
`magics-activate.sh: line 3: MAGPLUS_HOME: unbound variable`
it comes from a site Conda activation hook when shell nounset (`set -u`) is active and `MAGPLUS_HOME` is undefined.
The provided `submit_workflow.slurm` already guards against this by initializing `MAGPLUS_HOME` and temporarily disabling nounset during `conda activate`.
