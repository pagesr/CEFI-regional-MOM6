# CGOA Forecast IC/OBC Automation — Detailed Guide

This document is the **full operational reference** for the automated CGOA forecast IC/OBC workflow under:

- `/work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/forecast_cgoa`

It complements `README.md` with deeper implementation details, troubleshooting notes, and maintenance guidance.

---

## 1) Purpose and Scope

The workflow automates the full generation cycle for forecast preparation:

- **Physics IC** (from hindcast)
- **BGC IC** (from hindcast)
- **Physics OBC** (hindcast at `t=0`, forecast at `t=1..11`, padded script)
- **BGC OBC** (forecast at `t=0..11`, with flooding/padding enabled by config)

for all requested combinations of:

- **Years:** 2012–2019
- **Restart months:** 01, 04, 07, 10
- **Forecast ensembles:** 01–05 (corresponding to e1..e5)

The framework is intentionally designed to **reuse existing scientific scripts** rather than rewrite science logic.

---

## 2) High-Level Workflow Order

For each `(year, month)`:

1. Generate Physics IC config and run Physics IC script.
2. Generate BGC IC config and run BGC IC script.
3. For each ensemble `01..05`:
   - Generate Physics OBC config and run padded PHY OBC script.
   - Generate BGC OBC config and run BGC OBC script.

Execution is serialized by default in this order so downstream steps only run after prior steps complete.

---

## 3) Directory Structure

```text
tools/forecast_cgoa/
├── README.md
├── README-DETAILS.md
├── run_workflow.py
├── generate_configs.py
├── run_ic.py
├── run_phy_obc.py
├── run_bgc_obc.py
├── submit_workflow.slurm
├── config_templates/
│   ├── ic_phy_template.yaml
│   ├── ic_bgc_template.yaml
│   ├── obc_phy_template.yaml
│   └── obc_bgc_template.yaml
├── utils/
│   ├── helpers.py
│   ├── logging_utils.py
│   ├── paths.py
│   └── slurm_utils.py
├── generated_configs/        # auto-created
├── outputs/                  # auto-created
└── logs/                     # auto-created
```

---

## 4) Scientific Scripts Called by the Automation

The automation wraps these existing scripts:

- IC (Physics): `tools/initial/nep_to_goa_phy_ic.py`
- IC (BGC): `tools/initial/nep_to_goa_bgc_ic.py`
- OBC (Physics, padded): `tools/boundary/forecast/PHY/write_CGOA_boundary_2Dfrc-padded.py`
- OBC (BGC): `tools/boundary/forecast/BGC/OBC_BGC.py`

### Important import behavior

Some scripts rely on sibling modules in their folder, e.g. `from boundary import Segment`.
The runners execute each script **from its own working directory and by local filename**, preserving those imports exactly like manual usage.

---

## 5) Configuration Generation Model

`generate_configs.py` renders YAML files from templates in `config_templates/` using token substitution.

### Template families

- `ic_phy_template.yaml`
- `ic_bgc_template.yaml`
- `obc_phy_template.yaml`
- `obc_bgc_template.yaml`

### Rendered output location

Generated per-case configs are written under:

```text
generated_configs/<YEAR>/<MONTH>/e<ENSEMBLE>/
  ic_phy.yaml
  ic_bgc.yaml
  obc_phy.yaml
  obc_bgc.yaml
```

### Main substitutions

- `YEAR`, `MONTH`, `ENSEMBLE`
- Output roots for IC/OBC
- Hindcast restart file/dir
- Time metadata for IC files
- Forecast and grid/static paths from `DEFAULTS`

### Where to edit defaults

Edit `DEFAULTS` in `generate_configs.py` to adjust cluster paths globally:

- GOA/NEP static files
- hindcast restart/history roots
- forecast roots
- GOA hgrid

---

## 6) Output Organization

By default outputs are organized as:

```text
outputs/
  <YEAR>/
    <MONTH>/
      IC/
        ic_phy_<YEAR><MONTH>01.nc
        ic_bgc_<YEAR><MONTH>01.nc
      OBC/
        PHY/
          e01/
          e02/
          ...
        BGC/
          e01/
          e02/
          ...
```

This layout cleanly separates:

- year
- restart month
- IC vs OBC
- PHY vs BGC
- ensemble

---

## 7) Restart / Skip / Recovery Behavior

Each step writes completion markers in its target output folder:

- `.ic_phy.done`
- `.ic_bgc.done`
- `.phy_obc_eXX.done`
- `.bgc_obc_eXX.done`

### On rerun

- If marker exists: step is skipped.
- If marker does not exist: step runs.
- `--force`: ignore markers and rerun everything selected.

This supports production restarts after interruption.

---

## 8) Logging Design

Each step writes a dedicated log file into `logs/`:

- `YYYY_MM_ic_phy.log`
- `YYYY_MM_ic_bgc.log`
- `YYYY_MM_eXX_phy_obc.log`
- `YYYY_MM_eXX_bgc_obc.log`

Each log includes:

- UTC start time
- executed command
- working directory
- stdout/stderr stream of the scientific script
- UTC end time and return code

---

## 9) Local Execution Examples

From `tools/forecast_cgoa`:

### Run default full domain (2012–2019, 01/04/07/10, 01..05)

```bash
python run_workflow.py
```

### Run a subset

```bash
python run_workflow.py \
  --years 2016 2017 \
  --months 01 04 \
  --ensembles 01 02
```

### Rerun selected cases from scratch

```bash
python run_workflow.py \
  --years 2018 \
  --months 10 \
  --ensembles 03 \
  --force
```

### Only generate configs (no execution)

```bash
python generate_configs.py \
  --years 2012 \
  --months 01 \
  --ensembles 01
```

---

## 10) Slurm Usage

Use `submit_workflow.slurm`:

```bash
sbatch submit_workflow.slurm
```

It currently:

- loads `miniforge` and `nco`
- activates `medpy311`
- runs `run_workflow.py` with full year/month/ensemble coverage

### Why `MAGPLUS_HOME` guard exists

Some cluster Conda activation hooks (e.g., `magics-activate.sh`) reference `MAGPLUS_HOME` and fail under nounset (`set -u`) when unset.
The script initializes `MAGPLUS_HOME` and temporarily disables nounset around `conda activate` to avoid false activation failures.

---

## 11) BGC Flooding/Padding Behavior

`tools/boundary/forecast/BGC/OBC_BGC.py` now accepts `use_flooding` via YAML.
In the default generated `obc_bgc.yaml`, it is set to `true` so BGC regridding applies flooding/padding logic as requested.

If needed, override per case by editing generated config:

```yaml
use_flooding: false
```

---

## 12) Physics OBC Logic Reminder

Physics OBC generation uses existing padded script behavior with forecast/hindcast split:

- `t = 0` from hindcast
- `t = 1..11` from forecast ensemble

This is preserved by using the existing `write_CGOA_boundary_2Dfrc-padded.py` science script.

---

## 13) Common Failure Modes and Fixes

### A) `MAGPLUS_HOME: unbound variable`

- Cause: Conda activation hook + `set -u`.
- Fix: already handled in `submit_workflow.slurm`.

### B) Import failures (`from boundary import Segment`)

- Cause: script launched outside its directory.
- Fix: runners already execute script from its own folder by filename.

### C) Missing input data files

- Cause: path mismatch in `DEFAULTS` or template values.
- Fix: update `generate_configs.py` defaults or template path fields.

### D) Partial run interruption

- Fix: rerun same command; markers skip completed work.
- To recompute: use `--force` or delete marker files for selected steps.

### E) NCO tools missing for optional concat

- Ensure `module load nco` is available when `ncrcat_years` is enabled.

---

## 14) Extending the Workflow

### Add a new ensemble

1. Update CLI list (`--ensembles`) or default in `utils/helpers.py`.
2. Ensure forecast path naming convention still matches data layout.

### Add a new restart month

1. Add month token to `RESTART_MONTHS` in `utils/helpers.py`.
2. Verify hindcast restart availability for that month.

### Add additional boundary segments

1. Edit `obc_phy_template.yaml` / `obc_bgc_template.yaml` `segments:` blocks.
2. Confirm scientific scripts support those segment IDs and borders.

### Change output root

Use `--output-root` when launching `run_workflow.py`.

---

## 15) Recommended Production Procedure

1. Start with a **single dry-run subset** (1 year, 1 month, 1 ensemble).
2. Check generated configs under `generated_configs/`.
3. Check output file naming and segment files under `outputs/`.
4. Review logs for clean return codes.
5. Scale up to full range via Slurm.
6. Keep logs and markers for restart traceability.

---

## 16) Quick Command Cheat Sheet

```bash
# Full workflow (local)
python run_workflow.py

# Full workflow (Slurm)
sbatch submit_workflow.slurm

# Subset only
python run_workflow.py --years 2012 --months 01 --ensembles 01

# Force rerun subset
python run_workflow.py --years 2012 --months 01 --ensembles 01 --force

# Generate configs only
python generate_configs.py --years 2012 --months 01 --ensembles 01
```

---

## 17) Final Notes

- The automation layer is intentionally thin around existing science scripts.
- Most maintenance should happen in templates/default path settings rather than in script internals.
- Keep this document updated when new boundaries, tracers, or environment modules are added.
