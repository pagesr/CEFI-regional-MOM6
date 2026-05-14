IC-OBC-FCST-clean — Full Workflow Documentation (IC + OBC, multi-process)
==========================================================================

Scope
-----
This document describes ONLY what exists under:

  tools/IC-OBC-FCST-clean

It explains:
- workflow architecture and execution order
- how to run each part
- input/output/config locations
- interpolation/regridding methods used
- why UV is run with no rotation in this workflow
- IC and OBC source simulations (hindcast/forecast)


Paper-ready Materials & Methods summary (IC-OBC-FCST-clean)
------------------------------------------------------------
Use this section directly in a manuscript Methods section (edit paths/years as needed).

A) Workflow package and execution model
- Workflow used: `tools/IC-OBC-FCST-clean` (standalone Slurm workflow).
- Three stages are generated:
  1) Initial conditions (IC; physics + BGC),
  2) Physical open boundaries (PHY OBC),
  3) Biogeochemical open boundaries (BGC OBC).
- Slurm arrays execute stage tasks built from per-case YAML configs and TSV task lists.

B) Physical boundary treatment used in this workflow
- `zos` and `uv` are produced on a daily OBC timeline (hindcast day-0 + forecast day-1 onward, plus padded extra final step).
- `thetao` and `so` are produced on monthly timeline (restart + 11 forecast monthly files, plus padded final month).
- To ensure robust tracer production, PHY execution is split into two passes:
  - tracer pass (`thetao`,`so`) and
  - daily pass (`uv`,`zos`).
- Segment regridding is done with xESMF through `boundary.py` utilities.
- For UV, this workflow uses `rotate=False` in the PHY OBC driver.

C) Source simulations and static grids
- GOA static grid (target): CGOA static grid file configured in `generate_configs.py`.
- NEP static grid (source): NEP static file configured in `generate_configs.py`.
- Hindcast restart source (IC and initial OBC state): `MOM_YYYYMM01.res.nc`.
- Hindcast daily history source (daily `zos` anchor): `.../history/YYYY0101/ocean_daily.nc`.
- Forecast source root: `.../NEPbgc_fcst_dailyOB01/YYYY-MM-eNN/history/`.
- Forecast monthly files used for physics tracers/UV source:
  `oceanm_YYYY_MM.nc` sequence for the next 11 months.

D) Regridding / interpolation settings
- Regridding engine: xESMF.
- Default interpolation method in templates: `nearest_s2d`.
- Tracer OBC: `seg.regrid_tracer(... flood=False)`.
- Velocity OBC: `seg.regrid_velocity(... flood=False, rotate=False)`.
- Regridding weights are cached in shared `regrid_*` directories and reused.
- If a regridded field is detected all-zero, the corresponding weight file is removed and regenerated once.
- PHY OBC generation can write boundary-profile diagnostic PNGs comparing nearest NEP source samples against the GOA regridded OBC values for each configured segment/variable.

E) Reproducible command used to run all stages
From `tools/IC-OBC-FCST-clean`:

  python submit_workflow.py \
    --years 2013 \
    --months 01 \
    --ensembles 01 \
    --output-root ./output \
    --config-root ./generated_configs \
    --task-root ./tasks

Notes:
- `--output-root` is resolved to an absolute path at submit time.
- Add `--force` only when re-running existing outputs/markers.
- Do NOT pass `--no-ic-dependency` for normal IC -> PHY/BGC dependency behavior.

F) Output products and QC artifacts
- IC outputs: `output/YYYY/MM/IC/` (physics + BGC IC files, marker files).
- PHY OBC outputs: `output/YYYY/MM/OBC/PHY/eNN/` (thetao/so/uv/zos by segment).
- BGC OBC outputs: `output/YYYY/MM/OBC/BGC/eNN/` (tracer-by-segment outputs).
- Runtime logs: `output/logs/` including stage-specific logs
  (`*_ic_phy.log`, `*_ic_bgc.log`, `*_phy_obc_*.log`, `*_bgc_obc.log`).
- Optional QA:
  - `check_obc_all_zero.py` for all-zero diagnostics,
  - BGC postprocess scripts for merged output products.


1) High-level purpose
---------------------
This package builds MOM6-ready Initial Conditions (IC) and Open Boundary Conditions (OBC)
for CGOA/GOA runs, using NEP hindcast + forecast products.

The workflow is split into three stages:

1. IC stage (monthly per year/month):
   - Physical IC: temp/salt/ssh/u/v
   - BGC IC: COBALT tracer state variables

2. PHY OBC stage (monthly per year/month/ensemble):
   - OBC for thetao, so, zos, uv over configured segments

3. BGC OBC stage (monthly per year/month/ensemble):
   - OBC for BGC tracers over configured segments
   - optional merge/postprocess into single files

Parallel execution is done with Slurm arrays.


2) Directory map
----------------
Main files at tools/IC-OBC-FCST-clean:

- README.md
  Short overview.

- submit_workflow.py
  Top-level Slurm submitter. Builds task lists and submits IC/PHY/BGC arrays.

- build_task_lists.py
  Generates TSV task tables and per-case YAML configs.

- run_task_from_tsv.py
  Executes one row from a task TSV (used by array jobs).

- submit_ic_array.slurm
- submit_phy_array.slurm
- submit_bgc_array.slurm
  Slurm array wrappers (one row per array index).

- postprocess_bgc_obc_nco.sh
- submit_bgc_obc_postprocess.slurm
- bgc_obc_postprocess_tasks.txt
  Optional BGC OBC merge/postprocess tools.

- check_obc_all_zero.py
  QA utility for detecting all-zero OBC outputs.

- tasks/
  Generated task files:
  - ic_tasks.tsv
  - phy_tasks.tsv
  - bgc_tasks.tsv

- generated_configs/
  Generated per-case YAML files:
  YYYY/MM/eNN/{ic_phy.yaml, ic_bgc.yaml, obc_phy.yaml, obc_bgc.yaml}

- standalone/
  Self-contained scientific scripts and templates.

Important standalone subtrees:

- standalone/forecast_cgoa/
  - generate_configs.py
  - run_ic.py
  - run_phy_obc.py
  - run_bgc_obc.py
  - config_templates/*.yaml
  - utils/{paths.py,helpers.py,logging_utils.py,slurm_utils.py}

- standalone/initial/
  - nep_to_goa_phy_ic.py
  - nep_to_goa_bgc_ic.py

- standalone/boundary/PHY/
  - write_CGOA_boundary_2Dfrc-padded.py
  - boundary.py

- standalone/boundary/BGC/
  - OBC_BGC.py
  - boundary.py


3) Source simulations used (IC and OBC)
----------------------------------------
By default, configs are generated from values in:
  standalone/forecast_cgoa/generate_configs.py (DEFAULTS)

Static grids:
- GOA static (target grid):
  /archive/Remi.Pages/fre/Arc_12/2026_02.01/CGOA_BGC_2025_07_base_nep_phy_feb26/...
- NEP static (source grid):
  /archive/Liz.Drenkard/fre/cefi/NEP/2025_07/NEP10k_202507_physics_bgc/...

Hindcast sources:
- Restart root:
  /archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/restart
- History root:
  /archive/Dmitry.Dukhovskoy/fre/NEP/hindcast_bgc/NEPbgc_nudged_hindcast02/history

Forecast sources:
- Forecast physics/BGC root:
  /archive/Dmitry.Dukhovskoy/fre/NEP/forecast_bgc/NEPbgc_fcst_dailyOB01

Interpretation for stage data origin:

A) IC physical:
- Uses hindcast restart file for exact initialization date:
  MOM_YYYYMM01.res.nc (Temp, Salt, ave_ssh, u, v)

B) IC BGC:
- Scans restart directory restdate_YYYYMM01 for *res*.nc files
- Reads requested BGC_VARS from available files
- Missing expected variables in missing_to_zero are set to zero

C) PHY OBC (write_CGOA_boundary_2Dfrc-padded.py):
- time index 0 (initial month):
  - 3D T/S/U/V from hindcast restart MOM_YYYYMM01.res.nc
  - surface zos from hindcast monthly ocean_month.nc
- time index 1..11:
  - forecast monthly files from fct_dir/YYYY-MM-eNN/history
- time index 12:
  - padded copy of last available month

D) BGC OBC (OBC_BGC.py):
- Uses forecast monthly BGC tracer file from
  fct_dir/YYYY-MM-eNN/history/ocean_cobalt_tracers_month_z.nc
- Uses first 12 monthly steps by default (time_sel=first12)
- Pads extra last timestep by duplication


4) Interpolation / regridding methods used
------------------------------------------
Core engine:
- xESMF (xesmf.Regridder)

Default method in templates/configs:
- nearest_s2d for IC tracers and UV
- nearest_s2d defaults also in Segment regridding APIs for OBC

Where this is controlled:
- IC PHY template: tracer_method, uv_method (default nearest_s2d)
- IC BGC template: tracer_method (default nearest_s2d)
- OBC boundary Segment APIs default method='nearest_s2d'

IC PHY details (nep_to_goa_phy_ic.py):
- temp/salt/ssh: regrid on tracer grid geolon/geolat
- u: regrid on U grid geolon_u/geolat_u
- v: regrid on V grid geolon_v/geolat_v
- reusable weights enabled (reuse_weights=true)
- if V-weight file is suspiciously tiny or output all-zero, script rebuilds V weights once

IC BGC details (nep_to_goa_bgc_ic.py):
- all BGC tracers regridded with same tracer-grid mapping
- reusable weight file support
- missing_to_zero variables are filled with zeros on target grid

PHY OBC details:
- tracer fields use seg.regrid_tracer(..., flood=False)
- velocity fields use seg.regrid_velocity(..., flood=False, rotate=False)
- weight files are saved/reused in configured regrid_dir
- if all-zero output is detected for a segment/field, corresponding weights are removed and re-built once
- diagnostic_plots=true writes PNG profiles under OBC/PHY/eNN/diagnostics/boundary_profiles/ when matplotlib is available

BGC OBC details:
- tracer-only segment regridding via seg.regrid_tracer(...)
- unit conversion for selected tracers before write:
  dic/alk/sio4 from mol m-3 to mol kg-1 (divide by rho0=1026 kg m-3)
- per-variable per-segment outputs are padded to next-month first day


5) Why UV is set to no rotation in this workflow
------------------------------------------------
This workflow explicitly calls:

  seg.regrid_velocity(... rotate=False ...)

in the PHY OBC generator.

Meaning:
- No earth-to-model vector rotation is applied during OBC UV write.
- U and V are regridded component-wise on their respective source/target staggered
  U/V grids (with geolon_u/geolat_u and geolon_v/geolat_v coordinates attached).

Practical assumption used here:
- input and output U/V orientation are already treated as aligned for this pipeline,
  so additional rotation is intentionally skipped.

Notes:
- boundary.py includes rotation-capable utilities and a rotate=True default in generic APIs,
  but this specific PHY OBC driver overrides that default with rotate=False.


6) Time treatment and padding behavior
--------------------------------------
PHY OBC:
- Constructs a 13-step year-relative time axis:
  12 real monthly states + 1 padded final state
- Last time coordinate is moved to month-end behavior used by script
- Adds time bounds (time_bnds)

BGC OBC:
- Regrids selected monthly tracer times (default first 12)
- Appends one extra record by copying last timestep and assigning it to
  first day of next month


7) Inputs and outputs by stage
------------------------------
A) Input control files
- Config templates:
  standalone/forecast_cgoa/config_templates/*.yaml
- Generated concrete configs:
  generated_configs/YYYY/MM/eNN/*.yaml

B) Task list inputs for arrays
- tasks/ic_tasks.tsv
- tasks/phy_tasks.tsv
- tasks/bgc_tasks.tsv

C) Stage outputs (default structure under --output-root)
- IC:
  {output_root}/YYYY/MM/IC/
    - ic_phy_YYYYMM01.nc
    - ic_bgc_YYYYMM01.nc
    - marker files: .ic_phy.done, .ic_bgc.done

- PHY OBC:
  {output_root}/YYYY/MM/OBC/PHY/eNN/
    - per-variable per-segment yearly files (e.g., thetao_001_YYYY.nc, ...)
    - marker file: .phy_obc_eNN.done

- BGC OBC:
  {output_root}/YYYY/MM/OBC/BGC/eNN/
    - per-tracer per-segment files (then optionally merged)
    - marker file: .bgc_obc_eNN.done

D) Logs
- Slurm stdout/stderr:
  chosen by submit_workflow.py
  priority: --slurm-log-dir > {output_root}/logs > tools/forecast_multi_proc/logs
- stage command logs:
  written under FORECAST_LOG_ROOT (set by run_task_from_tsv.py to {output_root}/logs)

E) Regridding weights (shared folders inside tools/forecast_multi_proc)
- regrid_ic_phy_shared
- regrid_ic_bgc_shared
- regrid_obc_phy_shared
- regrid_obc_bgc_shared


8) How to run each part
-----------------------
8.1) Full workflow submission (recommended)

From repository root:

python tools/forecast_multi_proc/submit_workflow.py \
  --years 2015 \
  --months 10 \
  --ensembles 01 02 03 04 05 \
  --output-root /path/to/outputs \
  --config-root /path/to/generated_configs \
  --max-parallel 20

Behavior:
- builds task TSVs
- submits IC array
- submits PHY + BGC arrays with afterok dependency on IC (default)

Options:
- --no-ic-dependency  (submit PHY/BGC immediately)
- --force             (ignore marker skip)
- --dry-run           (show sbatch commands only)
- --slurm-log-dir     (custom location for sbatch .out/.err)
- --conda-env-path    (python env path used in slurm scripts)


8.2) Build tasks/configs only

python tools/forecast_multi_proc/build_task_lists.py \
  --years 2015 --months 10 --ensembles 01 02 03 \
  --output-root /path/to/outputs \
  --config-root /path/to/generated_configs \
  --task-root tools/forecast_multi_proc/tasks

This creates/updates:
- task TSVs
- generated YAML configs for each case


8.3) Run one task manually from TSV

IC example:
python tools/forecast_multi_proc/run_task_from_tsv.py \
  --stage ic \
  --task-file tools/forecast_multi_proc/tasks/ic_tasks.tsv \
  --task-id 0 \
  --output-root /path/to/outputs

PHY example:
python tools/forecast_multi_proc/run_task_from_tsv.py \
  --stage phy \
  --task-file tools/forecast_multi_proc/tasks/phy_tasks.tsv \
  --task-id 0 \
  --output-root /path/to/outputs

BGC example:
python tools/forecast_multi_proc/run_task_from_tsv.py \
  --stage bgc \
  --task-file tools/forecast_multi_proc/tasks/bgc_tasks.tsv \
  --task-id 0 \
  --output-root /path/to/outputs


8.4) Run stage scripts directly (without TSV wrapper)

IC wrapper:
python tools/forecast_multi_proc/standalone/forecast_cgoa/run_ic.py \
  --ic-phy-config tools/forecast_multi_proc/generated_configs/YYYY/MM/eNN/ic_phy.yaml \
  --ic-bgc-config tools/forecast_multi_proc/generated_configs/YYYY/MM/eNN/ic_bgc.yaml \
  --year YYYY --month MM --output-root /path/to/outputs

PHY wrapper:
python tools/forecast_multi_proc/standalone/forecast_cgoa/run_phy_obc.py \
  --config tools/forecast_multi_proc/generated_configs/YYYY/MM/eNN/obc_phy.yaml \
  --year YYYY --month MM --ensemble NN --output-root /path/to/outputs

BGC wrapper:
python tools/forecast_multi_proc/standalone/forecast_cgoa/run_bgc_obc.py \
  --config tools/forecast_multi_proc/generated_configs/YYYY/MM/eNN/obc_bgc.yaml \
  --year YYYY --month MM --ensemble NN --output-root /path/to/outputs


8.5) Postprocess BGC OBC (merge)

Automatic path:
- run_bgc_obc.py calls postprocess_bgc_obc_nco.sh when merge_to_single_file=true in config.

Manual:
- bash tools/forecast_multi_proc/postprocess_bgc_obc_nco.sh <output_dir> <year> <month> <ensemble> [final_output]

Slurm batch helper:
- sbatch tools/forecast_multi_proc/submit_bgc_obc_postprocess.slurm


9) Slurm array internals
------------------------
submit_*_array.slurm scripts do the same pattern:
- read TASK_FILE and SLURM_ARRAY_TASK_ID
- resolve python executable from conda env or fallback
- call run_task_from_tsv.py with stage=ic|phy|bgc

Dependencies:
- default: PHY/BGC arrays wait for IC success via afterok
- optional: remove dependency with --no-ic-dependency

Concurrency:
- controlled by --max-parallel -> --array=...%N


10) Config generation and naming conventions
--------------------------------------------
Default case space (when scripts use defaults):
- years: 2012..2019
- restart months: 01,04,07,10
- ensembles: 01..05

Config files are rendered from templates with placeholders:
- {YEAR}, {MONTH}, {ENSEMBLE}
- source paths from DEFAULTS in generate_configs.py
- output/regrid paths rooted under tools/forecast_multi_proc or user --output-root

Generated config location:
- generated_configs/YYYY/MM/eNN/


11) Restart/skip logic and robustness checks
--------------------------------------------
Skip markers:
- each successful stage writes a marker in its output folder:
  .ic_phy.done, .ic_bgc.done, .phy_obc_eNN.done, .bgc_obc_eNN.done
- if marker exists and expected outputs are present, stage is skipped
- --force overrides skip

Robustness checks implemented in scripts:
- verify forecast history directories exist before OBC stage runs
- detect and recover from suspicious/all-zero regridded outputs by rebuilding weights
- explicit config type normalization (year ints, month/ensemble zero-padding)


12) QA helper for zero fields
-----------------------------
Script:
  check_obc_all_zero.py

Purpose:
- scan generated OBC files for all-zero variables/files
- write report (default WARNNING.txt)

Use it after generating OBC to catch pathological regridding outcomes quickly.


13) Practical checklist
-----------------------
Before running:
1. Confirm source archives are accessible on your machine.
2. Confirm output-root and config-root are writable.
3. Confirm xESMF stack and netcdf4 are available in selected conda env.
4. Optionally dry-run submit_workflow.py.

After running:
1. Check Slurm logs under selected logs directory.
2. Verify marker files exist for completed stages.
3. Inspect output nc files under IC and OBC directories.
4. Run check_obc_all_zero.py for sanity.


14) Key assumptions in this workflow
------------------------------------
- NEP hindcast restart at YYYY-MM-01 is the authoritative IC anchor.
- Forecast supplies months after initialization for OBC continuation.
- One extra padded month is needed to extend boundary forcing window.
- UV rotation is intentionally disabled in the PHY OBC driver (rotate=False),
  consistent with the pipeline assumption that source and destination velocity
  orientations are already aligned for this case.


End of file
