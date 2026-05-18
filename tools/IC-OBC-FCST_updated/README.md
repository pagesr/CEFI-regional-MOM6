# IC-OBC-FCST_updated

Standalone workflow copied from `tools/IC-OBC-FCST-clean` with segment-002 OBC output updated for the CGOA MOM6 boundary definition:

```text
OBC_SEGMENT_002 = "I=0,J=N:0,FLATHER,ORLANSKI,NUDGED,ORLANSKI_TAN,NUDGED_TAN"
```

What this directory runs:
- IC stage
- OBC BGC stage
- OBC PHY stage with:
  - daily SSH (`zos`) from `ocean_daily.nc` (hindcast t=0 + forecast t=1..)
  - `uv` monthly-to-daily interpolation for OBC output
  - PHY OBC daily `zos`/`uv` behavior copied from `tools/OBC_PHY_DAILY`
  - segment 002 files written north-to-south along `ny_segment_002`

Segment 002 ordering update:
- All segment 002 NetCDF writes reverse only the horizontal `ny_segment_002` axis immediately before writing.
- `time`, `nz_segment_002`, and `constituent` axes are not reversed.
- Velocity signs are not changed.
- `lon_segment_002` and `lat_segment_002` are reversed with the data variables.
- `ny_segment_002` and `nz_segment_002` coordinates are reset to clean `0, 1, 2, ...` indices after the horizontal reversal.
- This applies to physical and tidal files generated through the `Segment.to_netcdf()` workflow, including `uv_002`, `zos_002`, `thetao_002`, `so_002`, `tu_002`, and `tz_002`.

Verify generated files with:

```bash
./verify_segment_002_order.py /path/to/OBC/PHY/e01
```

The verification checks north-to-south ordering, clean `ny_segment_002`/`nz_segment_002` coordinates, finite critical segment-002 variables such as `zos_segment_002`, and segment-001/segment-002 corner consistency.

Original directory was not modified:
- `tools/IC-OBC-FCST-clean/`
