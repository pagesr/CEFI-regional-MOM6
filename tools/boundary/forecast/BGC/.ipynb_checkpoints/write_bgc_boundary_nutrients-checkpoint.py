#!/usr/bin/env python3
"""
This script generates BGC OBC files for individual segments and parameters.
Run on analysis with `module load nco/5.0.1`.
How to use:
./write_bgc_boundary_nutrients.py --config bgc_obc.yaml
"""

import argparse
import os
import datetime as dt
import logging
import time
from glob import glob
from os import path
import numpy as np
import xarray
import yaml
from boundary import flood_missing, Segment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Adjustable parameters
#start_year = 1993
#end_year = 2019

# File paths and variables
monthly_cobalt_path = '/archive/Dmitry.Dukhovskoy/fre/NEP/forecast_bgc/NEPbgc_fcst_dailyOB01/2012-01-e01/history/'
output_dir = '/work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/boundary/forecast/BGC/outputs/'
vgrid_file = '/work/Remi.Pages/GRID/grid/vgrid_75_2m.nc'
static_file = '/archive/Liz.Drenkard/fre/cefi/NEP/2025_07/NEP10k_202507_physics_bgc/gfdl.ncrc6-intel23-repro/pp/ocean_daily/ocean_daily.static.nc'

def read_config(config_file):
    with open(config_file, 'r') as stream:
        return yaml.safe_load(stream)

def write_bgc(segments, year):
    cobalt_vars = [
    "alk",     # from alk
    "dic",   # from dic
    "po4",      # same
    "sio4",       # from sio4
    "o2",       # same
    "no3",      # same
    "nh4",      # same
    "fed",      # same
    "fedet",    # same
    "ndet",     # same
    "nbact",    # same
    "nsmz",     # same
    "nmdz",     # same
    "nlgz",     # same
    "chl",      # same
    "chl_Lg",   # same
    "chl_Md",   # same
    "chl_Sm",   # same
    "chl_Di",   # same
    "simd",     # same
    "silg",     # same
    "ndi",      # same
    "nlg",      # same
    "nsm",      # same
    "nmd",      # same
    "pdi",      # same
    "plg",      # same
    "pmd",      # same
    "psm"       # same
    ]
    logging.info(f"Processing year {year}...")

    # Load static file for 1D lat/lon
    static_ds = xarray.open_dataset(static_file)
    lat1d = static_ds['geolat'] if 'geolat' in static_ds else static_ds['yh']
    lon1d = static_ds['geolon'] if 'geolon' in static_ds else static_ds['xh']

    # Generate time coordinate
    time_values = np.array([
        dt.datetime(year, month, 1, 0, 0) for month in range(1, 13)
    ])
    time_coord = xarray.DataArray(
        np.arange(len(time_values)),
        dims="time",
        coords={"time": time_values},
        attrs={
            "units": f"days since {year}-01-01 00:00:00",
            "calendar": "gregorian",
        },
    )
    for var in cobalt_vars:  # Process each variable separately
        cobalt_file = f'{monthly_cobalt_path}/ocean_cobalt_tracers_month_z.nc'
        
        if not path.exists(cobalt_file):
            logging.warning(f"File not found: {cobalt_file}. Skipping...")
            continue
        # Load and rename grid indices
        ds = (
            xarray.open_dataset(cobalt_file,decode_cf=False)
            .rename({'z_l': 'z', 'jh': 'yh', 'ih': 'xh','si': 'sio4', 'talk': 'alk','dissic': 'dic'})
        )
        cobalt = ds[var]
        if var == 'alk' or var == 'dic':
            print('ALK or dic')
            #ds_ocean=xarray.open_dataset(ocean_file)#, decode_times=False)
            rho=1026.0
            cobalt=cobalt/rho

        # Add or fix time coordinate
        if 'time' not in cobalt.dims:
            cobalt = cobalt.expand_dims(time=time_coord)
        else:
            cobalt = cobalt.assign_coords(time=time_coord)

        # Assign 1D lat/lon
        print(lat1d.data.shape,lon1d.data.shape)
        print(cobalt)
        cobalt = cobalt.assign_coords(
            lat=('yh', lat1d.data),
            lon=('xh', lon1d.data)
        )
        cobalt['lat'].attrs.update({'long_name': 'latitude', 'units': 'degrees_north'})
        cobalt['lon'].attrs.update({'long_name': 'longitude', 'units': 'degrees_east'})

        logging.info(f"Processing variable {var}, dimensions: {cobalt.dims}")

        # Apply flooding
        cobalt_flooded = flood_missing(cobalt, xdim='xh', ydim='yh', zdim='z')
        cobalt_flooded = cobalt_flooded.load()
        cobalt_flooded = cobalt_flooded.assign_coords(lat=cobalt['lat'], lon=cobalt['lon'])

        print('cobalt_flooded', cobalt_flooded)
 
        # Process each segment for this variable and year
        for seg in segments:
            cobalt_seg = seg.regrid_tracer(cobalt_flooded, 
                                            regrid_suffix='cobalt', flood=False,
                                            periodic=True)
                
            # Ensure no negative values
            cobalt_seg = cobalt_seg.map(lambda var: np.clip(var, 0.0, None))
                
            # Add coordinates
            cobalt_seg = seg.add_coords(cobalt_seg)
                
            # Save to a unique file with padded segment ID
            segment_id_padded = f"{seg.num:03d}"
            output_file = f"{output_dir}/bgc_{var}_{year}_segment_{segment_id_padded}.nc"
            seg.to_netcdf(cobalt_seg, output_file)
            logging.info(f"File saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate BGC tracers obc.')
    parser.add_argument('--config', dest='config_file', default='bgc_obc.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--year', type=int, help='Year to process')  # Add year argument
    args = parser.parse_args()

    config = read_config(args.config_file)
    year = args.year  # Get the year from the batch job
    if year is None:
        raise ValueError("No year provided. Use --year argument.")
        
    global output_dir
    output_dir = config['output_dir']
    grid_file = config['grid_file']

    # Create output directory if it doesn't exist
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # Regional model domain and boundaries
    hgrid = xarray.open_dataset(grid_file)

    # Load segments
    segments = []
    for seg_config in config.get('segments', []):
        segment = Segment(seg_config['id'], seg_config['border'], hgrid, output_dir=output_dir)
        segments.append(segment)
        
    time0 = dt.datetime.strptime(str(config['time0']), '%Y-%m-%d')
    last_time = dt.datetime.strptime(str(config['last_time']), '%Y-%m-%d')
    write_bgc(segments, year)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Processing took {end_time - start_time} seconds.")
