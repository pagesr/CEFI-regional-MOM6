#!/usr/bin/env python3
"""
This script generated T,S, ssh, u, v OBC from Glorys
Have to make sure nan values in Glorys have been filled by non-nan values
Also this script require nco tools (optional) if you want to concatenat
multiple years results. 
Run on analysis, with module load nco/5.0.1
How to use:
./write_glorys_boundary_south.py --config glorys_obc_CGOA.yaml 
"""
from subprocess import run
from os import path
import xarray
import yaml
from boundary import Segment
import argparse
import os

# xarray gives a lot of unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def write_year(year, glorys_dir, segments, variables, is_first_year=False, is_last_year=False):
    ds = xarray.open_dataset(path.join(glorys_dir, f'{year}0101.goa_obcs_east.nc')) # EAST in NEP mean south boudary in GOA
    ds = ds.rename({'z_l': 'z', 'ssh': 'zos'})

    # Adjust time for matching initial conditions
    if is_first_year:
        tnew = xarray.concat((ds['time'][0].dt.floor('1d'), ds['time'][1:]), dim='time')
        ds['time'] = ('time', tnew.data)
    elif is_last_year:
        tnew = xarray.concat((ds['time'][0:-1], ds['time'][-1].dt.ceil('1d')), dim='time')
        ds['time'] = ('time', tnew.data)

    # Regrid zos (formerly ssh) first
    if 'zos' in variables and 'zos' in ds:
        for seg in segments:
            print(f'{seg.border} zos')
            tracer = ds['zos']
            tracer.coords['lon'] = ds['geolon']
            tracer.coords['lat'] = ds['geolat']
            tracer['lon'].attrs.update({'standard_name': 'longitude', 'units': 'degrees_east'})
            tracer['lat'].attrs.update({'standard_name': 'latitude', 'units': 'degrees_north'})
            seg.regrid_tracer(tracer, suffix=year, flood=False,weight_save=True)

    # Then regrid uo and vo
    if 'uv' in variables and 'uo' in ds and 'vo' in ds:
        for seg in segments:
            print(f'{seg.border} uv')
            uo = ds['uo']
            vo = ds['vo']
            uo.coords['lon'] = ds['geolon_u']
            uo.coords['lat'] = ds['geolat_u']
            vo.coords['lon'] = ds['geolon_v']
            vo.coords['lat'] = ds['geolat_v']
            uo['lon'].attrs.update({'standard_name': 'longitude', 'units': 'degrees_east'})
            uo['lat'].attrs.update({'standard_name': 'latitude', 'units': 'degrees_north'})
            vo['lon'].attrs.update({'standard_name': 'longitude', 'units': 'degrees_east'})
            vo['lat'].attrs.update({'standard_name': 'latitude', 'units': 'degrees_north'})
            seg.regrid_velocity(uo, vo, suffix=year, flood=False, rotate=False, weight_save=True)

    # Finally, regrid other tracers (e.g., thetao, so)
    for var in variables:
        if var in ['zos', 'uv']:
            continue
        if var in ds:
            for seg in segments:
                print(f'{seg.border} {var}')
                tracer = ds[var]
                tracer.coords['lon'] = ds['geolon']
                tracer.coords['lat'] = ds['geolat']
                tracer['lon'].attrs.update({'standard_name': 'longitude', 'units': 'degrees_east'})
                tracer['lat'].attrs.update({'standard_name': 'latitude', 'units': 'degrees_north'})
                seg.regrid_tracer(tracer, suffix=year, flood=False, weight_save=True)
        else:
            raise ValueError(f"{var} not found in dataset")
def ncrcat_years(nsegments, output_dir, variables, ncrcat_names):
    if not ncrcat_names:
        ncrcat_names = variables[:]

    for var,var_name in zip(variables,ncrcat_names):
        for seg in range(1, nsegments+1):
            run([f'ncrcat -O {var}_{seg:03d}_* {var_name}_{seg:03d}.nc'], cwd=output_dir, shell=True)

def main(config_file):
    # Load configuration from YAML file
    config = load_config(config_file)

    # Extract configuration parameters
    first_year = config.get('first_year', 1997)
    last_year = config.get('last_year', 1997)
    glorys_dir = config.get('glorys_dir', '/work/Remi.Pages/GOA2p5k/HINDCAST')
    output_dir = config.get('output_dir', './outputs_CGOA_dec25')
    hgrid_file = config.get('hgrid', '/work/Remi.Pages/GOA2p5k/GRID/CGOA_2.5k/ocean_hgrid.nc')
    ncrcat_years_flag = config.get('ncrcat_years', False)
    ncrcat_names = config.get('ncrcat_names', [])

    # Create output directory if it doesn't exist
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # Load hgrid
    hgrid = xarray.open_dataset(hgrid_file)

    # Load variables
    variables = config.get('variables', [])

    # Load segments
    segments = []
    for seg_config in config.get('segments', []):
        segment = Segment(seg_config['id'], seg_config['border'], hgrid, output_dir=output_dir)
        segments.append(segment)

    for y in range(first_year, last_year+1):
        print(y)
        write_year(y, glorys_dir, segments, variables, is_first_year=y == first_year, is_last_year=y == last_year)

    # Optional step: ncrcat_years
    if ncrcat_years_flag:
        assert len(ncrcat_names) == len(variables), ("Could not concatenate annual files because the the "
                                                     "number of file output names did not match the number "
                                                     "of variables provided. Please concatenate the files manually.")

        ncrcat_years(len(segments), output_dir, variables, ncrcat_names)

if __name__ == '__main__':
    # Set default config file name
    default_config_file = 'glorys_obc_CGOA.yaml'

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate obc from Glorys')
    parser.add_argument('--config', type=str, default='glorys_obc_CGOA.yaml',
                        help='Specify the YAML configuration file name')
    args = parser.parse_args()

    # Run the main function with the specified or default config file
    main(args.config)
