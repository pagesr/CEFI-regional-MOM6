#!/usr/bin/env python3
"""
script for preparing model IC (ssh,T,S,u,v) from Glorys data
How to use
./write_NEP_initial-v2.py --config_file nep_ic.yaml
"""
import sys
import os
import argparse
import yaml

import numpy as np
import xarray
import xesmf
import re

sys.path.append('HCtFlood')
from HCtFlood import kara as flood

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

#
#sys.path.append(os.path.join(script_dir, './depths'))
from depths import vgrid_to_interfaces, vgrid_to_layers

#
sys.path.append(os.path.join(script_dir, '../boundary'))
from boundary_nep import rotate_uv, rotate_uv_model_to_earth


def write_initial(config):
    glorys_file = config['glorys_file']
    vgrid_file = config['vgrid_file']
    grid_file = config['grid_file']
    grid_file_nep = config['grid_file_nep']
    static_nep = config['static_nep']
    output_file = config['output_file']
    reuse_weights = config.get('reuse_weights', False)

    variable_names = config.get('variable_names', {})
    temp_var = variable_names.get('temperature', 'temp')
    sal_var = variable_names.get('salinity', 'salt')
    ssh_var = variable_names.get('sea_surface_height', 'ssh')
    u_var = variable_names.get('zonal_velocity', 'u')
    v_var = variable_names.get('meridional_velocity', 'v')

    vgrid = xarray.open_dataarray(vgrid_file)
    z = vgrid_to_layers(vgrid)
    ztarget = xarray.DataArray(
        z,
        name='zl',
        dims=['zl'], 
        coords={'zl': z}, 
    )
    # Need to add geolon geolat to the rst files from NEP
    print(glorys_file)
    ds = xarray.open_dataset(glorys_file, decode_times=False)
    ds = ds.rename({'Layer': 'depth','Time': 'time'})
    glorys = ds[[temp_var, sal_var, ssh_var, u_var, v_var]]
    # RP : THIS SECTION TAKE THE U V from NEP --> super grid NEP
    # Then roatate it. Then use those directly to be interpolated on CGOA. 
    static_nep = xarray.open_dataset(config['static_nep'])
    hgrid_nep=xarray.open_dataset(grid_file_nep)
    angle_dx = hgrid_nep['angle_dx']
    print(static_nep)
    # === Get geolon/geolat for U and V ===
    geolon_u = static_nep['geolon_u']
    geolat_u = static_nep['geolat_u']
    geolon_v = static_nep['geolon_v']
    geolat_v = static_nep['geolat_v']
    # === Prepare U ===
    glorys_u = glorys[[u_var]].copy()
    glorys_u['lon'] = (('yh', 'xq'), geolon_u.data)
    glorys_u['lat'] = (('yh', 'xq'), geolat_u.data)

    # === Prepare V ===
    glorys_v = glorys[[v_var]].copy()
    glorys_v['lon'] = (('yq', 'xh'), geolon_v.data)
    glorys_v['lat'] = (('yq', 'xh'), geolat_v.data)

    # === 1. Load supergrid from ocean_hgrid.nc ===
    supergrid = xarray.open_dataset(grid_file_nep)  # e.g., ocean_hgrid.nc
    supergrid_uv = xarray.Dataset(
        {
            "lon": (("nyp", "nxp"), supergrid["x"].data),
            "lat": (("nyp", "nxp"), supergrid["y"].data),
        }
    )
    print(supergrid_uv)
    # === Regrid U ===
    regrid_u = xesmf.Regridder(glorys_u, supergrid_uv, method='nearest_s2d', filename='regrid_u_nep.nc', reuse_weights=True)
    interp_u = regrid_u(glorys_u[[u_var]])

    # === Regrid V ===
    regrid_v = xesmf.Regridder(glorys_v, supergrid_uv, method='nearest_s2d', filename='regrid_v_nep.nc', reuse_weights=True)
    interp_v = regrid_v(glorys_v[[v_var]])

    # === Check shapes before rotation ===
    print("interp_u", interp_u[u_var].shape)
    print("interp_v", interp_v[v_var].shape)
    print("angle_dx", angle_dx.shape)



    # Convert NEP grid-aligned (model) velocities -> earth-relative (east/north)
    # NOTE: angle_dx here is on the NEP supergrid (nyp,nxp), which matches interp_u/interp_v.
    u_e_nep, v_n_nep = rotate_uv_model_to_earth(interp_u[u_var], interp_v[v_var], angle_dx)
    
    u_e_nep = u_e_nep.transpose('time', 'depth', 'nyp', 'nxp')
    v_n_nep = v_n_nep.transpose('time', 'depth', 'nyp', 'nxp')
    
    print("u_e_nep, v_n_nep", u_e_nep.shape, v_n_nep.shape)

    ds = xarray.open_dataset(glorys_file,decode_times=False)
    ds = ds.rename({'Layer': 'depth','Time': 'time'})
    glorys = ds[[temp_var, sal_var, ssh_var]]
    # Round time down to midnight


    def _date_from_restart_path(path):
        m = re.search(r"MOM_(\d{8})\.res\.nc", os.path.basename(path))
        if not m:
            raise ValueError(f"Could not infer YYYYMMDD from restart filename: {path}")
        ymd = m.group(1)
        return np.datetime64(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}")
    
    ic_date = _date_from_restart_path(glorys_file)
    glorys = glorys.assign_coords(time=("time", [ic_date]))
   
    # Interpolate GLORYS vertically onto target grid.
    # Depths below bottom of GLORYS are filled by extrapolating the deepest available value.
    revert = glorys.interp(depth=ztarget, kwargs={'fill_value': 'extrapolate'}).ffill('zl', limit=None)
    print('revert')
    print(revert)

    flooded = xarray.merge((
        flood.flood_kara(revert[v], zdim='zl', ydim='lath', xdim='lonh')
        for v in [temp_var, sal_var]
    ))
    
    # flood sfc separately (no zl)
    flooded[ssh_var] = (
        flood.flood_kara(revert[ssh_var], ydim='lath', xdim='lonh')
        .isel(z=0).drop('z')
    )



    # Horizontally interpolate the vertically interpolated and flooded data onto the MOM grid. 
    target_grid = xarray.open_dataset(grid_file)
    target_t = (
        target_grid
        [['x', 'y']]
        .isel(nxp=slice(1, None, 2), nyp=slice(1, None, 2))
        .rename({'y': 'lat', 'x': 'lon', 'nxp': 'xh', 'nyp': 'yh'})
    )
    # Interpolate u and v onto supergrid to make rotation possible
    target_uv = (
        target_grid
        [['x', 'y']]
        .rename({'y': 'lat', 'x': 'lon'})
    )
    
    regrid_kws = dict(method='nearest_s2d', reuse_weights=reuse_weights, periodic=False)

    glorys_to_t = xesmf.Regridder(glorys, target_t, filename='regrid_glorys_tracers.nc', **regrid_kws)
    nep_super=(
        xarray.open_dataset(grid_file_nep)
        [['x','y']]
        .rename({'x': 'lon', 'y': 'lat'})
    )
    glorys_to_uv = xesmf.Regridder(nep_super, target_uv, filename='regrid_glorys_uv.nc', **regrid_kws)

    interped_t = glorys_to_t(flooded[[temp_var, sal_var, ssh_var]])

    # Regrid earth-relative velocities (east/north) onto target supergrid
    interped_uv = glorys_to_uv(xarray.Dataset({'u_e': u_e_nep, 'v_n': v_n_nep}))
    
    # Convert earth-relative (east/north) -> target grid-aligned (model) using target angle_dx
    urot, vrot = rotate_uv(interped_uv['u_e'], interped_uv['v_n'], target_grid['angle_dx'])


    
    uo = urot.isel(nxp=slice(0, None, 2), nyp=slice(1, None, 2))
    uo = uo.rename({'nxp': 'xq', 'nyp': 'yh', 'depth': 'zl'}).transpose('time', 'zl', 'yh', 'xq')
    uo.name = 'uo'

    vo = vrot.isel(nxp=slice(1, None, 2), nyp=slice(0, None, 2))
    vo = vo.rename({'nxp': 'xh', 'nyp': 'yq', 'depth': 'zl'}).transpose('time', 'zl', 'yq', 'xh')
    vo.name = 'vo'
    print("uo,vo",uo.shape,vo.shape)
    print(vo)
    # Make velocity time coordinate match tracer time (dtype + values)
    t_time = interped_t['time']
    uo = uo.assign_coords(time=t_time)
    vo = vo.assign_coords(time=t_time)
        
    interped = (
        xarray.merge((interped_t, uo, vo))
        .transpose('time', 'zl', 'yh', 'yq', 'xh', 'xq')
    )
    print(interped)

    # Rename to match MOM expectations.
    interped = interped.rename({
        temp_var: 'temp',
        sal_var: 'salt',
        ssh_var: 'ssh',
        'uo': 'u',
        'vo': 'v'
    })

    # Fix output metadata, including removing all _FillValues.
    all_vars = list(interped.data_vars.keys()) + list(interped.coords.keys())
    encodings = {v: {'_FillValue': None} for v in all_vars}
    encodings['time'].update({'dtype':'float64', 'calendar': 'gregorian'})
    interped['zl'].attrs = {
        'long_name': 'Layer pseudo-depth, -z*',
         'units': 'meter',
         'cartesian_axis': 'Z',
         'positive': 'down'
    }

    # Extract the directory from the output_file pat
    output_folder = os.path.dirname(output_file)
    
    # Check if the output folder exists, and if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   

    # output results
    interped.to_netcdf(
        output_file,
        format='NETCDF3_64BIT',
        engine='netcdf4',
        encoding=encodings,
        unlimited_dims='time'
    )


def main():

    parser = argparse.ArgumentParser(description='Generate ICs from Glorys.')
    parser.add_argument('--config_file', type=str, default='nep_ic.yaml' , help='Path to the YAML config file')
    args = parser.parse_args()

    if not args.config_file:
        parser.error('Please provide the path to the YAML config file.')

    with open(args.config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    if not all(key in config for key in ['glorys_file', 'vgrid_file', 'grid_file', 'output_file']):
        parser.error('Please provide all required parameters in the YAML config file.')

    write_initial(config)

if __name__ == '__main__':
    main()
