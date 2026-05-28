import xarray as xr
import numpy as np
import glob
from datetime import datetime
import argparse, os, re
import sys

#%% Functions
def remove_zonal_mean(data, vartime, varlevel, varlat, varlon, output_file=None):
    """
    Removes the zonal mean from a 4D variable (time, lev, lat, lon) in an xarray DataArray or Dataset.

    Parameters:
        data (xarray.Dataset or xarray.DataArray): Input dataset or data array with dimensions (time, lev, lat, lon).
        output_file (str, optional): Path to save the output NetCDF file. If None, the result is not saved.

    Returns:
        xarray.DataArray: Data array with the zonal mean removed.
    """

    if not isinstance(data, xr.DataArray):
        data = xr.DataArray(data)  # Convert to DataArray if it's not already
    
    # Ensure the input has the correct dimensions
    required_dims = {vartime, varlevel, varlat, varlon}
    if not required_dims.issubset(data.dims):
        raise ValueError(f"Input data must have dimensions {required_dims}, but got {data.dims}.")

    # Step 1: Calculate the zonal mean (mean along the 'lon' axis)
    zonal_mean = data.mean(dim=varlon)

    # Step 2: Subtract the zonal mean from the original data
    data_anomaly = data - zonal_mean

    # Step 3: Save to file if requested
    if output_file:
        data_anomaly.to_netcdf(output_file)
        print(f"Zonal mean removed and result saved to '{output_file}'")

    return data_anomaly


def split_zonal_wavenumbers(x, lon_axis=-1, kcut=4, return_sum_check=False):
    """
    Split a field into low- and high-zonal-wavenumber components.

    Parameters
    ----------
    x : ndarray
        Input array, shape (..., lon)
    lon_axis : int
        Longitude axis
    kcut : int
        Threshold (default 4):
        - low:  |k| < kcut  (planetary)
        - high: |k| >= kcut (synoptic)
    return_sum_check : bool
        If True, also return reconstruction error

    Returns
    -------
    x_low : ndarray
        Low wavenumber component (|k| < kcut)
    x_high : ndarray
        High wavenumber component (|k| >= kcut)
    (optional) err : float
        max reconstruction error
    """
    import numpy as np

    # FFT
    Xf = np.fft.fft(x, axis=lon_axis)

    nlon = x.shape[lon_axis]
    k = np.fft.fftfreq(nlon) * nlon
    k_abs = np.abs(k)

    # masks
    mask_low = (k_abs < kcut)
    mask_high = (k_abs >= kcut)

    # reshape for broadcasting
    shape = [1] * x.ndim
    shape[lon_axis] = nlon
    mask_low = mask_low.reshape(shape)
    mask_high = mask_high.reshape(shape)

    # apply masks
    Xf_low = Xf * mask_low
    Xf_high = Xf * mask_high

    # inverse FFT
    x_low = np.fft.ifft(Xf_low, axis=lon_axis).real.astype(np.float32)
    x_high = np.fft.ifft(Xf_high, axis=lon_axis).real.astype(np.float32)

    if return_sum_check:
        err = np.max(np.abs(x - (x_low + x_high)))
        return x_low, x_high, err

    return x_low, x_high

#%% dataset settings -------------------------------------------------------------
datasets = ["ERA5", "MERRA2", "JRA3Q"]

OUT_DIR_List = {
    "ERA5": "/scratch/bell/hu1029/Data/processed/ERA5_EKE_total_1979_2025",
    "MERRA2": "/scratch/bell/hu1029/Data/processed/MERRA2_EKE_total_1980_2025",
    "JRA3Q": "/scratch/bell/hu1029/Data/processed/JRA3Q_EKE_total_1979_2025"
}
IN_DIR_List_u = {
    "ERA5": "/depot/wanglei/data/ERA5_uvT/u_component_of_wind_*.nc",
    "MERRA2": "/scratch/bell/hu1029/Data/processed/MERRA2_UV_1dg/MERRA2_U_6hr_*.nc",
    "JRA3Q": "/scratch/bell/hu1029/Data/processed/JRA3Q_UV_1dg/JRA3Q_U_6hr_*.nc"
}
IN_DIR_List_v = {
    "ERA5": "/depot/wanglei/data/ERA5_uvT/v_component_of_wind_*.nc",
    "MERRA2": "/scratch/bell/hu1029/Data/processed/MERRA2_UV_1dg/MERRA2_V_6hr_*.nc",
    "JRA3Q": "/scratch/bell/hu1029/Data/processed/JRA3Q_UV_1dg/JRA3Q_V_6hr_*.nc"
}
varnameList_u = {
    "ERA5": "u",
    "MERRA2": "U",
    "JRA3Q": "ugrd-pres-an-ll125"
}
varnameList_v = {
    "ERA5": "v",
    "MERRA2": "V",
    "JRA3Q": "vgrd-pres-an-ll125"
}
levelnameList = {
    "ERA5": "pressure_level", # for 2024,2025 u and v in ERA5, level name is "pressure_level" instead of "level"
    "MERRA2": "lev",
    "JRA3Q": "pressure_level"
}
yearnameList = {
    "ERA5": "1979_2025",
    "MERRA2": "1980_2025",
    "JRA3Q": "1979_2025"
}

def find_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.variables or name in ds.dims:
            return name
    raise KeyError(f"Cannot find any of {candidates} in dataset. Available: {list(ds.variables)}")

# lat_name, lon_name, time_name = "lat", "lon", "valid_time" # for 2024,2025 u and v in ERA5, time_name is "valid_time" instead of "time"

#%% Calculate EKE and save to NetCDF
def process_one_year(dtname, target_year):

    # 00 define the dataset-specific settings -----------------------------------------------
    print(f"Processing dataset: {dtname}", flush=True)
    OUTDIR = OUT_DIR_List[dtname]
    IN_DIR_u = IN_DIR_List_u[dtname]
    IN_DIR_v = IN_DIR_List_v[dtname]
    varname_u = varnameList_u[dtname]
    varname_v = varnameList_v[dtname]
    # level_name = levelnameList[dtname]
    year_name = yearnameList[dtname]
    # make the output dir
    os.makedirs(OUTDIR, exist_ok=True)
    # Read multiple yearly NetCDF files for u and v components
    u_file_paths = glob.glob(IN_DIR_u)  # Replace with the path to your u files
    v_file_paths = glob.glob(IN_DIR_v)  # Replace with the path to your v files
    u_file_paths.sort()  # Ensure u files are sorted by year
    v_file_paths.sort()  # Ensure v files are sorted by year

    # Filter files for the range of target years (from year_name)
    start_year = int(year_name.split('_')[0])
    end_year = int(year_name.split('_')[1])
    print(start_year, end_year, flush=True)
    if not (start_year <= target_year <= end_year):
        print(f"{dtname}: year {target_year} outside valid range {start_year}-{end_year}", flush=True)
        return
    filtered_u_files = [
        f for f in u_file_paths
        if int(re.search(r'_(\d{4})', os.path.basename(f)).group(1)) == target_year
    ]
    filtered_v_files = [
        f for f in v_file_paths
        if int(re.search(r'_(\d{4})', os.path.basename(f)).group(1)) == target_year
    ]
    if len(filtered_u_files) != 1 or len(filtered_v_files) != 1:
        print(f"{dtname}: cannot uniquely find files for year {target_year}", flush=True)
        print("u files:", filtered_u_files, flush=True)
        print("v files:", filtered_v_files, flush=True)
        return
    
    u_file = filtered_u_files[0]
    v_file = filtered_v_files[0]

    print(filtered_u_files, flush=True)
    print('----------------------------', flush=True)

    year = int(target_year)
    # check if the output file already exists
    outfile_check = os.path.join(OUTDIR, f"planetaryEKE_{dtname}_NH_TROP_{year}.nc")
    if os.path.exists(outfile_check):
        print(f"Output file {outfile_check} already exists. Skipping year {year}.", flush=True)
        return
    
    # 01 open the dataset and get the coordinate names -----------------------------------------------
    ds_u = xr.open_dataset(u_file)
    ds_v = xr.open_dataset(v_file)
    
    # get the coordinate names for level, lat, lon, time 
    level_name = find_coord_name(ds_u, ["pressure_level", "level", "lev", "plev"])
    lat_name   = find_coord_name(ds_u, ["lat", "latitude"])
    lon_name   = find_coord_name(ds_u, ["lon", "longitude"])
    time_name  = find_coord_name(ds_u, ["valid_time", "time"])

    # daily average
    ds_u_dm = ds_u.resample({time_name: '1D'}).mean()
    ds_v_dm = ds_v.resample({time_name: '1D'}).mean()
    # times
    # time = ds_u.variables[time_name][2::4] # get the daily time points (every 4th point starting from the 3rd)
    time = ds_u_dm[time_name].values

    for hemi_key in ['NH']:

        # 01 get the coordinates and subset the levels and latitudes
        lev = ds_u.variables[level_name][:]
        lon = ds_u.variables[lon_name][:]
        lev_mask = (lev >= 100) & (lev <= 1000)
        lev_indices = np.where(lev_mask)[0]
        lev_selected = lev[lev_indices]
        sort_idx = np.argsort(lev_selected) # make sure the levels are in ascending order (from surface to upper levels)
        lev_indices = lev_indices[sort_idx]
        lat_all = ds_u.coords[lat_name].values

        # half hemisphere lat indices
        if hemi_key == 'SH':
            lat_mask = lat_all < 0
        else:
            lat_mask = lat_all > 0
        lat_indices = np.where(lat_mask)[0]

        # lat and lon values
        levi = ds_u.coords[level_name].values[lev_indices]
        lati = ds_u.coords[lat_name].values[lat_indices]
        loni = ds_u.coords[lon_name].values

        # 02 get the subset data and calculate the anomalies and EKE
        # get the subset data
        u_data = ds_u_dm.variables[varname_u][:, lev_indices, lat_indices, :].astype(np.float32)
        v_data = ds_v_dm.variables[varname_v][:, lev_indices, lat_indices, :].astype(np.float32)

        # remove zonal mean and calculate EKE
        u_anom = remove_zonal_mean(u_data, time_name, level_name, lat_name, lon_name, output_file=None)
        v_anom = remove_zonal_mean(v_data, time_name, level_name, lat_name, lon_name, output_file=None)
        energy = 0.5 * (u_anom**2 + v_anom**2)
        print(f"Calculated EKE for year {year}, now splitting into synoptic and planetary components", flush=True)
        
        # get synoptic component (high wavenumber) and large-scale component (low wavenumber)
        u_low, u_high = split_zonal_wavenumbers(u_anom, lon_axis=-1, kcut=4)
        v_low, v_high = split_zonal_wavenumbers(v_anom, lon_axis=-1, kcut=4)
        EKE_low = 0.5 * (u_low**2 + v_low**2)
        EKE_high = 0.5 * (u_high**2 + v_high**2)
        print(f"Split EKE into synoptic and planetary components for year {year}", flush=True)
        
        # 03 Create an xarray DataArray with metadata and save to NetCDF
        # var_name = "EKE"
        # ds_out = xr.Dataset(
        #     {
        #         var_name: (["time", "level", "lat", "lon"], energy.data)  # Define the variable and its dimensions
        #     },
        #     coords={
        #         "time": time,       # Time dimension
        #         "level": levi,     # Vertical level dimension
        #         "lat": lati,         # Latitude dimension
        #         "lon": loni,         # Longitude dimension
        #     },
        # )
        # outfile = os.path.join(OUTDIR, f"{dtname}_{hemi_key}_TROP_{year}.nc")
        # ds_out.to_netcdf(outfile)
        # print(f"{dtname}: Processed year {year} for hemisphere {hemi_key}, saved to {outfile}", flush=True)
        # ds_out.close()

        # synoptic component (high wavenumber)
        var_name = "EKE"
        ds_high = xr.Dataset(
            {
                var_name: (["time", "level", "lat", "lon"], EKE_high.data)  # Define the variable and its dimensions
            },
            coords={
                "time": time,       # Time dimension
                "level": levi,     # Vertical level dimension
                "lat": lati,         # Latitude dimension
                "lon": loni,         # Longitude dimension
            },
        )
        outfile = os.path.join(OUTDIR, f"synopticEKE_{dtname}_{hemi_key}_TROP_{year}.nc")
        ds_high.to_netcdf(outfile)
        ds_high.close()

        # planetary component (low wavenumber)
        var_name = "EKE"
        ds_low = xr.Dataset(
            {
                var_name: (["time", "level", "lat", "lon"], EKE_low.data)  # Define the variable and its dimensions
            },
            coords={
                "time": time,       # Time dimension
                "level": levi,     # Vertical level dimension
                "lat": lati,         # Latitude dimension
                "lon": loni,         # Longitude dimension
            },
        )
        outfile = os.path.join(OUTDIR, f"planetaryEKE_{dtname}_{hemi_key}_TROP_{year}.nc")
        ds_low.to_netcdf(outfile)
        ds_low.close()

        print(f"{dtname}: Processed year {year} for hemisphere {hemi_key}, saved to {outfile}", flush=True)

    # close datasets
    ds_u.close(); ds_v.close()
    print(f"{dtname}: Completed processing for year {year}", flush=True)

if __name__ == "__main__":
    dtname = sys.argv[1]
    year = int(sys.argv[2])
    process_one_year(dtname, year)
