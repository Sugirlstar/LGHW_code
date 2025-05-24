import xarray as xr
import numpy as np
from windspharm.standard import VectorWind
from tqdm import tqdm
import sys
sys.stdout.reconfigure(line_buffering=True) # print at once in slurm


ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_geopotential500_subtractseasonal_6hr_1979_2021.nc')
Z = ds['var129'].values / 9.80665  # 转换为高度，单位：米
lats = ds['lat'].values
lons = ds['lon'].values
nt, nlat, nlon = Z.shape

Z_filtered = np.full_like(Z, np.nan)

def spectral_filter_scalar(field2d, truncate_below=5):
    vwt = VectorWind(np.zeros_like(field2d), field2d) 
    _, spec = vwt._grdtospec(vwt.vcomp)
    spec[:truncate_below, :] = 0
    return vwt._spectogrd(spec)

for t in tqdm(range(nt), desc='Filtering Z500'):
    Z_filtered[t] = spectral_filter_scalar(Z[t])

np.save('ERA5_geopotentialheight500_T42filtered_6hr_1979_2021_F128.npy', Z_filtered)

print('done')
