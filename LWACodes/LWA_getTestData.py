## main code for LWA calculation from Z500 ##
#%%
import netCDF4 as nc
import glob
import numpy as np
import xarray as xr
import os
import time

LWA_td = np.load("/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr_float32.npy")
LWA_td_test = LWA_td[0:7304,:,:]  # Test with first 10 time steps
np.save("/scratch/bell/hu1029/LGHW/LWA_td_1979_1983_ERA5_6hr_float32_test.npy", LWA_td_test)

LWA_td_A = np.load("/scratch/bell/hu1029/LGHW/LWA_td_A_1979_2021_ERA5_6hr_float32.npy")
LWA_td_A_test = LWA_td_A[0:7304,:,:]  # Test with first 10 time steps
np.save("/scratch/bell/hu1029/LGHW/LWA_td_A_1979_1983_ERA5_6hr_float32_test.npy", LWA_td_A_test)

LWA_td_C = np.load("/scratch/bell/hu1029/LGHW/LWA_td_C_1979_2021_ERA5_6hr_float32.npy")
LWA_td_C_test = LWA_td_C[0:7304,:,:]  # Test with first 10 time steps
np.save("/scratch/bell/hu1029/LGHW/LWA_td_C_1979_1983_ERA5_6hr_float32_test.npy", LWA_td_C_test)


