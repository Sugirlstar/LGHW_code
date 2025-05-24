## main code for LWA calculation from Z500 ##
#%%
import netCDF4 as nc
import glob
import numpy as np
import xarray as xr
import os
import time

lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
print('lat lon loaded successfully.', flush=True)

path = "/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy"
print(f"File size: {os.path.getsize(path) / 1024**3:.2f} GB")
start = time.time()
LWA_td = np.load(path)
end = time.time()
print(f"Load time: {end - start:.2f} seconds")
print("Shape:", LWA_td.shape)
print("Dtype:", LWA_td.dtype)
print("Memory usage in RAM (estimated):", LWA_td.nbytes / 1024**3, "GB")

LWA_td_f32 = LWA_td.astype(np.float32)
np.save("/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr_float32.npy", LWA_td_f32)
print( "Converted LWA_td to float32 and saved successfully.", flush=True)
# ------
LWA_td_A = np.load("/scratch/bell/hu1029/LGHW/LWA_td_A_1979_2021_ERA5_6hr.npy")
np.save( "/scratch/bell/hu1029/LGHW/LWA_td_A_1979_2021_ERA5_6hr_float32.npy", LWA_td_A.astype(np.float32))
print("Converted LWA_td_A to float32 and saved successfully.", flush=True)
# ------
LWA_td_C = np.load("/scratch/bell/hu1029/LGHW/LWA_td_C_1979_2021_ERA5_6hr.npy")
np.save("/scratch/bell/hu1029/LGHW/LWA_td_C_1979_2021_ERA5_6hr_float32.npy", LWA_td_C.astype(np.float32))
print("Converted LWA_td_C to float32 and saved successfully.", flush=True)

#%% Plot
import matplotlib.pyplot as plt
Plot = np.nanmean(LWA_td, axis=0)  # Sum over time to get total LWA
fig = plt.figure(figsize=[12,7])
plt.contourf(lon,lat,Plot, 50, extend="both", cmap='Reds') 
cb=plt.colorbar()

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('LWAtotalMean_test.png')  
plt.show()
plt.close()
