import numpy as np
import datetime as dt
from datetime import date
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *

from netCDF4 import Dataset
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.colors
import os
import cartopy
from cartopy import crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
from scipy import ndimage
from multiprocessing import Pool, Manager
import cartopy.feature as cfeature
from scipy.ndimage import convolve
from scipy.signal import detrend
import pickle
import xarray as xr
import regionmask
from matplotlib.patches import Polygon
import matplotlib.path as mpath
from matplotlib.lines import Line2D

filepath = "/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc"
ds = xr.open_dataset(filepath)

z500 = ds['z'].squeeze()/9.80665 # unit: m

print(z500.shape)
# 02 calculate anomaly
zonal_mean = z500.mean(dim='lon') # zonal mean
zonal_mean_expanded = np.expand_dims(zonal_mean, axis=2) # expand the zonal mean 
Z_anomaly = z500-zonal_mean_expanded # z-zonal mean; (time,lat,lon)
print(Z_anomaly.shape,flush=True)
# get the monthly mean 
Z_anomaly_month = Z_anomaly.resample(time='1MS').mean() # monthly mean; (months,lat,lon) 
Z_anomaly_month_smoothed = Z_anomaly_month.rolling(time=3, center=True).mean() #(months,lat,lon)
print(Z_anomaly_month_smoothed.shape,flush=True)
# get the climatology
climatology = Z_anomaly_month_smoothed.groupby("time.month").mean(dim='time')
print(climatology.shape,flush=True)
# get the final Zanomaly
print(climatology['month'])
months = Z_anomaly['time'].dt.month 
print(months)
Zclimatology = climatology.sel(month=months) # expand the climatology to the same shape as the original data
Z_anomaly_corrected = Z_anomaly - Zclimatology

# 03 save data into netCDF file
# delete the original z
ds = ds.drop_vars('z')
ds['z'] = Z_anomaly_corrected
ds = ds.drop_vars('month')

# write to the nc file
ds.to_netcdf('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
print('Calculated')

# 04 plot anomaly
climatologyarr = np.array(climatology) # flip along the latitude axis
climatologyarr = np.flip(climatologyarr, axis=1)
lat = np.array(climatology['lat'])
lat = np.flip(lat)
lon = np.array(climatology['lon'])

fig = None  # No fig at first
nrows, ncols = 3, 4
axes = []
for i in range(1,nrows * ncols+1):
    fig, ax, cf = create_Polarmap(lon,lat,climatologyarr[i-1,:,:],fill=True,fig=fig, nrows=nrows, ncols=ncols, index=i, 
                                minv=-200, maxv=200, interv=11, 
                                centralLon=0, colr='seismic',
                         title=f"ZanomClim Mon {i}")
    axes.append(ax)
    addPatch(40,70,190,250,ax)
    addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='darkred',linewidth=1)
plt.colorbar(cf,ax=axes,orientation='horizontal',label='Z500 anomaly (m)',fraction=0.04, pad=0.1)

plt.show()
plt.savefig('ZanomClim12Months_1dg.png')

print('Done')

