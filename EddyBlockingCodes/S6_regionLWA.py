import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
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
from multiprocessing import Pool, Manager
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
import sys
import imageio

sys.stdout.reconfigure(line_buffering=True) # print at once in slurm

# get the LWA for the blocking and check the LWA for the track
# plot1: the map of the LWA
# plot2: the contour of the track's LWA
# plot3: the line of the LWA for both blocking and track

# %% 00 function preparation --------------------------------
regions = ["ATL", "NP", "SP"]
seasons = ["DJF", "JJA", "ALL"]
seasonsmonths = [[12, 1, 2], [6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["CC", "AC"]

def Region_ERA(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max = 45, 75, 300, 60
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max = 40, 70, 130, 250
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max = -75, -45, 180, 300
    return lat_min, lat_max, lon_min, lon_max

def PlotBoundary(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max, loncenter = 30, 90, 250, 90, 350
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max, loncenter = 30, 90, 80, 280, 180
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max, loncenter = -90, -30, 130, 330, 230

    return lat_min, lat_max, lon_min, lon_max, loncenter

def findClosest(lati, latids):

    if isinstance(lati, (np.ndarray, list)):  # if lat is an array or list
        closest_indices = []
        for l in lati:  
            diff = np.abs(l - latids)
            closest_idx = np.argmin(diff) 
            closest_indices.append(closest_idx)
        return closest_indices
    else:
        # if lat is a single value
        diff = np.abs(lati - latids)
        return np.argmin(diff) 

# %% 01 get the data ------------------------
typeid = 1
rgname = "ATL"
ss = "ALL"
cyc = "AC"

lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)

# read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

# read in the LWA data
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

# read in lat and lon for LWA
lon = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
lat = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order 90~-90
# get the lat and lon for LWA, grouped by NH and SH
lat_mid = int(len(lat)/2) + 1 
if rgname == "SP":
    latLWA = lat[lat_mid:len(lat)]
    LWA_td = LWA_td_origin[:,lat_mid:len(lat),:] 
else:
    latLWA = lat[0:lat_mid-1]
    LWA_td = LWA_td_origin[:,0:lat_mid-1,:]
latLWA = np.flip(latLWA) # make it ascending order (from south to north)
LWA_td = np.flip(LWA_td, axis=1) 
print(latLWA, flush=True)
print('LWA shape: ', LWA_td.shape, flush=True)
lonLWA = lon 

# read in the blocking event index array
blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
# transfer to 6-hourly
blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
blockingEidArr = np.repeat(blockingEidArr, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

# get the blocking days
Blockingday = np.load(f'/scratch/bell/hu1029/LGHW/Blockingday_1979_2021_Type{typeid}_{rgname}_{ss}.npy') # the day index of blockings 
print('Blockingday: ', Blockingday, flush=True)
Blocking6h_idx = np.repeat(Blockingday, 4) * 4 + np.tile(np.arange(4), len(Blockingday))

# make the region mask
rgmask = np.zeros((len(latLWA), len(lonLWA)))
lat_min_ix = np.argmin(np.abs(latLWA - lat_min))
lat_max_ix = np.argmin(np.abs(latLWA - lat_max))
lon_min_ix = np.argmin(np.abs(lonLWA - lon_min))
lon_max_ix = np.argmin(np.abs(lonLWA - lon_max))
if lon_min < lon_max:
    rgmask[lat_min_ix:lat_max_ix+1, lon_min_ix:lon_max_ix+1] = 1
else:
    # if the region crosses the 360/0 degree longitude
    rgmask[lat_min_ix:lat_max_ix+1, lon_min_ix:len(lonLWA)] = 1
    rgmask[lat_min_ix:lat_max_ix+1, 0:lon_max_ix+1] = 1

regionLWA = LWA_td * rgmask  # apply the region mask to the LWA data
dailyLWA = np.nansum(regionLWA, axis=(1, 2))  # sum over lat and lon to get the daily LWA for the region
# get the every blocking day's regional LWA
blockingLWA = dailyLWA[Blocking6h_idx]
nonblockingLWA = dailyLWA[~Blocking6h_idx]  # get the non-blocking days' regional LWA

# plot the boxplot for the blocking LWA and nonblocking LWA
plt.figure(figsize=(8, 6))
plt.boxplot([blockingLWA, nonblockingLWA], labels=['Blocking', 'Non-blocking'])
plt.title(f'Regional LWA ({rgname}) - {ss} - Type {typeid}')
plt.ylabel('LWA (m^2)')
plt.grid(True)
plt.savefig(f'BlockingRegionLWA_Boxplot_Type{typeid}_{rgname}_{ss}.png', bbox_inches='tight')
plt.close()


plt.figure(figsize=(8, 6))
sns.kdeplot(blockingLWA, label='Blocking', color='blue', linewidth=2)
sns.kdeplot(nonblockingLWA, label='Non-blocking', color='red', linewidth=2)
plt.title(f'Regional LWA PDF ({rgname}) - {ss} - Type {typeid}')
plt.xlabel('LWA (m²)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'BlockingRegionLWA_PDF_Type{typeid}_{rgname}_{ss}.png', dpi=300)
plt.close()

print('done')
