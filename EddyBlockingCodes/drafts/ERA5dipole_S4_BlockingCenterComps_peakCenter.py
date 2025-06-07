# get the density of the tracks during blocking / non-blocking days and plot the map

import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
import math

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
import imageio
from scipy import stats
from collections import defaultdict
import dask

# 00 function --------------------------------
def findClosest(lati, latids):

    if isinstance(lati, np.ndarray):  # if lat is an array
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

# 01 read data --------------------------------------------------------------
# attributes for tracks
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
lon = np.array(ds['lon'])
lat = np.array(ds['lat'])
lat = np.flip(lat)
latNH = lat[(findClosest(0,lat)+1):len(lat)] # print(len(latNH))
print(len(lon))

# get the Z500anom, 1dg (same as LWA)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
Zanom = np.array(ds['z'].squeeze())  # [0]~[-1] it's from north to south
Zanom = Zanom[:,91:181,:] 
print('-------- Zanom loaded --------', flush=True)

# read in LWA
LWA_td = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td = LWA_td/100000000 # change the unit to 1e8 
LWA_td = LWA_td[:,0:90,:] # keep only the NH
LWA_td = np.flip(LWA_td, axis=1) # make it from north to south
print('-------- LWA loaded --------', flush=True)
print(LWA_td.shape, flush=True)
# blocking lat and lon
lonLWA = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
latLWA = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order
latLWA = latLWA[0:90] # keep only the NH
latLWA = np.flip(latLWA) # make it ascending order (from south to north)
print(latLWA, flush=True)
print(LWA_td.shape, flush=True) # check the shape

# time management -----------
times = np.array(ds['time'])
datetime_array = pd.to_datetime(times)
timei = list(datetime_array)
LWAtimei = timei

# get the composites centered at the peak
for typeid in [1,2,3]:

    # track points
    CCtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/CCtrackPoints_array_inBlkType{typeid}.npy')
    ACtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/ACtrackPoints_array_inBlkType{typeid}.npy')
    print('all trackpoints loaded', flush=True)

    # get the peaking date and location of blocking events
    with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingPeakDateList_blkType{typeid}.pkl", "rb") as fp:
        peakdateIndex = pickle.load(fp)
    with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingPeakDateLatList_blkType{typeid}.pkl", "rb") as fp:
        peakdatelatV = pickle.load(fp) # lat
    with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingPeakDateLonList_blkType{typeid}.pkl", "rb") as fp:
        peakdatelonV = pickle.load(fp) # lon
    print('peakblocking date and location loaded', flush=True)

    # use the peaking index to select the center LWA region using relative lats and lons
    ttblklen = len(peakdateIndex)
    centeredLWA = []  # create an empty list to store the centered LWA
    centeredZ500 = []
    centeredAC = []
    centeredCC = []

    def getSlice(LWA_dayi, latid, lonid, latLWA, lonLWA, LWA_td, timeslice=False):
        LWAlatStart = latid-40 if latid-40 >= 0 else 0 
        LWAlatEnd = latid+40 if latid+40 <= len(latLWA) else len(latLWA)
        if timeslice:
            # lat
            # LWALatSlice = LWA_td[(LWA_dayi-20):(LWA_dayi+21),LWAlatStart:LWAlatEnd,:]
            # make sure the same target length
            target_length = 41  
            time_dim = LWA_td.shape[0]
            start_idx = LWA_dayi - 20
            end_idx   = LWA_dayi + 21 
            valid_start = max(start_idx, 0)
            valid_end   = min(end_idx, time_dim)
            pad_before = valid_start - start_idx if start_idx < 0 else 0
            pad_after  = end_idx - valid_end if end_idx > time_dim else 0
            valid_slice = LWA_td[valid_start:valid_end, LWAlatStart:LWAlatEnd, :]
            LWALatSlice = np.pad(valid_slice, 
                                    pad_width=((pad_before, pad_after), (0, 0), (0, 0)), 
                                    mode='constant', constant_values=np.nan)

            if latid-40 < 0:
                num_new_rows = int(abs(latid-40))
                new_rows = np.full((LWALatSlice.shape[0], num_new_rows, LWALatSlice.shape[2]), np.nan)  
                LWALatSlice = np.concatenate((new_rows,LWALatSlice),axis=1)
                print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
            if latid+40 > len(latLWA):
                num_new_rows = int(abs(latid+40-len(latLWA)))
                new_rows = np.full((LWALatSlice.shape[0], num_new_rows, LWALatSlice.shape[2]), np.nan)
                LWALatSlice = np.concatenate((LWALatSlice,new_rows),axis=1)
                print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
            # lon
            start = lonid - 40
            end = lonid + 40
            if start < 0:  
                indices = list(range(start + len(lonLWA), len(lonLWA))) + list(range(0, end))
            elif end >= len(lonLWA):
                indices = list(range(start, len(lonLWA))) + list(range(0, end - len(lonLWA)))
            else:
                indices = list(range(start, end))
            slice_LWA = LWALatSlice[:,:,indices]
        else:
            # lat
            LWALatSlice = LWA_td[LWA_dayi,LWAlatStart:LWAlatEnd,:]  # get the lat slice
            if latid-40 < 0:
                num_new_rows = int(abs(latid-40))
                new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)  
                LWALatSlice = np.vstack((new_rows,LWALatSlice))
                print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
            if latid+40 > len(latLWA):
                num_new_rows = int(abs(latid+40-len(latLWA)))
                new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)
                LWALatSlice = np.vstack((LWALatSlice,new_rows))
                print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
            # lon
            start = lonid - 40
            end = lonid + 40
            if start < 0:  
                indices = list(range(start + len(lonLWA), len(lonLWA))) + list(range(0, end))
            elif end >= len(lonLWA):
                indices = list(range(start, len(lonLWA))) + list(range(0, end - len(lonLWA)))
            else:
                indices = list(range(start, end))
            slice_LWA = LWALatSlice[:,indices]

            print(f'sliced shape: {slice_LWA.shape}',flush=True)

        return slice_LWA

    for i in range(ttblklen):

        LWA_dayi = LWAtimei.index(peakdateIndex[i])
        track_dayi = timei.index(peakdateIndex[i])
        # get the lat and lon values
        latid = findClosest(peakdatelatV[i], latLWA)
        lonid = findClosest(peakdatelonV[i], lonLWA)
        latTrackid = findClosest(peakdatelatV[i], latNH)  # get the index of the lat
        lonTrackid = findClosest(peakdatelonV[i], lon)  # get the index of the lon

        # get the LWA slices -----------
        slice_LWA = getSlice(LWA_dayi, latid, lonid, latLWA, lonLWA, LWA_td, timeslice=True)  # get the LWA slice
        slice_Z500 = getSlice(track_dayi, latid, lonid, latLWA, lonLWA, Zanom, timeslice=True)  # get the Z500 slice
        slice_AC = getSlice(track_dayi, latTrackid, lonTrackid, latNH, lon, ACtrackpoints, timeslice=True)  # get the trackpoints
        slice_CC = getSlice(track_dayi, latTrackid, lonTrackid, latNH, lon, CCtrackpoints, timeslice=True)  # get the trackpoints

        centeredLWA.append(slice_LWA) 
        centeredZ500.append(slice_Z500)
        centeredAC.append(slice_AC)  
        centeredCC.append(slice_CC)  

    centeredLWAnpy = np.array(centeredLWA)
    centeredZ500npy = np.array(centeredZ500)
    centeredACnpy = np.array(centeredAC)
    centeredCCnpy = np.array(centeredCC)
    # four dimensions: event, relativetime(len=41), relativelat, relativelon

    np.save('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredLWA_timewindow41_BlkType'+str(typeid)+'.npy', centeredLWAnpy)
    np.save('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredZ500_timewindow41_BlkType'+str(typeid)+'.npy', centeredZ500npy)
    np.save('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredAC_timewindow41_BlkType'+str(typeid)+'.npy', centeredACnpy)
    np.save('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredCC_timewindow41_BlkType'+str(typeid)+'.npy', centeredCCnpy)

    centeredLWAnpy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredLWA_timewindow41_BlkType'+str(typeid)+'.npy')
    centeredZ500npy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredZ500_timewindow41_BlkType'+str(typeid)+'.npy')
    centeredACnpy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredAC_timewindow41_BlkType'+str(typeid)+'.npy')
    centeredCCnpy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredCC_timewindow41_BlkType'+str(typeid)+'.npy')

    # get each point location, not sum
    # centeredAC: total length 0-20 to 0+20, timsesteps 5: -20, -15, -10, -5, 0, 5, 10, 15, 20, np.arange(-20,21,5) 
    timestep = np.arange(-20,21,4)
    print(timestep, flush=True)

    latarr = np.arange((0-40),(0+40))
    lonarr = np.arange((0-40),(0+40))
    def makeMap(lonarr, latarr, centeredLWAnpy, AC_lat_idx, AC_lon_idx, CC_lat_idx, CC_lon_idx, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        cs = ax.contour(lonarr, latarr, centeredLWAnpy, levels=10, colors='k', linewidths=1.5)
        ax.scatter(lonarr[AC_lon_idx], latarr[AC_lat_idx], c='blue', marker='o', s=90, edgecolors='none', alpha=0.6)
        ax.scatter(lonarr[CC_lon_idx], latarr[CC_lat_idx], c='red', marker='o', s=90, edgecolors='none', alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=12)
        # plt.scatter(0, 0, color='red', s=100)
        ax.set_xlabel('relative longitude')
        ax.set_ylabel('relative latitude')
        plt.show()
        plt.savefig(filename, dpi=300)
        plt.close()

    for tid in timestep:

        ti = tid+20
        
        centeredLWAnpyti = np.nanmean(centeredLWAnpy, axis=0)[ti,:,:]  # average over the time dimension
        centeredZ500npyti = np.nanmean(centeredZ500npy, axis=0)[ti,:,:]  # average over the time dimension
        _, AC_lat_idx, AC_lon_idx = np.where(centeredACnpy[:,ti,:,:] == 1)
        _, CC_lat_idx, CC_lon_idx = np.where(centeredCCnpy[:,ti,:,:] == 1)
        print(f'AC_lat_idx: {AC_lat_idx}, AC_lon_idx: {AC_lon_idx}', flush=True)
        print(f'CC_lat_idx: {CC_lat_idx}, CC_lon_idx: {CC_lon_idx}', flush=True)

        makeMap(lonarr, latarr, centeredLWAnpyti, AC_lat_idx, AC_lon_idx, CC_lat_idx, CC_lon_idx, f'ERA5dipole_ACtrackdensityLWA_CenterOverlap_BlkType{typeid}_day{tid/4}.png')
        makeMap(lonarr, latarr, centeredZ500npyti, AC_lat_idx, AC_lon_idx, CC_lat_idx, CC_lon_idx, f'ERA5dipole_ACtrackdensityZ500anom_CenterOverlap_BlkType{typeid}_day{tid/4}.png')

print('done')


# added figure - all composites

timestep = np.arange(-20,21,4)
print(timestep, flush=True)

ncols, nrows = 3, len(timestep)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))

latarr = np.arange((0-40),(0+40))
lonarr = np.arange((0-40),(0+40))
titles = ['Ridge', 'Trough', 'Dipole']

# axes 是一个 (11, 3) 的 2D array
for i,tid in enumerate(timestep):
    ti = tid + 20  # Adjust index to match the actual position in the array (tid = -20 to 20 maps to ti = 0 to 40)
    for j in range(ncols):

        typeid = j+1 # typeid: 1,2,3
        idx = i * ncols + j  # Index of the current graph, from 0 to 32

        centeredZ500npy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredZ500_timewindow41_BlkType'+str(typeid)+'.npy')
        centeredACnpy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredAC_timewindow41_BlkType'+str(typeid)+'.npy')
        centeredCCnpy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_CenteredCC_timewindow41_BlkType'+str(typeid)+'.npy')

        centeredZ500npyti = np.nanmean(centeredZ500npy, axis=0)[ti,:,:]  # average over the time dimension
        _, AC_lat_idx, AC_lon_idx = np.where(centeredACnpy[:,ti,:,:] == 1)
        _, CC_lat_idx, CC_lon_idx = np.where(centeredCCnpy[:,ti,:,:] == 1)
        
        ax = axes[i, j]
        cs = ax.contour(lonarr, latarr, centeredZ500npyti, levels=10, colors='k', linewidths=1.5)
        ax.scatter(lonarr[AC_lon_idx], latarr[AC_lat_idx], c='blue', marker='o', s=90, edgecolors='none', alpha=0.6)
        ax.scatter(lonarr[CC_lon_idx], latarr[CC_lat_idx], c='red', marker='o', s=90, edgecolors='none', alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=10)

        # row title
        if j == 0:
            ax.set_ylabel(f'Day {tid/4:.0f}', fontsize=12)

        # column title
        if i == 0:
            ax.set_title(titles[j], fontsize=13)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.savefig('centerComposites_allpanels_peakingBlkDay.png', dpi=300)
plt.show()
plt.close()

print('done')
