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

# %% 00 prepare the environment and functions
regions = ["ATL", "NP", "SP"]
seasons = ["ALL", "DJF", "JJA"]
seasonsmonths = [[12, 1, 2], [6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
blkTypes = ["Ridge", "Trough", "Dipole"]

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

# %% 01 prepare the data --------------------------------------------------------------
# get the Z500anom, 1dg (same as LWA)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
Zanom_origin = np.array(ds['z'].squeeze())  # [0]~[-1] it's from north to south
print('-------- Zanom loaded --------', flush=True)
# lat and lon for blockings
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")

# attributes for tracks
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
tracklon_o = np.array(ds['lon'])
tracklat_o = np.array(ds['lat'])
tracklat_o = np.flip(tracklat_o) # make lat ascending -90~90

# time management -----------
times = np.array(ds['time'])
datetime_array = pd.to_datetime(times)
timei = list(datetime_array)

# %% get the composites centered at the peak
for typeid in [1,2,3]:
    for ss in seasons:
        for rgname in regions:

            # 01 read the data and attributes
            # attributes for z500
            lat_mid = int(len(lat)/2) + 1 
            if rgname == "SP":
                latLWA = lat[lat_mid:len(lat)]
                Zanom = Zanom_origin[:,lat_mid:len(lat),:] 
            else:
                latLWA = lat[0:lat_mid-1]
                Zanom = Zanom_origin[:,0:lat_mid-1,:]
            latLWA = np.flip(latLWA) # make it ascending order (from south to north)
            Zanom = np.flip(Zanom, axis=1) 
            print(latLWA, flush=True)
            print('LWA shape: ', Zanom.shape, flush=True)
            lonLWA = lon 

            # attributes for tracks
            if rgname == "SP":
                tracklat = tracklat_o[0:(findClosest(0,tracklat_o)+1)] 
            else:
                tracklat = tracklat_o[(findClosest(0,tracklat_o)+1):len(tracklat_o)] 
            print('lat for tracks:', tracklat, flush=True)
            tracklon = tracklon_o

            # track points
            CCtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/CCtrackInteracting_array_Type{typeid}_{rgname}_{ss}.npy')
            ACtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/ACtrackInteracting_array_Type{typeid}_{rgname}_{ss}.npy')
            CCtrackpoints = CCtrackpoints.astype(float)
            ACtrackpoints = ACtrackpoints.astype(float)
            print('all trackpoints loaded', flush=True)

            # get the 1st date and location of blocking events
            with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
                peakdateIndex = pickle.load(fp)
            with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayLatList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
                peakdatelatV = pickle.load(fp)
            with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayLonList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
                peakdatelonV = pickle.load(fp)
            print('peakblocking date and location loaded', flush=True)
            print(len(peakdateIndex))
            print(len(peakdatelatV))
            print(len(peakdatelonV))

            # use the peaking index to select the center LWA region using relative lats and lons
            ttblklen = len(peakdateIndex)
            centeredZ500 = []
            centeredAC = []
            centeredCC = []

            def getSlice(LWA_dayi, latid, lonid, latLWA, lonLWA, LWA_td, timeslice=False):
                LWAlatStart = latid-40 if latid-40 >= 0 else 0 
                LWAlatEnd = latid+40 if latid+40 <= len(latLWA) else len(latLWA)
                if timeslice:
                    # lat
                    # make sure the same target length
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
                    start = lonid - 80
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
                    start = lonid - 80
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

                LWA_dayi = timei.index(peakdateIndex[i])
                track_dayi = timei.index(peakdateIndex[i])
                # get the center lat and lon location (index)
                latid = findClosest(peakdatelatV[i], latLWA)
                lonid = findClosest(peakdatelonV[i], lonLWA)
                latTrackid = findClosest(peakdatelatV[i], tracklat)  # get the index of the lat
                lonTrackid = findClosest(peakdatelonV[i], tracklon)  # get the index of the lon

                # get the LWA slices -----------
                slice_Z500 = getSlice(track_dayi, latid, lonid, latLWA, lonLWA, Zanom, timeslice=True)  # get the Z500 slice
                slice_AC = getSlice(track_dayi, latTrackid, lonTrackid, tracklat, tracklon, ACtrackpoints, timeslice=True)  # get the trackpoints
                slice_CC = getSlice(track_dayi, latTrackid, lonTrackid, tracklat, tracklon, CCtrackpoints, timeslice=True)  # get the trackpoints

                centeredZ500.append(slice_Z500)
                centeredAC.append(slice_AC)  
                centeredCC.append(slice_CC)  

            centeredZ500npy = np.array(centeredZ500)
            centeredACnpy = np.array(centeredAC)
            centeredCCnpy = np.array(centeredCC)
            # four dimensions: event, relativetime(len=41), relativelat, relativelon

            np.save(f'/scratch/bell/hu1029/LGHW/CenteredZ500_timewindow41_BlkType_Type{typeid}_{rgname}_{ss}.npy', centeredZ500npy)
            np.save(f'/scratch/bell/hu1029/LGHW/CenteredAC_timewindow41_BlkType_Type{typeid}_{rgname}_{ss}.npy', centeredACnpy)
            np.save(f'/scratch/bell/hu1029/LGHW/CenteredCC_timewindow41_BlkType_Type{typeid}_{rgname}_{ss}.npy', centeredCCnpy)

            print('centered composites saved for typeid:', typeid, 'region:', rgname, 'season:', ss, flush=True)

print('composites done')

# Figure - all composites
for typeid in [1,2,3]:
    for ss in seasons:
        for rgname in regions:

            timestep = np.arange(-16,17,4)
            print(timestep, flush=True)

            ncols, nrows = 3, len(timestep)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))

            latarr = np.arange((0-40),(0+40))
            lonarr = np.arange((0-80),(0+40))
            titles = ['Ridge', 'Trough', 'Dipole']

            # axes is a 2D array with shape (11, 3)
            for i,tid in enumerate(timestep):
                ti = tid + 20  # Adjust index to match the actual position in the array (tid = -20 to 20 maps to ti = 0 to 40)
                for j in range(ncols):

                    typeid = j+1 # typeid: 1,2,3
                    idx = i * ncols + j  # Index of the current graph, from 0 to 32

                    centeredZ500npy = np.load(f'/scratch/bell/hu1029/LGHW/CenteredZ500_timewindow41_BlkType_Type{typeid}_{rgname}_{ss}.npy')
                    centeredACnpy = np.load(f'/scratch/bell/hu1029/LGHW/CenteredAC_timewindow41_BlkType_Type{typeid}_{rgname}_{ss}.npy')
                    centeredCCnpy = np.load(f'/scratch/bell/hu1029/LGHW/CenteredCC_timewindow41_BlkType_Type{typeid}_{rgname}_{ss}.npy')

                    centeredZ500npyti = np.nanmean(centeredZ500npy, axis=0)[ti,:,:]  # average over the time dimension
                    _, AC_lat_idx, AC_lon_idx = np.where(centeredACnpy[:,ti,:,:] >= 1)
                    _, CC_lat_idx, CC_lon_idx = np.where(centeredCCnpy[:,ti,:,:] >= 1)
                    
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
            plt.savefig(f'centerComposites_allpanels_1stBlkDay_extendedLon_Type{typeid}_{rgname}_{ss}.png', dpi=300)
            plt.show()
            plt.close()

            print('done with typeid:', typeid, 'region:', rgname, 'season:', ss, flush=True)
