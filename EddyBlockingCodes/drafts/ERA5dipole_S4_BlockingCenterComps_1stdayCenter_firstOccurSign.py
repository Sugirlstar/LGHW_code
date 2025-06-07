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

# # 00 function --------------------------------
# def findClosest(lati, latids):

#     if isinstance(lati, np.ndarray):  # if lat is an array
#         closest_indices = []
#         for l in lati:  
#             diff = np.abs(l - latids)
#             closest_idx = np.argmin(diff) 
#             closest_indices.append(closest_idx)
#         return closest_indices
#     else:
#         # if lat is a single value
#         diff = np.abs(lati - latids)
#         return np.argmin(diff) 

# # 01 read data --------------------------------------------------------------
# # attributes for tracks
# ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
# lon = np.array(ds['lon'])
# lat = np.array(ds['lat'])
# lat = np.flip(lat)
# latNH = lat[(findClosest(0,lat)+1):len(lat)] # print(len(latNH))
# print(len(lon))

# # time management -----------
# times = np.array(ds['time'])
# datetime_array = pd.to_datetime(times)
# timei = list(datetime_array)
# LWAtimei = timei

# # get the composites centered at the peak
# for typeid in [1,2,3]:

#     # track points
#     CCtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/CCtrackPoints_TrackIDarray_inBlkType{typeid}.npy')
#     ACtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/ACtrackPoints_TrackIDarray_inBlkType{typeid}.npy')
#     print('all trackpoints loaded', flush=True)

#     # get the 1st date and location of blocking events
#     with open(f"/scratch/bell/hu1029/LGHW/ERA5dipole_Blocking1stdayDateList_blkType{typeid}", "rb") as fp:
#         peakdateIndex = pickle.load(fp)
#     with open(f"/scratch/bell/hu1029/LGHW/ERA5dipole_Blocking1stdayLatList_blkType{typeid}", "rb") as fp:
#         peakdatelatV = pickle.load(fp)
#     with open(f"/scratch/bell/hu1029/LGHW/ERA5dipole_Blocking1stdayLonList_blkType{typeid}", "rb") as fp:
#         peakdatelonV = pickle.load(fp)
#     print('peakblocking date and location loaded', flush=True)
#     print(len(peakdateIndex))
#     print(len(peakdatelatV))
#     print(len(peakdatelonV))

#     # use the peaking index to select the center LWA region using relative lats and lons
#     ttblklen = len(peakdateIndex)
#     centeredAC = []
#     centeredCC = []

#     def getSlice(LWA_dayi, latid, lonid, latLWA, lonLWA, LWA_td, timeslice=False):
#         LWAlatStart = latid-40 if latid-40 >= 0 else 0 
#         LWAlatEnd = latid+40 if latid+40 <= len(latLWA) else len(latLWA)
#         if timeslice:
#             # lat
#             # LWALatSlice = LWA_td[(LWA_dayi-20):(LWA_dayi+21),LWAlatStart:LWAlatEnd,:]
#             # make sure the same target length
#             target_length = 41  
#             time_dim = LWA_td.shape[0]
#             start_idx = LWA_dayi - 20
#             end_idx   = LWA_dayi + 21 
#             valid_start = max(start_idx, 0)
#             valid_end   = min(end_idx, time_dim)
#             pad_before = valid_start - start_idx if start_idx < 0 else 0
#             pad_after  = end_idx - valid_end if end_idx > time_dim else 0
#             valid_slice = LWA_td[valid_start:valid_end, LWAlatStart:LWAlatEnd, :]
#             LWALatSlice = np.pad(valid_slice, 
#                                     pad_width=((pad_before, pad_after), (0, 0), (0, 0)), 
#                                     mode='constant', constant_values=np.nan)

#             if latid-40 < 0:
#                 num_new_rows = int(abs(latid-40))
#                 new_rows = np.full((LWALatSlice.shape[0], num_new_rows, LWALatSlice.shape[2]), np.nan)  
#                 LWALatSlice = np.concatenate((new_rows,LWALatSlice),axis=1)
#                 print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
#             if latid+40 > len(latLWA):
#                 num_new_rows = int(abs(latid+40-len(latLWA)))
#                 new_rows = np.full((LWALatSlice.shape[0], num_new_rows, LWALatSlice.shape[2]), np.nan)
#                 LWALatSlice = np.concatenate((LWALatSlice,new_rows),axis=1)
#                 print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
#             # lon
#             start = lonid - 40
#             end = lonid + 40
#             if start < 0:  
#                 indices = list(range(start + len(lonLWA), len(lonLWA))) + list(range(0, end))
#             elif end >= len(lonLWA):
#                 indices = list(range(start, len(lonLWA))) + list(range(0, end - len(lonLWA)))
#             else:
#                 indices = list(range(start, end))
#             slice_LWA = LWALatSlice[:,:,indices]
#         else:
#             # lat
#             LWALatSlice = LWA_td[LWA_dayi,LWAlatStart:LWAlatEnd,:]  # get the lat slice
#             if latid-40 < 0:
#                 num_new_rows = int(abs(latid-40))
#                 new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)  
#                 LWALatSlice = np.vstack((new_rows,LWALatSlice))
#                 print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
#             if latid+40 > len(latLWA):
#                 num_new_rows = int(abs(latid+40-len(latLWA)))
#                 new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)
#                 LWALatSlice = np.vstack((LWALatSlice,new_rows))
#                 print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
#             # lon
#             start = lonid - 40
#             end = lonid + 40
#             if start < 0:  
#                 indices = list(range(start + len(lonLWA), len(lonLWA))) + list(range(0, end))
#             elif end >= len(lonLWA):
#                 indices = list(range(start, len(lonLWA))) + list(range(0, end - len(lonLWA)))
#             else:
#                 indices = list(range(start, end))
#             slice_LWA = LWALatSlice[:,indices]

#             print(f'sliced shape: {slice_LWA.shape}',flush=True)

#         return slice_LWA

#     for i in range(ttblklen):

#         track_dayi = timei.index(peakdateIndex[i])
#         # get the lat and lon values
#         latTrackid = findClosest(peakdatelatV[i], latNH)  # get the index of the lat
#         lonTrackid = findClosest(peakdatelonV[i], lon)  # get the index of the lon
#         # get the LWA slices -----------
#         slice_AC = getSlice(track_dayi, latTrackid, lonTrackid, latNH, lon, ACtrackpoints, timeslice=True)  # get the trackpoints
#         slice_CC = getSlice(track_dayi, latTrackid, lonTrackid, latNH, lon, CCtrackpoints, timeslice=True)  # get the trackpoints

#         centeredAC.append(slice_AC)  
#         centeredCC.append(slice_CC)  

#     centeredACnpy = np.array(centeredAC)
#     centeredCCnpy = np.array(centeredCC)
#     # four dimensions: event, relativetime(len=41), relativelat, relativelon

#     np.save('/scratch/bell/hu1029/LGHW/ERA5dipole_1stBlkDay_CenteredAC_TrackIDarr_timewindow41_BlkType'+str(typeid)+'.npy', centeredACnpy)
#     np.save('/scratch/bell/hu1029/LGHW/ERA5dipole_1stBlkDay_CenteredCC_TrackIDarr_timewindow41_BlkType'+str(typeid)+'.npy', centeredCCnpy)

# print('done')



# added figure - all composites

timestep = np.arange(-20,21,4)
print(timestep, flush=True)

ncols, nrows = 3, len(timestep)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))

latarr = np.arange((0-40),(0+40))
lonarr = np.arange((0-40),(0+40))
titles = ['Ridge', 'Trough', 'Dipole']

# axes 是一个 (11, 3) 的 2D array
seen_AC_ids = set()
seen_CC_ids = set()
for i,tid in enumerate(timestep):
    ti = tid + 20  # Adjust index to match the actual position in the array (tid = -20 to 20 maps to ti = 0 to 40)
    for j in range(ncols):

        typeid = j+1 # typeid: 1,2,3
        idx = i * ncols + j  # Index of the current graph, from 0 to 32

        centeredZ500npy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_1stBlkDay_CenteredZ500_timewindow41_BlkType'+str(typeid)+'.npy')
        centeredACnpy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_1stBlkDay_CenteredAC_TrackIDarr_timewindow41_BlkType'+str(typeid)+'.npy')
        centeredCCnpy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_1stBlkDay_CenteredCC_TrackIDarr_timewindow41_BlkType'+str(typeid)+'.npy')

        # get the points that have been seen
        centeredZ500npyti = np.nanmean(centeredZ500npy, axis=0)[ti,:,:]  # average over the time dimension
        mask = (centeredACnpy[:,ti,:,:] > 0) & np.isin(centeredACnpy[:,ti,:,:], list(seen_AC_ids))
        _, AC_lat_idx, AC_lon_idx = np.where(mask)
        mask = (centeredCCnpy[:,ti,:,:] > 0) & np.isin(centeredCCnpy[:,ti,:,:], list(seen_CC_ids))
        _, CC_lat_idx, CC_lon_idx = np.where(mask)

        # get the id that first occurs 
        ACtrackids = np.unique(centeredACnpy[:,ti,:,:])
        ACtrackids = ACtrackids[ACtrackids > 0]  # remove the zero values
        new_ac = [id for id in ACtrackids if id not in seen_AC_ids]
        seen_AC_ids.update(ACtrackids)
        newaclen = len(new_ac)
        if (newaclen > 0):
            new_ac_set = set(new_ac)
            mask_new = np.isin(centeredACnpy[:,ti,:,:], list(new_ac_set))
            _, AC_lat_idx_new, AC_lon_idx_new = np.where(mask_new)
        
        # get the id that first occurs 
        CCtrackids = np.unique(centeredCCnpy[:,ti,:,:])
        CCtrackids = CCtrackids[CCtrackids > 0] # remove the zero values
        new_cc = [id for id in CCtrackids if id not in seen_CC_ids]
        seen_CC_ids.update(CCtrackids)
        newcclen = len(new_cc)
        if (newcclen > 0):
            new_cc_set = set(new_cc)
            mask_new = np.isin(centeredCCnpy[:,ti,:,:], list(new_cc_set))
            _, CC_lat_idx_new, CC_lon_idx_new = np.where(mask_new)

        subtitle = f'newEnter AC: {newaclen}, CC: {newcclen}'
        
        ax = axes[i, j]
        cs = ax.contour(lonarr, latarr, centeredZ500npyti, levels=10, colors='k', linewidths=1.5)
        ax.scatter(lonarr[AC_lon_idx], latarr[AC_lat_idx], c='blue', marker='o', s=90, edgecolors='none', alpha=0.4)
        ax.scatter(lonarr[CC_lon_idx], latarr[CC_lat_idx], c='red', marker='o', s=90, edgecolors='none', alpha=0.5)
        if i > 0:
            if newaclen > 0:
                ax.scatter(lonarr[AC_lon_idx_new], latarr[AC_lat_idx_new], c='blue', marker='x', s=110, edgecolors='none')
            if newcclen > 0:
                ax.scatter(lonarr[CC_lon_idx_new], latarr[CC_lat_idx_new], c='red', marker='x', s=110, edgecolors='none')
        ax.clabel(cs, inline=True, fontsize=10)
        ax.set_title(subtitle, fontsize=12)

        ax.clabel(cs, inline=True, fontsize=10)
        ax.set_title(subtitle, fontsize=12)

        # row title
        if j == 0:
            if tid == 0:
                ax.set_ylabel(f'1st Block Day (Day 0)', fontsize=14)
            else:
                ax.set_ylabel(f'Day {tid/4:.0f}', fontsize=14)

        # column title
        if i == 0:
            # ax.set_title(titles[j], fontsize=13)
            fig.text(ax.get_position().x0 + ax.get_position().width / 2,
                ax.get_position().y1 + 0.01, titles[j],
                ha='center', va='bottom', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.savefig('centerComposites_allpanels_1stBlkDay_OccurSign.png', dpi=300)
plt.show()
plt.close()

print('done')
