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
import cv2
import copy
from collections import defaultdict

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

def get_lon_width(finalLabel,nlon):

    _, x_idx = np.where(finalLabel > 0)
    x_sorted = np.sort(np.unique(x_idx))
    gaps = np.diff(np.concatenate((x_sorted, [x_sorted[0] + nlon])))
    max_gap_idx = np.argmax(gaps)
    west_idx = (x_sorted[(max_gap_idx + 1) % len(x_sorted)]) % nlon
    east_idx = (x_sorted[max_gap_idx]) % nlon
    if west_idx > east_idx:
        lon_width = 360 - west_idx + east_idx
    else:
        lon_width = east_idx - west_idx 
    return lon_width

def getSliceSingle(LWA_dayi, latid, lonid, latLWA, lonLWA, LWA_td, 
                latup=40, latdown=40, lonleft=80, lonright=40):
    
    LWAlatStart = latid-latdown if latid-latdown >= 0 else 0 
    LWAlatEnd = latid+latup if latid+latup <= len(latLWA) else len(latLWA)
    LWALatSlice = LWA_td[LWA_dayi, LWAlatStart:LWAlatEnd, :]

    if latid-latdown < 0:
        num_new_rows = abs(latid - latdown)
        new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)
        LWALatSlice = np.vstack((new_rows, LWALatSlice))
        print(f'Added {num_new_rows} rows at the top. Now shape: {LWALatSlice.shape}', flush=True)

    if latid + latup > len(latLWA):
        num_new_rows = latid + latup - len(latLWA)
        new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)
        LWALatSlice = np.vstack((LWALatSlice, new_rows))
        print(f'Added {num_new_rows} rows at the bottom. Now shape: {LWALatSlice.shape}', flush=True)

    # lon
    start = lonid - lonleft
    end = lonid + lonright
    if start < 0:  
        indices = list(range(start + len(lonLWA), len(lonLWA))) + list(range(0, end))
    elif end >= len(lonLWA):
        indices = list(range(start, len(lonLWA))) + list(range(0, end - len(lonLWA)))
    else:
        indices = list(range(start, end))
    slice_LWA = LWALatSlice[:,indices]

    return slice_LWA

# read in the LWA data
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

# ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
# Zanom_origin = np.array(ds['z'].squeeze())  # [0]~[-1] it's from north to south
# print(np.shape(Zanom_origin), flush=True)
# print('-------- Zanom loaded --------', flush=True)
# # attributes for tracks
# lon = np.array(ds['lon'])
# lat = np.array(ds['lat']) # -90~90
# print(lat, flush=True)

# # %% 01 get the LWA events ------------------------
# # read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

for rgname in ["ATL"]:

    dstrack = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
    tracklon = np.array(ds['lon'])
    tracklat = np.array(ds['lat'])
    tracklat = np.flip(tracklat)
    if rgname == "SP":
        tracklat = tracklat[0:(findClosest(0,tracklat)+1)]
    else:
        tracklat = tracklat[(findClosest(0,tracklat)+1):len(tracklat)]

    lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
    lat_min1, lat_max1, lon_min1, lon_max1, loncenter = PlotBoundary(rgname)
    print(lat_min1, lat_max1, lon_min1, lon_max1)

    if rgname == "SP":
        HMi = '_SH'
    else:
        HMi = ''

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

    # get the wave event
    T = LWA_td.shape[0]
    print('Total number of timesteps:', T, flush=True)

    T4 = T // 4
    LWA_Z = LWA_td[:T4*4]
    LWA_Z = LWA_Z.reshape(T4, 4, len(latLWA), len(lonLWA)).mean(axis=1) # daily mean of LWA

    print('LWA_Z loaded, shape: ', LWA_Z.shape, flush=True)

    # # get the Z500 anomaly data -----------------
    # # attributes for track array and blockings
    # lat_mid = int(len(lat)/2) + 1 
    # if rgname == "SP":
    #     Blklat = lat[0:lat_mid-1]
    #     Zanom = Zanom_origin[:, 0:lat_mid-1, :]
    # else:
    #     Blklat = lat[lat_mid:len(lat)]
    #     Zanom = Zanom_origin[:, lat_mid:len(lat), :]

    # T = Zanom.shape[0]
    # print('Total number of timesteps:', T, flush=True)

    # T4 = T // 4
    # Zanom_Z = Zanom[:T4*4]
    # Zanom_Z = Zanom_Z.reshape(T4, 4, len(latLWA), len(lonLWA)).mean(axis=1) # daily mean of LWA

ss = 'ALL'
rgname = "ATL"


# %% read in necessary data for the composite
RealEmbryoIDs = np.load(f'/scratch/bell/hu1029/LGHW/embryo_RealEmbryoIDs_{rgname}.npy')

embryoArr = np.load(f'/scratch/bell/hu1029/LGHW/EmbryoIndexArrayDaily_{rgname}.npy') # the embryo event index array, daily value
# get the days of the embryo events
nday = np.shape(embryoArr)[0] # number of days (not 6-hourly)
embryovalues = np.unique(embryoArr)  # get the unique embryo values
embryovalues = embryovalues[embryovalues >= 0]  # remove the -1 values
print(embryovalues, flush=True)  # print the unique embryo values
embryo_id_to_days = defaultdict(list)
for t in range(embryoArr.shape[0]):
    if np.any(embryoArr[t] >= 0):
        ids = np.unique(embryoArr[t])
        for eid in ids:
            if eid >= 0:
                embryo_id_to_days[int(eid)].append(t)

# # %% make the composite of embryo v.s. blocking events ------------------------

# latarr = np.arange((0-40),(0+40))
# lonarr = np.arange((0-80),(0+40))

# embryoCompArr = np.zeros((len(RealEmbryoIDs), len(latarr), len(lonarr)))  # the composite array for the embryo events
# for k, ebyroid in enumerate(RealEmbryoIDs):
#     print(f'Embryo ID: {ebyroid}', flush=True)
#     emdays = embryo_id_to_days[ebyroid]  # get the days of the embryo event
#     lastday = emdays[0]  # the 1st day of the embryo event
#     # get the day slice of the embryo event
#     embryoslice = embryoArr[lastday]
#     mask0 = embryoslice == ebyroid # transform the slice to a boolean array
#     center0 = np.unravel_index(np.argmax(LWA_Z[lastday,:,:] * mask0, axis=None), LWA_Z[lastday,:,:].shape)
#     centerlat = center0[0]  # the latitude index of the center
#     centerlon = center0[1]  # the longitude index of the center

#     # get the LWA slices -----------
#     slice_k = getSliceSingle(lastday, centerlat, centerlon, latLWA, lonLWA, Zanom_Z)  # get the Z500 slice
#     embryoCompArr[k,:,:] = slice_k

#     if k % 100 == 0:
#         plt.figure(figsize=(12, 8))
#         ax = plt.subplot(1, 1, 1)
#         cs = ax.contour(lonarr, latarr, slice_k, levels=10, colors='k', linewidths=1.5)
#         ax.clabel(cs, inline=True, fontsize=10, fmt='%1.1f')
#         plt.tight_layout()
#         plt.subplots_adjust(hspace=0.3, wspace=0.2)
#         plt.savefig(f'embryo_centerCase{k}_{rgname}.png', dpi=300)
#         plt.show()
#         plt.close()

# centeredEmbryo = np.nanmean(embryoCompArr, axis=0)  # average over the time dimension

# plt.figure(figsize=(12, 8))
# ax = plt.subplot(1, 1, 1)
# cs = ax.contour(lonarr, latarr, centeredEmbryo, levels=10, colors='k', linewidths=1.5)
# ax.clabel(cs, inline=True, fontsize=10, fmt='%1.1f')
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.3, wspace=0.2)
# plt.savefig(f'embryo_centerComposites_{rgname}.png', dpi=300)
# plt.show()
# plt.close()

# %% check AC and CC probability ------------------------
# for each type of embryo, get the AC and CC related to it.
# for each related event, get the probability of if it successfully develops into a blocking event
SuccssEmbryoID = np.load(f'/scratch/bell/hu1029/LGHW/embryo_SuccssEmbryoID_{rgname}.npy')  # the embryo IDs that successfully develop into a blocking event
all_included = np.all(np.isin(SuccssEmbryoID, RealEmbryoIDs))
if not all_included:
    print('Warning: Not all successful embryo IDs are included in the real embryo IDs.', flush=True)

embryoArr = np.repeat(embryoArr, 4, axis=0)  # transfer to 6-hourly index

AC_trackPoints_array = np.load(f'/scratch/bell/hu1029/LGHW/ACtrackPoints_array{HMi}.npy')
CC_trackPoints_array = np.load(f'/scratch/bell/hu1029/LGHW/CCtrackPoints_array{HMi}.npy')
# AC_trackPoints_array_thinned = AC_trackPoints_array[::4, :, :] # get the daily value
# CC_trackPoints_array_thinned = CC_trackPoints_array[::4, :, :] # get the daily value

# test plot the CC points
theday = 411*4 # 1980-02-16 00:00
timedate = timei[theday]
print('test date:', timedate, flush=True)
embryoArrtest = embryoArr[theday]
print(np.unique(embryoArrtest), flush=True)  # print the unique embryo values in the slice
print('it should be in the RidgeSuccssEmbryo')

mask0 = embryoArrtest >= 0 # transform the slice to a boolean array
fig, ax, cf = create_Map(lonLWA,latLWA,LWA_td[theday,:,:],fill=True,fig=None,
                    leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
                        minv=0, maxv=np.nanmax(LWA_td[theday,:,:]), interv=11, figsize=(12,5),
                        centralLon=loncenter, colr='PuBu', extend='max',title=f'{timei[theday]}')
ax.contour(lonLWA, latLWA, mask0.astype(float), levels=[0.5], 
            colors='darkblue', linewidths=1.5, transform=ccrs.PlateCarree())
AAlat,AAlon = np.where(AC_trackPoints_array[theday]>=1)
AAlatvalue = tracklat[AAlat]
AAlonvalue = tracklon[AAlon]
ax.scatter(AAlonvalue, AAlatvalue, color='blue', s=20, label='AC points', transform=ccrs.PlateCarree(),zorder=10)

plt.show()
plt.savefig(f'testACtrackPoint.png', bbox_inches='tight', dpi=300)
plt.close()

# get the successful blocking event number for each blocking type
successNum1 = np.load(f'/scratch/bell/hu1029/LGHW/embryo_SuccessNumbersEachEmbryo_BlkType1_{rgname}.npy', )
successNum2 = np.load(f'/scratch/bell/hu1029/LGHW/embryo_SuccessNumbersEachEmbryo_BlkType2_{rgname}.npy', )
successNum3 = np.load(f'/scratch/bell/hu1029/LGHW/embryo_SuccessNumbersEachEmbryo_BlkType3_{rgname}.npy', )
# the global ids of the embryo that successfully develop into a blocking event for type 1, 2, 3
success1ids = np.where(successNum1 > 0)[0]  
success2ids = np.where(successNum2 > 0)[0]  
success3ids = np.where(successNum3 > 0)[0] 

withTrackID = []
withACID = []
withCCID = []
relatedACnumber = []
relatedCCnumber = []
withTrackID2 = []
withTrackID3 = []

ExcludedEmbryoID = [eid for eid in embryovalues if eid not in RealEmbryoIDs]
successNum = np.load(f'/scratch/bell/hu1029/LGHW/embryo_SuccessNumbersEachEmbryo_BlkType1_{rgname}.npy')
successNum[ExcludedEmbryoID] = 0 
RidgeSuccssEmbryo = np.where(successNum > 0)[0]  # the global ids of the embryo that successfully develop into a blocking event for type 1
print("RidgeEmbryo:", RidgeSuccssEmbryo, flush=True)
all_included = np.all(np.isin(RidgeSuccssEmbryo, RealEmbryoIDs))
if not all_included:
    print('Warning: Not all successful Ridge embryo IDs are included in the real embryo IDs.', flush=True)

successNum = np.load(f'/scratch/bell/hu1029/LGHW/embryo_SuccessNumbersEachEmbryo_BlkType2_{rgname}.npy')
successNum[ExcludedEmbryoID] = 0 
TroughSuccssEmbryo = np.where(successNum > 0)[0]  # the global ids of the embryo that successfully develop into a blocking event for type 1
print("DipoleEmbryo:", TroughSuccssEmbryo, flush=True)
all_included = np.all(np.isin(TroughSuccssEmbryo, RealEmbryoIDs))
if not all_included:
    print('Warning: Not all successful Trough embryo IDs are included in the real embryo IDs.', flush=True)

# timeiday = timei[::4]
for k, ebyroid in enumerate(RealEmbryoIDs):
    print(f'Embryo ID: {ebyroid}', flush=True)
    emdays = embryo_id_to_days[ebyroid]  # get the days of the embryo event
    emdaysfirstinhour = emdays[0]*4  # the 1st day of the embryo event, transfer to 6-hourly index
    emdays = np.arange(emdaysfirstinhour, emdaysfirstinhour + 4*3)  # get the first 3 days

    ACnumber = 0
    CCnumber = 0

    for i in np.arange(len(emdays)):
        theday = emdays[i]  # the ist day of the embryo event
        dateid = timei[theday]
        # get the day slice of the embryo event
        embryoslice = embryoArr[theday]
        mask0 = embryoslice == ebyroid # transform the slice to a boolean array
        center0 = np.unravel_index(np.argmax(LWA_td[theday,:,:] * mask0, axis=None), LWA_td[theday,:,:].shape)
        centerlat = center0[0]  # the latitude index of the center
        centerlon = center0[1]  # the longitude index of the center
        print(f'Center latitude index: {centerlat}, Center longitude index: {centerlon}', flush=True)
        centerlatvalue = latLWA[centerlat]  # the latitude value of the center
        centerlonvalue = lonLWA[centerlon]  # the longitude value of the center
        print(f'Center latitude value: {centerlatvalue}, Center longitude value: {centerlonvalue}', flush=True)

        centerlatidx = findClosest(centerlatvalue, tracklat)  # get the latitude index of the center
        centerlonidx = findClosest(centerlonvalue, tracklon)  # get the longitude index of the center

        ACdaySlice = getSliceSingle(theday, centerlatidx, centerlonidx, tracklat, tracklon, AC_trackPoints_array, 
                                    latup=10, latdown=10, lonleft=20, lonright=0)
        CCdaySlice = getSliceSingle(theday, centerlatidx, centerlonidx, tracklat, tracklon, CC_trackPoints_array, 
                                    latup=10, latdown=10, lonleft=20, lonright=0)

        # get the total number of AC and CC points
        ACnumber = ACnumber + np.nansum(ACdaySlice)
        CCnumber = CCnumber + np.nansum(CCdaySlice)

        # if k in [0,50,100,150,200]:  # only plot the first 8 embryos as examples
                
        #     # make the case plot to see the embryo
        #     AAlatloc, AAlatlon = np.where(ACdaySlice > 0)
        #     CClatloc, CClonloc = np.where(CCdaySlice > 0)
        #     AAlatloc = AAlatloc + centerlatidx -10
        #     CClatloc = CClatloc + centerlatidx -10
        #     AAlonloc = AAlatlon + centerlonidx -30
        #     CClonloc = CClonloc + centerlonidx -30

        #     AAlonvalue = tracklon[AAlonloc]
        #     AAlatvalue = tracklat[AAlatloc]
        #     CClonvalue = tracklon[CClonloc]
        #     CClatvalue = tracklat[CClatloc]

        #     print(f'AC points latitude value: {AAlatvalue}, longitude value: {AAlonvalue}', flush=True)
        #     print(f'CC points latitude value: {CClatvalue}, longitude value: {CClonvalue}', flush=True)

        #     if ebyroid in SuccssEmbryoID:
        #         titletag = '_Blocked'
        #     else:
        #         titletag = ''

        #     fig, ax, cf = create_Map(lonLWA,latLWA,LWA_td[theday,:,:],fill=True,fig=None,
        #                         leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
        #                             minv=0, maxv=np.nanmax(LWA_td[theday,:,:]), interv=11, figsize=(12,5),
        #                             centralLon=loncenter, colr='PuBu', extend='max',title=f'{timei[theday]}')
        #     ax.contour(lonLWA, latLWA, mask0.astype(float), levels=[0.5], 
        #                     colors='darkblue', linewidths=1.5, transform=ccrs.PlateCarree())
        #     addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1)  
        #     if np.nansum(ACdaySlice) > 0 or np.nansum(CCdaySlice) > 0:
        #         ax.scatter(AAlonvalue, AAlatvalue, color='blue', s=20, label='AC points', transform=ccrs.PlateCarree(),zorder=10)
        #         ax.scatter(CClonvalue, CClatvalue, color='red', s=20, label='CC points', transform=ccrs.PlateCarree(),zorder=10)
            
        #     plt.show()
        #     plt.savefig(f'embryoCaseCheck_ID{ebyroid}{titletag}_day{i}.png', bbox_inches='tight', dpi=300)
        #     plt.close()

    print(f'Embryo ID: {ebyroid}, AC number: {ACnumber}, CC number: {CCnumber}', flush=True)

    relatedACnumber.append(ACnumber)
    relatedCCnumber.append(CCnumber)

    if ACnumber > 0 or CCnumber > 0:
        withTrackID.append(ebyroid)
    if ACnumber + CCnumber >= 2:
        withTrackID2.append(ebyroid)
    if ACnumber + CCnumber >= 3:
        withTrackID3.append(ebyroid)
    if ACnumber > 0:
        withACID.append(ebyroid)
    if CCnumber > 0:
        withCCID.append(ebyroid)

# calculate the probability of  Block | withEddy, Block | withoutEddy
print('total number of embryos:', len(RealEmbryoIDs), flush=True)
print('total number of embryos with track:', len(withTrackID), flush=True)
print(f'probability of blocking developed from an embryo: {len(SuccssEmbryoID)/len(RealEmbryoIDs)}', flush=True)
common_ids = np.intersect1d(SuccssEmbryoID, withTrackID)
print(f'probability of blocking developed from an embryo with track: {len(common_ids)/len(withTrackID)}', flush=True)
common_ids2 = np.intersect1d(SuccssEmbryoID, withTrackID2)
common_ids3 = np.intersect1d(SuccssEmbryoID, withTrackID3)
print(f'probability of blocking developed from an embryo with 2 or more tracks: {len(common_ids2)/len(withTrackID2)}', flush=True)
print(f'probability of blocking developed from an embryo with 3 or more tracks: {len(common_ids3)/len(withTrackID3)}', flush=True)

common_ridgeAC = np.intersect1d(RidgeSuccssEmbryo, withACID)
print(f'probability of Ridge blocking developed from an embryo with AC: {len(common_ridgeAC)/len(withACID)}', flush=True)
common_troughCC = np.intersect1d(TroughSuccssEmbryo, withCCID)
print(f'probability of Trough blocking developed from an embryo with CC: {len(common_troughCC)/len(withCCID)}', flush=True)
