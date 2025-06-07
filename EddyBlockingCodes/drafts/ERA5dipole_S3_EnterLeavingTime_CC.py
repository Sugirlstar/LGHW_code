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
from scipy.stats import pearsonr

def findCloset(lat,latids):
    closest_indices = []
    for latid in latids:
        diff = np.abs(lat - latid)
        closest_idx = np.argmin(diff)
        closest_indices.append(closest_idx)
    return closest_indices

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

def extendList(desired_length,centorinlist,test):
    targetlist = np.full(desired_length*2+1, np.nan)
    targetlist[desired_length] = test[centorinlist]
    leftlen = centorinlist
    rightlen = len(test) - leftlen -1
    left_padding = [np.nan] * (desired_length - leftlen)
    left_values = left_padding + list(test[np.nanmax([0, leftlen-desired_length]):centorinlist])
    right_padding = [np.nan] * (desired_length - rightlen)
    if rightlen-desired_length>0:
        rightend = centorinlist+1+desired_length
    else: rightend = len(test)
    right_values = list(test[centorinlist+1:rightend]) + right_padding
    targetlist[:desired_length] = left_values
    targetlist[desired_length+1:] = right_values
    return targetlist

# 01 - read data -------------------------------------------------------------
# lon and lat for the track
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
lon = np.array(ds['lon'])
lat = np.array(ds['lat'])
lat = np.flip(lat)
lat = lat[(findClosest(0,lat)+1):len(lat)]
# time management
times = np.array(ds['time'])
datetime_array = pd.to_datetime(times)
timei = list(datetime_array)
# lat and lon for blocking
lonBLK = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
latBLK = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order
latBLK = latBLK[0:90]
latBLK = np.flip(latBLK) # make it ascending order (from south to north)
print(latBLK)

# get all the tracks
with open('/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks.pkl', 'rb') as file:
    track_data = pickle.load(file)

for typeid in [1,2,3]:
    
    # blocking persistence
    Sec2BlockPersis = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EventPersisSector2_1979_2021.npy')
    # get the blocking index each track contribute to
    BlockIndexSec2 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipole_TrackBlockingType{typeid}_IndexSector2_1979_2021_CC.npy') # the related blocking event id, -1 represent no blocking related
    # get the blocking event index list
    with open(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EventListSector2_1979_2021.pkl', 'rb') as file:
        Sec2BlockEvent = pickle.load(file)
    blkdays = sum(len(sublist) for sublist in Sec2BlockEvent)
    print('total blocking day length:',blkdays)

    # the arr of the blocking event index
    BlockingEIndexSec2 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingFlagSector2maskClusters_1979_2021_Type{typeid}_newATLdefine.npy')
    BlockingEIndexSec2 = np.flip(BlockingEIndexSec2, axis=1) # flip the lat
    BlockingEIndexSec2 = np.repeat(BlockingEIndexSec2, 4, axis=0)

    # the number of blocking events within the Sec2 region --------------------------------
    numBlockEvent = len(Sec2BlockPersis)
    # define if the block is related to a track
    BlockwithTrack = 0
    BlockwithoutTrack = 0
    for i in range(numBlockEvent):
        if i in BlockIndexSec2:
            BlockwithTrack += 1
        else:
            BlockwithoutTrack += 1

    # the number of all the tracks that has ever enter into the Sec2 region --------------------------------
    lat_min = 45; lat_max = 75; lon_min = 330; lon_max = 30
    lat_indices = (lat >= lat_min) & (lat <= lat_max)
    lon_indices = (lon >= lon_min) & (lon <= lon_max)
    if lon_min > lon_max:
        lon_indices = (lon >= lon_min) | (lon <= lon_max)
    lat_indices = np.where(lat_indices)[0]
    lon_indices = np.where(lon_indices)[0]
    lat_indices, lon_indices = np.ix_(lat_indices, lon_indices)
    region_mask = np.zeros((len(lat), len(lon)), dtype=bool)
    region_mask[lat_indices, lon_indices] = True

    n = 0
    for _,pointlist in track_data:
        latids = [lati for _, _, lati in pointlist]
        latids = findCloset(lat,latids)
        lonids = [loni for _, loni, _ in pointlist]
        lonids = findCloset(lon,lonids)
        for k in range(len(latids)):
            if region_mask[latids[k], lonids[k]]:
                n = n+1
                break

    print('Number of CC tracks that has ever enter into the Sec2 region:')
    print(n)
    print('Number of CC tracks that has ever interact with blocking:')
    print(np.sum(BlockIndexSec2 != -1))
    print('done')

    # calculate the correlation between the blocking persistence and the related track numbers --------------------------------
    tknumlist = []
    for k in range(len(Sec2BlockPersis)):
        tknum = np.sum(BlockIndexSec2 == k)
        tknumlist.append(tknum)
    tknumlist = np.array(tknumlist)
    # remove nan values
    valid_mask = ~np.isnan(tknumlist) & ~np.isnan(Sec2BlockPersis)
    tknumlist_clean = tknumlist[valid_mask]
    Sec2BlockPersis_clean = Sec2BlockPersis[valid_mask]

    corr, p_value = pearsonr(tknumlist_clean, Sec2BlockPersis_clean)

    print(f'correlation coefficient between the blocking type{typeid} persistence and the related CC track numbers:')
    print(corr,flush=True)
    print(f'p-value: {p_value}',flush=True)

    # # calculate the distribution of entering and leaving time of the blocking events, relative to the blocking's life cycle --------------------------------
    # intopercentList = []
    # leavepercentList = []
    # targetTrackIndexCommon = range(len(track_data))
    # for k,i in enumerate(targetTrackIndexCommon):
    #     blockid = BlockIndexSec2[i]
    #     if blockid >=0:
    #         realtimeindex = Sec2BlockEvent[blockid]
    #         track = track_data[i] # the target track's information
    #         # tracklwa = LWAtrack_Sec2[i] # the target track's LWA
    #         _, pointlist = track # get the point list
    #         times = [ti for ti, _, _ in pointlist]
    #         latids = [lati for _, _, lati in pointlist]
    #         latids = findCloset(latBLK,latids)
    #         lonids = [loni for _, loni, _ in pointlist]
    #         lonids = findCloset(lonBLK,lonids)
    #         timeids = [timei.index(j) for j in times] # the time index in the real all time list
    #         # print(times)
    #         # print([lati for _, _, lati in pointlist])
    #         # print([loni for _, loni, _ in pointlist])
    #         # print('timeids:',timeids)
    #         # print('latids:',latids)
    #         # print('lonids:',lonids)
    #         blockingValue = BlockingEIndexSec2[np.array(timeids), np.array(latids), np.array(lonids)] # True or False
    #         print(blockingValue)
    #         firstinto = np.argmax(blockingValue) # the first index of blockingValue == True
    #         lastleave = np.where(blockingValue)[0][-1] # the last index of blockingValue == True
    #         # find these two points in the realtimeindex (relative position)
    #         firstinto_realtime = timeids[firstinto]
    #         lastleave_realtime = timeids[lastleave]
    #         intopercent = np.searchsorted(realtimeindex, firstinto_realtime) / len(realtimeindex)
    #         leavepercent = np.searchsorted(realtimeindex, lastleave_realtime) / len(realtimeindex)

    #         intopercentList.append(intopercent)
    #         leavepercentList.append(leavepercent)

    # bins = np.arange(0, 1.1, 0.1) 
    # plt.hist(intopercentList, bins=bins, alpha=0.5, color='orangered', label='Enter Time', edgecolor='black', align='mid')
    # plt.hist(leavepercentList, bins=bins, alpha=0.5, color='royalblue', label='Leaving Time', edgecolor='black', align='mid')
    # plt.xlabel('Relative Time in the Blocking Life Cycle')
    # plt.ylabel('Count')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'ERA5dipole_CC_BlockingEventEnterLeaveTimeDistribution_blkType{typeid}.png')
    # plt.close()

    # # group by the type of the tracks
    # InterTypeSec2 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_InteractionTypeSec2_1979_2021_CC.npy')
    # Throughid = np.where(InterTypeSec2 == 'T')[0]
    # Absorbedid = np.where(InterTypeSec2 == 'A')[0]
    # Edgeid = np.where(InterTypeSec2 == 'E')[0]
    # print(f'number of CC through: {len(Throughid)}')
    # print(f'number of CC absorbed: {len(Absorbedid)}')
    # print(f'number of CC edge: {len(Edgeid)}')

    # intopercentList = []
    # leavepercentList = []
    # targetTrackIndexCommon = Absorbedid
    # for k,i in enumerate(targetTrackIndexCommon):
    #     blockid = BlockIndexSec2[i]
    #     if blockid >=0:
    #         realtimeindex = Sec2BlockEvent[blockid]
    #         track = track_data[i] # the target track's information
    #         # tracklwa = LWAtrack_Sec2[i] # the target track's LWA
    #         _, pointlist = track # get the point list
    #         times = [ti for ti, _, _ in pointlist]
    #         latids = [lati for _, _, lati in pointlist]
    #         latids = findCloset(latBLK,latids)
    #         lonids = [loni for _, loni, _ in pointlist]
    #         lonids = findCloset(lonBLK,lonids)
    #         timeids = [timei.index(j) for j in times] # the time index in the real all time list
    #         blockingValue = BlockingEIndexSec2[np.array(timeids), np.array(latids), np.array(lonids)] # True or False
    #         firstinto = np.argmax(blockingValue) # the first index of blockingValue == True
    #         lastleave = np.where(blockingValue)[0][-1] # the last index of blockingValue == True
    #         # find these two points in the realtimeindex (relative position)
    #         firstinto_realtime = timeids[firstinto]
    #         lastleave_realtime = timeids[lastleave]
    #         intopercent = np.searchsorted(realtimeindex, firstinto_realtime) / len(realtimeindex)
    #         leavepercent = np.searchsorted(realtimeindex, lastleave_realtime) / len(realtimeindex)

    #         intopercentList.append(intopercent)
    #         leavepercentList.append(leavepercent)

    # bins = np.arange(0, 1.1, 0.1) 
    # plt.hist(intopercentList, bins=bins, alpha=0.5, color='orangered', label='Enter Time', edgecolor='black', align='mid')
    # plt.hist(leavepercentList, bins=bins, alpha=0.5, color='royalblue', label='Leaving Time', edgecolor='black', align='mid')
    # plt.xlabel('Relative Time in the Blocking Life Cycle')
    # plt.ylabel('Count')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'ERA5dipole_BlockingEventEnterLeaveTimeDistribution_AbsorbedTracks_blkType{typeid}.png')
    # plt.close()

