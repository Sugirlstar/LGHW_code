import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
import math
import time

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
from scipy.ndimage import label
from scipy.interpolate import interp2d

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

def getSectorValues(data, lat_min, lat_max, lon_min, lon_max, lat, lon):
    lat_indices = (lat >= lat_min) & (lat <= lat_max)
    lon_indices = (lon >= lon_min) & (lon <= lon_max)
    if lon_min > lon_max:
        lon_indices = (lon >= lon_min) | (lon <= lon_max)
    lat_indices = np.where(lat_indices)[0]
    lon_indices = np.where(lon_indices)[0]

    lat_indices, lon_indices = np.ix_(lat_indices, lon_indices)

    region_mask = np.zeros_like(data, dtype=bool)
    region_mask[:, lat_indices, lon_indices] = True

    data_filtered = np.where(region_mask, data, 0)

    return region_mask[0,:,:], data_filtered

# 01 prepare the data --------------------------------
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily", "rb") as fp:
    Blocking_diversity_label = pickle.load(fp)   
# structure: 
# Blocking_diversity_label[0/1/2]: three types of blocks
# Blocking_diversity_label[0][...]: a single block events
# Blocking_diversity_label[0][0][0]: 2d array, True/False label for blocked or not, at each location
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily", "rb") as fp:
    Blocking_diversity_date = pickle.load(fp)      

# get lon and lat
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 #90
Blklat = lat[0:lat_mid-1]
Blklon = lon
print(Blklat)

# time management 
Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
timestamp = list(Date0['date'])
timestamparr = np.array(timestamp)

# 02 transfer into 3d label array --------------------------------
for type_idx in range(3):

    # ATL blocking id list
    ATLlist = []
    print(type_idx,flush=True)
    print('---------------------------',flush=True)
    blocking_array = np.zeros((len(timestamp), len(Blklat), len(Blklon)))
    for event_idx in range(len(Blocking_diversity_date[type_idx])):
        event_dates = Blocking_diversity_date[type_idx][event_idx] # a list of dates
        timeindex = np.where(np.isin(timestamparr, event_dates))[0]  # find the time index in the total len
        blklabelarr = np.array(Blocking_diversity_label[type_idx][event_idx])
    
        _, Sector2FG = getSectorValues(blklabelarr, 45, 75, 330, 30, Blklat, Blklon)
        flagnum = np.nansum(Sector2FG, axis=(1,2))
        if np.nansum(flagnum > 0) >= 5:
        # if np.nansum(flagnum > 180) >= 1:
            print('blocking within the ATL region',flush=True)
            ATLlist.append(event_idx)
            blocking_array[timeindex, :, :] = np.logical_or(blocking_array[timeindex, :, :],blklabelarr) # fill into the 3d array, flip along the lat axis
            # blocking_array[timeindex, :, :] = blklabelarr # fill into the 3d array, flip along the lat axis
            print('Event_idx:',event_idx,flush=True)

    np.save(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_{type_idx+1}_newATLdefine.npy", blocking_array)
    # save the ATL blocking id list
    with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_{type_idx+1}_ATLeventList_newATLdefine", "wb") as fp:
        pickle.dump(ATLlist, fp)
    print(f'blockingType{type_idx+1} total ATL frequency: {len(ATLlist)}',flush=True)

# read in the blocking numpy
BlkFlag1 = np.load("/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_1_newATLdefine.npy")
BlkFlag2 = np.load("/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_2_newATLdefine.npy")
BlkFlag3 = np.load("/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_3_newATLdefine.npy")
print('blockingarr loaded ----------------',flush=True)


# 03 filter the ATL region ---------------------------------------
def getTargetRegion(typeid, BlkFlag1):
    # 001 masked by the target region
    regionMask, Sector2FG = getSectorValues(BlkFlag1, 45, 75, 330, 30, Blklat, Blklon)
    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingFlagSector2mask_1979_2021_Type{typeid}_newATLdefine.npy', Sector2FG) # the 1 only for sector 2 blocking days, other areas set to 0
    # 002 get the blocking days in sector 2 region (at least 1 grid point blocked)
    flagnum = np.nansum(Sector2FG, axis=(1,2))
    BlockingdaySector2 = np.where(flagnum > 0)[0]
    print(f"totoal blocking length, typeid {typeid}: {len(BlockingdaySector2)}",flush=True)
    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingdaySector2_1979_2021_Type{typeid}_newATLdefine.npy', BlockingdaySector2) # the day index of blockings for sector 2
    # 003 for all the blking days, recover the whole blocking area
    Sector2FGCluster_Blk1 = np.zeros_like(Sector2FG)
    for day in BlockingdaySector2:  
        # return to the cluster label and number of clusters 
        print(day,flush=True)
        labeled_array, num_features = label(BlkFlag1[day, :, :])
        newFlag = np.zeros((len(Blklat),len(Blklon)))
        # search for each cluster    
        for region_label in range(1, num_features + 1): # start from 1
            # find all the positions of the current region
            clusterx = labeled_array == region_label
            # check if the cluster has an overlap with the region mask, if yes, keep the cluster
            maskmatch = clusterx * regionMask
            if np.sum(maskmatch) > 0:
                newFlag[np.where(clusterx==1)] = 1
        Sector2FGCluster_Blk1[day,:,:] = newFlag
    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingFlagSector2maskClusters_1979_2021_Type{typeid}_newATLdefine.npy', Sector2FGCluster_Blk1) # the 1 only for sector 2 blocking days, other areas set to 0
    print(f'blocking{typeid} clusters saved',flush=True)

    return Sector2FGCluster_Blk1

Sector2FGCluster_Blk1 = getTargetRegion(1, BlkFlag1)
Sector2FGCluster_Blk2 = getTargetRegion(2, BlkFlag2)
Sector2FGCluster_Blk3 = getTargetRegion(3, BlkFlag3)

# plot the map
# calculate the sum over time -------------------
Blk1sum = np.sum(Sector2FGCluster_Blk1, axis=0)
Blk2sum = np.sum(Sector2FGCluster_Blk2, axis=0)
Blk3sum = np.sum(Sector2FGCluster_Blk3, axis=0)

# plot the map -------------------
fig, ax, cf = create_Map(Blklon,Blklat,Blk1sum,fill=True,fig=None,
                            minv=0, maxv=round(np.nanmax(Blk1sum)), interv=11, figsize=(12,5),
                            centralLon=0, colr='PuRd', extend='max',title=f'Blocking frequency')
addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
plt.show()
plt.tight_layout()
plt.savefig(f'ERA5dipole_Blk1_Frequency_Climatology_newATLdefine.png')
plt.close()

fig, ax, cf = create_Map(Blklon,Blklat,Blk2sum,fill=True,fig=None,
                            minv=0, maxv=round(np.nanmax(Blk2sum)), interv=11, figsize=(12,5),
                            centralLon=0, colr='PuRd', extend='max',title=f'Blocking frequency')
addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
plt.show()
plt.tight_layout()
plt.savefig(f'ERA5dipole_Blk2_Frequency_Climatology_newATLdefine.png')
plt.close()

fig, ax, cf = create_Map(Blklon,Blklat,Blk3sum,fill=True,fig=None,
                            minv=0, maxv=round(np.nanmax(Blk3sum)), interv=11, figsize=(12,5),
                            centralLon=0, colr='PuRd', extend='max',title=f'Blocking frequency')
addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
plt.show()
plt.tight_layout()
plt.savefig(f'ERA5dipole_Blk3_Frequency_Climatology_newATLdefine.png')
plt.close()


#  add the peaking points ---------------------------
# readin blocking peaking data
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_date_daily", "rb") as fp:
    Blocking_diversity_peaking_date = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_lon_daily", "rb") as fp:
    Blocking_diversity_peaking_lon = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_lat_daily", "rb") as fp:
    Blocking_diversity_peaking_lat = pickle.load(fp)
    
# readin peaking event list
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_1_ATLeventList_newATLdefine", "rb") as fp:
    ATLlist1 = pickle.load(fp)
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_2_ATLeventList_newATLdefine", "rb") as fp:
    ATLlist2 = pickle.load(fp)
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_3_ATLeventList_newATLdefine", "rb") as fp:
    ATLlist3 = pickle.load(fp)
peakinglat1 = [Blocking_diversity_peaking_lat[0][i] for i in ATLlist1]
peakinglat2 = [Blocking_diversity_peaking_lat[1][i] for i in ATLlist2]
peakinglat3 = [Blocking_diversity_peaking_lat[2][i] for i in ATLlist3]
peakinglon1 = [Blocking_diversity_peaking_lon[0][i] for i in ATLlist1]
peakinglon2 = [Blocking_diversity_peaking_lon[1][i] for i in ATLlist2]
peakinglon3 = [Blocking_diversity_peaking_lon[2][i] for i in ATLlist3]

# plot the map -------------------
fig, ax, cf = create_Map(Blklon,Blklat,Blk1sum,fill=True,fig=None,
                            minv=0, maxv=round(np.nanmax(Blk1sum)), interv=11, figsize=(12,5),
                            centralLon=0, colr='PuRd', extend='max',title=f'Blocking frequency')
addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
ax.scatter(peakinglon1, peakinglat1, color='orange', s=20, marker='x', alpha=0.7, transform=ccrs.PlateCarree())
plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
plt.show()
plt.tight_layout()
plt.savefig(f'ERA5dipole_Blk1_Frequency_Climatology_WithPeakingPoints_newATLdefine.png')
plt.close()

fig, ax, cf = create_Map(Blklon,Blklat,Blk2sum,fill=True,fig=None,
                            minv=0, maxv=round(np.nanmax(Blk2sum)), interv=11, figsize=(12,5),
                            centralLon=0, colr='PuRd', extend='max',title=f'Blocking frequency')
addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
ax.scatter(peakinglon2, peakinglat2, color='orange', s=20, marker='x', alpha=0.7, transform=ccrs.PlateCarree())
plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
plt.show()
plt.tight_layout()
plt.savefig(f'ERA5dipole_Blk2_Frequency_Climatology_WithPeakingPoints_newATLdefine.png')
plt.close()

fig, ax, cf = create_Map(Blklon,Blklat,Blk3sum,fill=True,fig=None,
                            minv=0, maxv=round(np.nanmax(Blk3sum)), interv=11, figsize=(12,5),
                            centralLon=0, colr='PuRd', extend='max',title=f'Blocking frequency')
addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
ax.scatter(peakinglon3, peakinglat3, color='orange', s=20, marker='x', alpha=0.7, transform=ccrs.PlateCarree())
plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
plt.show()
plt.tight_layout()
plt.savefig(f'ERA5dipole_Blk3_Frequency_Climatology_WithPeakingPoints_newATLdefine.png')
plt.close()

