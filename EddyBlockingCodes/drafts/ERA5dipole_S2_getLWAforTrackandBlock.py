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
sys.stdout.reconfigure(line_buffering=True) # print at once in slurm

# get the area weight over the NH
R = 6371  # in km
nlat = 90
nlon = 360
lats = np.linspace(90 - 0.5*180/nlat, 0 + 0.5*180/nlat, nlat)
lons = np.linspace(0, 360, nlon, endpoint=False)
dlat = np.deg2rad(180 / nlat)
dlon = np.deg2rad(360 / nlon)
lat_radians = np.deg2rad(lats)
lat_radians_2d = np.repeat(lat_radians[:, np.newaxis], nlon, axis=1)
area_grid = (R**2) * dlat * dlon * np.cos(lat_radians_2d)
area_weights = area_grid / np.sum(area_grid)

# print(area_weights[:,1])
# print(np.sum(area_weights))
# print(area_weights.shape)
# lon_grid, lat_grid = np.meshgrid(lons, lats)
# plt.figure(figsize=(12, 5))
# plt.pcolormesh(lon_grid, lat_grid, area_weights, shading='auto')
# plt.title('Normalized Area Weights (Northern Hemisphere)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.colorbar(label='Weight')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig('areaweights.png')
# plt.show()

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

# 01 load the data -------------------------------------------------------------
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

# track lat and lon
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
lon = np.array(ds['lon'])
lat = np.array(ds['lat'])
lat = np.flip(lat)
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)


# 02 get LWA for blocking events per day -----------------------------------------------
for typeid in [1,2,3]:

    # blockings
    blockingSec2 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingFlagSector2maskClusters_1979_2021_Type{typeid}_newATLdefine.npy') # the blocking flag in sector 2, True only for blocked area in Sector 2
    # transfer to 6-hourly
    blockingSec2 = np.flip(blockingSec2, axis=1) # flip the lat
    blockingSec2 = np.repeat(blockingSec2, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)
    print(blockingSec2.shape, flush=True)
    blklwadaily = blockingSec2 * LWA_td # * area_weights # the LWA of each point
    blklwadaily1dvalue = np.nansum(blklwadaily, axis=(1, 2)) # the LWA of each day
    blklwadaily1dvalue[blklwadaily1dvalue == 0] = np.nan # set non-blocked days to nan
    print(blklwadaily1dvalue[0:500], flush=True)
    LWAblock_Sec2 = np.array(blklwadaily1dvalue) # LWA sum value for each blocked day

    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockedDayLWA_1979_2021_Type{typeid}.npy', LWAblock_Sec2)

    # plot the pdf of LWA for blocked days
    plt.figure()
    sns.kdeplot(LWAblock_Sec2, fill=True)
    plt.title('Probability Density Function (PDF)')
    plt.xlabel('LWA')
    plt.ylabel('Density')
    plt.show()
    plt.savefig(f'ERA5dipoleDaily_ATL_blockedDay_LWA_PDF_Type{typeid}.png')

LWAblock_Sec1 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockedDayLWA_1979_2021_Type1.npy')
LWAblock_Sec2 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockedDayLWA_1979_2021_Type2.npy')
LWAblock_Sec3 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockedDayLWA_1979_2021_Type3.npy')
plt.figure()
sns.kdeplot(LWAblock_Sec1, fill=True, color='blue', alpha=0.5, label='Ridge')
sns.kdeplot(LWAblock_Sec2, fill=True, color='red', alpha=0.5, label='Low-pressure')
sns.kdeplot(LWAblock_Sec3, fill=True, color='orange', alpha=0.5, label='Dipole')
plt.title('Probability Density Function (PDF)')
plt.xlabel('LWA')
plt.ylabel('Density')
plt.legend()  
plt.savefig(f'ERA5dipoleDaily_blockedDay_LWA_3PDFs_3Types.png')
plt.show()

# 03 calculate the tracks'LWA -------------------------------------------------------------
def getTrackLWA(track_data):
    
    LWAtrack_Sec2 = []
    for index, pointlist in track_data:
        print('-------------------',flush=True)
        print(index,flush=True)

        times = [ti for ti, _, _ in pointlist]
        latids = [lati for _, _, lati in pointlist]
        lonids = [loni for _, loni, _ in pointlist]
        timeids = [timei.index(i) for i in times]

        latinLWA = findClosest(np.array(latids), latLWA)
        loninLWA = findClosest(np.array(lonids), lonLWA)

        eachTrackLWA=[]
        for j in range(len(latinLWA)):

            lat_idx = latinLWA[j]
            lon_idx = loninLWA[j]
            timeid = timeids[j]
            radius = 5
            # get the range
            lat_range = slice(max(lat_idx - radius, 0), min(lat_idx + radius + 1, LWA_td.shape[1]))
            lon_range = [(lon_idx - i) % 360 for i in range(radius, -radius-1, -1)]
            if j<5:
                print(lat_idx, flush=True)
                print(lon_idx, flush=True)
                print(lat_range, flush=True)
                print(lon_range, flush=True)
            # extracted
            extracted_data = LWA_td[timeid, lat_range, lon_range]
            lwa_sum = np.nansum(extracted_data)
            eachTrackLWA.append(lwa_sum) # a list of each day's LWA

        LWAtrack_Sec2.append(eachTrackLWA)

    return LWAtrack_Sec2

with open('/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks.pkl', 'rb') as file:
    track_data_CC = pickle.load(file)
with open('/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks.pkl', 'rb') as file:
    track_data_AC = pickle.load(file)

LWAtrack_CC = getTrackLWA(track_data_CC)
LWAtrack_AC = getTrackLWA(track_data_AC)
print(LWAtrack_CC[0], flush=True)
print(LWAtrack_AC[1], flush=True)

with open('/scratch/bell/hu1029/LGHW/ERA5dipole_CCTrackLWA_1979_2021.pkl', 'wb') as file:
    pickle.dump(LWAtrack_CC, file)
with open('/scratch/bell/hu1029/LGHW/ERA5dipole_ACTrackLWA_1979_2021.pkl', 'wb') as file:
    pickle.dump(LWAtrack_AC, file)

# plot the pdf of CC and AC LWA
flat_LWAtrack_CC = [np.nanmean(sublist) for sublist in LWAtrack_CC if len(sublist) > 0]
flat_LWAtrack_AC = [np.nanmean(sublist) for sublist in LWAtrack_AC if len(sublist) > 0]

plt.figure()
sns.kdeplot(flat_LWAtrack_CC, fill=True)
plt.title('Probability Density Function (PDF)')
plt.xlabel('TRACK LWA')
plt.ylabel('Density')
plt.show()
plt.savefig('ERA5dipole_CCtracksLWA_PDF.png')

plt.figure()
sns.kdeplot(flat_LWAtrack_AC, fill=True)
plt.title('Probability Density Function (PDF)')
plt.xlabel('TRACK LWA')
plt.ylabel('Density')
plt.show()
plt.savefig('ERA5dipole_ACtracksLWA_PDF.png')

plt.figure()
sns.kdeplot(flat_LWAtrack_CC, fill=True, color='blue', alpha=0.5, label='CC')
sns.kdeplot(flat_LWAtrack_AC, fill=True, color='orange', alpha=0.5, label='AC')
plt.title('Probability Density Function (PDF)')
plt.xlabel('TRACK LWA')
plt.ylabel('Density')
plt.legend()  
plt.savefig('ERA5dipole_CCandACtrackLWA_2PDF.png')
plt.show()



# 05 plot the LWA time series of block with eddies - one block event as a sample -----------------------------------
# get the blocking index each track contribute to
BlockIndexSec2 = np.load('/scratch/bell/hu1029/LGHW/ERA5_TrackBlockingIndexSector2_1979_2021.npy') # the related blocking event id, -1 represent no blocking related
# get the blocking event index list
with open('/scratch/bell/hu1029/LGHW/ERA5_BlockingEventListSector2_1979_2021.pkl', 'rb') as file:
    Sec2BlockEvent = pickle.load(file)

BlockingEIndexSec2 = np.load('/scratch/bell/hu1029/LGHW/ERA5_BlockingEventIndexArrSector2_1979_2021.npy')

# get the blocking with most contributing tracks
unique_values, counts = np.unique(BlockIndexSec2, return_counts=True)
mask = unique_values != -1
filtered_values = unique_values[mask]
filtered_counts = counts[mask]
max_count = np.max(filtered_counts)
most_frequent_value = filtered_values[np.argmax(filtered_counts)]
print(most_frequent_value) # this is the index of the blocking event
random_values = np.random.choice(filtered_values, size=15, replace=True)
random_values[0] = most_frequent_value # replace using the most frequent value

for m in random_values:
    print('-------------------',flush=True)
    print(m,flush=True)
    # find the index of the track
    indices_of_max_value = np.where(BlockIndexSec2 == m)[0]
    print(indices_of_max_value)
    # find the index of the blocking event
    targetblockDaysIndex = Sec2BlockEvent[m]
    print(targetblockDaysIndex) # a time series

    # get the time range of the plot
    totaltracktimeid = []
    for tkindex in indices_of_max_value:
        _, pointlist = track_data[tkindex]
        times = [ti for ti, _, _ in pointlist]
        TrackTimeids = [timei.index(i) for i in times]
        print(TrackTimeids)
        totaltracktimeid.extend(TrackTimeids)
    max_time_id = max(totaltracktimeid)
    min_time_id = min(totaltracktimeid)
    max_time_id = max(max_time_id,max(targetblockDaysIndex))
    min_time_id = min(min_time_id,min(targetblockDaysIndex))
    stid = min_time_id - 5
    edid = max_time_id + 5

    # make the plot
    trackcolor = ['red','blue','green','purple','orange','pink','brown','grey','black','yellow','cyan','magenta','olive','lime','teal','aqua','maroon','navy','silver','gold','crimson','darkred','darkorange','darkgreen','darkblue','darkviolet','darkcyan','darkmagenta','darkolive','darklime','darkteal','darkaqua','darkmaroon','darknavy','darksilver','darkgold','darkcrimson']
    fig, ax1 = plt.subplots()

    ax1.plot(np.arange(stid,edid+1), LWAblock_Sec2[stid:edid+1], color='black', linestyle='-', label='Blocking')
    ax1.set_xlabel('TimeIndex')
    ax1.set_ylabel('Blocking LWA', color='black')
    ax1.set_ylim(0, 1.1*np.nanmax(LWAblock_Sec2[stid:edid+1]))
    ax1.tick_params(axis='y', labelcolor='black')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Track LWA', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    alltracksLWA = [LWAtrack_Sec2[tkindex] for tkindex in indices_of_max_value]
    flat_LWAtrack_Sec2 = np.array([value for sublist in alltracksLWA for value in sublist])
    ax2.set_ylim(0, 1.1*np.nanmax(flat_LWAtrack_Sec2))

    k = 0
    # maybe more than 1 track
    for tkindex in indices_of_max_value:
        _, pointlist = track_data[tkindex]
        track_LWA = LWAtrack_Sec2[tkindex] # the lwa values of the track
        times = [ti for ti, _, _ in pointlist]
        TrackTimeids = [timei.index(i) for i in times] # the time index of the track
        latids = [lati for _, _, lati in pointlist]
        latids = findClosest(np.array(latids),lat) # latindex
        lonids = [loni for _, loni, _ in pointlist]
        lonids = findClosest(np.array(lonids),lon) # lonindex
        blockingValue = BlockingEIndexSec2[np.array(TrackTimeids), np.array(latids), np.array(lonids)] # arr of blocking values of each track grid
        print(blockingValue)
        blockingif = blockingValue > 0 # a mask of track/blocking interaction
        track_LWA_masked = track_LWA * blockingif # mask the track LWA with blocking interaction
        track_LWA_masked[np.where(blockingif == 0)] = np.nan # set non-blocking interaction to nan

        ax2.plot(TrackTimeids, track_LWA, color=trackcolor[k], linestyle='--', label=f'Track {k+1}')
        ax2.plot(TrackTimeids, track_LWA_masked, 'o', color=trackcolor[k], label=f'Track {k+1} in Blocking')  

        k=k+1

    fig.legend(loc='upper right')
    plt.show()
    plt.savefig(f'Sample{m}_LWA_Timeseries.png')

