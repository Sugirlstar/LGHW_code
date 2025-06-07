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
from matplotlib.dates import DateFormatter

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

# read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

# read in lat and lon for LWA
lon = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
lat = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order 90~-90
# get the lat and lon for LWA, grouped by NH and SH
lat_mid = int(len(lat)/2) + 1 
if rgname == "SP":
    latLWA = lat[lat_mid:len(lat)]
else:
    latLWA = lat[0:lat_mid-1]
latLWA = np.flip(latLWA) # make it ascending order (from south to north)
print(latLWA, flush=True)
lonLWA = lon 

# get the event's label
if rgname == "SP":
    with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily_SH", "rb") as fp:
        Blocking_diversity_label = pickle.load(fp)
    with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily_SH", "rb") as fp:
        Blocking_diversity_date = pickle.load(fp)
else:
    with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily", "rb") as fp:
        Blocking_diversity_label = pickle.load(fp)
    with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily", "rb") as fp:
        Blocking_diversity_date = pickle.load(fp)
Blocking_diversity_label = Blocking_diversity_label[typeid-1] # get the blocking event list for the typeid
Blocking_diversity_date = Blocking_diversity_date[typeid-1] # get the blocking event date for the typeid

blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
# transfer to 6-hourly
blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
blockingEidArr = np.repeat(blockingEidArr, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

# read in the event's LWA list
with open(f"/scratch/bell/hu1029/LGHW/BlockEventDailyLWAList_1979_2021_Type{typeid}_{rgname}_{ss}.pkl", "rb") as f:
    BlkeventLWA = pickle.load(f) 

# the id of each blocking event (not all events! just the events within the target region)
with open(f'/scratch/bell/hu1029/LGHW/BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
    blkEindex = pickle.load(f)
blkEindex = np.array(blkEindex)

# read in the track's interaction information
if rgname == "SP":
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}TrackLWA_1979_2021_SH.pkl', 'rb') as file:
        LWAtrack = pickle.load(file)
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks_SH.pkl', 'rb') as file:
        track_data = pickle.load(file)
else:
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}TrackLWA_1979_2021.pkl', 'rb') as file:
        LWAtrack = pickle.load(file)
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks.pkl', 'rb') as file:
        track_data = pickle.load(file)

# EddyNumber = np.load(f'/scratch/bell/hu1029/LGHW/BlockingType{typeid}_EventEddyNumber_1979_2021_{rgname}_{ss}_{cyc}.npy')
eddyBlockIndex = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy')

# get the enter/leaving time for each eddy [the location relative to the blocking]
entertime = np.load(f'/scratch/bell/hu1029/LGHW/EnterTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy')
leavetime = np.load(f'/scratch/bell/hu1029/LGHW/LeaveTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy')
                  
# %% 02 test plotting for one event ------------------------
interactingBlkID = eddyBlockIndex
interactingBlkID = interactingBlkID[np.where(interactingBlkID >= 0)]
unique_ids, counts = np.unique(interactingBlkID, return_counts=True) # counts is the number of eddies in each block (unique_ids)
count_vals, count_freq = np.unique(counts, return_counts=True) # count_vals is the number of eddies in each block, count_freq is the frequency of each count value
for c, freq in zip(count_vals, count_freq):
    print(f"number of blocking that have {c} times eddy interaction: {freq}", flush=True)
idx = np.where(counts == 3)[0][0]
testblkid = unique_ids[idx] # the blocking with 3 eddies
print(f'Test blocking id: {testblkid}', flush=True)

# the time Dates for the target blocking event
maskday = np.any(blockingEidArr == testblkid, axis=(1, 2))  #shape=(time,) 
BlkDates = np.array(timei)[np.where(maskday)[0]] 
mind = np.min(BlkDates) # the first day of the blocking event
maxd = np.max(BlkDates) # the last day of the blocking event
BlkLWA = BlkeventLWA[np.where(blkEindex==testblkid)[0][0]] # get the LWA series for the target blocking event
print("testblkid:", testblkid, type(testblkid))

# find the eddy indices for the target blocking event
eddyindices = np.where(eddyBlockIndex == testblkid)[0] # the eddy indices for the target blocking event
print('related eddy indices:', eddyindices, flush=True)
targetTracks = [track_data[i] for i in eddyindices] # the track data for the target blocking event
TrackLWAlist = [LWAtrack[i] for i in eddyindices] # the LWA list for the target blocking event
TrackDates = [np.array([ti for ti, _, _ in pointlist]) for _, pointlist in targetTracks]
print('blocking dates:', BlkDates, flush=True)
print('track dates:', TrackDates, flush=True)
# find the enter relative timeindex
for i in range(len(TrackDates)):
    entertimei = entertime[eddyindices[i]] # the relative index
    print(entertimei, flush=True)
    enterdate = TrackDates[i][0] # the date of the eddy entering the blocking

# plot2: the process line of the LWA
fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot blocking LWA on left y-axis
ax1.fill_between(BlkDates, BlkLWA, color='tab:blue', alpha=0.3, label='Blocking LWA')
ax1.plot(BlkDates, BlkLWA, color='tab:blue', linewidth=2)  
ax1.set_xlabel('Date')
ax1.set_ylabel('Blocking LWA', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.xaxis.set_major_locator(mticker.MaxNLocator(8))
date_form = DateFormatter('%m-%d %H:%M')  #
ax1.xaxis.set_major_formatter(date_form)
fig.autofmt_xdate()
# Create right y-axis for track LWA
cols = ['orange', 'orangered', 'red']  # colors for different tracks
ax2 = ax1.twinx()
for i, track_lwa in enumerate(TrackLWAlist):
    # Some TrackLWAlist may be shorter than BlkDates, so align by length
    track_dates = TrackDates[i]
    ax2.plot(track_dates, track_lwa, label=f'Track {i+1} LWA',color=cols[i], linestyle='--', marker='o')
    ax2.axvline(track_dates[entertime[eddyindices[i]]], color=cols[i], linestyle='--', linewidth=1.5, label='Enter Time')
ax2.set_ylabel('Track LWA', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Blocking and Track LWA Time Series')
plt.tight_layout()
plt.show()
plt.savefig(f'LWAseries_Type{typeid}_{rgname}_{ss}_{cyc}_testblkid{testblkid}.png', dpi=300, bbox_inches='tight')

