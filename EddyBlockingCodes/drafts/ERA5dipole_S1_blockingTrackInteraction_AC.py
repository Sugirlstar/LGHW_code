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

# plot the scatter plot 
def ScatterPlot(persistence,blkarea,eddynumber,Figname):
    q1, q2, q3 = np.percentile(persistence, [25, 50, 75])
    # correlation calculation
    corr_persistence_area = np.corrcoef(persistence, blkarea)[0, 1]
    corr_persistence_eddynumber = np.corrcoef(persistence, eddynumber)[0, 1]
    corr_area_eddynumber = np.corrcoef(blkarea, eddynumber)[0, 1]

    bounds = [0, 1, 2, 3, 4, 5, 6] # colorbar bounds
    cmap = ListedColormap(plt.cm.YlOrRd(np.linspace(0, 1, len(bounds))))  # divide the color map into 6 parts
    norm = BoundaryNorm(bounds, ncolors=cmap.N, extend='max')

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        persistence, blkarea, c=eddynumber, cmap=cmap, norm=norm)

    cbar = plt.colorbar(scatter, extend="max")
    cbar.set_label("Eddy Number")

    plt.xticks([0, 20, 40, 60], labels=["0", "20", "40", "60"])
    plt.yticks(range(1, 7))
    plt.xlabel("Persistence")
    plt.ylabel("Area")

    plt.axvline(q1, color="gray", linestyle="--", label=f"Q1 = {q1:.1f}")
    plt.axvline(q2, color="gray", linestyle="--", label=f"Q2 = {q2:.1f}")
    plt.axvline(q3, color="gray", linestyle="--", label=f"Q3 = {q3:.1f}")

    corr_text = (
        f"Corr(N, P) = {corr_persistence_eddynumber:.2f}\n"
        f"Corr(N, A) = {corr_area_eddynumber:.2f}\n"
        f"Corr(P, A) = {corr_persistence_area:.2f}"
    )
    plt.text(0.95, 0.95, corr_text, fontsize=10, transform=plt.gca().transAxes,
            ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))

    plt.title("Scatter Plot: Persistence vs Area with Eddy Number Coloring")
    plt.grid(alpha=0.3)
    plt.show()
    plt.savefig(Figname+'.png')
# Groups of consecutive day index (list of arr), and the length of each group
def getDurationArea(pos_hw, sumareaSec1):
    #input:
    # pos_hw - the positions of Blocking days
    diff = np.diff(pos_hw)
    group_boundaries = np.where(diff > 1)[0] + 1  
    # get the length of each group
    groups = np.split(pos_hw, group_boundaries)
    group_lengths = [len(group)/4 for group in groups]
    group_areas = []
    for i in groups:
        group_areas.append(np.nanmean(sumareaSec1[i]))
    return groups, group_lengths, group_areas
# get the closest index of the lat/lon
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

# 01 read data --------------------------------------------------------------
# attributes for TRACKs
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
lon = np.array(ds['lon'])
lat = np.array(ds['lat'])
lat = np.flip(lat)
latNH = lat[(findClosest(0,lat)+1):len(lat)]
# lat and lon for blockings
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 #90
Blklat = lat[0:lat_mid-1]
Blklat = np.flip(Blklat)
Blklon = lon
# Time management
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)
print(len(timei))
# load tracks
with open('/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks.pkl', 'rb') as file:
    track_data = pickle.load(file)
print('track loaded-----------------------',flush=True)

# 02 blocking statistics --------------------------------------------------------------
# blockings
for typeid in [1,2,3]:

    print('############################ processing blocking type',typeid,flush=True)

    # 001 transfer to 6-hourly
    blockingSec2 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingFlagSector2maskClusters_1979_2021_Type{typeid}_newATLdefine.npy') # the blocking value (z500) in ATL
    # transfer to 6-hourly
    blockingSec2 = np.flip(blockingSec2, axis=1) # flip the lat
    blockingSec2 = np.repeat(blockingSec2, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

    # 002 blocking statistics
    blockingSec2Days = np.where(np.any(blockingSec2 != 0, axis=(1, 2)))[0]
    print(f'blocking{typeid} days total length:', len(blockingSec2Days),flush=True)
    # blocking area calculation
    sumareaSec2 = np.nansum((blockingSec2>0), axis=(1,2)) # the blocking area of each day
    # identify blocking events
    Sec2BlockEvent, Sec2BlockPersis, _ = getDurationArea(blockingSec2Days, sumareaSec2)
    # get the id of each blocking event
    BlockingEIndexSec2 = np.full_like(blockingSec2, np.nan, dtype=float)
    for i in range(len(Sec2BlockPersis)):
        eventindex = Sec2BlockEvent[i] # the indices of the day when each blocking event occur
        ti, row_indices, col_indices = np.where(blockingSec2[eventindex, :, :] > 0)
        BlockingEIndexSec2[eventindex[ti],row_indices,col_indices] = i

    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EventIndexArrSector2_1979_2021.npy', BlockingEIndexSec2)
    with open(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EventListSector2_1979_2021.pkl', 'wb') as file:
        pickle.dump(Sec2BlockEvent, file)
    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EventPersisSector2_1979_2021.npy', np.array(Sec2BlockPersis))
    print('blocking event identified and saved-----------------',flush=True)

    # 03 define four scenarios --------------------------------------------------------------
    # for each single track, check its start and end time
    Sec2BlockPersis = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EventPersisSector2_1979_2021.npy')
    BlockingEIndexSec2 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EventIndexArrSector2_1979_2021.npy')
    Sec2BlockEvent = pickle.load(open(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EventListSector2_1979_2021.pkl', 'rb'))

    EddyNumberSec2 = [0] * len(Sec2BlockPersis) # a list of the eddy number that each block is related to; = length of Sec2BlockPersis
    BlockIndexSec2 = [-1] * len(track_data) # a list of the blockingid that each track is related to; = length of track_data
    InterTypeSec2 = ['N'] * len(track_data) # a list of the interaction type that each track is related to

    def eddyBlockInteraction(track_data,BlockingEIndexSec1,EddyNumberSec1,BlockIndex,tpIndex):
        ThroughTrack = []
        EdgeTrack = []
        AbsorbedTrack = []
        InternalTrack = []
        SpawnedTrack = []
        tracknum = 0

        for index, pointlist in track_data:

            print('-------------------',flush=True)
            print(index,flush=True)
            tp = 'N'
            bkids = -1

            times = [ti for ti, _, _ in pointlist]
            if any(ti not in timei for ti in times):
                print("Found an element not in timei, continue...")
                BlockIndex[tracknum] = bkids
                tpIndex[tracknum] = tp
                tracknum += 1
                continue

            # get the lat and lon id for blockings
            latids = [lati for _, _, lati in pointlist]
            latids = findCloset(Blklat,latids)
            lonids = [loni for _, loni, _ in pointlist]
            lonids = findCloset(Blklon,lonids)
            timeids = [timei.index(i) for i in times]
            blockingValue = BlockingEIndexSec1[np.array(timeids), np.array(latids), np.array(lonids)]

            # 001 Through (include Edge)
            if np.isnan(blockingValue[0]) and np.isnan(blockingValue[-1]):
                if np.any(blockingValue >= 0):
                    indices = np.where(blockingValue >= 0)[0]
                    if np.all(np.diff(indices) == 1):
                        print(blockingValue,flush=True)
                        ThroughTrack.append(index)
                        bkEindex = np.unique(blockingValue) # the blockingid that the track is related to
                        bkEindex = bkEindex[~np.isnan(bkEindex)].astype(int)
                        bkids = bkEindex[0] # only get the first one that interacts with
                        for nn in bkEindex:
                            EddyNumberSec1[nn] += 1
                        tp = 'T'
                        print(f'{bkids}: Through Identified',flush=True)
            
            # 002 Edge
            if np.isnan(blockingValue[0]) and np.isnan(blockingValue[-1]):
                if np.any(blockingValue >= 0):
                    if not np.all(np.diff(indices) == 1):
                        print(blockingValue,flush=True)
                        EdgeTrack.append(index)
                        bkEindex = np.unique(blockingValue) # the blockingid that the track is related to
                        bkEindex = bkEindex[~np.isnan(bkEindex)].astype(int)
                        bkids = bkEindex[0] # only get the first one that interacts with
                        for nn in bkEindex:
                            EddyNumberSec1[nn] += 1
                        tp = 'E'
                        print(f'{bkids}: Edge Identified',flush=True)
            
            # 003 Absorbed
            if np.isnan(blockingValue[0]) and blockingValue[-1] >= 0:
                print(blockingValue,flush=True)
                AbsorbedTrack.append(index)
                bkEindex = np.unique(blockingValue)
                bkEindex = bkEindex[~np.isnan(bkEindex)].astype(int)
                print(bkEindex,flush=True)
                bkids = bkEindex[0]
                for nn in bkEindex:
                    EddyNumberSec1[nn] += 1
                tp = 'A'
                print(f'{bkids}: Absorbed Identified',flush=True)

            # 004 Internal
            if blockingValue[0] >= 0:
                print(f'{bkids}: Internal or Spawned Identified',flush=True)

            BlockIndex[tracknum] = bkids
            tpIndex[tracknum] = tp
            tracknum += 1

        FilteredIndex = ThroughTrack + EdgeTrack + AbsorbedTrack  # all the contributing tracks id
        print(BlockIndex)

        return EddyNumberSec1, BlockIndex, FilteredIndex, ThroughTrack, AbsorbedTrack, EdgeTrack, tpIndex

    EddyNumberSec2, BlockIndexSec2, _, ThroughTrackSec2, AbsorbedTrackSec2, EdgeTrackSec2, InterTypeSec2 = eddyBlockInteraction(track_data,BlockingEIndexSec2,EddyNumberSec2,BlockIndexSec2,InterTypeSec2)

    print('length of the through interaction:', len(ThroughTrackSec2),flush=True)
    print('length of the absorbed interaction:', len(AbsorbedTrackSec2),flush=True)
    print('length of the edge interaction:', len(EdgeTrackSec2),flush=True)

    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EventEddyNumberSector2_1979_2021_AC.npy', np.array(EddyNumberSec2))
    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipole_TrackBlockingType{typeid}_IndexSector2_1979_2021_AC.npy', np.array(BlockIndexSec2))
    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_ThroughTrackSec2_1979_2021_AC.npy', np.array(ThroughTrackSec2))
    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_AbsorbedTrackSec2_1979_2021_AC.npy', np.array(AbsorbedTrackSec2))
    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_EdgeTrackSec2_1979_2021_AC.npy', np.array(EdgeTrackSec2))
    np.save(f'/scratch/bell/hu1029/LGHW/ERA5dipole_BlockingType{typeid}_InteractionTypeSec2_1979_2021_AC.npy', np.array(InterTypeSec2))



