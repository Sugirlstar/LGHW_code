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

# 00 functions -------------------------------------------------------------
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

# 01 read the data -------------------------------------------------------------
LWA_td = np.load('/scratch/bell/hu1029/Data/processed/LWA_all.npy')  # [0]~[-1] it's from south to north
LWA_td = LWA_td/100000000 # change the unit to 1e8 
print('-------- LWA loaded --------', flush=True)
print(LWA_td.shape)
lonLWA = np.load('/home/hu1029/LGHW_code/ERA5_lon.npy')
latLWA = np.load('/home/hu1029/LGHW_code/ERA5_lat.npy') # descending order
latLWA = np.flip(latLWA) # make it ascending order (from south to north)
LWA_td = LWA_td[:,(findClosest(0,latLWA)):len(latLWA), :]
latLWA = latLWA[(findClosest(0,latLWA)):len(latLWA)]
print(latLWA)
timei = []
start_date = dt.datetime(1940, 1, 1)
end_date = dt.datetime(2022, 12, 31)
for ordinal in range(start_date.toordinal(), end_date.toordinal() + 1):
    current_date = dt.datetime.fromordinal(ordinal)
    date_str = str(current_date.year).zfill(4)+current_date.strftime('%m%d')
    timei.append(date_str)
start_target = '19800101'
end_target = '20211231'
start_index = timei.index(start_target)
end_index = timei.index(end_target)
LWA_td = LWA_td[start_index:(end_index+1), :, :]
LWA = np.repeat(LWA_td, 4, axis=0) # turn 
print(LWA.shape, flush=True)

ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
lon = np.array(ds['lon'])
lat = np.array(ds['lat'])
lat = np.flip(lat)
lat = lat[(findClosest(0,lat)+1):len(lat)]

times = np.array(ds['time'])
datetime_array = pd.to_datetime(times)
timei = list(datetime_array)
timei = [t for t in timei if pd.Timestamp("1980-01-01") <= t <= pd.Timestamp("2022-01-01")] # get the time for dipole blocking

for typeid in [3,1]:

    # blocking persistence
    Sec2BlockPersis = np.load(f'/scratch/bell/hu1029/LGHW/Dipole_Merra2_BlockingType{typeid}_EventPersisSector2_1979_2021.npy', allow_pickle=True) # blocking duration for each event, corresponding to the event number
    print('most common persistance:', flush=True)
    print(stats.mode(Sec2BlockPersis).mode, flush=True)
    print('count:', flush=True) 
    print(stats.mode(Sec2BlockPersis).count, flush=True)
    # the arr of the blocking event index
    BlockingEIndexSec2 = np.load(f'/scratch/bell/hu1029/LGHW/Dipole_Merra2_BlockingFlagSector2maskClusters_1979_2021_Type{typeid}.npy')
    BlockingEIndexSec2 = np.repeat(BlockingEIndexSec2, 4, axis=0)
    # get the blocking index each track contribute to
    BlockIndexSec2 = np.load(f'/scratch/bell/hu1029/LGHW/Dipole_Merra2_TrackBlockingType{typeid}_IndexSector2_1979_2021_CC.npy') # the related blocking event id, -1 represent no blocking related
    # get the blocking event index list
    with open(f'/scratch/bell/hu1029/LGHW/Dipole_Merra2_BlockingType{typeid}_EventListSector2_1979_2021.pkl', 'rb') as file:
        Sec2BlockEvent = pickle.load(file)
    # get all the tracks
    with open('/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks.pkl', 'rb') as file:
        track_data = pickle.load(file)

    print('all loaded ----------', flush=True)

    # 02 get the target event id ---------------------------------------
    # get the blocking persistence each track related to, if not related, np.nan
    allpersis = []
    for i in BlockIndexSec2:
        persis = np.nan
        if i>=0:
            persis = Sec2BlockPersis[i]
        allpersis.append(persis)
    # # persistence between 5 to 10
    # # BlockIndexSec2[np.where(BlockIndexSec2 == -1)] = np.nan
    # targetTrackIndexCommon = [i for i,v in enumerate(BlockIndexSec2) if not (Sec2BlockPersis[v] > 10 or Sec2BlockPersis[v] < 5)]
    # print(len(targetTrackIndexCommon),flush=True)
    # targetTrackIndexSevere = [i for i,v in enumerate(BlockIndexSec2) if Sec2BlockPersis[v] > 40]
    # print(len(targetTrackIndexSevere),flush=True)

    # class of different interaction types
    InterTypeSec2 = np.load(f'/scratch/bell/hu1029/LGHW/Dipole_Merra2_BlockingType{typeid}_InteractionTypeSec2_1979_2021_CC.npy')
    Throughid = np.where(InterTypeSec2 == 'T')[0]
    Absorbedid = np.where(InterTypeSec2 == 'A')[0]
    Edgeid = np.where(InterTypeSec2 == 'E')[0]

    extendlen = 30
    def getCompFig(fname, extendlen, targetTrackIndexCommon, BlockIndexSec2, BlockingEIndexSec2):

        allpersistence = []
        for tks in targetTrackIndexCommon:
            blockpersist = allpersis[tks]
            allpersistence.append(blockpersist)
        maxpersistence = np.nanmax(allpersistence) * 4
        print(f'the maximum persistence of all these blocks (timesteps): {maxpersistence}', flush=True)

        if extendlen is False:
            extendlen = maxpersistence
            extendlen = math.ceil(extendlen)
            extendlen = int(extendlen)

        # get the LWA composite of each day for all tracks
        daylens = 2 * extendlen + 1
        blockLWAarr = np.full((len(targetTrackIndexCommon),daylens, len(latLWA), len(lonLWA)), np.nan)
        blockingFREarr = np.full((len(targetTrackIndexCommon),daylens,len(lat),len(lon)), np.nan)
        considerperiodArr = np.full((len(targetTrackIndexCommon),daylens), np.nan)

        for k, tks in enumerate(targetTrackIndexCommon):
            print('-------------------',flush=True)
            print('related blocking event id:', BlockIndexSec2[tks], flush=True)
            track = track_data[tks] # the target track
            _, pointlist = track
            times = [ti for ti, _, _ in pointlist]
            latids = [lati for _, _, lati in pointlist]
            latids = findCloset(lat,latids)
            lonids = [loni for _, loni, _ in pointlist]
            lonids = findCloset(lon,lonids)
            timeids = [timei.index(i) for i in times] # the time index in the real all time list
            blockingValue = BlockingEIndexSec2[np.array(timeids), np.array(latids), np.array(lonids)] # using bool value arr to save time
            enterindex = np.argmax(blockingValue)  # the index of the selected blocking values, not the real time
            realenterindex = timeids[enterindex] # get the real time index of the entering
            print('the real time index when entering the blocking event:')
            print(realenterindex,flush=True)

            considerperiod = np.arange(realenterindex-extendlen, realenterindex+extendlen+1)
            # get the realenterindex day's LWA regime (a 2d map)
            if np.any(considerperiod < 0) or np.any(considerperiod >= len(LWA)): # if out of bounds, replace them with np.nan
                considerperiod = np.where(considerperiod < 0, np.nan, considerperiod)
                considerperiod = np.where(considerperiod >= len(LWA), np.nan, considerperiod)
                result = []
                blkfreresult = []
                for idx in considerperiod:
                    if np.isnan(idx):
                        result.append(np.full_like(LWA[0, :, :], np.nan))
                        blkfreresult.append(np.full_like(BlockingEIndexSec2[0,:,:], np.nan))
                    else:
                        result.append(LWA[int(idx), :, :])
                        blkfreresult.append(BlockingEIndexSec2[int(idx),:,:])

                enterdayLWA = np.array(result, dtype=object)
                blkfre = np.array(blkfreresult, dtype=object)
            else:
                enterdayLWA = LWA[considerperiod, :, :]
                blkfre = BlockingEIndexSec2[considerperiod, :, :]
            
            print('the considered period:', flush=True)
            print(considerperiod, flush=True)

            considerperiodArr[k, :] = considerperiod
            blockLWAarr[k,:,:,:] = enterdayLWA
            blockingFREarr[k,:,:,:] = blkfre

        TrackCompSeries = np.nanmean(blockLWAarr, axis=0) # 3d, daylens, lat, lon
        TrackCompBlkFre = np.nansum(blockingFREarr, axis = 0)
        print(TrackCompSeries.shape, flush=True)
        print(considerperiodArr.shape, flush=True)

        # get the tracks every target day
        # for each day, there is a list of the track points
        print('start to get the target period tracks points -----------------', flush=True)
        filtered_track = [track_data[i] for i in targetTrackIndexCommon] # get all the target tracks
        for k, (index, points) in enumerate(filtered_track):
            filtered_track[k] = (k, points)
        InvolvedTrackLen = len(filtered_track)

        TargetperiodTrackList = [] # a list of the length of the target period
        daystart = considerperiodArr[:,0] # the first day of the target period for each track, len = the number of target tracks
        print(f'the first day index for each track: {daystart}', flush=True)
        titlename = np.arange(-extendlen, extendlen+1, 1)

        fig, ax, cf = create_Map(lonLWA,latLWA,TrackCompSeries[0,:,:],fill=True,fig=None,
                                    minv=0, maxv=21, interv=11, figsize=(12,5),
                                    centralLon=0, colr='Blues', extend='max',title=f'getColorbar {fname}')
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)  
        cbar.set_label('LWA')    
        plt.show()
        plt.savefig(f'Dipole_CC_getCompositeColorBarSec2_{fname}_BlkType{typeid}.png')
        plt.close()         

        for i in range(considerperiodArr.shape[1]):  
            print(i) # day i - not all tracks have points on day i (i exceeds the total length)
            # daystart: a list for every tracks' first day (some are NAs)
            # get the day i' index for every track
            dayiIDX = considerperiodArr[:,i]
            print(f"the end day index for each track: {dayiIDX}", flush=True)
            track_points = [
            (index, [(lon, lat) for time, lon, lat in points if timei[int(daystart[index])] <= time <= timei[int(dayiIDX[index])] ])
            for index, points in filtered_track
            if not np.isnan(daystart[index]) and not np.isnan(dayiIDX[index])
            ]
            TargetperiodTrackList.append(track_points)
            print(track_points)

        # plot the map+tracks for each day
        for i in range(len(TargetperiodTrackList)):

            track_points = TargetperiodTrackList[i]
            last_points = [pointlist[-1] for _, pointlist in track_points if pointlist]

            fig, ax, cf = create_Map(lonLWA,latLWA,TrackCompSeries[i,:,:],fill=True,fig=None,
                                        minv=0, maxv=round(np.nanmax(TrackCompSeries)), interv=11, figsize=(12,5),
                                        centralLon=0, colr='Blues', extend='max',title=f'TrackNumber: {InvolvedTrackLen}, Timestep {str(titlename[i])}')
            _, _, cf1 = create_Map(lon,lat, TrackCompBlkFre[i,:,:], 0, 10,continterv=10, ax=ax, fill=False,
                        centralLon=0, extend='both', alpha=1, fig=fig, figsize=(10, 6))
            addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)

            # add points
            if not last_points:
                continue  
            lonps, latps = zip(*last_points)  # unzip the list of tuples
            ax.scatter(lonps, latps, color='orange', marker='o', s=14, alpha=0.9, edgecolors='none',transform=ccrs.PlateCarree())        

            plt.show()
            plt.savefig(f'Dipole_CC_AnimationCompositePoints_{fname}_BlkType{typeid}_{i}.png')
            plt.close()

        # make animation ---------------------
        image_folder = '/home/hu1029/LGHW_code'
        # get all the image files in the folder and sort them
        frames = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.startswith(f'Dipole_CC_AnimationCompositePoints_{fname}_BlkType{typeid}_') and f.endswith('.png')
        ], key=lambda f: int(f.split('_')[-1].split('.')[0]))
        print(frames)
        # get all the images
        gif_filename = f"Dipole_CC_Animation_compositePoints_{fname}_BlkType{typeid}.gif"
        with imageio.get_writer(gif_filename, mode="I", fps=6) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)
        print(f"GIF saved as {gif_filename}")
        # remove the individual frames
        for f in os.listdir(image_folder):
            if f.startswith(f'Dipole_CC_AnimationCompositePoints_{fname}_BlkType{typeid}_') and f.endswith('.png'):
                os.remove(os.path.join(image_folder, f))

    getCompFig('Through', extendlen, Throughid, BlockIndexSec2, BlockingEIndexSec2)
    getCompFig('Absorbed', extendlen, Absorbedid, BlockIndexSec2, BlockingEIndexSec2)
    getCompFig('Edge', extendlen, Edgeid, BlockIndexSec2, BlockingEIndexSec2)
