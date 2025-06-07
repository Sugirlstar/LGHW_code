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
# read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

# read in the LWA data
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

ss = "ALL"

for typeid in [1,2]:
    for rgname in regions:

        dstrack = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
        tracklon = np.array(ds['lon'])
        tracklat = np.array(ds['lat'])
        tracklat = np.flip(tracklat)
        if rgname == "SP":
            tracklat = tracklat[0:(findClosest(0,tracklat)+1)]
        else:
            tracklat = tracklat[(findClosest(0,tracklat)+1):len(tracklat)]

        lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
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

        if rgname == "SP":
            HMi = '_SH'
        else:
            HMi = ''

        if typeid == 1:
            cyc = 'AC'
        elif typeid == 2:
            cyc = 'CC'

        trackPoints_array = np.load(f'/scratch/bell/hu1029/LGHW/{cyc}trackPoints_array{HMi}.npy')
        trackPoints_middle = np.load(f'/scratch/bell/hu1029/LGHW/{cyc}trackPoints_middle{HMi}.npy')

        # make the region mask
        rgmask = np.zeros((len(tracklat), len(tracklon)))
        lat_min_ix = np.argmin(np.abs(tracklat - lat_min))
        lat_max_ix = np.argmin(np.abs(tracklat - lat_max))
        lon_min_ix = np.argmin(np.abs(tracklon - lon_min))
        lon_max_ix = np.argmin(np.abs(tracklon - lon_max))
        if lon_min < lon_max:
            rgmask[lat_min_ix:lat_max_ix+1, lon_min_ix:lon_max_ix+1] = 1
        else:
            # if the region crosses the 360/0 degree longitude
            rgmask[lat_min_ix:lat_max_ix+1, lon_min_ix:len(tracklon)] = 1
            rgmask[lat_min_ix:lat_max_ix+1, 0:lon_max_ix+1] = 1

        # %% 02 get the 1st day index of blocking events ------------------------
        with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}", "rb") as fp: 
            firstday_Date = pickle.load(fp) # hourly list
        firstblkIndex = [(timei.index(i)) for i in firstday_Date] # transfer to location index, 6-hourly
        blkFlag = np.zeros(len(timei)) # the blocking flag for each time step
        blkFlag[firstblkIndex] = 1 # set the blocking flag to 1 for the first day of each blocking event
        backgroundP = np.nansum(blkFlag==1) / len(blkFlag) # the background probability of blocking events
        print('the background probability of blocking events:', backgroundP, flush=True)

        window_size = 5
        hits = []
        for t in range(len(timei)):
            end = min(t + window_size, len(timei))
            window = blkFlag[t:end]
            hits.append(np.any(window))
        # calculate the probability of blocking events in the window
        hits = np.array(hits)
        probability = np.mean(hits)
        print(f"blocking in 5-day window：{probability:.4f}")

        # %% 03 start to monitor the eddies ------------------------
        regionEddyArr = trackPoints_array * rgmask # the eddy array for the region
        regionEddyMidID = trackPoints_middle * rgmask # the eddy middle point ID for the region
        regionEddyNumber = np.sum(regionEddyArr, axis=(1, 2)) # the number of eddies in the region for each time step
        dailyuniqueEddyNum = np.zeros(len(timei))
        for i in np.arange(len(timei)):
            # get the unique eddy ID for each time step
            dailyuniqueEddyNum[i] = len(np.unique(regionEddyMidID[i, :, :][regionEddyMidID[i, :, :] > 0]))
            
        regiondailyNumber = np.unique(regionEddyMidID, axis=0) # the unique number of eddies in the region for each time step
        delta = np.diff(regionEddyNumber)
        prev = regionEddyNumber[:-1]
        # find the index with X → X+1 
        transbase = [0,1,2,3,4]  # the base states for the transitions
        trans_loc = []
        for k in transbase:
            trans = np.where((prev == k) & (delta == 1))[0] + 1  # +1 to match the next time step
            trans_loc.append(trans)
            # print the results
            print(f"{k} → {k+1}:", trans, flush=True)
        # find the index with eddies' presence
        presenbase = [0,1,2,3,4] # ==0, ==1, ==2, ==3, >=4
        presentNum = []
        for k in presenbase:
            if k == 4:
                present = np.where(dailyuniqueEddyNum >= k)[0]
            else:
                present = np.where(dailyuniqueEddyNum == k)[0]
            presentNum.append(present)

        # enter plot
        predictstartday = [0,4,8,12] # the prediction window in days
        windlen = 5*4 # five days predict window
        prematrix = np.zeros((len(transbase), len(predictstartday))) # the matrix to store the prediction results
        for pi,predictDay in enumerate(predictstartday):
            for k in transbase:
                trans = trans_loc[k]
                print(f"predicting {k} → {k+1} for {predictDay} timesteps later:", flush=True)
                print('eddy locations:', trans, flush=True)
                # find the locations where the number of eddies increases by the prediction window
                window_indices = [list(range(t + predictDay, t + predictDay + windlen)) for t in trans]
                blckoccur = [int(np.any(np.isin(item, firstblkIndex))) for item in window_indices]
                print('prediction window indices:', blckoccur, flush=True)
                trueRate = np.nansum(blckoccur) / len(trans) # the true rate of blocking events in the prediction window
                prematrix[k, pi] = trueRate

        # make the plot of the prediction success rate matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(prematrix, cmap='viridis_r')
        ax.set_xticks(np.arange(len(predictstartday)))
        ax.set_xticklabels(predictstartday)
        ax.set_yticks(np.arange(len(transbase)))
        ax.set_yticklabels([f"{k}→{k+1}" for k in transbase])

        ax.set_xlabel('Prediction lead day (days)')
        ax.set_ylabel('Transition window')

        for i in range(prematrix.shape[0]):
            for j in range(prematrix.shape[1]):
                ax.text(j, i, f'{prematrix[i, j]:.2f}', ha='center', va='center',
                        color='w' if prematrix[i, j] < 0.5 else 'black')

        ax.set_title(f'Prediction Success Rate, bg: {probability}')
        fig.colorbar(im, ax=ax)
        plt.show()
        plt.savefig(f'BasePredicRate_Type{typeid}_{rgname}.png')
        plt.close()
        print('Prediction success fig saved', flush=True)

        # present plot
        predictstartday = [0,4,8,12] # the prediction window in days
        windlen = 5*4 # five days predict window
        prematrix = np.zeros((len(presenbase), len(predictstartday))) # the matrix to store the prediction results
        for pi,predictDay in enumerate(predictstartday):
            for k in presenbase:
                trans = presentNum[k]
                # find the locations where the number of eddies increases by the prediction window
                window_indices = [list(range(t + predictDay, t + predictDay + windlen)) for t in trans]
                blckoccur = [int(np.any(np.isin(item, firstblkIndex))) for item in window_indices]
                print('prediction window indices:', blckoccur, flush=True)
                trueRate = np.nansum(blckoccur) / len(trans) # the true rate of blocking events in the prediction window
                prematrix[k, pi] = trueRate
