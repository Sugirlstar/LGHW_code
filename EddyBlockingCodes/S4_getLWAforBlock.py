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

# %% dataset settings -------------------------------------------------------------
datasets = ["ERA5", "MERRA2", "JRA3Q"]

OUT_DIR_List = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2",
    "JRA3Q": "/scratch/bell/hu1029/LGHW/interm_JRA3Q"
}
yearnameList = {
    "ERA5": "1979_2025",
    "MERRA2": "1980_2025",
    "JRA3Q": "1979_2025"
}
timerefFile = {
    "ERA5": "/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2025_1dg.nc",
    "MERRA2": "/scratch/bell/hu1029/Data/processed/MERRA2_Z500_6hr_1980_2025_1dg.nc",
    "JRA3Q": "/scratch/bell/hu1029/Data/processed/JRA3Q_Z500_6hr_1979_2025_1dg.nc"
}
latrefFile = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5/ERA5_LWA_lat_1979_2025_6hr.npy",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2/MERRA2_LWA_lat_1980_2025_6hr.npy",
    "JRA3Q": "/scratch/bell/hu1029/LGHW/interm_JRA3Q/JRA3Q_LWA_lat_1979_2025_6hr.npy"
}
lonrefFile = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5/ERA5_LWA_lon_1979_2025_6hr.npy",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2/MERRA2_LWA_lon_1980_2025_6hr.npy",
    "JRA3Q": "/scratch/bell/hu1029/LGHW/interm_JRA3Q/JRA3Q_LWA_lon_1979_2025_6hr.npy"
}

# %% 00 function preparation --------------------------------
regions = ["ATL", "NP", "SP"]
seasons = [ "ALL", "DJF", "JJA"]
seasonsmonths = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1, 2], [6, 7, 8]] 
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["AC", "CC"]

def Region_ERA(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max = 45, 75, 300, 60
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max = 40, 70, 130, 250
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max = -75, -45, 180, 300
    return lat_min, lat_max, lon_min, lon_max

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

# %% processing --------------------------------
for dtname in datasets:

    print(f"Processing dataset: {dtname}", flush=True)
    OUTDIR = OUT_DIR_List[dtname]
    INDIR = OUTDIR
    yearname = yearnameList[dtname]
    latfile = latrefFile[dtname]
    lonfile = lonrefFile[dtname]
    timefile = timerefFile[dtname]

    # read in LWA
    LWA_td_origin = np.load(f'{INDIR}/{dtname}_LWA_td_{yearname}_6hr.npy')  # -90～90, it's from south to north
    LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 
    print('-------- LWA loaded --------', flush=True)

    # time management
    ds = xr.open_dataset(timefile)
    timesarr = np.array(ds['time'])
    datetime_array = pd.to_datetime(timesarr)
    timei = list(datetime_array)
    # read in lat and lon
    lon = np.load(lonfile)
    lat = np.load(latfile) # increasing

    # get LWA for each blocking event -----------------------------------------------
    for ss in seasons:
        for typeid in [1,2,3]:
            for rgname in regions:

                # get the lat and lon for LWA, grouped by NH and SH
                lat_mid = int(len(lat)/2) + 1 
                if rgname == "SP":
                    latLWA = lat[0:lat_mid-1] 
                    LWA_td = LWA_td_origin[:,0:lat_mid-1,:]
                else:
                    latLWA = lat[lat_mid:len(lat)]
                    LWA_td = LWA_td_origin[:,lat_mid:len(lat),:] 
                print(latLWA, flush=True)
                print('LWA shape: ', LWA_td.shape, flush=True)
                lonLWA = lon 

                # read in blocking event lists
                if rgname == "SP":
                    with open(f"{INDIR}/{dtname}_SD_Blocking_diversity_label_daily_SH", "rb") as fp:
                        Blocking_diversity_label = pickle.load(fp)
                    with open(f"{INDIR}/{dtname}_SD_Blocking_diversity_date_daily_SH", "rb") as fp:
                        Blocking_diversity_date = pickle.load(fp)
                else:
                    with open(f"{INDIR}/{dtname}_SD_Blocking_diversity_label_daily_NH", "rb") as fp:
                        Blocking_diversity_label = pickle.load(fp)
                    with open(f"{INDIR}/{dtname}_SD_Blocking_diversity_date_daily_NH", "rb") as fp:
                        Blocking_diversity_date = pickle.load(fp)
                Blocking_diversity_label = Blocking_diversity_label[typeid-1] # get the blocking event list for the typeid
                Blocking_diversity_date = Blocking_diversity_date[typeid-1] # get the blocking event date for the typeid

                # blockings
                # the id of each blocking event (not all events! just the events within the target region)
                with open(f'{INDIR}/{dtname}_SD_BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
                    blkEindex = pickle.load(f)
                print(f'blocking id of each event: {blkEindex}',flush=True)
                
                blkFirstday = [] # the first day index of each blocking event in 6-hourly data
                eventLWA = []
                for k,event in enumerate(blkEindex):
                    
                    event_dates = Blocking_diversity_date[event]
                    event_label = np.array(Blocking_diversity_label[event]) # the label of the blocking event
                    event_label = np.repeat(event_label, 4, axis=0)
                    
                    # get the corresponding arr index
                    stindex = timei.index(event_dates[0]) # the first date of the blocking event
                    edinex = timei.index(event_dates[-1]) + 3 # the last date of the blocking event, but add 3 to match the 6-hourly data
                    LWAvaluesArr = LWA_td[stindex:edinex+1, :, :] * event_label # get the LWA values of the blocking event
                    # daily sum
                    daily_list = np.nansum(LWAvaluesArr, axis=(1, 2)) # sum over the time axis (6-hourly to daily)
                    d1stindex = stindex
                    blkFirstday.append(d1stindex)
                    eventLWA.append(daily_list) # a list of LWA values for each blocking event

                with open(f"{OUTDIR}/{dtname}_BlockEventDailyLWAList_{yearname}_Type{typeid}_{rgname}_{ss}.pkl", "wb") as f:
                    pickle.dump(eventLWA, f) 
                np.save(f'{OUTDIR}/{dtname}_BlockEvent1stDayIndex_in6hourly_Type{typeid}_{rgname}_{ss}.npy', np.array(blkFirstday))

                eventAVGLWA = np.array([np.nanmean(event) for event in eventLWA]) # average LWA for each blocking event
                np.save(f'{OUTDIR}/{dtname}_BlockEventAvgedLWA_Type{typeid}_{rgname}_{ss}.npy', eventAVGLWA)

                print('Averaged LWA for each blocking event: \n', eventAVGLWA, flush=True)
                # plot the pdf of LWA for blocked days
                plt.figure()
                sns.kdeplot(eventAVGLWA, fill=True)
                plt.title('Probability Density Function (PDF)')
                plt.xlabel('LWA')
                plt.ylabel('Density')
                plt.show()
                plt.savefig(f'./checkPlots/{dtname}_BlockEventLWAPDF_Type{typeid}_{rgname}_{ss}.png')

                print(f'Finished processing {dtname} {rgname} {ss} Type{typeid}', flush=True)

