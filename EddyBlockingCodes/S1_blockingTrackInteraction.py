import numpy as np
import datetime as dt
from datetime import date, datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
from netCDF4 import Dataset
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import cartopy
from cartopy import crs as ccrs
from scipy import ndimage
from scipy.ndimage import convolve
from scipy.signal import detrend
from scipy.stats import pearsonr
import pickle
import xarray as xr
import regionmask
from matplotlib.patches import Polygon
import sys
import matplotlib.colors
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
import matplotlib.path as mpath
from multiprocessing import Pool
from itertools import product

# %% 00 prepare the environment and functions --------------------------------------------------------------
regions = ["ATL", "NP", "SP"]
seasons = ["ALL", "DJF", "JJA"]
seasonsmonths = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],[12, 1, 2], [6, 7, 8]]
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

# Groups of consecutive day index (list of arr), and the length of each group
def getDuration(pos_hw):
    #input:
    # pos_hw - the positions of Blocking days
    diff = np.diff(pos_hw)
    group_boundaries = np.where(diff > 1)[0] + 1  
    # get the length of each group
    groups = np.split(pos_hw, group_boundaries)
    group_lengths = [len(group)/4 for group in groups]

    return groups, group_lengths

# get the closest index of the lat/lon
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

#%% dataset settings -------------------------------------------------------------
datasets = ["ERA5", "MERRA2", "JRA3Q"]

OUT_DIR_List = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2",
    "JRA3Q": "/scratch/bell/hu1029/LGHW/interm_JRA3Q"
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
varnameList = {
    "ERA5": "z",
    "MERRA2": "H",
    "JRA3Q": "hgt-pres-an-ll125"
}
yearnameList = {
    "ERA5": "1979_2025",
    "MERRA2": "1980_2025",
    "JRA3Q": "1979_2025"
}
lat_name, lon_name, time_name = "lat", "lon", "time"


def process_combination(args):
    rgname, cyc, typeid, ss, dtname = args

    OUT_DIR = OUT_DIR_List[dtname]
    yearname = yearnameList[dtname]
    tag = f"{dtname}-{cyc}-Type{typeid}_{rgname}_{ss}"
    # check if the tag has been processed
    cor_summary_file = f"{dtname}_BlkPersis_EddyNumber_Cor.txt"
    if os.path.exists(cor_summary_file):
        with open(cor_summary_file, "r") as f:
            if any(tag in line for line in f):
                print(f"[SKIP] {tag} already processed.")
                return  # skip
            else:
                print(f"[PROCESSING] {tag} ")

    # %% 01 read lat and lon --------------------------------------------------------------
    # attributes for TRACKs
    
    # lat and lon for blockings
    lat = np.load(latrefFile[dtname])
    lon = np.load(lonrefFile[dtname])
    lat_mid = int(len(lat)/2) + 1

    if rgname == "SP":
        Blklat = lat[0:lat_mid-1] # increasing lat from -90 to 0
        k = 'SH'
    else:
        Blklat = lat[lat_mid:len(lat)] # increasing lat from 0 to 90
        k = 'NH'
    Blklon = lon

    # Time management
    # time for the tracks: 6-hourly
    ds = xr.open_dataset(timerefFile[dtname])
    timesarr = np.array(ds[time_name])
    datetime_array = pd.to_datetime(timesarr)
    timei = list(datetime_array)
    
    # load tracks
    with open(f'{OUT_DIR}/{dtname}_{cyc}Zanom_allyearTracks_{k}.pkl', 'rb') as file:
        track_data = pickle.load(file)

    print('track loaded-----------------------',flush=True)

    # read in blocking event lists
    with open(f"{OUT_DIR}/{dtname}_SD_Blocking_diversity_date_daily_{k}", "rb") as fp:
        Blocking_diversity_date = pickle.load(fp)
    Blocking_diversity_date = Blocking_diversity_date[typeid-1] # get the blocking event list for the typeid

    # the id of each blocking event (not all events! just the events within the target region)
    with open(f'{OUT_DIR}/{dtname}_SD_BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
        targetblockingeventID = pickle.load(f)
    print(f'blocking id of each event (10 for example): {targetblockingeventID[:10]}',flush=True)
    targetblockingeventID = np.array(targetblockingeventID)

    # %% 02 eddy-blocking interaction identify --------------------------------------------------------------
    # 001 transfer to 6-hourly
    blockingSec2 = np.load(f'{OUT_DIR}/{dtname}_SD_BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
    # transfer to 6-hourly
    BlockingEIndex = np.repeat(blockingSec2, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

    # 002 blocking event characteristics: persistence
    eventPersistence = [len(Blocking_diversity_date[evid]) for evid in targetblockingeventID]
    print(f'blocking persistence of each event (10 for example): {eventPersistence[:10]}',flush=True)
    np.save(f'{OUT_DIR}/{dtname}_BlockingEventPersistence_Type{typeid}_{rgname}_{ss}.npy', np.array(eventPersistence))

    # 003* define four scenarios and check --------------------------------------------------------------
    EddyNumber = [0] * len(targetblockingeventID) # a list of the eddy number that each block is related to; length = blocking event number
    BlockIndex = [-1] * len(track_data) # a list of the blockingid that each track is related to; length = track number
    tpIndex = ['N'] * len(track_data) # a list of the interaction type that each track is related to; length = track number
    ThroughTrack = []
    EdgeTrack = []
    AbsorbedTrack = []

    tracknum = 0
    for index, pointlist in track_data:

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
        latids = findClosest(latids,Blklat)
        lonids = [loni for _, loni, _ in pointlist]
        lonids = findClosest(lonids,Blklon)
        timeids = [timei.index(i) for i in times]
        blockingValue = BlockingEIndex[np.array(timeids), np.array(latids), np.array(lonids)]
        blockingValue = blockingValue.astype(float) # convert to float for np.isnan check
        blockingValue[np.where(blockingValue == -1)] = np.nan  # replace -1 with NaN 

        # 001 Through (and Edge)
        if np.isnan(blockingValue[0]) and np.isnan(blockingValue[-1]):
            if np.any(blockingValue >= 0):
                indices = np.where(blockingValue >= 0)[0]
                if np.all(np.diff(indices) == 1):
                    print(blockingValue,flush=True)
                    ThroughTrack.append(index)
                    bkEindex = np.unique(blockingValue) # the blockingid that the track is related to (global id)
                    bkEindex = bkEindex[~np.isnan(bkEindex)].astype(int)
                    bkids = bkEindex[0] # only get the first one that interacts with
                    if len(bkEindex) > 1:
                        print(f'Warning: {len(bkEindex)} blocking events identified for track {index}', flush=True)
                    for nn in bkEindex:
                        nnidx = np.where(targetblockingeventID == nn)[0][0] # the location of the event id in the region's collection
                        EddyNumber[nnidx] += 1
                    tp = 'T'
                    print(f'{bkids}: Through Identified',flush=True)
                else:
                    print(blockingValue,flush=True)
                    EdgeTrack.append(index)
                    bkEindex = np.unique(blockingValue) # the blockingid that the track is related to
                    bkEindex = bkEindex[~np.isnan(bkEindex)].astype(int)
                    bkids = bkEindex[0] # only get the first one that interacts with
                    for nn in bkEindex:
                        nnidx = np.where(targetblockingeventID == nn)[0][0]
                        EddyNumber[nnidx] += 1
                    tp = 'E'
                    print(f'{bkids}: Edge Identified',flush=True)

        # 002 Absorbed
        if np.isnan(blockingValue[0]) and blockingValue[-1] >= 0:
            print(blockingValue,flush=True)
            AbsorbedTrack.append(index)
            bkEindex = np.unique(blockingValue)
            bkEindex = bkEindex[~np.isnan(bkEindex)].astype(int)
            print(bkEindex,flush=True)
            bkids = bkEindex[0]
            if len(bkEindex) > 1:
                print(f'Warning: {len(bkEindex)} blocking events identified for track {index}', flush=True)
            for nn in bkEindex:
                nnidx = np.where(targetblockingeventID == nn)[0][0]
                EddyNumber[nnidx] += 1
            tp = 'A'
            print(f'{bkids}: Absorbed Identified',flush=True)

        # 003 Internal
        if blockingValue[0] >= 0:
            print(f'{bkids}: Internal or Spawned Identified',flush=True)

        BlockIndex[tracknum] = bkids
        tpIndex[tracknum] = tp
        tracknum += 1

    # if tag already in the summary file, skip
    if os.path.exists(f'./checkPlots/{dtname}_Interaction_summary.txt'):
        with open(f'{dtname}_Interaction_summary.txt', "r") as f:
            if not any(tag in line for line in f):
                with open(f"{dtname}_Interaction_summary.txt", "a") as f:  
                    f.write(f'Blocking type{typeid} - {cyc} - {rgname} - {ss} total length: {len(eventPersistence)}\n')
                    f.write(f'Length of the Through interaction, {cyc}-Type{typeid}_{rgname}_{ss}: {len(ThroughTrack)}\n')
                    f.write(f'Length of the Edge interaction, {cyc}-Type{typeid}_{rgname}_{ss}: {len(EdgeTrack)}\n')
                    f.write(f'Length of the Absorbed interaction, {cyc}-Type{typeid}_{rgname}_{ss}: {len(AbsorbedTrack)}\n')
                    f.write(f'Length of the total interaction, {cyc}-Type{typeid}_{rgname}_{ss}: {len(ThroughTrack)+len(AbsorbedTrack)+len(EdgeTrack)}\n')
                    f.write('-----------------------------------------------------\n')
    else:
        with open(f"./checkPlots/{dtname}_Interaction_summary.txt", "a") as f:  
            f.write(f'Blocking type{typeid} - {cyc} - {rgname} - {ss} total length: {len(eventPersistence)}\n')
            f.write(f'Length of the Through interaction, {cyc}-Type{typeid}_{rgname}_{ss}: {len(ThroughTrack)}\n')
            f.write(f'Length of the Edge interaction, {cyc}-Type{typeid}_{rgname}_{ss}: {len(EdgeTrack)}\n')
            f.write(f'Length of the Absorbed interaction, {cyc}-Type{typeid}_{rgname}_{ss}: {len(AbsorbedTrack)}\n')
            f.write(f'Length of the total interaction, {cyc}-Type{typeid}_{rgname}_{ss}: {len(ThroughTrack)+len(AbsorbedTrack)+len(EdgeTrack)}\n')
            f.write('-----------------------------------------------------\n')

    rr, pp = pearsonr(eventPersistence, EddyNumber)
    if os.path.exists(f'./checkPlots/{dtname}_BlkPersis_EddyNumber_Cor.txt'):
        with open(f'./checkPlots/{dtname}_BlkPersis_EddyNumber_Cor.txt', "r") as f:
            if not any(tag in line for line in f):
                with open(f"./checkPlots/{dtname}_BlkPersis_EddyNumber_Cor.txt", "a") as f:  
                    f.write(f"{cyc}-Type{typeid}_{rgname}_{ss}, Pearson r = {rr:.4f}, p-value = {pp:.4e}\n")
    else:
        with open(f"./checkPlots/{dtname}_BlkPersis_EddyNumber_Cor.txt", "a") as f:  
            f.write(f"{cyc}-Type{typeid}_{rgname}_{ss}, Pearson r = {rr:.4f}, p-value = {pp:.4e}\n")

    np.save(f'{OUT_DIR}/{dtname}_BlockingType{typeid}_EventEddyNumber_{yearname}_{rgname}_{ss}_{cyc}.npy', np.array(EddyNumber))
    np.save(f'{OUT_DIR}/{dtname}_TrackBlockingType{typeid}_Index_{yearname}_{rgname}_{ss}_{cyc}.npy', np.array(BlockIndex))
    np.save(f'{OUT_DIR}/{dtname}_BlockingType{typeid}_ThroughTrack_{yearname}_{rgname}_{ss}_{cyc}.npy', np.array(ThroughTrack))
    np.save(f'{OUT_DIR}/{dtname}_BlockingType{typeid}_AbsorbedTrack_{yearname}_{rgname}_{ss}_{cyc}.npy', np.array(AbsorbedTrack))
    np.save(f'{OUT_DIR}/{dtname}_BlockingType{typeid}_EdgeTrack_{yearname}_{rgname}_{ss}_{cyc}.npy', np.array(EdgeTrack))
    np.save(f'{OUT_DIR}/{dtname}_BlockingType{typeid}_InterType_{yearname}_{rgname}_{ss}_{cyc}.npy', np.array(tpIndex))

    print(f'Blocking type{typeid}-{cyc}_{rgname}_{ss} interaction saved',flush=True)

# parallel processing ---------------------------------------------
# if __name__ == "__main__":
#     param_combinations = list(product(regions, cycTypes, [1, 2, 3], seasons))
#     with Pool(processes=64) as pool:
#         pool.map(process_combination, param_combinations)

# serial processing ---------------------------------------------
# for dtname in datasets:
#     for ss in seasons:
#         for cyc in cycTypes:
#             for rgname in regions:
#                     for typeid in [1, 2, 3]:
#                         process_combination((rgname, cyc, typeid, ss, dtname))

# multi-tasks processing, using multiple nodes ---------------------------------------------
import sys
rgname = sys.argv[1]
cyc = sys.argv[2]
typeid = int(sys.argv[3])
ss = sys.argv[4]
dtname = sys.argv[5]
process_combination((rgname, cyc, typeid, ss, dtname))
