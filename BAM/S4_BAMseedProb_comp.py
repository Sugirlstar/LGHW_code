from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import glob
import copy
import pickle
import matplotlib.path as mpath
from netCDF4 import Dataset
import xarray as xr

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
TRACK_latrefFile = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5/ERA5_TRACK_lat_1979_2025_6hr.npy",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2/MERRA2_TRACK_lat_1980_2025_6hr.npy",
    "JRA3Q": "/scratch/bell/hu1029/LGHW/interm_JRA3Q/JRA3Q_TRACK_lat_1979_2025_6hr.npy"
}
TRACK_lonrefFile = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5/ERA5_TRACK_lon_1979_2025_6hr.npy",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2/MERRA2_TRACK_lon_1980_2025_6hr.npy",
    "JRA3Q": "/scratch/bell/hu1029/LGHW/interm_JRA3Q/JRA3Q_TRACK_lon_1979_2025_6hr.npy"
}

# %% 00 function preparation --------------------------------
regions = ["ATL", "NP", "SP"]
seasons = [ "ALL", "DJF", "JJA"]
seasonsmonths = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1, 2], [6, 7, 8]]
blkTypes = ["Ridge", "Trough", "Dipole"]

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


# start ---
for dtname in datasets: 

    OUTDIR = OUT_DIR_List[dtname]
    timereff = timerefFile[dtname]
    latref = latrefFile[dtname]
    lonref = lonrefFile[dtname]
    yearname = yearnameList[dtname]

    #%% 01 read the data - time management
    ### Time Management ###
    start_year, end_year = map(int, yearname.split('_'))
    Datestamp = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
    Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
    Date = list(Date0['date'])
    nday = len(Date)
    Month = Date0['date'].dt.month
    Year = Date0['date'].dt.year
    Day = Date0['date'].dt.day
    # get the DJF date indices
    BAMDate = Date0
    DJF_index = np.array(BAMDate[BAMDate['date'].dt.month.isin([12, 1, 2])].index)
    print('DJF_index:', DJF_index, flush=True)
    print(len(DJF_index), flush=True)
    offsets = np.arange(-5, 1)  
    DJF_index_preceding = np.unique(np.concatenate([DJF_index + offset for offset in offsets]))
    print('DJF preceding index:', DJF_index_preceding, flush=True)

    # lat and lon
    lat = np.load(latref) # increasing order (-90~90)
    lon = np.load(lonref)
    lat1dg_mid = int(len(lat)/2) + 1 #91
    Blklon = lon
    tracklon = lon

    # 
    ss = 'ALL'
    regions_B_B_freq = []
    regions_B_LB_freq = []
    regions_B_freq_clima = []
    regions_S_B_freq = []
    regions_S_LB_freq = []
    regions_S_freq_clima = []
    for type_idx in [0,1,2]: # Ridge, Trough, Dipole
        for rgname in regions:

            nametag = f'{blkTypes[type_idx]}_{rgname}_{ss}'

            # 01 prepare the lat  
            if rgname == 'SP':
                k = 'SH'
                latNH = lat[0:lat1dg_mid-1] # increasing order (-90~0), exclude equator
                Blklat = latNH
            else:
                k = 'NH'
                latNH = lat[lat1dg_mid:len(lat)] # increasing order (1~90), exclude equator
                Blklat = latNH

            print(f'Processing {type_idx} blocking in {rgname} region for {ss} season', flush=True)
            print('latNH:', latNH, flush=True)
            print('latNH length:', latNH.shape, flush=True)

            # 02 read in the Blocking and Seeding Flag (lat decreasing)
            B_freq_3d = np.load(f'{OUTDIR}/{dtname}_SD_BlockingFlagmaskClusters_Type{type_idx+1}_{rgname}_{ss}.npy')
            S_freq_3d = np.load(f'{OUTDIR}/{dtname}_SD_SeedingFlagmaskClusters_Type{type_idx+1}_{rgname}_{ss}.npy')
            blocking3dArr = B_freq_3d
            seeding3dArr = S_freq_3d
            blockingEidArr_2D = np.any(blocking3dArr, axis=1).astype(int)
            print('blockingEidArr_2D shape:', blockingEidArr_2D.shape, flush=True) # (nday*4, 360), 0 or 1
            seedingEidArr_2D = np.any(seeding3dArr, axis=1).astype(int)
            print('seedingEidArr_2D shape:', seedingEidArr_2D.shape, flush=True) # (nday*4, 360), 0 or 1
            nt = blockingEidArr_2D.shape[0]
            if ss == 'DJF':
                # keep only the DJF dates
                mask = np.ones(nt, dtype=bool)
                mask[DJF_index_preceding] = False  
                blockingEidArr_2D[mask, :] = 0
                seedingEidArr_2D[mask, :] = 0

            # 04 read in the high/low BAM state index 
            # read the BAM index
            if k == 'NH':
                with open(f'{OUTDIR}/synoptic_{dtname}_NH_BAM_event_peak_list.pkl', 'rb') as f:
                    BAM_event_all = pickle.load(f)
                # read the BAM index values
                BI = np.load(f'{OUTDIR}/synoptic_{dtname}_NH_BAM_index_total_with_leap.npy')
            else:
                with open(f'{OUTDIR}/{dtname}_SH_BAM_event_peak_list.pkl', 'rb') as f:
                    BAM_event_all = pickle.load(f)
                # read the BAM index values
                BI = np.load(f'{OUTDIR}/{dtname}_SH_BAM_index_total_with_leap.npy')


            # get the high and low BAM state list
            if ss != 'ALL':
                BAM_dates = BAMDate['date'].iloc[BAM_event_all]
                BAM_event_DJF = [idx for idx, date in zip(BAM_event_all, BAM_dates) if date.month in [12, 1, 2]]

                T=1
                n_BAM = len(BAM_event_DJF)
                # B_B and B_LB is to store the BAM or Low BAM condition days
                B_B = np.zeros((B_freq_3d.shape[0],B_freq_3d.shape[2]))  ## This array is to store the blocking which falls into +-T day of BAM peaking date
                B_LB = np.zeros((B_freq_3d.shape[0],B_freq_3d.shape[2])) ## This array is to store the blocking whihc falls into +-T day of low BAM date
                for i in np.arange(n_BAM):
                    BAMloc = BAM_event_DJF[i]
                    print('BAMloc:' ,BAMloc, flush=True)
                    B_B[BAMloc-T:BAMloc+T+1,:] = 1
                    B_LB[BAMloc-12-T:BAMloc-12+T+1,:] = 1
                print(B_B.shape, flush=True)
            else:
                # find the low BAM states
                T=1
                n_BAM = len(BAM_event_all)
                # B_B and B_LB is to store the BAM or Low BAM condition days
                B_B = np.zeros((B_freq_3d.shape[0],B_freq_3d.shape[2]))  ## This array is to store the blocking which falls into +-T day of BAM peaking date
                B_LB = np.zeros((B_freq_3d.shape[0],B_freq_3d.shape[2])) ## This array is to store the blocking whihc falls into +-T day of low BAM date
                for i in np.arange(n_BAM):
                    BAMloc = BAM_event_all[i]
                    BAMday = Date0.iloc[BAMloc]
                    B_B[BAMloc-T:BAMloc+T+1,:] = 1
                print("B_B shape:", B_B.shape, flush=True) # high BAM center = 1, others = 0

                BAM_low_days = set()
                for idx in BAM_event_all:
                    move = 12
                    day_st = idx - move
                    day_ed = idx + move + 1
                    if day_st>=0 and day_ed<=len(Date0):
                        # find the 3 lowest values and locations within the 25-day period
                        # save the global index of the three lowest days as BAM_low_days
                        window = BI[day_st:day_ed]
                        lowest_indices = np.argsort(window)[:3]  # indices of the three lowest values
                        for li in lowest_indices:
                            BAM_low_days.add(int(day_st + li))
                # convert the set to a sorted numpy array
                BAM_low_days = np.array(sorted(BAM_low_days))
                print('BAM_low_days:', BAM_low_days, flush=True)
                B_LB[BAM_low_days, :] = 1

            # 05 check: now we have blockingEidArr_2D and BAM index (1D)
            print('blockingEidArr_2D shape:', blockingEidArr_2D.shape, flush=True)
            print('BAM index B_B shape:', B_B.shape, flush=True)

            # 07 check the probability of Blocking under HB or LB
            blk_under_BAM = blockingEidArr_2D * B_B
            blk_under_LB = blockingEidArr_2D * B_LB
            sed_under_BAM = seedingEidArr_2D * B_B
            sed_under_LB = seedingEidArr_2D * B_LB
            lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
            ntime = len(Date0) # total length
            print('total time length:', ntime, flush=True)
            if ss == 'DJF':
                nDJF = len(DJF_index_preceding) # number of DJF days
                ntime = nDJF
            B_B_day = n_BAM*(2*T+1) # number of days under high BAM condition
            print('B_B_day length:', B_B_day, flush=True)
            B_B_num = np.nansum(blk_under_BAM,axis=0)
            B_LB_num = np.nansum(blk_under_LB,axis=0)
            S_B_num = np.nansum(sed_under_BAM,axis=0)
            S_LB_num = np.nansum(sed_under_LB,axis=0)
            B_B_freq = (B_B_num/B_B_day)
            B_LB_freq = (B_LB_num/B_B_day) # frequency of blocking under low BAM condition
            S_B_freq = (S_B_num/B_B_day)
            S_LB_freq = (S_LB_num/B_B_day) # frequency of seeding under low BAM condition
            B_freq_clima = np.nansum(blockingEidArr_2D,axis=0)/ntime
            S_freq_clima = np.nansum(seedingEidArr_2D,axis=0)/ntime

            regions_B_B_freq.append(B_B_freq)
            regions_B_LB_freq.append(B_LB_freq)
            regions_B_freq_clima.append(B_freq_clima)
            regions_S_B_freq.append(S_B_freq)
            regions_S_LB_freq.append(S_LB_freq)
            regions_S_freq_clima.append(S_freq_clima)

        np.save(f'{OUTDIR}/{dtname}_{blkTypes[type_idx]}_{ss}_regions_S_B_freq_synoptic.npy', regions_S_B_freq)
        np.save(f'{OUTDIR}/{dtname}_{blkTypes[type_idx]}_{ss}_regions_S_LB_freq_synoptic.npy', regions_S_LB_freq)
        np.save(f'{OUTDIR}/{dtname}_{blkTypes[type_idx]}_{ss}_regions_S_freq_clima_synoptic.npy', regions_S_freq_clima)
        np.save(f'{OUTDIR}/{dtname}_{blkTypes[type_idx]}_{ss}_regions_B_B_freq_synoptic.npy', regions_B_B_freq)
        np.save(f'{OUTDIR}/{dtname}_{blkTypes[type_idx]}_{ss}_regions_B_LB_freq_synoptic.npy', regions_B_LB_freq)
        np.save(f'{OUTDIR}/{dtname}_{blkTypes[type_idx]}_{ss}_regions_B_freq_clima_synoptic.npy', regions_B_freq_clima)

        print(f'Finished processing {blkTypes[type_idx]} blocking for {ss} season', flush=True)

