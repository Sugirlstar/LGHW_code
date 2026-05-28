###### This code is to track all blocking events ######
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import cv2
import copy
import matplotlib.path as mpath
import pickle
import glob
from netCDF4 import Dataset
from skimage import feature, segmentation, measure
import sys
from collections import defaultdict
import xarray as xr

### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt
 
# 00 function --------------------------------

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

def haversine(lon1, lat1, lon2, lat2): # 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # transform decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine equation
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # earth radius
    return c * r * 1000

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

# ---- dataset info ---
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

# ---- start

for dtname in datasets:

    yearname = yearnameList[dtname]
    OUTDIR = OUT_DIR_List[dtname]
    reflat = latrefFile[dtname]
    reflon = lonrefFile[dtname]
    timereff = timerefFile[dtname]

    ### Time Management ###
    start_year, end_year = map(int, yearname.split('_'))
    Datestamp = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
    Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
    Date = list(Date0['date'])
    nday = len(Date)
    print(nday)

    # ridge seed only
    type_idx = 0

    composite_3regions = []

    for rgname in regions:

        # load the seeding event id in each region
        lat = np.load(reflat)
        lon = np.load(reflon)
        lat_mid = int(len(lat)/2) + 1 #91
        
        if rgname == 'SP':
            lat_NH = lat[0:lat_mid-1] # lat increasing
            kk = 'SH'
        else:
            lat_NH = lat[lat_mid:len(lat)]  # lat increasing
            kk = 'NH'
        print(lat_NH)
        print(lon)
        nlon = len(lon)
        nlat = len(lat)
        nlat_NH =len(lat_NH)
        
        # get the BAM index
        # time management
        start_year, end_year = map(int, yearname.split('_'))
        Datestamp = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
        Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
        Date = list(Date0['date'])
        nday = len(Date)
        print(nday)
        Month = Date0['date'].dt.month
        Year = Date0['date'].dt.year
        Day = Date0['date'].dt.day
        Datelist = list(Date0)
        # BAM dates
        feb_29_ind = Date0[(Month == 2) & (Day == 29)].index
        print(Date0.shape, flush=True)
        BAMDate = Date0

        # read the BAM index
        if kk == 'NH':
            with open(f'{OUTDIR}/synoptic_{dtname}_NH_BAM_event_peak_list.pkl', 'rb') as f:
                BAM_event_all = pickle.load(f)
            # read the BAM index values
            BI = np.load(f'{OUTDIR}/synoptic_{dtname}_NH_BAM_index_total_with_leap.npy')
        else:
            with open(f'{OUTDIR}/{dtname}_SH_BAM_event_peak_list.pkl', 'rb') as f:
                BAM_event_all = pickle.load(f)
            # read the BAM index values
            BI = np.load(f'{OUTDIR}/{dtname}_SH_BAM_index_total_with_leap.npy')
        # note: BAM_event_all is the index of BAMDate (the peak 1 day; need to expand it to 3-day high BAM period)

        # get high and low BAM states
        # first, expand the BAM_event_all to 3-day high BAM period
        BAM_high_days = set()
        for idx in BAM_event_all:
            for offset in [-1, 0, 1]:
                day = idx + offset
                if 0 <= day < len(Date0):
                    BAM_high_days.add(day)
        print('BAM_high_days:', sorted(BAM_high_days))
        print('len of BAM_high_days:', len(BAM_high_days))

        # find the lowest three values during a BAM period (-+12 days before the peak)
        # BAM index values: BI
        BAM_low_days = set()
        move = 12
        # for each event, if there is no 3 picked events, print its index
        for idx in BAM_event_all:
            day_st = max(0, idx-12)
            day_ed = min(len(Date0), idx+12+1)
            window = BI[day_st:day_ed]
            cand_sorted = np.argsort(window)  # from low to high
            picked = 0
            for li in cand_sorted:
                cand_day = day_st + int(li)
                if cand_day not in BAM_low_days:
                    BAM_low_days.add(cand_day)
                    picked += 1
                    if picked == 3:  # find three unrepeated lowest days
                        break
        print('BAM_low_days:', sorted(BAM_low_days))
        print('len of BAM_low_days:', len(BAM_low_days))


        # find the number of seeding events each day
        seedingflag_tp1 = np.load(f"{OUTDIR}/{dtname}_SD_SeedingClustersEventID_Type{type_idx+1}_{rgname}_ALL.npy")
        def getseedNumber(seedingflag):
            seeding_events_per_day = np.zeros(nday, dtype=int)
            seeding_LWA_per_day = np.zeros(nday, dtype=float)
            for day in range(nday):
                uniquevalues = np.unique(seedingflag[day, :])
                # also get the average daily LWA total value (total LWA over the grid with seeding events)
                # seedingflag_01 = seedingflag[day, :] > 0
                # totalLWA = np.nansum(LWA_Z[day, :, :]*seedingflag_01) / 100000000
                # seeding_LWA_per_day[day] = totalLWA
                eventlen = len(uniquevalues) - 1
                seeding_events_per_day[day] = eventlen

            return seeding_events_per_day, seeding_LWA_per_day
        
        seeding_events_per_day_tp1, _ = getseedNumber(seedingflag_tp1)
        seeding_events_per_day = seeding_events_per_day_tp1
        composite_3regions.append(seeding_events_per_day)
        print(f"Seeding events per day for region {rgname}:")
        print(seeding_events_per_day)

    # combine ATL and NP as NH --------------------------------------------------------------
    seeding_events_per_day_NH = composite_3regions[0] + composite_3regions[1]
    seeding_events_per_day_SH = composite_3regions[2]

    print(seeding_events_per_day_NH)
    print(seeding_events_per_day_SH)

    # read the BAM index
    with open(f'{OUTDIR}/synoptic_{dtname}_NH_BAM_event_peak_list.pkl', 'rb') as f:
        BAM_event_all_NH = pickle.load(f)
    with open(f'{OUTDIR}/{dtname}_SH_BAM_event_peak_list.pkl', 'rb') as f:
        BAM_event_all_SH = pickle.load(f)

    BI_NH = np.load(f'{OUTDIR}/synoptic_{dtname}_NH_BAM_index_total_with_leap.npy')
    BI_SH = np.load(f'{OUTDIR}/{dtname}_SH_BAM_index_total_with_leap.npy')

    ###
    # get all the peaking seeding days, center at the peak, and extend +/- 12 days
    # make a composite of seeding numbers for each day
    # make the composite array
    peak_days = BAM_event_all_NH
    composite_days = 25  # -12 to +12
    seeding_composite_NH = np.zeros((len(peak_days), composite_days))
    BMindex_NH = np.zeros((len(peak_days), composite_days))
    for i, peak_day in enumerate(peak_days):
        for offset in range(-12, 13):
            day = peak_day + offset
            if 0 <= day < nday:
                seeding_composite_NH[i, offset + 12] = seeding_events_per_day_NH[day]
                BMindex_NH[i, offset + 12] = BI_NH[day]

    peak_days = BAM_event_all_SH
    composite_days = 25  # -12 to +12
    seeding_composite_SH = np.zeros((len(peak_days), composite_days))
    BMindex_SH = np.zeros((len(peak_days), composite_days))
    for i, peak_day in enumerate(peak_days):
        for offset in range(-12, 13):
            day = peak_day + offset
            if 0 <= day < nday:
                seeding_composite_SH[i, offset + 12] = seeding_events_per_day_SH[day]
                BMindex_SH[i, offset + 12] = BI_SH[day]

    # get the composite mean
    seeding_composite_mean_NH = np.nanmean(seeding_composite_NH, axis=0)
    seeding_composite_std_NH  = np.nanstd(seeding_composite_NH, axis=0)
    N = np.sum(~np.isnan(seeding_composite_NH), axis=0) 
    seeding_composite_se_NH = seeding_composite_std_NH / np.sqrt(np.maximum(N, 1))
    BAMindex_composite_mean_NH = np.nanmean(BMindex_NH, axis=0)
    BAMindex_composite_std_NH  = np.nanstd(BMindex_NH, axis=0)
    N_BI_NH = np.sum(~np.isnan(BMindex_NH), axis=0) 
    BAMindex_composite_se_NH = BAMindex_composite_std_NH / np.sqrt(np.maximum(N_BI_NH, 1))

    seeding_composite_mean_SH = np.nanmean(seeding_composite_SH, axis=0)
    seeding_composite_std_SH  = np.nanstd(seeding_composite_SH, axis=0)
    N = np.sum(~np.isnan(seeding_composite_SH), axis=0) 
    seeding_composite_se_SH = seeding_composite_std_SH / np.sqrt(np.maximum(N, 1))
    BAMindex_composite_mean_SH = np.nanmean(BMindex_SH, axis=0)
    BAMindex_composite_std_SH  = np.nanstd(BMindex_SH, axis=0)
    N_BI_SH = np.sum(~np.isnan(BMindex_SH), axis=0) 
    BAMindex_composite_se_SH = BAMindex_composite_std_SH / np.sqrt(np.maximum(N_BI_SH, 1))

    np.save(f"{OUTDIR}/{dtname}_synoptic_BAMseedNumber_composite_NH_mean.npy", seeding_composite_mean_NH)
    # np.save(f"{OUTDIR}/{dtname}_BAMseedNumber_composite_NH_se.npy", seeding_composite_se_NH)
    np.save(f"{OUTDIR}/{dtname}_synoptic_BAMindex_composite_NH_mean.npy", BAMindex_composite_mean_NH)
    # np.save(f"{OUTDIR}/{dtname}_BAMindex_composite_NH_se.npy", BAMindex_composite_se_NH)
    np.save(f"{OUTDIR}/{dtname}_BAMseedNumber_composite_SH_mean.npy", seeding_composite_mean_SH)
    # np.save(f"{OUTDIR}/{dtname}_BAMseedNumber_composite_SH_se.npy", seeding_composite_se_SH)
    np.save(f"{OUTDIR}/{dtname}_BAMindex_composite_SH_mean.npy", BAMindex_composite_mean_SH)
    # np.save(f"{OUTDIR}/{dtname}_BAMindex_composite_SH_se.npy", BAMindex_composite_se_SH)

