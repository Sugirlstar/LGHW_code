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
from math import radians, cos, sin, asin, sqrt
from scipy.ndimage import zoom

import sys

dtname = sys.argv[1]
kk = sys.argv[2]
blktype = int(sys.argv[3])
trackType = sys.argv[4]

print(dtname, kk, blktype, trackType, flush=True)

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

# get all the seeding indicies and make the composite
def getSliceSingle(LWA_dayi, latid, lonid, latLWA, lonLWA, LWA_td, 
                latup=30, latdown=30, lonleft=30, lonright=30):
    
    LWAlatStart = latid-latdown if latid-latdown >= 0 else 0 
    LWAlatEnd = latid+latup if latid+latup <= len(latLWA) else len(latLWA)
    LWALatSlice = LWA_td[LWA_dayi, LWAlatStart:LWAlatEnd, :]

    if latid-latdown < 0:
        num_new_rows = abs(latid - latdown)
        new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)
        LWALatSlice = np.vstack((new_rows, LWALatSlice))

    if latid + latup > len(latLWA):
        num_new_rows = latid + latup - len(latLWA)
        new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)
        LWALatSlice = np.vstack((LWALatSlice, new_rows))

    # lon
    start = lonid - lonleft
    end = lonid + lonright
    if start < 0:  
        indices = list(range(start + len(lonLWA), len(lonLWA))) + list(range(0, end))
    elif end >= len(lonLWA):
        indices = list(range(start, len(lonLWA))) + list(range(0, end - len(lonLWA)))
    else:
        indices = list(range(start, end))
    slice_LWA = LWALatSlice[:,indices]

    return slice_LWA


def findcenter(LWA_Z_timelist, masks_list):
    centers_list = []
    for k, masks in enumerate(masks_list):
        rgmask = LWA_Z_timelist[k] * masks
        y0, x0 = np.unravel_index(np.argmax(rgmask), rgmask.shape)
        centers_list.append((y0, x0))
    return centers_list

def _precompute_event(dates_list, masks_list):
    """
    preprocess one event (day by day):
      - date_to_days: date -> [day_idx, ...]
      - masks_bool: daily mask (bool array)
      - areas: daily count of 1s (for coverage pruning)
      - bboxes: daily minimum bounding box (i0,i1,j0,j1) or None (all 0)
    """

    LWA_Z_timelist = [LWA_Z[Date.index(dt)] for dt in dates_list]

    date_to_days = defaultdict(list)
    for d_idx, d in enumerate(dates_list):
        date_to_days[d].append(d_idx)

    masks_bool = [np.asarray(m, dtype=bool) for m in masks_list]

    areas = np.array([int(m.sum()) for m in masks_bool], dtype=np.int32)
    bboxes = []
    for m in masks_bool:
        if m.any():
            r, c = np.where(m)
            bboxes.append((r.min(), r.max(), c.min(), c.max()))
        else:
            bboxes.append(None)
    
    centers = findcenter(LWA_Z_timelist, masks_bool)

    return {
        "dates": dates_list,
        "date_to_days": date_to_days,
        "masks": masks_bool,
        "areas": areas,
        "bboxes": bboxes,
        "centers": centers
    }

# 01 ------------------------------------------------------------------------------------------------------
# %%
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
varnameList = {
    "ERA5": "z",
    "MERRA2": "H",
    "JRA3Q": "hgt-pres-an-ll125"
}

# 02 ------------------------------------------------------------------------------------------------------
# %%
# dtname = 'JRA3Q'
# kk = 'NH' # SH
# blktype = 1
# trackType = 'AC'

yearname = yearnameList[dtname]
OUTDIR = OUT_DIR_List[dtname]
reflat = latrefFile[dtname]
reflon = lonrefFile[dtname]
timereff = timerefFile[dtname]

# %%
# 03 read in the data ------------------------------------------------------------------------------------------------------
### 1 - Read the Z500-based LWA ###
lat = np.load(reflat)
lon = np.load(reflon)
LWA_td = np.load(f"{OUTDIR}/{dtname}_LWA_td_{yearname}_6hr.npy") 
# daily averaged data:
T = LWA_td.shape[0]
T4 = T // 4
LWA_td = LWA_td[:T4*4]
LWA_td = LWA_td.reshape(T4, 4, len(lat), len(lon)).mean(axis=1)
print(LWA_td.shape)
lat_mid = int(len(lat)/2) + 1 #91

### Time Management ###
start_year, end_year = map(int, yearname.split('_'))
Datestamp = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Date = list(Date0['date'])
nday = len(Date)

if kk == 'SH':
    lat_NH = lat[0:lat_mid-1]  # lat increasing
    LWA_Z = LWA_td[:,0:lat_mid-1,:] # SH only!
else:
    lat_NH = lat[lat_mid:len(lat)] # lat increasing
    LWA_Z = LWA_td[:,lat_mid:len(lat),:] # NH only!

print(lat_NH)
print(lon)
nlon = len(lon)
nlat = len(lat)
nlat_NH =len(lat_NH)

### 2- read in the seeding and blocking events ###
with open(f"{OUTDIR}/{dtname}_SD_SeedingTotal_date_{kk}", "rb") as fp:
    SeedingTotal_date = pickle.load(fp)
with open(f"{OUTDIR}/{dtname}_SD_SeedingTotal_label_{kk}", "rb") as fp:
    SeedingTotal_label = pickle.load(fp)
with open(f"{OUTDIR}/{dtname}_SD_SeedingTypeI_{kk}", "rb") as fp:
    SeedingTypeI = pickle.load(fp)

with open(f"{OUTDIR}/{dtname}_SD_BlockingTotal_date_{kk}", "rb") as fp:
    BlockingTotal_date = pickle.load(fp)
with open(f"{OUTDIR}/{dtname}_SD_BlockingTotal_label_{kk}", "rb") as fp:
    BlockingTotal_label = pickle.load(fp)
with open(f"{OUTDIR}/{dtname}_SD_BlockingTypeI_{kk}", "rb") as fp:
    BlockingTypeI = pickle.load(fp)

### 3 - read in the Z500 anomaly data ###
varn = varnameList[dtname]
# load the Z500 anomaly data
# Z500 anomaly data
ds = xr.open_dataset(f'/scratch/bell/hu1029/Data/processed/{dtname}_Z500anomaly_subtractseasonal_6hr_{yearname}_1dg.nc')
latZ500 = np.array(ds['lat']) # lat increasing
print(latZ500)
Zanom_origin = np.array(ds[varn].squeeze())  
print(np.shape(Zanom_origin), flush=True)
print('-------- Zanom loaded --------', flush=True)
lat_mid = int(len(latZ500)/2) + 1
print(kk)
if kk == 'SH':
    Zanom = Zanom_origin[:, 0:lat_mid-1, :]
    latZ500_NH = latZ500[0:lat_mid-1]
else:
    Zanom = Zanom_origin[:, lat_mid:len(latZ500), :]
    latZ500_NH = latZ500[lat_mid:len(latZ500)]
print(latZ500_NH)
# daily averaged data:
T = Zanom.shape[0]
T4 = T // 4
Zanom = Zanom[:T4*4]
Zanom_daily = Zanom.reshape(T4, 4, len(latZ500_NH), len(lon)).mean(axis=1)
print(Zanom_daily.shape)

#%%
# 04 Match seeding and blocking events ------------------------------------------------------------------------------------------------------
print(BlockingTypeI[0:10])
# get the sublist of BlockingTotal_date and BlockingTotal_label corresponding to BlockingType1
BlockingType1_indices = [i for i, B in enumerate(BlockingTypeI) if B == blktype]
BlockingTotal_date = [BlockingTotal_date[i] for i in BlockingType1_indices]
BlockingTotal_label = [BlockingTotal_label[i] for i in BlockingType1_indices]

# === Preprocess all blocking / seeding events ===
B_events = [_precompute_event(dts, lbls) for dts, lbls in zip(BlockingTotal_date, BlockingTotal_label)]
S_events = [_precompute_event(dts, lbls) for dts, lbls in zip(SeedingTotal_date,  SeedingTotal_label)]

# === Main loop: Match blocking events for each seeding event by "same day" and perform coverage judgment ===
seed_hit_block_idx = []
hit_in_blocking = []
hit_in_seed = []

for s_idx, S in enumerate(S_events):
    stats = {}  # b_idx -> [min_block_day, min_seed_day]
    S_masks = S["masks"]
    S_dates = S["dates"]
    S_areas = S["areas"]
    S_bboxes = S["bboxes"]
    S_centers = S["centers"]

    for s_day, (s_date, s_mask, s_area, s_bbox, s_center) in enumerate(zip(S_dates, S_masks, S_areas, S_bboxes, S_centers)):

        # only consider "same day" blocking candidates: all b_day ∈ B.date_to_days[s_date] for b_idx
        for b_idx, B in enumerate(B_events):
            if b_idx in stats:
                continue  # if already hit, skip

            b_day_list = B["date_to_days"].get(s_date, [])
            if not b_day_list:
                continue

            for b_day in b_day_list:

                b_mask  = B["masks"][b_day]
                b_area  = B["areas"][b_day]
                b_bbox  = B["bboxes"][b_day]
                b_center = B["centers"][b_day]

                # first check: if the center matches
                if s_center == b_center:
                    if b_idx not in stats:
                        stats[b_idx] = [b_day, s_day]
                    break  # this blocking's day has been hit, no need to look at other b_days on the same day (if any ties exist)

                else:
                    # second check: if the coverage matches
                    # branch cutting 1: area (blocking cannot be larger than seeding)
                    if b_area > s_area:
                        continue
                    # branch cutting 2: bbox (blocking bbox must fall within seeding bbox)
                    if s_bbox is not None and b_bbox is not None:
                        i0s, i1s, j0s, j1s = s_bbox
                        i0b, i1b, j0b, j1b = b_bbox
                        if not (i0b >= i0s and i1b <= i1s and j0b >= j0s and j1b <= j1s):
                            continue

                    # coverage judgment: all 1s of block must fall within 1s of seed <=> (b_mask & ~s_mask).any() == False
                    if not np.bitwise_and(b_mask, np.logical_not(s_mask)).any():
                        # hit: record the "earliest b_day" in the same day (b_day_list is already in original order, usually the earliest)
                        if b_idx not in stats:
                            stats[b_idx] = [b_day, s_day]
                        break  # this blocking's day has been hit, no need to look at other b_days on the same day (if any ties exist)

    if stats:
        ids = sorted(stats.keys())
        seed_hit_block_idx.append(ids)
        hit_in_blocking.append([stats[i][0] for i in ids])
        hit_in_seed.append([stats[i][1] for i in ids])
    else:
        seed_hit_block_idx.append([-1])
        hit_in_blocking.append([-1])
        hit_in_seed.append([-1])
    
    if (s_idx + 1) % 20 == 0:
        print(f"{(s_idx+1)/len(SeedingTotal_label)*100:.1f}% done", flush=True)

print("-------- seed_hit_block_idx：", seed_hit_block_idx[:5], " ...")
print("-------- hit_in_blocking：",   hit_in_blocking[:5], " ...")
print("-------- hit_in_seed：",       hit_in_seed[:5], " ...")

# get the hitted seed id
allhitblockingindices = [
    b for seed_list in seed_hit_block_idx
    for b in seed_list
    if b != -1
]
print('allhitblockingindices:', np.unique(allhitblockingindices))
# get the hitted blocking id
hit_seed_indices = [
    i for i, bids in enumerate(seed_hit_block_idx)
    if bids != [-1]
]
print('hit_seed_indices:', hit_seed_indices)

with open(f'{OUTDIR}/{dtname}_{kk}_{blktype}_seed_hit_block_idx.pkl', 'wb') as f:
    pickle.dump(seed_hit_block_idx, f)
with open(f'{OUTDIR}/{dtname}_{kk}_{blktype}_hit_in_blocking.pkl', 'wb') as f:
    pickle.dump(hit_in_blocking, f)
with open(f'{OUTDIR}/{dtname}_{kk}_{blktype}_hit_in_seed.pkl', 'wb') as f:
    pickle.dump(hit_in_seed, f)

# %%
# 05 statistics of seed developing into blocking ------------------------------------------------------------------------------------------------------
len_develop = len(hit_seed_indices)
percent_develop = len_develop / len(SeedingTotal_date) * 100
# find the percentage of seeding events that develop into blocking on its first day
seed_develop_firstday = [
    i
    for i, (bids, days_in_seed) in enumerate(zip(seed_hit_block_idx, hit_in_seed))
    if bids != [-1] and all(d == 0 for d in days_in_seed)
]
len_develop_firstday = len(seed_develop_firstday)
print('seed_develop_firstday indices:', seed_develop_firstday)
percent_develop_firstday = len_develop_firstday / len_develop * 100
blk_developfromseed = np.unique(allhitblockingindices)
len_blk_developfromseed = len(blk_developfromseed)
percent_blk_developfromseed = len_blk_developfromseed / len(BlockingTotal_date) * 100
print(f'number of seeding events: {len(SeedingTotal_date)}')
print(f'number of seeding events that develop into blocking: {len_develop}, percentage: {percent_develop:.2f}%')
print(f'number of seeding events that develop into blocking on the first day: {len_develop_firstday}, percentage: {percent_develop_firstday:.2f}%')
print(f'number of blocking events: {len(BlockingTotal_date)}')
print(f'number of blocking events that develop from seeding: {len_blk_developfromseed}, percentage: {percent_blk_developfromseed:.2f}%')

# 06 read in track data ------------------------------------------------------------------------------------------------------
# 1 - track lat/lon values
lon_track = lon
latSH_track = lat_NH # the track data's lat is increasing order
print(latSH_track)
# time of track data (6-hourly)
ds = xr.open_dataset(timereff)
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)
timeiarr = np.array(timei)
trackPoints_array = np.load(f'{OUTDIR}/{dtname}_{trackType}trackPoints_array1dg_{kk}.npy') # 3d-array: time, lat, lon, 1/0 mask for track points
# 
# time management
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
nday = len(Date0)
Month = Date0['date'].dt.month
Year = Date0['date'].dt.year
Day = Date0['date'].dt.day
Datelist = list(Date0)

# %%
# 06 check the Seeding with Track points ------------------------------------------------------------------------------------------------------
# for each seeding event, check if there is any track point within the seeding mask on the same day

# 1 - compress the trackPoints_array to daily data (from 6-hourly to daily)
nlat_track = len(latSH_track)
nlon_track = len(lon_track)
T_track = trackPoints_array.shape[0]
T4_track = T_track // 4
trackPoints_array = trackPoints_array[:T4_track*4]
# take the maximum value of the 4 6-hourly values (if any 1, then 1)
trackPoints_array_daily = trackPoints_array.reshape(T4_track, 4, nlat_track, nlon_track).max(axis=1) # daily max, shape: time, lat, lon
trackPoints_array_daily = trackPoints_array_daily.astype(bool)
print('trackPoints_array shape (daily):', trackPoints_array_daily.shape)
# 2 - convert the track to bool
trackPoints_array_daily_bool = (trackPoints_array_daily != 0)  #  bool

# %%
# 07 get the two conditions ------------------------------------------------------------------------------------------------------
ttlen = len(SeedingTotal_date)
s1 = hit_seed_indices      # developed seeding events (matched to a blocking event)
s2 = [i for i in np.arange(ttlen) if i not in hit_seed_indices]  # undeveloped seeding events
situationlist = [s1, s2]
print(len(s1), len(s2))
print(len(SeedingTotal_date))
situationnamelist = ['Developed', 'Decayed']

for ii,targetindices in enumerate(situationlist):

    print(targetindices)
    # %% 04 make the composite of seeds ------------------------
    latarr = np.arange((0-25),(0+25))
    latarr = latarr[::-1]  # from high to low
    lonarr = np.arange((0-30),(0+30))
    seedCompArr = np.full((10, len(targetindices), len(latarr), len(lonarr)), np.nan)  # the composite array for the seeds
    trackCompArr = np.zeros((10, len(targetindices), len(latarr), len(lonarr)))  # the composite array for the track points
    comptracklat_loc = [[] for _ in range(10)]
    comptracklon_loc = [[] for _ in range(10)]
    print(len(comptracklat_loc), len(comptracklon_loc))

    for k, sindex in enumerate(targetindices):

        seedDates = SeedingTotal_date[sindex]
        seedLabels = SeedingTotal_label[sindex]

        firstday = seedDates[0]
        firstdayidx = Date.index(firstday) 

        # get the day slice of the seed
        mask = seedLabels[0] # get the first day as the center reference
        mask0 = (mask>0)  # convert to binary mask

        center0 = np.unravel_index(np.argmax(LWA_Z[firstdayidx,:,:] * mask0, axis=None), LWA_Z[firstdayidx,:,:].shape)
        centerlat = center0[0]  # the latitude index of the center
        centerlon = center0[1]  # the longitude index of the center

        # print(f'Center lat index: {centerlat}, lon index: {centerlon}', flush=True)
        # print(f'Center lat value: {lat_NH[centerlat]}, lon value: {lon[centerlon]}', flush=True)
        if firstdayidx+9 >= nday:
            print('Skipping event at the end of the dataset')
            continue

        # get the Z500 slices -----------
        for j in [0,1,2,3,4,5,6,7,8,9]:  # get 0-9 days
            slice_k = getSliceSingle(firstdayidx+j, centerlat, centerlon, lat_NH, lon, LWA_Z,
                                        latup=25, latdown=25, lonleft=30, lonright=30)
            seedCompArr[j+0,k,:,:] = slice_k / 100000000

            slice_track = getSliceSingle(firstdayidx+j, centerlat, centerlon, lat_NH, lon, trackPoints_array_daily_bool,
                                            latup=25, latdown=25, lonleft=30, lonright=30)  # get the track slice
            tracklat, tracklon = np.where(slice_track==1)
            comptracklat_loc[j+0].extend(tracklat)
            comptracklon_loc[j+0].extend(tracklon)

            # Only add to trackCompArr where both the target and destination are not nan
            trackCompArr[j+0, k, :, :] += slice_track == 1
            # valid_mask = ~np.isnan(slice_track) & ~np.isnan(trackCompArr[j+3, k, :, :])
            # trackCompArr[j+3, k, :, :][valid_mask] += np.where((slice_track == 1) & valid_mask, 1, 0)[valid_mask]

    print(np.shape(seedCompArr))
    centeredComp = np.nanmean(seedCompArr, axis=1)  # average over the event dimension
    centeredCompTrack = np.nansum(trackCompArr, axis=1) # sum over the event dimension
    centeredCompTrack = centeredCompTrack/len(targetindices)  # calculate the average track points per event
    print(np.shape(centeredComp))
    print(len(comptracklat_loc), len(comptracklon_loc))

    # get the lat average
    centeredComp_latavg = np.nanmean(centeredComp, axis=1)  # average over the lat dimension
    centeredCompTrack_latavg = np.nansum(centeredCompTrack, axis=1)  # sum over the lat dimension
    print(np.nanmax(centeredCompTrack_latavg), np.nanmin(centeredCompTrack_latavg))

    np.save(f'{OUTDIR}/{dtname}_{kk}_{blktype}_{situationnamelist[ii]}_seedCompArr.npy', seedCompArr) # time, event, lat, lon (LWA)
    np.save(f'{OUTDIR}/{dtname}_{kk}_{blktype}_{trackType}_{situationnamelist[ii]}_trackCompArr.npy', trackCompArr) # time, event, lat, lon (track points)
    # save targetindices as .pickle
    with open(f'{OUTDIR}/{dtname}_{kk}_{blktype}_{situationnamelist[ii]}_targetindices.pkl', 'wb') as f:
        pickle.dump(targetindices, f)
    np.save(f'{OUTDIR}/{dtname}_{kk}_{blktype}_centeredComp_latavg_{situationnamelist[ii]}.npy', centeredComp_latavg) # time, lon (LWA)
    np.save(f'{OUTDIR}/{dtname}_{kk}_{blktype}_{trackType}_centeredCompTrack_latavg_{situationnamelist[ii]}.npy', centeredCompTrack_latavg) # time, lon (track points)

print(f'{dtname}, {kk}, {blktype}, {trackType}  -  All done!')
