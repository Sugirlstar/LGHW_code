# %%
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
from collections import defaultdict
import sys

### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt
 
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

# %%
dtname = 'JRA3Q'
yearname = yearnameList[dtname]
OUTDIR = OUT_DIR_List[dtname]
reflat = latrefFile[dtname]
reflon = lonrefFile[dtname]
timereff = timerefFile[dtname]


# %%
#%%
### Read the Z500-based LWA ###
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
lat_NH = lat[lat_mid:len(lat)] # NH only!
LWA_Z = LWA_td[:,lat_mid:len(lat),:] # NH only!

print(lat_NH)
print(lon)
nlon = len(lon)
nlat = len(lat)
nlat_NH =len(lat_NH)

### Time Management ###
start_year, end_year = map(int, yearname.split('_'))
Datestamp = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Date = list(Date0['date'])
nday = len(Date)

# read the LWA_td_A and LWA_td_C 
LWA_td_A = np.load(f"{OUTDIR}/{dtname}_LWA_td_A_{yearname}_6hr.npy")
LWA_td_C = np.load(f"{OUTDIR}/{dtname}_LWA_td_C_{yearname}_6hr.npy")
# daily averaged:
T = LWA_td_A.shape[0]
T4 = T // 4
LWA_td_A = LWA_td_A[:T4*4]
LWA_td_A = LWA_td_A.reshape(T4, 4, len(lat), len(lon)).mean(axis=1)
# daily averaged:
T = LWA_td_C.shape[0]
T4 = T // 4
LWA_td_C = LWA_td_C[:T4*4]
LWA_td_C = LWA_td_C.reshape(T4, 4, len(lat), len(lon)).mean(axis=1)
# get the NH part
LWA_Z_A = LWA_td_A[:,lat_mid:len(lat),:] 
LWA_Z_C = LWA_td_C[:,lat_mid:len(lat),:] 


### Time Management ###
start_year, end_year = map(int, yearname.split('_'))
Datestamp = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Date = list(Date0['date'])
nday = len(Date)

# %%
# Parameters ###
dlat = dlon = 1
BlockingDuration = 5
SeedingDuration = 3
BlockingLonWidth = 15
SeedingLonWidth = 15
valueBlockingThresh = 50  # percentile
valueSeedingThresh = 25    # percentile

Duration = BlockingDuration; LonWidth = BlockingLonWidth; ThreshPercentile =valueBlockingThresh

print('Duration:', Duration)
print('LonWidth:', LonWidth)
print('valueBlockingThresh:', ThreshPercentile)

# %%

# get the clusters and connect labels across the dateline

def getConnectedLabels(LWA_Z, ThreshPercentile):

    #%% 02 Get Threshold and Duration --------------------------------
    LWA_max_lon = np.zeros((nlon*nday))
    for t in np.arange(nday):    
        for lo in np.arange(nlon):
            LWA_max_lon[t*nlon+lo] = np.max(LWA_Z[t,:,lo])
    Thresh = np.percentile(LWA_max_lon[:],ThreshPercentile)

    print('Threshold:', Thresh)

    ### Wave Event ###
    WE = np.zeros((nday,nlat_NH,nlon),dtype='uint8') 
    WE[LWA_Z>Thresh] = 255                    # Wave event

    #%% 03 connected component-labeling algorithm -----------------------
    num_labels = np.zeros(nday)
    labels = np.zeros((nday,nlat_NH,nlon))
    for d in np.arange(nday):
        num_labels[d], labels[d,:,:], stats, centroids  = cv2.connectedComponentsWithStats(WE[d,:,:], connectivity=4)

    # ------- Connect labels wrapping across the dateline (lon=0 and lon=360) -------
    labels_new = labels.copy()
    for d in range(nday):
        left  = labels_new[d, :, 0].astype(int)    # lon=0
        right = labels_new[d, :, -1].astype(int)   # lon=360

        mask = (left > 0) & (right > 0)
        if not np.any(mask):
            continue

        # pairs that need to be merged (left_label, right_label)
        pairs = np.unique(np.stack([left[mask], right[mask]], axis=1), axis=0)

        # get the minimal set of merges 
        labs = np.unique(pairs)
        rep = {int(x): int(x) for x in labs}   # initially each label represents itself

        changed = True
        while changed:
            changed = False
            # merge to the smaller label
            for a, b in pairs:
                ra, rb = rep[a], rep[b]
                r = ra if ra < rb else rb
                if rep[a] != r: rep[a] = r; changed = True
                if rep[b] != r: rep[b] = r; changed = True
            # path compression
            for x in list(rep.keys()):
                while rep[x] != rep[rep[x]]:
                    rep[x] = rep[rep[x]]

        # apply the merges
        lab = labels_new[d]
        for src, dst in rep.items():
            if src == dst: 
                continue
            lab[lab == src] = dst

        # compress labels to 1..K
        uniq = np.unique(lab); uniq = uniq[uniq > 0]
        if uniq.size > 0:
            OFFSET = 10**9
            for i, old in enumerate(uniq):
                lab[lab == old] = OFFSET + (i + 1)
            for i in range(uniq.size):
                lab[lab == OFFSET + (i + 1)] = i + 1
            num_labels[d] = uniq.size + 1  # include the background label 0

        labels_new[d] = lab

    return labels_new

connectedlabels_seeding = getConnectedLabels(LWA_Z, valueSeedingThresh)
connectedlabels_blocking = getConnectedLabels(LWA_Z, valueBlockingThresh)

# %%
# Parameters ###
DX_THRESH = int(18*3)
MIN_DIST = 5
TH_ABS = None

# based on the peaking value, do watershed segmentation to further split the large clusters
def group_peaks_circular_strict(peaks, dx_thresh, wrap_len, lon_col=1):
    
    K = len(peaks)
    if K == 0:
        return []
    if K == 1:
        return [np.asarray(peaks)]

    # 1) sort by lon
    order = np.argsort(peaks[:, lon_col].astype(float))
    P = peaks[order]
    lons = P[:, lon_col].astype(float)

    # 2) find the largest gap, cut the circle there
    diffs = np.diff(lons, append=lons[0] + wrap_len)   
    cut = int(np.argmax(diffs))
    P = np.concatenate([P[cut+1:], P[:cut+1]], axis=0)
    lons = P[:, lon_col].astype(float)

    # 3) unwrap, make sure lon is non-decreasing
    lons_unwrap = lons.copy()
    for i in range(1, K):
        if lons_unwrap[i] < lons_unwrap[i-1]:
            lons_unwrap[i:] += wrap_len

    # 4) greedy grouping: ensure each group's "span" <= dx_thresh
    groups = []
    start_idx = 0
    start_lon = lons_unwrap[0]
    for i in range(1, K):
        if (lons_unwrap[i] - start_lon) > dx_thresh:
            groups.append(P[start_idx:i])     # close the previous group
            start_idx = i
            start_lon = lons_unwrap[i]
    groups.append(P[start_idx:K])             # the last group

    return [np.array(g) for g in groups]

labels_final = np.zeros_like(connectedlabels_blocking, dtype=np.int32)
num_labels_final = np.zeros(nday, dtype=int)

for d in range(nday):
    lab0 = connectedlabels_blocking[d].astype(np.int32)
    field_s = LWA_Z[d].astype(float)

    next_id = 1  # 0 is the background label
    for comp_id in range(1, lab0.max() + 1):
        comp_mask = (lab0 == comp_id)
        if not np.any(comp_mask):
            continue

        # === longitude periodicity (0/360) local maxima finding ===
        # 1) 3 times tiling along lon
        F3 = np.concatenate([field_s, field_s, field_s], axis=1)
        M3 = np.concatenate([comp_mask, comp_mask, comp_mask], axis=1)
        # 2) find peaks on the 3x domain, allowing border peaks
        peaks3 = feature.peak_local_max(
            F3,
            min_distance=MIN_DIST,
            threshold_abs=(TH_ABS if TH_ABS is not None else None),
            labels=M3,
            exclude_border=False
        )
        # 3) keep only the "middle" peaks, map lon back to original domain
        if peaks3.size == 0:
            peaks = peaks3
        else:
            yy3 = peaks3[:, 0]
            xx3 = peaks3[:, 1]
            keep = (xx3 >= nlon) & (xx3 < 2*nlon)
            peaks = np.stack([yy3[keep], (xx3[keep] - nlon)], axis=1).astype(int)
            # 4) delete repeated peaks
            if peaks.size:
                peaks = np.unique(peaks, axis=0)

        if len(peaks) <= 1:
            # if no peaks or only 1 peak, do not cut
            labels_final[d][comp_mask] = next_id
            next_id += 1
            continue

        # group peaks by longitude distance
        groups = group_peaks_circular_strict(
            peaks=peaks,
            dx_thresh=DX_THRESH,
            wrap_len=nlon,  
            lon_col=1
        )

        if len(groups) == 1:
            labels_final[d][comp_mask] = next_id
            next_id += 1
            continue

        # pick one representative peak (the strongest) from each group as marker
        markers = np.zeros_like(lab0, dtype=np.int32)
        for k, grp in enumerate(groups, start=1):
            vals = [field_s[lat_i, lon_i] for (lat_i, lon_i) in grp]
            lat_i, lon_i = grp[int(np.argmax(vals))]
            markers[lat_i, lon_i] = k

        # masked watershed segmentation, do the watershed on -field_s
        # 1) 3 times tiling along lon
        F3 = np.concatenate([field_s, field_s, field_s], axis=1)
        M3 = np.concatenate([comp_mask, comp_mask, comp_mask], axis=1)
        # 2) only keep markers on the middle copy
        markers3 = np.zeros_like(M3, dtype=np.int32)
        yy, xx = np.where(markers > 0)
        markers3[yy, xx] = markers[yy, xx]
        markers3[yy, xx + nlon] = markers[yy, xx]
        markers3[yy, xx + 2*nlon] = markers[yy, xx]
        # 3) watershed on the 3x domain
        ws3 = segmentation.watershed(-F3, markers=markers3, mask=M3)
        # 4) cut the region
        ws = ws3[:, nlon:2*nlon].astype(np.int32)

        # write to final labels, re-assign IDs
        for k in range(1, ws.max() + 1):
            submask = (ws == k)
            if np.any(submask):
                labels_final[d][submask] = next_id
                next_id += 1

    num_labels_final[d] = labels_final[d].max()  # not include the background label 0

labels_blocking_watershed = labels_final


# %%
#%% 04 Get the maximum LWA location of each individule event and it's area and width --------------

def getClusterFeature(labels_new, LonWidth):

    lat_d = []; lon_d = []
    lon_w = []; lon_e = []; lon_wide = []
    labelx = []

    for d in np.arange(nday):

        # get the label values in this day
        labelvalue = np.unique(labels_new[d])
        labelvalue = labelvalue[labelvalue>0]

        if len(labelvalue)==0:
            lat_d.append([]); lon_d.append([])
            lon_w.append([]); lon_e.append([]); lon_wide.append([])
            labelx.append([])
            continue
        
        lat_list=[]; lon_list=[]
        lon_w_list=[]; lon_e_list=[]; lon_wide_list=[]
        label_list = []

        for n in np.arange(0,len(labelvalue)):

            lbv = labelvalue[n]
            LWA_d = np.zeros((nlat_NH, nlon))
            LWA_d[labels_new[d]==lbv]=LWA_Z[d][labels_new[d]==lbv]   ### isolate that wave event ###
            ### Get the maximum location, always get the first point ###
            ix, iy = np.argwhere(LWA_d == LWA_d.max())[0]  # get the maximum location
            
            ### Get the west east boundary longitude ###
            ixs, iys = np.where(labels_new[d]==lbv)
            lon_vals = np.sort(lon[iys] % 360)
            # calculate the width considering periodic boundary
            gaps = np.diff(np.r_[lon_vals, lon_vals[0] + 360])
            k = np.argmax(gaps)
            # the point after the largest gap is the west boundary
            # the point before the largest gap is the east boundary
            lon_w_i = lon_vals[(k+1) % lon_vals.size]
            lon_e_i = lon_vals[k]
            width = 360 - gaps[k]

            #%% 04 Single Event Filtering ---------------------------------
            if (abs(lat_NH[ix]) < 30) or (width < LonWidth): #or (width > 120):
                labels_new[d][labels_new[d]==lbv] = 0
                continue
            else:
                lat_list.append(lat_NH[ix])
                lon_list.append(lon[iy])
                lon_w_list.append(float(lon_w_i))
                lon_e_list.append(float(lon_e_i))
                lon_wide_list.append(float(width))
                label_list.append(lbv)

        lat_d.append(lat_list);   lon_d.append(lon_list)
        lon_w.append(lon_w_list); lon_e.append(lon_e_list)
        lon_wide.append(lon_wide_list)
        labelx.append(label_list)

    print('Identify 2: single event Feature Get done')
    return lat_d, lon_d, lon_w, lon_e, lon_wide, labelx, labels_new

lat_d, lon_d, lon_w, lon_e, lon_wide, labelx, labels_blocking_filtered = getClusterFeature(labels_blocking_watershed, BlockingLonWidth)


# %%
#%% 06 Pairing events during consecutive two days  (with the shortest dististance) --------

def pair2days(lat_d, lon_d):
    ########### the distance is limited to 18 degree longitude and 13.5 degree latitude #########
    next_index = [] # for each label cluster in day d, store the index of its pair in day d+1, if no pair, then nan
    lon_thresh = 18
    lat_thresh = 13.5
    for d in np.arange(nday-1):
        next_index_day = np.full(len(lon_d[d]), np.nan)
        ### create a matrix that contains the distance between each WE during the consective two days ###
        shift_L = np.zeros((len(lon_d[d]),len(lon_d[d+1]) ))
        for i in np.arange(len(lon_d[d])):
            for j in np.arange(len(lon_d[d+1])):
                shift_L[i,j] = haversine(lon_d[d][i], lat_d[d][i], lon_d[d+1][j], lat_d[d+1][j])    

        ### pair the events from the shortest distance among them ###
        for dd in np.arange( min(len(lon_d[d]), len(lon_d[d+1])) ):
            WE_i, WE_j = np.unravel_index(np.argmin(shift_L), shift_L.shape)

            ### note the distance between two paris is also limited a thrshold ###
            # lon_shift = abs(lon_d[d+1][WE_j] - lon_d[d][WE_i])
            # lat_shift = abs(lat_d[d+1][WE_j] - lat_d[d][WE_i])
            # ### correct the longitude shift due the periodic boundary ###        
            # if lon_shift > 180:
            #     lon_shift = lon_shift-360
            # if lon_shift < -180:
            #     lon_shift = lon_shift + 360
            delta = ( (lon_d[d+1][WE_j] - lon_d[d][WE_i] + 540) % 360 ) - 180
            lon_shift = abs(delta)
            lat_shift = abs(lat_d[d+1][WE_j] - lat_d[d][WE_i])
            
            if lon_shift<lon_thresh and lat_shift<lat_thresh:
                next_index_day[WE_i] = WE_j  ### these two events can be paired! ###
                shift_L[WE_i, :] = np.inf    ### make the distance related to these two events to infinity so that we can search the next shortest distance and avoid this pair ###
                shift_L[:, WE_j] = np.inf
            else:
                shift_L[WE_i, :] = np.inf
                shift_L[:, WE_j] = np.inf
                
        next_index.append(next_index_day)

    print('Identify 3: distance filter done -------------- ')
    return next_index

next_index_blocking = pair2days(lat_d, lon_d)

print('blocking paired index:')
print(next_index_blocking[100])


# %%
lat_d_0 = copy.deepcopy(lat_d)
lon_d_0 = copy.deepcopy(lon_d)

def trackBlocking(lat_d, lon_d, lon_wide, labelx, next_index, labels_new, Duration, stationary=True):

    lon_thresh = 18
    lat_thresh = 13.5

    #%% 07 Tracking Wave Events ---------------------------------
    B_freq = np.zeros((nlat_NH, nlon))         ### The final blocking frequency/total number 
    B_freq2 = np.zeros((nday ,nlat_NH, nlon))  ### Record whether a grid point is under a blocking for a certain day (only 1 and 0 in this 3D array)

    ########### Now we begin to track wave events #########
    Blocking_lat = []; Blocking_lon = []; Blocking_date = []
    Blocking_lon_wide = []; Blocking_area = []; Blocking_label = []
            
    for d in np.arange(nday-1):
        ### if all wave events within this day are tracked, then go to the next day ###
        if len(lon_d[d]) == 0:
            continue
                
        for i in np.arange(len(lon_d[d])):

            lbv = labelx[d][i]
            ### if this wave evnet is tracked, then go to the next wave event ###
            if np.isnan(lon_d[d][i]):
                continue

            ### tracking starts ###
            day = 0
            track_lon = []; track_lat = []; track_date = []
            track_lon_index = []; track_lat_index = []
            track_lon_wide = []; track_area = []; track_lat_wide = []; track_label = []
            B_count = np.zeros((nlat_NH, nlon))
            B = np.zeros((nlat_NH, nlon))
            B_count2 = []
            
            B_count[labels_new[d+day]==lbv]+=1
            B[labels_new[d+day]==lbv]=1            
            B_count2.append(B)
            
            track_lon.append(lon_d[d+day][i]); track_lon_index.append(i)               
            track_lat.append(lat_d[d+day][i]); track_lat_index.append(i)            
            track_date.append(Date[d+day])
            track_lon_wide.append(lon_wide[d+day][i])
            track_label.append(labels_new[d+day]==lbv)
            
            next_index_pair= next_index[d+day][i] ### find the pair event at next day, it could be nan ### the i-th event at day d -> j-th the next day
            
            if ~np.isnan(next_index_pair):
                next_index_pair = int(next_index_pair) # the index of the cluster at the next day
                
            # while ~np.isnan(next_index_pair) and abs(lon_d[d+day+1][next_index_pair]-track_lon[0]) < 1.5*18 and abs(lat_d[d+day+1][next_index_pair]-track_lat[0]) < 1.5*13.5:
            while ~np.isnan(next_index_pair):
                if stationary:
                    # —— compare with the initial point ——
                    lon_diff0 = ((lon_d[d+day+1][next_index_pair] - track_lon[0] + 540) % 360) - 180
                    lat_diff0 =  (lat_d[d+day+1][next_index_pair] - track_lat[0])
                    if (abs(lon_diff0) >= 1.5*lon_thresh) or (abs(lat_diff0) >= 1.5*lat_thresh):
                        break   

                ### if this event does have a pair next day, and the displacement is within 1.5*18 lons and 1.5*13.5 lats, then keep tracking ###
                nextpair_label = labelx[d+day+1][next_index_pair] # the label value of that cluster at the next day
                track_date.append(Date[d+day+1])
                B_count[labels_new[d+day+1]==nextpair_label]+=1
                B = np.zeros((nlat_NH, nlon))
                B[labels_new[d+day+1]==nextpair_label]=1            
                B_count2.append(B)
                    
                track_lon.append(lon_d[d+day+1][next_index_pair])
                track_lon_index.append(next_index_pair)
                track_lat.append(lat_d[d+day+1][next_index_pair])
                track_lat_index.append(next_index_pair)
                track_lon_wide.append(lon_wide[d+day+1][next_index_pair])
                track_label.append(labels_new[d+day+1]==nextpair_label)
                
                ### interate the day ###
                day+=1
                
                ### if this the last day, then jump out ###
                if d+day+1>nday-1:
                    break
                
                ### if not, then find a next pair ###
                next_index_pair = next_index[d+day][next_index_pair]
                if ~np.isnan(next_index_pair):
                    next_index_pair = int(next_index_pair)
            
            ### when there are no pair events in the next day, the track is end ###
            ### make sure the wave event is large enough ###
            if day+1 >= Duration:
                Blocking_lon.append(track_lon)
                Blocking_lat.append(track_lat)
                Blocking_date.append(track_date)
                Blocking_label.append(track_label)
                B_freq += B_count
                for dd in np.arange(len(B_count2)):
                    B_freq2[d+dd,:,:]+= B_count2[dd]
            
            ### Once we successfully detected a wave event, we mark that with nan to avoid repeating ###
            for dd in np.arange(day+1):
                lon_d[d+dd][track_lon_index[dd]] = np.nan
                lat_d[d+dd][track_lat_index[dd]] = np.nan
        
        if d % 1000 == 0:
            print('day', d, flush=True)

    print('Identify 4: event tracking done')

    return Blocking_lat, Blocking_lon, Blocking_date, Blocking_label, B_freq, B_freq2

Blocking_lat, Blocking_lon, Blocking_date, Blocking_label, B_freq, B_freq2 = trackBlocking(lat_d, lon_d, lon_wide, labelx, next_index_blocking, labels_blocking_filtered, Duration=1, stationary=True)


# %%

#%% 08 get the blocking peaking date and location and wave activity -----------------
Blocking_peaking_date = []
Blocking_peaking_date_index = []
Blocking_peaking_lon = []
Blocking_peaking_lat = []

for n in np.arange(len(Blocking_date)):
    start = Date.index(Blocking_date[n][0])
    end = Date.index(Blocking_date[n][-1])
    duration = len(Blocking_date[n])
    LWA_event_max = np.zeros((duration))

    for d in np.arange(duration):
        index = start+d
        lo = np.squeeze(np.array(np.where( lon == Blocking_lon[n][d])))
        la = np.squeeze(np.array(np.where( lat_NH == Blocking_lat[n][d])))    
        LWA_event_max[d]  = LWA_Z[index, la, lo]
    
    Blocking_peaking_date_index=int(np.squeeze(np.array(np.where( LWA_event_max==np.max(LWA_event_max) ))))
    Blocking_peaking_date.append(Blocking_date[n][Blocking_peaking_date_index])
    Blocking_peaking_lon.append(Blocking_lon[n][Blocking_peaking_date_index])
    Blocking_peaking_lat.append(Blocking_lat[n][Blocking_peaking_date_index])

print('Identify 5: peaking date and location done')

# %% 09 get the blocking types -----------------
# separate 3 types of blocks (ridge, trough, dipole) 
# Method: Focus on the peaking date, calculate the total LWA_AC and LWA_C of the block region 
BlockingTypeI = []
lat_range=int(int((90-np.max(Blocking_peaking_lat))/dlat)*2+1)
lon_range=int(30/dlon)+1

for n in np.arange(len(Blocking_lon)):

    LWA_AC_sum = 0
    LWA_C_sum = 0
    Blocking_A = []
    Blocking_C = []
        
    ### peaking date information ###
    peaking_date_index = Date.index(Blocking_peaking_date[n])
    peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_peaking_lon[n])))
    peaking_lat_index = np.squeeze(np.array(np.where( lat_NH[:]==Blocking_peaking_lat[n]))) 
    
    t = np.squeeze(np.where(np.array(Blocking_date[n]) == np.array(Blocking_peaking_date[n] )))
    
    LWA_max = LWA_Z[peaking_date_index,peaking_lat_index,peaking_lon_index]

    ### date LWA_AC and  date LWA_C ###
    LWA_AC  = LWA_Z_A[peaking_date_index,:,:]
    LWA_C  = LWA_Z_C[peaking_date_index,:,:]
    
    ### shift the field to make the block center location at the domain center ###
    LWA_AC = np.roll(LWA_AC, int(nlon/2)-peaking_lon_index, axis=1)
    LWA_C = np.roll(LWA_C,   int(nlon/2)-peaking_lon_index, axis=1)
    WE = np.roll(Blocking_label[n][t], int(nlon/2)-peaking_lon_index, axis=1)
    lon_roll = np.roll(lon,   int(nlon/2)-peaking_lon_index)

    LWA_AC = LWA_AC[  :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    LWA_C = LWA_C[    :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    WE = WE[   :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]       ### Note that we need to confine the blocking region to +-15 degrees
    
    LWA_AC_d = np.zeros((nlat_NH, lon_range))
    LWA_C_d = np.zeros((nlat_NH, lon_range))
    LWA_AC_d[WE == True]  = LWA_AC[WE== True]
    LWA_C_d[WE == True]  =  LWA_C[WE == True]
    
    LWA_AC_sum += LWA_AC_d.sum()
    LWA_C_sum += LWA_C_d.sum()
    Blocking_A.append(LWA_AC_d.sum())
    Blocking_C.append(LWA_C_d.sum())

    ### if the anticyclonic LWA is much stronger than cytclonic LWA, then it is defined as ridge ###
    ### if the anticyclonic LWA is comparable with cyclonic LWA, then it is defined as dipole ###
    ### if the anticyclonic LWA is weaker than cyclonic LWA, then it is defined as trough events ###
    
    if LWA_AC_sum > 10 * LWA_C_sum :
        BlockingTypeI.append(1)  # 1 for ridge
    elif LWA_C_sum > 2 * LWA_AC_sum:
        BlockingTypeI.append(2)  # 2 for trough
    else:
        BlockingTypeI.append(3)  # 3 for dipole

    print(n)

print(len(BlockingTypeI))

# %%
print(len(BlockingTypeI))

# %%
BlockingTotal_date = Blocking_date
BlockingTotal_label = Blocking_label
Blocking_freq = B_freq
Blocking_freq2 = B_freq2

# %%
# BlockingTotal_date: only select BlockingTypeI == 1
BlockingTotal_date_ridge = [BlockingTotal_date[i] for i in range(len(BlockingTypeI)) if BlockingTypeI[i] == 1]
BlockingTotal_label_ridge = [BlockingTotal_label[i] for i in range(len(BlockingTypeI)) if BlockingTypeI[i] == 1]

# save as list
with open(f"{OUTDIR}/{dtname}_SingleRidge_BlockingTotal_date_ridge.pkl", "wb") as f:
    pickle.dump(BlockingTotal_date_ridge, f)
with open(f"{OUTDIR}/{dtname}_SingleRidge_BlockingTotal_label_ridge.pkl", "wb") as f:
    pickle.dump(BlockingTotal_label_ridge, f)
