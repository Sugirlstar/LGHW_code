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
import cv2
import copy

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

beforelen = 4*4 # the number of timesteps before the blocking event (4 days)
typeid = 1
rgname = "ATL"
ss = "ALL"
cyc = "AC"

if rgname == "SP":
    HMi = '_SH'
else:
    HMi = ''

lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)

# %% 01 get the LWA events ------------------------
# read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

# read in the LWA data
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

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

# # get the wave event
# T = LWA_td.shape[0]
# T4 = T // 4
# LWA_Z = LWA_td[:T4*4]
# LWA_Z = LWA_Z.reshape(T4, 4, len(latLWA), len(lonLWA)).mean(axis=1)

# nlon = len(lonLWA)
# nlat = len(latLWA)
# nday = np.shape(LWA_Z)[0] # the number of days
# LWA_max_lon = np.zeros((nlon*nday))
# for t in np.arange(nday):    
#     for lo in np.arange(nlon):
#         LWA_max_lon[t*nlon+lo] = np.max(LWA_Z[t,:,lo])
# Thresh = np.median(LWA_max_lon[:])
# Duration = 5 
# print('Threshold:', Thresh)
# ### Wave Event ###
# WEvent = np.zeros((nday,nlat,nlon),dtype='uint8') 
# WEvent[LWA_Z>Thresh] = 1                    # Wave event daily
# WEvent = np.repeat(WEvent, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)
# nday = nday * 4 # update the number of days 

# ### connected component-labeling algorithm ###
# num_labels = np.zeros(nday)
# labels = np.zeros((nday,nlat,nlon))
# for d in np.arange(nday):
#     num_labels[d], labels[d,:,:], stats, centroids  = cv2.connectedComponentsWithStats(WEvent[d,:,:], connectivity=4)

# ####### connect the label around 180, since they are labeled separately ########
# ####### but actually they should belong to the same label  ########
# labels_new = copy.copy(labels)
# for d in np.arange(nday):
#     if np.any(labels_new[d,:,0]) == 0 or np.any(labels_new[d,:,-1]) == 0:   ## If there are no events at either -180 or 179.375, then WEvent don't need to do any connection
#         continue
            
#     column_0 = np.zeros((nlat,3))       ## WEvent assume there are at most three wave events at column 0 (-180) (actuaaly most of the time there is just one)
#     column_end = np.zeros((nlat,3))
#     label_0 = np.zeros(3)
#     label_end = np.zeros(3)
    
#     ## Get the wave event at column 0 (0) ##
#     start_lat0 = 0
#     for i in np.arange(3):
#         for la in np.arange(start_lat0, nlat):
#             if labels_new[d,la,0]==0:
#                 continue
#             if labels_new[d,la,0]!=0:
#                 label_0[i]=labels_new[d,la,0]
#                 column_0[la,i]=labels_new[d,la,0]
#                 if labels_new[d,la+1,0]!=0:
#                     continue
#                 if labels_new[d,la+1,0]==0:
#                     start_lat0 = la+1
#                     break 

#         ## Get the wave event at column -1 (359) ## 
#         start_lat1 = 0
#         for j in np.arange(3):
#             for la in np.arange(start_lat1, nlat):
#                 if labels_new[d,la,-1]==0:
#                     continue
#                 if labels_new[d,la,-1]!=0:
#                     label_end[j]=labels_new[d,la,-1]
#                     column_end[la,j]=labels_new[d,la,-1]
#                     if labels_new[d,la+1,-1]!=0:
#                         continue
#                     if labels_new[d,la+1,-1]==0:
#                         start_lat1 = la+1
#                         break                       
#             ## Compare the two cloumns at 0 and 359, and connect the label if the two are indeed connected
#             if (column_end[:,i]*column_0[:,j]).mean() == 0:
#                 continue                
#             if (column_end*column_0).mean() != 0:
#                 num_labels[d]-=1
#                 if label_0[i] < label_end[j]:
#                     labels_new[d][labels_new[d]==label_end[j]] = label_0[i]
#                     labels_new[d][labels_new[d]>label_end[j]] = (labels_new[d]-1)[labels_new[d]>label_end[j]]            
#                 if label_0[i] > label_end[j]:
#                     labels_new[d][labels_new[d]==label_0[i]] = label_end[j]
#                     labels_new[d][labels_new[d]>label_0[i]] = (labels_new[d]-1)[labels_new[d]>label_0[i]]

# print('labels -> labels_new, num_labels -> num_labels', flush=True)

# np.save(f'/scratch/bell/hu1029/LGHW/Blk_LWAconnectedLabels_{HMi}.npy', LWA_td)
# print('connected labels saved', flush=True)

# # %% 02 get the blocking embryo for each event ------------------------

# # read in blocking event information
# # the id of each blocking event (not all events! just the events within the target region)
# with open(f'/scratch/bell/hu1029/LGHW/BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
#     blkEindex = pickle.load(f)

# # Center of 1st day of each blocking event
# with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
#     centerDate1 = pickle.load(fp)
# with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayLatList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
#     centerLat1 = pickle.load(fp)
# with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayLonList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
#     centerLon1 = pickle.load(fp)

# embryoBlk = np.zeros((nday, nlat, nlon), dtype=np.int64) # the embryo blocking event array, shape=(nblk, nlat, nlon)
# embryoWidthList = []
# embryoDistList = []
# embryoLWA = []
# for ki,k in enumerate(blkEindex):

#     print('Processing blocking event:', ki, flush=True)
#     eventcenterlat = centerLat1[ki]
#     eventcenterlon = centerLon1[ki]
#     eventcenterlati = np.argmin(np.abs(latLWA - eventcenterlat))
#     eventcenterloni = np.argmin(np.abs(lonLWA - eventcenterlon))

#     # find the 3d slice in the BlockingEIndex array where the first dimension matches the target_value
#     blk1stday = timei.index(centerDate1[ki]) # the index of the first day of the blocking event
#     # get the target LWA 
#     before_start = blk1stday - beforelen # the target blocking event index
#     before_end = blk1stday-1 # the target blocking event index
#     if before_start < 0: # if the first day of the embryo event is before the start of the LWA data
#         print('The first day of the blocking event is before the start of the LWA data, skipping this event.', flush=True)
#         embryoWidthList.append([]) 
#         embryoDistList.append([]) 
#         embryoLWA.append([]) 
#         continue
#     # LWA cluster
#     # labelbool = labels_new>0
#     # LWAcluster = LWA_td * labelbool # LWA cluster, shape=(nday, nlat, nlon)
#     # for each day, get each cluster's LWA maximum location
#     eventlonwidth = []
#     eventdist = []
#     dayLWA = []

#     for d in np.arange(before_end, before_start-1, -1): # from before_end to before_start, step -1
#         print('     Processing day:', d, flush=True)
#         labelvals = np.unique(labels_new[d,:,:])
#         labelvals = labelvals[labelvals>0]  # remove the background label (0)
#         if len(labelvals) == 0:  # if there are no labels, skip this day
#             print('     No labels found for this day, skipping.', flush=True)
#             eventlonwidth.append(np.nan)
#             eventdist.append(np.nan)
#             dayLWA.append(0)
#             continue

#         targetk = 0
#         min_dist = np.inf
#         for lbv in labelvals:
#             LWAcluster = np.where(labels_new[d,:,:]==lbv, LWA_td[d,:,:], 0) # get the LWA cluster for each day
#             # get the maximum LWA location for each cluster
#             maxloc = np.unravel_index(np.argmax(LWAcluster, axis=None), LWAcluster.shape)
#             dist = np.sqrt((maxloc[0] - eventcenterlati)**2 + (maxloc[1] - eventcenterloni)**2)
#             if dist < min_dist:
#                 min_dist = dist
#                 targetk = lbv
#         finalLabel = labels_new[d,:,:] == targetk # get the target label for the blocking event
#         y_idx, x_idx = np.where(finalLabel > 0)
#         finalmaxloc = np.unravel_index(np.argmax(LWA_td[d,:,:] * finalLabel, axis=None), LWA_td[d,:,:].shape)
#         eventcenterlati = finalmaxloc[0] # the latitude index of the center of the blocking event
#         eventcenterloni = finalmaxloc[1] # the longitude index of the center of the blocking event
#         lat_in = lat_min <= eventcenterlati <= lat_max
#         if lon_min <= lon_max:
#             lon_in = lon_min <= eventcenterloni <= lon_max
#         else:
#             lon_in = (eventcenterloni >= lon_min) or (eventcenterloni <= lon_max)
#         if(lat_in and lon_in):
#             # calculate the lon width and distance to the center of the blocking event
#             x_sorted = np.sort(np.unique(x_idx))
#             print('lon values sorted:', x_sorted, flush=True)
#             gaps = np.diff(np.concatenate((x_sorted, [x_sorted[0] + nlon])))
#             max_gap_idx = np.argmax(gaps)
#             west_idx = (x_sorted[(max_gap_idx + 1) % len(x_sorted)]) % nlon
#             east_idx = (x_sorted[max_gap_idx]) % nlon
#             if west_idx > east_idx:
#                 lon_width = 360 - west_idx + east_idx
#             else:
#                 lon_width = east_idx - west_idx 
#             print('east:', east_idx, 'west:', west_idx, 'lon width:', lon_width, flush=True)

#             eventdist.append(min_dist) # the distance to the center of the blocking event
#             eventlonwidth.append(lon_width) # the lon width of the blocking event
#             dayLWA.append(np.nansum(finalLabel * LWA_td[d,:,:]))  # get the LWA sum
#             embryoBlk[d, y_idx, x_idx] = k # get the LWA cluster for the target blocking event
#         else:
#             eventdist.append(np.nan) # if the blocking event is not in the target region, set the distance to 0
#             eventlonwidth.append(np.nan) # if the blocking event is not in the target region, set the lon width to 0
#             dayLWA.append(0) # if the blocking event is not in the target region, set the LWA sum to 0

#     embryoWidthList.append(eventlonwidth) # the lon width of the blocking event
#     embryoDistList.append(eventdist) # the distance to the center of the blocking event
#     embryoLWA.append(dayLWA) # the LWA sum for each day of the blocking event

# np.save(f'/scratch/bell/hu1029/LGHW/BlockingEmbryoIndex_Type{typeid}_{rgname}_{ss}_{cyc}.npy', embryoBlk)
# with open(f'/scratch/bell/hu1029/LGHW/BlockingEmbryoWidthList_Type{typeid}_{rgname}_{ss}_{cyc}.pkl', 'wb') as f:
#     pickle.dump(embryoWidthList, f)
# with open(f'/scratch/bell/hu1029/LGHW/BlockingEmbryoDistList_Type{typeid}_{rgname}_{ss}_{cyc}.pkl', 'wb') as f:
#     pickle.dump(embryoDistList, f)
# with open(f'/scratch/bell/hu1029/LGHW/BlockingEmbryoLWA_Type{typeid}_{rgname}_{ss}_{cyc}.pkl', 'wb') as f:
#     pickle.dump(embryoLWA, f)

# print('Embryo data saved', flush=True)

embryoBlk = np.load(f'/scratch/bell/hu1029/LGHW/BlockingEmbryoIndex_Type{typeid}_{rgname}_{ss}_{cyc}.npy')
with open(f'/scratch/bell/hu1029/LGHW/BlockingEmbryoWidthList_Type{typeid}_{rgname}_{ss}_{cyc}.pkl', 'rb') as f:
    embryoWidthList = pickle.load(f)
with open(f'/scratch/bell/hu1029/LGHW/BlockingEmbryoDistList_Type{typeid}_{rgname}_{ss}_{cyc}.pkl', 'rb') as f:
    embryoDistList = pickle.load(f)
with open(f'/scratch/bell/hu1029/LGHW/BlockingEmbryoLWA_Type{typeid}_{rgname}_{ss}_{cyc}.pkl', 'rb') as f:
    embryoLWA = pickle.load(f)

# print the statistics of the embryo blocking events
embryoWidthListavg = [np.nanmean(i) if len(i) > 0 else 0 for i in embryoWidthList]
embryoDistListavg = [np.nanmean(i) if len(i) > 0 else 0 for i in embryoDistList]
embryoLWAavg = [np.nanmean(i) if len(i) > 0 else 0 for i in embryoLWA]
lastdayWidth = [i[0] if len(i) > 0 else 0 for i in embryoWidthList]
lastdayDist = [i[0] if len(i) > 0 else 0 for i in embryoDistList]
lastdayWidth = [lastdayWidth[i] for i in range(len(lastdayWidth)) if not np.isnan(lastdayWidth[i])]
lastdayDist = [lastdayDist[i] for i in range(len(lastdayDist)) if not np.isnan(lastdayDist[i])]

# print the length of non-nan values in embryoWidthList and embryoDistList
embryoWidthList = [len([x for x in sublist if x]) for sublist in embryoWidthList]
print('Number of non-nan values in embryoWidthList:', embryoWidthList, flush=True)
total_count = sum(x > 1 for sublist in embryoWidthList for x in sublist)
print('Total number of blocking events with more than 1 embryo:', total_count, flush=True)

print('Embryo blocking events statistics:', flush=True)
print('Average lon width:', np.nanmean(embryoWidthListavg), '±', np.nanstd(embryoWidthListavg), flush=True)
print('Average distance to center:', np.nanmean(embryoDistListavg), '±', np.nanstd(embryoDistListavg), flush=True)
print('Average LWA sum:', np.nanmean(embryoLWAavg), '±', np.nanstd(embryoLWAavg), flush=True)

bin_width = 1
bins = np.arange(np.floor(min(lastdayDist)), np.ceil(max(lastdayDist)) + bin_width, bin_width)
plt.hist(lastdayDist, bins=bins, edgecolor='black', align='left')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of lastdayDist')
plt.show()
plt.savefig(f'Embryo_lastdayDist_{typeid}_{rgname}_{ss}_{cyc}.png', bbox_inches='tight', dpi=300)
plt.close()

bin_width = 5
bins = np.arange(np.floor(min(lastdayWidth)), np.ceil(max(lastdayWidth)) + bin_width, bin_width)
plt.hist(lastdayWidth,  bins=bins, edgecolor='black', align='left')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('lastdayWidth Distribution')
plt.show()
plt.savefig(f'Embryo_lastdayWidth_{typeid}_{rgname}_{ss}_{cyc}.png', bbox_inches='tight', dpi=300)
plt.close()

# %% 03 get the event's label ------------------------
# check the embryo blocking event label - make the comparing plot
evid = 45
ti, _, _ = np.where(embryoBlk == 45) # get the time index of the target blocking event
tivales = np.unique(ti) # get the unique time index of the target blocking event

# plot1: the map of the LWA
lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
lat_min1, lat_max1, lon_min1, lon_max1, loncenter = PlotBoundary(rgname)
for dayi, i in enumerate(tivales):

    print('Plotting day:', i, flush=True)
    fig, ax, cf = create_Map(lonLWA,latLWA,LWA_td[i,:,:],fill=True,fig=None,
                             leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
                                minv=0, maxv=np.nanmax(LWA_td[i,:,:]), interv=11, figsize=(12,5),
                                centralLon=loncenter, colr='PuBu', extend='max',title=f'Blocking LWA on {timei[i]}',)
    ax.contour(lonLWA, latLWA, embryoBlk[i,:,:].astype(float), levels=[0.5], 
                     colors='darkred', linewidths=1.5, transform=ccrs.PlateCarree())
    
    addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1)  
    plt.colorbar(cf,ax=ax,orientation='horizontal',label='LWA',fraction=0.04, pad=0.1)

    plt.show()
    plt.savefig(f'Embryo_EventID{evid}_Timestep_{dayi}.png', bbox_inches='tight', dpi=300)
    plt.close()

