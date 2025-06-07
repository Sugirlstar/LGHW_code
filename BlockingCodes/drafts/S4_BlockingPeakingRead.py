import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
import math
import time

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
from collections import defaultdict
from scipy.ndimage import label
from scipy.interpolate import interp2d

# 00 function --------------------------------
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

# 01 read in block centers data --------------------------------
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_date_daily", "rb") as fp:
    Blocking_diversity_peaking_date = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_lon_daily", "rb") as fp:
    Blocking_diversity_peaking_lon = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_lat_daily", "rb") as fp:
    Blocking_diversity_peaking_lat = pickle.load(fp)
    
print(len(Blocking_diversity_peaking_date))

# structure: 
# Blocking_diversity_peaking[0/1/2]: three types of blocks, each type has a list of events, each point represents a blocking peaking date and location
# time: 1979-2021
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_1_ATLeventList_newATLdefine", "rb") as fp:
    ATLlist1 = pickle.load(fp)
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_2_ATLeventList_newATLdefine", "rb") as fp:
    ATLlist2 = pickle.load(fp)
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_3_ATLeventList_newATLdefine", "rb") as fp:
    ATLlist3 = pickle.load(fp)

all_lists = [ATLlist1, ATLlist2, ATLlist3]

# 02 put into the 3d array --------------------------------
# make the time list 
start_date = dt.datetime(1979, 1, 1)
end_date = dt.datetime(2021, 12, 31)
timei = []
timestamp = []
for ordinal in range(start_date.toordinal(), end_date.toordinal() + 1):
    current_date = dt.datetime.fromordinal(ordinal)
    date_str = current_date.strftime('%Y%m%d')
    timei.append(date_str)
    timestamp.append(current_date)
timestamparr = np.array(timestamp)

# loop for 3 types
for type_idx in range(len(Blocking_diversity_peaking_date)):

    print(type_idx,flush=True)
    peakdateIndex = []
    peakdatelatV = []
    peakdatelonV = []
    
    BLKdate = Blocking_diversity_peaking_date[type_idx]
    BLKlon = Blocking_diversity_peaking_lon[type_idx]
    BLKlat = Blocking_diversity_peaking_lat[type_idx]
    ATLlist = all_lists[type_idx]

    for k, eventdate in enumerate(BLKdate):  # loop for each event
        
        # keep the event only if it is in the ATL list
        if k not in ATLlist:
            continue

        # get the lat and lon values
        latvalue = BLKlat[k]
        lonvalue = BLKlon[k]
        lonvalue = lonvalue + 360 if lonvalue < 0 else lonvalue

        peakdateIndex.append(eventdate)  # get the index of the date
        peakdatelatV.append(latvalue)  # get the index of the lat
        peakdatelonV.append(lonvalue)  # get the index of the lon

    print(len(peakdateIndex),flush=True)
    print(peakdateIndex,flush=True)
    print(peakdatelatV,flush=True)
    print(peakdatelonV,flush=True)
    print('---------------------------',flush=True)

    with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingPeakDateList_blkType{type_idx+1}.pkl", "wb") as f:
        pickle.dump(peakdateIndex, f)   
    with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingPeakDateLatList_blkType{type_idx+1}.pkl", "wb") as f:
        pickle.dump(peakdatelatV, f)
    with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingPeakDateLonList_blkType{type_idx+1}.pkl", "wb") as f:
        pickle.dump(peakdatelonV, f)


print('done')