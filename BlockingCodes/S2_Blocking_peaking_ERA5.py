#%%
###### This code is to study the blocking diversity (separation of 3 types of blocks) ######
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
import scipy.stats as stats
import cartopy

### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # earth radius
    return c * r * 1000

### Read basic variables ###
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 #90
lat_NH = lat[0:lat_mid-1]
nlat_NH =len(lat_NH)
LWA_td = np.load("/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr_float32.npy") # test data !!REPLACE!!
LWA_Z = LWA_td[:,0:lat_mid-1,:] # NH only!
print(LWA_Z.shape)

### read blocking data ###
with open("/scratch/bell/hu1029/LGHW/Blocking_date", "rb") as fp:
    Blocking_date = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_lat", "rb") as fp:
    Blocking_lat = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_lon", "rb") as fp:
    Blocking_lon = pickle.load(fp)    
with open("/scratch/bell/hu1029/LGHW/Blocking_lon_wide", "rb") as fp:
    Blocking_lon_wide = pickle.load(fp) 
with open("/scratch/bell/hu1029/LGHW/Blocking_area", "rb") as fp:
    Blocking_area = pickle.load(fp) 
with open("/scratch/bell/hu1029/LGHW/Blocking_label", "rb") as fp:
    Blocking_label = pickle.load(fp)  

### Time Management ###
Datestamp = pd.date_range(start="1979-01-01 00:00", end="2021-12-31 18:00", freq="6H")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Date = list(Date0['date'])
# Date = Date[0:7304] # test data !!REMOVE!!
nday = len(Date)

#%%
### get the blocking peaking date and location and wave activity ###
Blocking_peaking_date = []
Blocking_peaking_date_index = []
Blocking_peaking_lon = []
Blocking_peaking_lat = []
Blocking_peaking_LWA = []
Blocking_duration =[]
Blocking_velocity = [] 
Blocking_peaking_lon_wide = []
Blocking_peaking_area = []

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
    Blocking_peaking_LWA.append(np.max(LWA_event_max))
    Blocking_peaking_date.append(Blocking_date[n][Blocking_peaking_date_index])
    Blocking_peaking_lon.append(Blocking_lon[n][Blocking_peaking_date_index])
    Blocking_peaking_lat.append(Blocking_lat[n][Blocking_peaking_date_index])
    Blocking_peaking_lon_wide.append(Blocking_lon_wide[n][Blocking_peaking_date_index])
    Blocking_peaking_area.append(Blocking_area[n][Blocking_peaking_date_index])
    Blocking_duration.append(duration)
    Blocking_velocity.append( haversine(Blocking_lon[n][0], Blocking_lat[n][0], Blocking_lon[n][-1], Blocking_lat[n][-1])/(duration*(24/4)*60*60) )
    
    print(n)

#%% Save results
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_date_index", "wb") as fp:
    pickle.dump(Blocking_peaking_date_index, fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_LWA", "wb") as fp:
    pickle.dump(Blocking_peaking_LWA, fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_date", "wb") as fp:
    pickle.dump(Blocking_peaking_date, fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_lon", "wb") as fp:
    pickle.dump(Blocking_peaking_lon, fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_lat", "wb") as fp:
    pickle.dump(Blocking_peaking_lat, fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_lon_wide", "wb") as fp:
    pickle.dump(Blocking_peaking_lon_wide, fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_area", "wb") as fp:
    pickle.dump(Blocking_peaking_area, fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_duration", "wb") as fp:
    pickle.dump(Blocking_duration, fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_velocity", "wb") as fp:
    pickle.dump(Blocking_velocity, fp)
