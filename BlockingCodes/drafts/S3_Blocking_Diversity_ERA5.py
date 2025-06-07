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

# 00 readin data
LWA_td = np.load("/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr_float32.npy")  # test data !!REPLACE!!
LWA_td_A = np.load("/scratch/bell/hu1029/LGHW/LWA_td_A_1979_2021_ERA5_6hr_float32.npy")  # test data !!REPLACE!!
LWA_td_C = np.load("/scratch/bell/hu1029/LGHW/LWA_td_C_1979_2021_ERA5_6hr_float32.npy")  # test data !!REPLACE!!

lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 #90
lat_NH = lat[0:lat_mid-1]
nlat_NH =len(lat_NH)
nlon = len(lon)
LWA_Z = LWA_td[:,0:lat_mid-1,:] # NH only!
LWA_Z_A = LWA_td_A[:,0:lat_mid-1,:] # NH only!
LWA_Z_C = LWA_td_C[:,0:lat_mid-1,:] # NH only!

with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_date", "rb") as fp:
    Blocking_peaking_date = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_lon", "rb") as fp:
    Blocking_peaking_lon = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_lat", "rb") as fp:
    Blocking_peaking_lat = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_peaking_area", "rb") as fp:
    Blocking_peaking_area = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_velocity", "rb") as fp:
    Blocking_velocity = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_date", "rb") as fp:
    Blocking_date = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_lat", "rb") as fp:
    Blocking_lat = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_lon", "rb") as fp:
    Blocking_lon = pickle.load(fp)    
with open("/scratch/bell/hu1029/LGHW/Blocking_label", "rb") as fp:
    Blocking_label = pickle.load(fp)  

### Time Management ###
Datestamp = pd.date_range(start="1979-01-01 00:00", end="2021-12-31 18:00", freq="6H")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Date = list(Date0['date'])
# Date = Date[0:7304] # test data !!REMOVE!!
nday = len(Date)
dlat = dlon = 1

###### Code for separate 3 types of blocks (ridge, trough, dipole) ######
#### Method: Focus on the peaking date, calculate the total LWA_AC and LWA_C of the block region ####
Blocking_ridge_date = [];  Blocking_ridge_lon = []; Blocking_ridge_lat=[];  Blocking_ridge_peaking_date = [];   Blocking_ridge_peaking_lon = []; Blocking_ridge_peaking_lat=[];  Blocking_ridge_duration = [];  Blocking_ridge_velocity = [];    Blocking_ridge_area = [];   Blocking_ridge_peaking_LWA = [];   Blocking_ridge_A = []; Blocking_ridge_C = [];   Blocking_ridge_label =[]
Blocking_trough_date = []; Blocking_trough_lon =[]; Blocking_trough_lat=[]; Blocking_trough_peaking_date = [];  Blocking_trough_peaking_lon =[]; Blocking_trough_peaking_lat=[]; Blocking_trough_duration = []; Blocking_trough_velocity = [];   Blocking_trough_area = [];  Blocking_trough_peaking_LWA = [];  Blocking_trough_A = []; Blocking_trough_C = []; Blocking_trough_label =[]
Blocking_dipole_date = []; Blocking_dipole_lon =[]; Blocking_dipole_lat=[]; Blocking_dipole_peaking_date = [];  Blocking_dipole_peaking_lon =[]; Blocking_dipole_peaking_lat=[]; Blocking_dipole_duration= []; Blocking_dipole_velocity =[];    Blocking_dipole_area = [];   Blocking_dipole_peaking_LWA = [];  Blocking_dipole_A = []; Blocking_dipole_C = []; Blocking_dipole_label = []
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
        Blocking_ridge_date.append(Blocking_date[n]);                 Blocking_ridge_lon.append(Blocking_lon[n]);                  Blocking_ridge_lat.append(Blocking_lat[n])
        Blocking_ridge_peaking_date.append(Blocking_peaking_date[n]); Blocking_ridge_peaking_lon.append(Blocking_peaking_lon[n]);  Blocking_ridge_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_ridge_peaking_LWA.append(LWA_max)
        Blocking_ridge_duration.append(len(Blocking_date[n]));        Blocking_ridge_velocity.append(Blocking_velocity[n]);        Blocking_ridge_area.append(Blocking_peaking_area[n]); Blocking_ridge_label.append(Blocking_label[n])
        Blocking_ridge_A.append(Blocking_A);                          Blocking_ridge_C.append(Blocking_C)
    elif LWA_C_sum > 2 * LWA_AC_sum:
        Blocking_trough_date.append(Blocking_date[n]);                 Blocking_trough_lon.append(Blocking_lon[n]);                 Blocking_trough_lat.append(Blocking_lat[n])
        Blocking_trough_peaking_date.append(Blocking_peaking_date[n]); Blocking_trough_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_trough_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_trough_peaking_LWA.append(LWA_max)
        Blocking_trough_duration.append(len(Blocking_date[n]));        Blocking_trough_velocity.append(Blocking_velocity[n]);       Blocking_trough_area.append(Blocking_peaking_area[n]); Blocking_trough_label.append(Blocking_label[n])
        Blocking_trough_A.append(Blocking_A);                          Blocking_trough_C.append(Blocking_C)
    else:
        Blocking_dipole_date.append(Blocking_date[n]);                 Blocking_dipole_lon.append(Blocking_lon[n]);                 Blocking_dipole_lat.append(Blocking_lat[n])
        Blocking_dipole_peaking_date.append(Blocking_peaking_date[n]); Blocking_dipole_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_dipole_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_dipole_peaking_LWA.append(LWA_max)
        Blocking_dipole_duration.append(len(Blocking_date[n]));        Blocking_dipole_velocity.append(Blocking_velocity[n]);       Blocking_dipole_area.append(Blocking_peaking_area[n]); Blocking_dipole_label.append(Blocking_label[n])
        Blocking_dipole_A.append(Blocking_A);                          Blocking_dipole_C.append(Blocking_C)

    print(n)
    
Blocking_diversity_date= []; Blocking_diversity_label = []
Blocking_diversity_date.append(Blocking_ridge_date);  Blocking_diversity_label.append(Blocking_ridge_label)
Blocking_diversity_date.append(Blocking_trough_date); Blocking_diversity_label.append(Blocking_trough_label)    
Blocking_diversity_date.append(Blocking_dipole_date); Blocking_diversity_label.append(Blocking_dipole_label)

with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date", "wb") as fp:
    pickle.dump(Blocking_diversity_date, fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label", "wb") as fp:
    pickle.dump(Blocking_diversity_label, fp)

print(Blocking_diversity_date[0])
print(Blocking_diversity_date[1])
print(Blocking_diversity_date[2])

print(len(Blocking_diversity_date[0]))
print(len(Blocking_diversity_date[1]))
print(len(Blocking_diversity_date[2]))

