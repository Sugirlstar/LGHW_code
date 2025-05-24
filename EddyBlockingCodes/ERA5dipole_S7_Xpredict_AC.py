import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
import math

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

# 00 read data --------------------------------------------------------------
# attributes
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
lon = np.array(ds['lon'])
lat = np.array(ds['lat'])
lat = np.flip(lat)
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)
timeiarr = np.array(timei)
grid_area = calculate_grid_area_from_bounds(lat, lon) # calculate the grid area array
# tracks
with open('/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks.pkl', 'rb') as file:
    track_data = pickle.load(file)

# extract all the point elements
time_list = np.array([time for _, track in track_data for (time, _, _) in track])
y_list = np.array([y for _, track in track_data for (_, y, _) in track])
x_list = np.array([x for _, track in track_data for (_, _, x) in track])

# # 01 find how many track points are in the Sec2 region for each day
# lon_min, lon_max = 330, 30  
# lat_min, lat_max = 45, 75 
# track_points_num = []
# for target_time in timeiarr:
#     print(target_time,flush=True)
#     indices = np.where((time_list == target_time) & ((lon_min <= y_list) | (y_list <= lon_max)) & (lat_min <= x_list) & (x_list <= lat_max))[0]
#     print(len(indices),flush=True)
#     track_points_num.append(len(indices))
    
# track_points_num = np.array(track_points_num)
# np.save('/scratch/bell/hu1029/LGHW/CCZanom_trackPointsNum.npy', track_points_num)

# # 02 find how many track points are in the upstream region for each day
# lon_min, lon_max = 240, 330  
# lat_min, lat_max = 40, 80 
# track_points_num = []
# for target_time in timeiarr:
#     print(target_time,flush=True)
#     indices = np.where((time_list == target_time) & ((lon_min <= y_list) & (y_list <= lon_max)) & (lat_min <= x_list) & (x_list <= lat_max))[0]
#     print(len(indices),flush=True)
#     track_points_num.append(len(indices))
    
# track_points_num = np.array(track_points_num)
# np.save('/scratch/bell/hu1029/LGHW/CCZanom_trackPointsNum_W240to330N40to80.npy', track_points_num)

# print('Done', flush=True)

# 03 lead day0 - at that day, what's the relationship between the track points and the blocking events
track_points_num = np.load('/scratch/bell/hu1029/LGHW/CCZanom_trackPointsNum.npy')
BlkIndex = np.load('/scratch/bell/hu1029/LGHW/ERA5_BlockingdaySector2_1979_2021.npy') # load the index of blocking days
BlkDayflag = np.zeros_like(track_points_num)
BlkDayflag[BlkIndex] = 1 # 1 for blocking day, 0 for non-blocking day
uniqpointnum = np.unique(track_points_num)
backgroundP = len(BlkIndex)/len(BlkDayflag)
print(f'backgroundProbability: {backgroundP}') # 0.3935120336177257

predicArr = np.zeros((4, len(range(0,31))))
for i in uniqpointnum:
    for j in range(0,31):
        indices = np.where(track_points_num == i)[0]
        indices = [val for val in indices if val < len(BlkDayflag)-j] # prevent outside the length
        predicDay = [item + j for item in indices]
        BGdays = len(indices)
        Blkdays = np.sum(BlkDayflag[predicDay])
        prob = Blkdays/BGdays - backgroundP
        predicArr[i,j] = prob

# plot 
plt.imshow(predicArr, aspect='auto', cmap='RdBu_r',vmin=-np.max(np.abs(predicArr)), vmax=np.max(np.abs(predicArr)))
plt.colorbar(label="Blocking Probability")
plt.xlabel("Lead days")
plt.yticks([0, 1, 2, 3])
plt.ylabel("Track number")
plt.title("Heatmap")
plt.show()
plt.savefig('CCZanom_PredictHeatmap_countEveryone_minusBackground.png')
plt.close()

# 0
# 22903
# 6820
# 0.29777758372265645

# 1
# 35518
# 16081
# 0.4527563488935188

# 2
# 4313
# 1795
# 0.41618363088337584

# 3
# 90
# 26
# 0.28888888888888886

# 04 find all the place where Xi = Xi + 1
diffIndices = np.array(np.where(np.diff(track_points_num) == 1)[0] + 1)
predicArr = np.zeros((4, len(range(0,31))))
for i in [0,1,2,3]:
    for j in range(0,31):
        if i == 0:
            intersection = np.array(np.where(track_points_num == i)[0])
        else:
            valueIndices = np.array(np.where(track_points_num == i)[0])
            intersection = np.intersect1d(diffIndices, valueIndices)
        intersection = [val for val in intersection if val < len(BlkDayflag)-j] # prevent outside the length
        predicDay = [item + j for item in intersection]
        BGdays = len(intersection)
        Blkdays = np.sum(BlkDayflag[predicDay])
        prob = Blkdays/BGdays - backgroundP
        predicArr[i,j] = prob

# plot 
plt.imshow(predicArr, aspect='auto', cmap='RdBu_r',vmin=-np.max(np.abs(predicArr)), vmax=np.max(np.abs(predicArr)))
plt.colorbar(label="Blocking Probability")
plt.xlabel("Lead days")
plt.yticks([0, 1, 2, 3])
plt.xticks([4, 8, 12, 16, 20, 24, 28], [1, 2, 3, 4, 5, 6, 7])
plt.ylabel("Track number")
plt.title("Heatmap")
plt.show()
plt.savefig('CCZanom_PredictHeatmap_countEnterOnly_minusBackground.png')
plt.close()

print('Done', flush=True)



