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

# readin lat and lon
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 #90
Blklat = lat[0:lat_mid-1]
Blklon = lon
print(Blklat)

# readin blocking data
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_date_daily", "rb") as fp:
    Blocking_diversity_peaking_date = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_lon_daily", "rb") as fp:
    Blocking_diversity_peaking_lon = pickle.load(fp)
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_lat_daily", "rb") as fp:
    Blocking_diversity_peaking_lat = pickle.load(fp)
    
# readin peaking event list
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_1_ATLeventList", "rb") as fp:
    ATLlist1 = pickle.load(fp)
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_3_ATLeventList", "rb") as fp:
    ATLlist3 = pickle.load(fp)
peakinglat1 = [Blocking_diversity_peaking_lat[0][i] for i in ATLlist1]
peakinglat3 = [Blocking_diversity_peaking_lat[2][i] for i in ATLlist3]
peakinglon1 = [Blocking_diversity_peaking_lon[0][i] for i in ATLlist1]
peakinglon3 = [Blocking_diversity_peaking_lon[2][i] for i in ATLlist3]

Sector2FGCluster_Blk1 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingFlagSector2maskClusters_1979_2021_Type1.npy')
Sector2FGCluster_Blk3 = np.load(f'/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingFlagSector2maskClusters_1979_2021_Type3.npy')

Blk1sum = np.sum(Sector2FGCluster_Blk1, axis=0)
Blk3sum = np.sum(Sector2FGCluster_Blk3, axis=0)

# plot the map -------------------
fig, ax, cf = create_Map(Blklon,Blklat,Blk1sum,fill=True,fig=None,
                            minv=0, maxv=round(np.nanmax(Blk1sum)), interv=11, figsize=(12,5),
                            centralLon=0, colr='PuRd', extend='max',title=f'Blocking frequency')
addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
ax.scatter(peakinglon1, peakinglat1, color='green', s=30, marker='o', transform=ccrs.PlateCarree())
plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
plt.show()
plt.tight_layout()
plt.savefig(f'Dipole_Blk1_Frequency_Climatology_WithPeakingPoints.png')
plt.close()

fig, ax, cf = create_Map(Blklon,Blklat,Blk3sum,fill=True,fig=None,
                            minv=0, maxv=round(np.nanmax(Blk3sum)), interv=11, figsize=(12,5),
                            centralLon=0, colr='PuRd', extend='max',title=f'Blocking frequency')
addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
ax.scatter(peakinglon3, peakinglat3, color='green', s=30, marker='o', transform=ccrs.PlateCarree())
plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
plt.show()
plt.tight_layout()
plt.savefig(f'Dipole_Blk3_Frequency_Climatology_WithPeakingPoints.png')
plt.close()

