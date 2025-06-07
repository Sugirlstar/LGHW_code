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
import sys
import os

# %% function --------------------------------
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


lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 
Blklon = lon 
latglobe = lat[lat != 0]

# %% load the data and make the plot -------------------
# get lon and lat

for typeid in [1,2,3]:
    for ss in seasons:
        for cyc in cycTypes:

            rg1 = np.load(f"/scratch/bell/hu1029/LGHW/BlockingFlagmaskClusters_Type{typeid}_ATL_{ss}.npy")
            rg2 = np.load(f"/scratch/bell/hu1029/LGHW/BlockingFlagmaskClusters_Type{typeid}_NP_{ss}.npy")
            rg3 = np.load(f"/scratch/bell/hu1029/LGHW/BlockingFlagmaskClusters_Type{typeid}_SP_{ss}.npy")
            rgNH = np.logical_or(rg1,rg2)
            rgworld = np.concatenate([rgNH, rg3], axis=1)
            # calculate the sum over time
            Blk1sum = np.sum(rgworld, axis=0)

            # get the interacting track points
            trackPoints_array_ATL = np.load(f'/scratch/bell/hu1029/LGHW/{cyc}trackInteracting_array_Type{typeid}_ATL_{ss}.npy')
            trackPoints_array_NP = np.load(f'/scratch/bell/hu1029/LGHW/{cyc}trackInteracting_array_Type{typeid}_NP_{ss}.npy')
            trackPoints_array_SP = np.load(f'/scratch/bell/hu1029/LGHW/{cyc}trackInteracting_array_Type{typeid}_SP_{ss}.npy')
            trackPointsNorth = np.logical_or(trackPoints_array_ATL, trackPoints_array_NP)
            trackPoints_array_Global = np.concatenate([trackPoints_array_SP, trackPointsNorth], axis=1)
            trackPoints_array_Global = np.flip(trackPoints_array_Global, axis=1) 

            trackPoints_frequency = np.nansum(trackPoints_array_Global, axis=0) 
            # find the locations
            lat_idx, lon_idx = np.where(trackPoints_frequency > 3) # larger than 3 times
            frequencies = trackPoints_frequency[lat_idx, lon_idx]

            # get the latitudes and longitudes for the scatter plot
            lat_vals = latglobe[lat_idx]
            lon_vals = Blklon[lon_idx]

            import matplotlib.cm as cm
            Cb_half = cm.get_cmap('OrRd')
            new_cmap = Cb_half(np.linspace(0.5, 1.0, 256))  
            from matplotlib.colors import LinearSegmentedColormap
            Cb_tail = LinearSegmentedColormap.from_list('Cb_tail', new_cmap)

            # plot the map, seperated for types -------------------
            fig, ax, cf = create_Map(Blklon,latglobe,Blk1sum,fill=True,fig=None,
                                        leftlon=-180, rightlon=180, lowerlat=-90, upperlat=90,
                                        minv=round(np.nanmin(Blk1sum)), maxv=round(np.nanmax(Blk1sum)), interv = 10, figsize=(12,5),
                                        centralLon=270,colr='Blues', extend='max', title=f'blocking interacting tracks density_{blkTypes[typeid]}_{ss}')
            cbar1 = plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
            
            sc = ax.scatter(lon_vals, lat_vals, marker='x',c=frequencies, cmap=Cb_tail, 
                            alpha=0.7, transform=ccrs.PlateCarree())
            cbar2 = plt.colorbar(sc, orientation='vertical')
            cbar2.set_label('Interaction Frequency')
            
            # add region boxes
            for k,rgname in enumerate(regions):
                lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
                addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1.5)
            
            # plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
            plt.show()
            plt.tight_layout()
            plt.savefig(f'Fig1_scatters_{blkTypes[typeid-1]}_{cyc}_{ss}.png')
            plt.close()

    print('Fig1 plotted ----------------',flush=True)


