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
from matplotlib import cm

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

def PlotBoundary(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max, loncenter = 30, 90, 250, 90, 350
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max, loncenter = 30, 90, 80, 280, 180
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max, loncenter = -90, -30, 130, 330, 230

    return lat_min, lat_max, lon_min, lon_max, loncenter

lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 
Blklon = lon 
latglobe = lat[lat != 0]
# %% load the data and make the plot -------------------
# get lon and lat

cclevels = np.arange(0, 26, 2)  # color levels for the colormap
aclevels = np.arange(0, 22, 2)  # color levels for the colormap

for typeid in [1,2,3]:
    for ss in ['ALL']:
        for cyc in cycTypes:

            if cyc == 'CC':
                cyccolor = 'Reds'
                lv = cclevels
            else: 
                cyccolor = 'Blues'
                lv = aclevels

            if typeid == 1:
                Blklevels = [100,350,600,850,1100,1350]
                Blklevels_SH = [100,350,600,850,1100,1350]
            elif typeid == 2:
                Blklevels = [100,350,600,850,1100,1350]
                Blklevels_SH = [100,350,600,850,1100,1350]
            elif typeid == 3:
                Blklevels = [100, 200, 300, 400]
                Blklevels_SH = [100, 200, 300, 400]

            # take .5 to 1.0 only
            cmap_half = cm.get_cmap(cyccolor, 256)  # 256 is the number of colors in the colormap
            cmap_half = cmap_half(np.linspace(0.1, 1.0, 256))
            from matplotlib.colors import ListedColormap
            cyccolor = ListedColormap(cmap_half)

            rg1 = np.load(f"/scratch/bell/hu1029/LGHW/BlockingFlagmaskClusters_Type{typeid}_ATL_{ss}.npy")
            rg2 = np.load(f"/scratch/bell/hu1029/LGHW/BlockingFlagmaskClusters_Type{typeid}_NP_{ss}.npy")
            rg3 = np.load(f"/scratch/bell/hu1029/LGHW/BlockingFlagmaskClusters_Type{typeid}_SP_{ss}.npy")
            rgNH = np.logical_or(rg1,rg2)

            rgSH = rg3
            # calculate the sum over time
            Blk1sumNH = np.sum(rgNH, axis=0)
            Blk1sumSH = np.sum(rgSH, axis=0)

            latNH = lat[0:lat_mid-1]  # North Hemisphere latitudes
            latSH = lat[lat_mid:]  # South Hemisphere latitudes

            # get the interacting track points
            trackPoints_array_ATL = np.load(f'/scratch/bell/hu1029/LGHW/{cyc}trackInteracting_array_Type{typeid}_ATL_{ss}.npy')
            trackPoints_array_NP = np.load(f'/scratch/bell/hu1029/LGHW/{cyc}trackInteracting_array_Type{typeid}_NP_{ss}.npy')
            trackPoints_array_SP = np.load(f'/scratch/bell/hu1029/LGHW/{cyc}trackInteracting_array_Type{typeid}_SP_{ss}.npy')
            trackPointsNorth = np.logical_or(trackPoints_array_ATL, trackPoints_array_NP)
            trackPointsNorth = np.flip(trackPointsNorth, axis=1)  # flip the North Hemisphere track points to match the global map
            trackPointsSouth = np.flip(trackPoints_array_SP, axis=1)  # flip the South Hemisphere track points to match the global map

            trackPoints_frequency_NH = np.nansum(trackPointsNorth, axis=0) 
            trackPoints_frequency_NH = trackPoints_frequency_NH.astype(float)
            trackPoints_frequency_NH[np.where(trackPoints_frequency_NH == 0)] = np.nan  # set zero values to NaN for better visualization

            trackPoints_frequency_SH = np.nansum(trackPointsSouth, axis=0) 
            trackPoints_frequency_SH = trackPoints_frequency_SH.astype(float)
            trackPoints_frequency_SH[np.where(trackPoints_frequency_SH == 0)] = np.nan  # set zero values to NaN for better visualization

            # plot the map, seperated for types -------------------
            fig = plt.figure(figsize=(12, 6), dpi=300)

            # set levels and color range
            max_freq = np.nanmax([np.nanmax(trackPoints_frequency_NH), np.nanmax(trackPoints_frequency_SH)])
            levels = np.linspace(0, max_freq, 12)  # set levels for the colormap
            levels = np.round(levels, 3) 

            # NH
            _, ax1, cf1 = create_Map(
                Blklon, latNH, trackPoints_frequency_NH,
                levels=levels, fill=True, fig=fig,
                leftlon=-180, rightlon=180, lowerlat=30, upperlat=90,
                minv=0, maxv=max_freq,nrows=2, ncols=1, index=1,
                centralLon=270, colr=cyccolor, extend='max',ybotomlabel = False
            )

            _, _, _ = create_Map(
                Blklon, latNH, Blk1sumNH, fill=False, fig=fig, ax=ax1,
                leftlon=-180, rightlon=180, lowerlat=30, upperlat=90,
                minv=0, maxv=round(np.nanmax(Blk1sumNH)), levels=Blklevels,
                centralLon=270, contourcolr='darkgreen', ybotomlabel = False
            )
            print(f'type{typeid}-{cyc} maximum Track frequency in NH: {np.nanmax(trackPoints_frequency_NH)}')
            print(f'type{typeid}-{cyc} maximum Block frequency in NH: {np.nanmax(Blk1sumNH)}')

            # SH
            _, ax2, cf2 = create_Map(
                Blklon, latSH, trackPoints_frequency_SH,
                levels=levels, fill=True, fig=fig, 
                leftlon=-180, rightlon=180, lowerlat=-90, upperlat=-30,
                minv=0, maxv=max_freq, nrows=2, ncols=1, index=2,
                centralLon=270, colr=cyccolor, extend='max', ybotomlabel = True
            )

            _, _, _ = create_Map(
                Blklon, latSH, Blk1sumSH, fill=False, fig=fig, ax=ax2,
                leftlon=-180, rightlon=180, lowerlat=-90, upperlat=-30,
                minv=0, maxv=round(np.nanmax(Blk1sumSH)), levels=Blklevels_SH,
                centralLon=270, contourcolr='darkgreen', ybotomlabel = True
            )
            print(f'type{typeid}-{cyc} maximum Track frequency in NH: {np.nanmax(trackPoints_frequency_SH)}')
            print(f'type{typeid}-{cyc} maximum Block frequency in NH: {np.nanmax(Blk1sumSH)}')

            # add region boxes
            for k,rgname in enumerate(['ATL', 'NP', 'SP']):
                lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
                if k == 2:
                    addSegments(ax2,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],
                                colr='black',linewidth=1.5)
                else:
                    addSegments(ax1,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],
                                colr='black',linewidth=1.5)

            ax1.set_position([0.05, 0.45, 0.9, 0.5])  
            ax2.set_position([0.05, 0.1, 0.9, 0.5]) 

            # add colorbar
            cbar_ax = fig.add_axes([0.20, 0.1, 0.6, 0.04])  # left, bottom, width, height 
            cbar = fig.colorbar(cf1, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Track Density (times)', fontsize=12)
            cbar.set_ticks(levels)
            cbar.ax.tick_params(labelsize=10)
            # from matplotlib.ticker import MultipleLocator
            # cbar.ax.xaxis.set_major_locator(MultipleLocator(2)) 

            plt.savefig(f'Fig1_EddyHeatmap{blkTypes[typeid-1]}_{cyc}.png', dpi=300)
            plt.close()
    
            print(f'Fig1_{blkTypes[typeid-1]}_{cyc} plotted ----------------',flush=True)

