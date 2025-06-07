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
from scipy.stats import pearsonr

# %% 00 prepare the environment and functions
regions = ["ATL", "NP", "SP"]
seasons = ["DJF", "JJA","ALL"]
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

def extendList(desired_length,centorinlist,test):
    targetlist = np.full(desired_length*2+1, np.nan)
    targetlist[desired_length] = test[centorinlist]
    leftlen = centorinlist
    rightlen = len(test) - leftlen -1
    left_padding = [np.nan] * (desired_length - leftlen)
    left_values = left_padding + list(test[np.nanmax([0, leftlen-desired_length]):centorinlist])
    right_padding = [np.nan] * (desired_length - rightlen)
    if rightlen-desired_length>0:
        rightend = centorinlist+1+desired_length
    else: rightend = len(test)
    right_values = list(test[centorinlist+1:rightend]) + right_padding
    targetlist[:desired_length] = left_values
    targetlist[desired_length+1:] = right_values
    return targetlist

for rgname in regions:
    
    # 01 - read data -------------------------------------------------------------
    # lon and lat for the track
    ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
    lon = np.array(ds['lon'])
    lat = np.array(ds['lat'])
    lat = np.flip(lat)
    if rgname == "SP":
        lat = lat[0:(findClosest(0,lat)+1)]
    else:
        lat = lat[(findClosest(0,lat)+1):len(lat)]
    # time management
    times = np.array(ds['time'])
    datetime_array = pd.to_datetime(times)
    timei = list(datetime_array)
    # lat and lon for blocking
    lonBLK = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
    latBLK = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order
    lat_mid = int(len(latBLK)/2) + 1 
    if rgname == "SP":
        latBLK = latBLK[lat_mid:len(latBLK)]
    else:
        latBLK = latBLK[0:lat_mid-1]
    latBLK = np.flip(latBLK)
    print(latBLK)


ss = 'ALL'  

for cyc in cycTypes:

    if cyc == 'CC':
        clr = 'Reds'
    else:
        clr = 'Blues'

    fig, axes = plt.subplots(
        3, 3,
        figsize=(18, 15),
        sharex=True,
        sharey=True,
        gridspec_kw={'right': 0.85,     
        'wspace': 0.1, # the space between columns
        'hspace': 0.1  } # the space between rows
    )

    all_collections = []
    for i, typeid in enumerate([1, 2, 3]):
        blkname = blkTypes[i]
        for j, rgname in enumerate(regions):
            
            ax = axes[j, i]

            intopercentList = np.load(f'/scratch/bell/hu1029/LGHW/intopercentList_type{typeid}_{cyc}_{rgname}_{ss}.npy')
            intopercentList = intopercentList*100
            interactingduration = np.load(f'/scratch/bell/hu1029/LGHW/interactingduration_type{typeid}_{cyc}_{rgname}_{ss}.npy')

            print('data loaded-----------------------',flush=True)

            # make the plot
            kde = sns.kdeplot(
                x=interactingduration,
                y=intopercentList,
                fill=True,
                cmap=clr,
                levels=np.linspace(0, 1, 40),  
                thresh=0.05,
                ax=ax
            )
            all_collections.append(kde.collections)

            ax.set_xlim(0, 12)
            ax.set_ylim(0, 100)
            ax.text(0.95, 0.95,f'N: {len(intopercentList)}',ha='right',va='top',transform=ax.transAxes, fontsize=18)  
            if j == 2:
                ax.set_xlabel('Eddy Stay time (days)', fontsize=18)
                ax.tick_params(axis='x', labelsize=16)  
            if i == 0:
                ax.set_ylabel('Entry Time (% of Blocking)', fontsize=18)
                ax.tick_params(axis='y', labelsize=16)  
                ax.text(-0.23, 0.5, rgname, va='center', ha='right',
                    fontsize=20, rotation=90, transform=ax.transAxes)
            # column title
            if j == 0:
                ax.set_title(blkname, fontsize=20, pad=20)

    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.7])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=0, vmax=0.6)
    sm = plt.cm.ScalarMappable(cmap=clr, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(0, 0.61, 0.1))
    cbar.set_label('Density', fontsize=18)
    cbar.ax.tick_params(labelsize=18)    

    fig.subplots_adjust(
    left=0.08,
    right=0.85,
    bottom=0.08,
    top=0.95
    )          

    plt.savefig(f'Fig2_EnterWithStay_3type3regionAllSeasons_{cyc}.png')
    plt.close()


                