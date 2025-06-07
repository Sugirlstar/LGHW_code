# get the density of the tracks during blocking / non-blocking days and plot the map

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
import dask

# added figure - all composites

timestep = np.arange(-20,21,4)
print(timestep, flush=True)

ncols, nrows = 3, len(timestep)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))

latarr = np.arange((0-40),(0+40))
lonarr = np.arange((0-40),(0+40))
titles = ['Ridge', 'Trough', 'Dipole']

# axes 是一个 (11, 3) 的 2D array
seen_AC_ids = set()
seen_CC_ids = set()
for i,tid in enumerate(timestep):
    ti = tid + 20  # Adjust index to match the actual position in the array (tid = -20 to 20 maps to ti = 0 to 40)
    for j in range(ncols):

        typeid = j+1 # typeid: 1,2,3
        idx = i * ncols + j  # Index of the current graph, from 0 to 32

        centeredZ500npy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_1stBlkDay_CenteredZ500_timewindow41_BlkType'+str(typeid)+'.npy')
        centeredACnpy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_1stBlkDay_CenteredAC_TrackIDarr_timewindow41_BlkType'+str(typeid)+'.npy')
        centeredCCnpy = np.load('/scratch/bell/hu1029/LGHW/ERA5dipole_1stBlkDay_CenteredCC_TrackIDarr_timewindow41_BlkType'+str(typeid)+'.npy')

        # get the points that have been seen
        centeredZ500npyti = np.nanmean(centeredZ500npy, axis=0)[ti,:,:]  # average over the time dimension
        mask = (centeredACnpy[:,ti,:,:] > 0) & np.isin(centeredACnpy[:,ti,:,:], list(seen_AC_ids))
        _, AC_lat_idx, AC_lon_idx = np.where(mask)
        mask = (centeredCCnpy[:,ti,:,:] > 0) & np.isin(centeredCCnpy[:,ti,:,:], list(seen_CC_ids))
        _, CC_lat_idx, CC_lon_idx = np.where(mask)

        # get the id that first occurs 
        ACtrackids = np.unique(centeredACnpy[:,ti,:,:])
        ACtrackids = ACtrackids[ACtrackids > 0]  # remove the zero values
        new_ac = [id for id in ACtrackids if id not in seen_AC_ids]
        seen_AC_ids.update(ACtrackids)
        newaclen = len(new_ac)
        print(newaclen, flush=True)
        if (newaclen > 0):
            new_ac_set = set(new_ac)
            mask_new = np.isin(centeredACnpy[:,ti,:,:], list(new_ac_set))
            _, AC_lat_idx_new, AC_lon_idx_new = np.where(mask_new)
        
        # get the id that first occurs 
        CCtrackids = np.unique(centeredCCnpy[:,ti,:,:])
        CCtrackids = CCtrackids[CCtrackids > 0] # remove the zero values
        new_cc = [id for id in CCtrackids if id not in seen_CC_ids]
        seen_CC_ids.update(CCtrackids)
        newcclen = len(new_cc)
        print(newcclen, flush=True)
        if (newcclen > 0):
            new_cc_set = set(new_cc)
            mask_new = np.isin(centeredCCnpy[:,ti,:,:], list(new_cc_set))
            _, CC_lat_idx_new, CC_lon_idx_new = np.where(mask_new)
        print(i, '--------')

        subtitle = f'newEnter AC: {newaclen}, CC: {newcclen}'
        
        ax = axes[i, j]
        cs = ax.contour(lonarr, latarr, centeredZ500npyti, levels=10, colors='k', linewidths=1.5)
        # ax.scatter(lonarr[AC_lon_idx], latarr[AC_lat_idx], c='blue', marker='o', s=90, edgecolors='none', alpha=0.6)
        # ax.scatter(lonarr[CC_lon_idx], latarr[CC_lat_idx], c='red', marker='o', s=90, edgecolors='none', alpha=0.7)
        if i > 0:
            if newaclen > 0:
                ax.scatter(lonarr[AC_lon_idx_new], latarr[AC_lat_idx_new], c='blue', marker='x', s=100, edgecolors='none')
            if newcclen > 0:
                ax.scatter(lonarr[CC_lon_idx_new], latarr[CC_lat_idx_new], c='red', marker='x', s=100, edgecolors='none')
        ax.clabel(cs, inline=True, fontsize=10)
        ax.set_title(subtitle, fontsize=12)

        # row title
        if j == 0:
            if tid == 0:
                ax.set_ylabel(f'1st Block Day (Day 0)', fontsize=14)
            else:
                ax.set_ylabel(f'Day {tid/4:.0f}', fontsize=14)

        # column title
        if i == 0:
            # ax.set_title(titles[j], fontsize=13)
            fig.text(ax.get_position().x0 + ax.get_position().width / 2,
                ax.get_position().y1 + 0.01, titles[j],
                ha='center', va='bottom', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.savefig('centerComposites_allpanels_1stBlkDay_EnterOnly.png', dpi=300)
plt.show()
plt.close()

print('done')
