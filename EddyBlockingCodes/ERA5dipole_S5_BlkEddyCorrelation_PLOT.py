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
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

# check two correlations: 
# 1. Blocking's persistence ~ the number of the related track points
# 2. Blocking's LWA ~ the eddy's LWA

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

def radiusExpand(arr, radius):
    
    structure = generate_binary_structure(2, 1)
    structure = iterate_structure(structure, radius)
    # wrap-around dilation
    expanded_arr = np.zeros_like(arr)
    for t in range(arr.shape[0]):
        slice2d = arr[t]
        padded = np.concatenate([slice2d[:, -radius:], slice2d, slice2d[:, :radius]], axis=1)
        # binary dilation on padded
        dilated = binary_dilation(padded, structure=structure).astype(int)
        expanded_arr[t] = dilated[:, radius:-radius]
    return expanded_arr

save_dir = '/scratch/bell/hu1029/LGHW/'
typeid = 3

# === variable list ===
variables_to_load = [
    'BlkPersistence',
    'BlkeventLWA',
    'BlkeventCCNums',
    'BlkeventACNums',
    'BlkeventCCLWA',
    'BlkeventACLWA',
    'BlkACbeforeNum',
    'BlkCCbeforeNum',
    'BlkACbeforeLWA',
    'BlkCCbeforeLWA'
]


# === readin ===
variables = {}
for name in variables_to_load:
    filepath = os.path.join(save_dir, f"ERA5dipoleDaily_typeid{typeid}_{name}.pkl")
    with open(filepath, 'rb') as f:
        variables[name] = pickle.load(f)
for name in variables:
    variables[name] = np.nan_to_num(variables[name], nan=0.0)

BlkPersistence = variables['BlkPersistence']
BlkeventLWA = variables['BlkeventLWA']
BlkeventCCNums = variables['BlkeventCCNums']
BlkeventACNums = variables['BlkeventACNums']
BlkeventCCLWA = variables['BlkeventCCLWA']
BlkeventACLWA = variables['BlkeventACLWA']
BlkACbeforeNum = variables['BlkACbeforeNum']
BlkCCbeforeNum = variables['BlkCCbeforeNum']
BlkACbeforeLWA = variables['BlkACbeforeLWA']
BlkCCbeforeLWA = variables['BlkCCbeforeLWA']

print(BlkPersistence)
print(BlkeventACNums)

# === plot ===
plt.figure(figsize=(8, 6))
plt.hist(BlkeventCCNums, bins=range(0, max(BlkeventCCNums+BlkeventACNums)+2), 
         color='red', alpha=0.6, label='CCNums')
plt.hist(BlkeventACNums, bins=range(0, max(BlkeventCCNums+BlkeventACNums)+2), 
         color='blue', alpha=0.6, label='ACNums')
plt.xlabel('Number of Events')
plt.ylabel('Frequency')
plt.title('Distribution of CC and AC Events')
plt.legend()
plt.tight_layout()
plt.savefig(f'test_BlkEddyDistribution_blktype{typeid}.png')
plt.show()
plt.close()

# CC AC distribution
all_values = BlkeventCCLWA + BlkeventACLWA
min_val = min(all_values)
max_val = max(all_values)
bins = np.linspace(min_val, max_val, 11)  # 或手动设置步长
plt.figure(figsize=(8, 6))
plt.hist(BlkeventCCLWA, bins=bins, color='red', alpha=0.6, label='CCLWA', align='mid')
plt.hist(BlkeventACLWA, bins=bins, color='blue', alpha=0.6, label='ACLWA', align='mid')
plt.xlabel('LWA')
plt.ylabel('Frequency')
plt.title('Distribution of CC and AC LWA')
plt.legend()
plt.tight_layout()
plt.savefig(f'test_BlkCCACLWAhist_blktype{typeid}.png')
plt.show()
plt.close()

# scatter plot --------
plt.figure(figsize=(8, 6))
sc = plt.scatter(BlkPersistence, BlkeventLWA, c=BlkeventACNums, cmap='hot_r', s=80, edgecolor='k')
plt.xlabel('BlkPersistence')
plt.ylabel('BlkeventLWA')
plt.title('LWA vs. Blocking Persistence Colored by ACNums')
# add colorbar
cbar = plt.colorbar(sc, ticks=sorted(set(BlkeventACNums)))
cbar.set_label('BlkeventACNums')
cbar.set_ticks(sorted(set(BlkeventACNums)))  # 显示整数标签
plt.tight_layout()
plt.show()
plt.savefig(f'test_scatterplot_blktype{typeid}.png')
plt.close()

print('done')
