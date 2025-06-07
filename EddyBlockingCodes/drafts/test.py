# get the density of the tracks during blocking / non-blocking days and plot the map

import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
import pickle
import os

typeid = 1
rgname = "ATL"  # "NH" or "SP"
ss = "ALL"  # "DJF", "JJA", or "All"
with open(f'/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}', 'rb') as file:
    firstday_Date = pickle.load(file)
with open(f'/scratch/bell/hu1029/LGHW/Blocking1stdayLatList_blkType{typeid}_{rgname}_{ss}', 'rb') as file:
    lat_values = pickle.load(file)
with open(f'/scratch/bell/hu1029/LGHW/Blocking1stdayLonList_blkType{typeid}_{rgname}_{ss}', 'rb') as file:
    lon_values = pickle.load(file)
    
print(firstday_Date[0:100], flush=True)
print(lat_values[0:100], flush=True)
print(lon_values[0:100], flush=True)
    
# ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
# timesarr = np.array(ds['time'])
# print(timesarr[0], flush=True)
# print(timesarr[-1], flush=True)
# ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
# timesarr = np.array(ds['time'])
# print(timesarr[0], flush=True)
# print(timesarr[-1], flush=True)

            
# # 所有变量名
# variable_names = [
#     'BlkPersistence',
#     'BlkeventLWA',
#     'BlkeventCCNums',
#     'BlkeventACNums',
#     'BlkeventCCLWA',
#     'BlkeventACLWA',
#     'BlkACbeforeNum',
#     'BlkCCbeforeNum',
#     'BlkACbeforeLWA',
#     'BlkCCbeforeLWA',
# ]

# # 设置路径和 typeid
# save_dir = '/scratch/bell/hu1029/LGHW/'
# typeid = 1  # 根据你实际的 typeid 修改

# # 逐个读取并打印长度
# for name in variable_names:
#     filename = os.path.join(save_dir, f"ERA5dipoleDaily_typeid{typeid}_{name}.pkl")
#     try:
#         with open(filename, 'rb') as f:
#             var = pickle.load(f)
#             # print(f"{name}: length = {len(var)}")
#             print(var[0:20])
#     except Exception as e:
#         print(f"{name}: Failed to load ({e})")

# arr = np.zeros((3, 50, 50), dtype=int)
# arr[0, 25, 0] = 1  # 在最左边设置一个点，测试 wrap-around
# arr[2, 25, 25] = 1
# arr[2, 48, 49] = 1

# def radiusExpand(arr, radius):
    
#     structure = generate_binary_structure(2, 1)
#     structure = iterate_structure(structure, radius)
#     # wrap-around dilation
#     expanded_arr = np.zeros_like(arr)
#     for t in range(arr.shape[0]):
#         slice2d = arr[t]
#         padded = np.concatenate([slice2d[:, -radius:], slice2d, slice2d[:, :radius]], axis=1)
#         # binary dilation on padded
#         dilated = binary_dilation(padded, structure=structure).astype(int)
#         expanded_arr[t] = dilated[:, radius:-radius]
#     return expanded_arr

# arrx = radiusExpand(arr, 5)

# # 可视化
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(arr[2], cmap='Greys', origin='lower')
# axs[0].set_title('Original (wrap test)')
# axs[1].imshow(arrx[2], cmap='Greys', origin='lower')
# axs[1].set_title('Expanded with Longitude Wrap')

# for ax in axs:
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout()
# plt.show()
# plt.savefig('expanded_array_example.png', dpi=300, bbox_inches='tight')
# print('done')