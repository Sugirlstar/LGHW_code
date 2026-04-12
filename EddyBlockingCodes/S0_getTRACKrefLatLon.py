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


# %% dataset settings -------------------------------------------------------------
datasets = ["ERA5", "MERRA2", "JRA55"]

TRACK_latrefFile = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5/ERA5_TRACK_lat_1979_2021_6hr.npy",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2/MERRA2_TRACK_lat_1980_2021_6hr.npy",
    "JRA55": "/scratch/bell/hu1029/LGHW/interm_JRA55/JRA55_TRACK_lat_1979_2021_6hr.npy"
}
TRACK_lonrefFile = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5/ERA5_TRACK_lon_1979_2021_6hr.npy",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2/MERRA2_TRACK_lon_1980_2021_6hr.npy",
    "JRA55": "/scratch/bell/hu1029/LGHW/interm_JRA55/JRA55_TRACK_lon_1979_2021_6hr.npy"
}
trackrefFile = {
    "ERA5": "/scratch/bell/hu1029/Data/processed/ERA5_Z500climatology_monthly_1979_2021_F128.nc",
    "MERRA2": "/scratch/bell/hu1029/Data/processed/MERRA2_Z500climatology_monthly_1980_2021_F128.nc",
    "JRA55": "/scratch/bell/hu1029/Data/processed/JRA55_Z500climatology_monthly_1979_2021_F128.nc"
}


for dtname in datasets:
    
    print(f"Processing dataset: {dtname}", flush=True)
    TRACK_lat_out = TRACK_latrefFile[dtname]
    TRACK_lon_out = TRACK_lonrefFile[dtname]
    TRACK_ref_file = trackrefFile[dtname]
    
    ds = xr.open_dataset(TRACK_ref_file)
    lon = np.array(ds['lon'])
    lat = np.array(ds['lat'])
    # save the lat and lon of the track reference file
    np.save(TRACK_lat_out, lat)
    np.save(TRACK_lon_out, lon)

    print(f"Saved TRACK reference lat and lon for {dtname} to {TRACK_lat_out} and {TRACK_lon_out}", flush=True)
