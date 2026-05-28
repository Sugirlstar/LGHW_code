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
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["CC", "AC"]

datasets = ["ERA5", "MERRA2", "JRA3Q"]

OUT_DIR_List = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2",
    "JRA3Q": "/scratch/bell/hu1029/LGHW/interm_JRA3Q"
}
yearnameList = {
    "ERA5": "1979_2025",
    "MERRA2": "1980_2025",
    "JRA3Q": "1979_2025"
}

for dtname in datasets:

    OUT_DIR = OUT_DIR_List[dtname]
    yearname = yearnameList[dtname]
    outfile = f"{dtname}_totalNumber.txt"

    with open(outfile, "w") as f:

        for ss in seasons:
            for eve in ['Blocking','Seeding']:
                for typeid in [1,2,3]:
                    for rgname in regions:

                        with open(f"{OUT_DIR}/{dtname}_SD_{eve}FlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}", "rb") as fp:
                            ATLlist = pickle.load(fp)

                        line = f"{dtname} {eve} Type{typeid}_{rgname}_{ss}: {len(ATLlist)}\n"
                        f.write(line)
                        print(line.strip())
                
    print(f"{dtname} done")
    