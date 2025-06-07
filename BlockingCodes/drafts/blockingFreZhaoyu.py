from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import cv2
import copy
import matplotlib.path as mpath
import pickle
import glob
from netCDF4 import Dataset

B_freq = np.load("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/B_freq.npy")
lat = np.linspace(0,90, 181)
lon = np.linspace(0,360,576,endpoint=False)
print(lat)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=[12,7])
plt.contourf(lon,lat,B_freq, 20, extend="both", cmap='Reds') 
cb=plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('blockingFreq_zhaoyu_MERRA.png')  
plt.show()
plt.close()

print('results saved!')


