import numpy as np
import pandas as pd


regions = ["ATL", "NP", "SP"]
seasons = ["DJF", "JJA", "ALL"]
seasonsmonths = [[12, 1, 2], [6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["CC", "AC"]

def Region_ERA(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max = 45, 75, 300, 30
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max = 45, 75, 120, 210
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max = -70, -50, 170, 300

# data files

# # lat/lon/time for LWA and Blocking
# lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy") # 90~-90 (181)
# lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy") # 0~359 (360)
# Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31") # (15706)
# Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
# Date = list(Date0['date'])


