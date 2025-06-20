# 📦 Output File Documentation Table

This document summarizes the key input and output files generated in the project, including their origin, format, and meaning.

### Input files

| 📄 File Name                       | 🛠 Generated By                  | 📐 Format (Shape, Dtype)         | 📈 Description |
|-----------------------------------|----------------------------------|----------------------------------|----------------|
| `/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc`| `S2_Blocking_transfer2array.py`| `nc files`| Gussian grid, 6hourly, global, -89.4~89.4, 0~359.2|


### Output files

| 📄 File Name                       | 🛠 Generated By                  | 📐 Format (Shape, Dtype)         | 📈 Description |
|-----------------------------------|----------------------------------|----------------------------------|----------------|
| `/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily`| `S1_Blocking_track_ERA5_daily.py`| `list, elements: list(2d array)`, 90~-90| a list of event 2d label |
| `/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily`| `S1_Blocking_track_ERA5_daily.py`| `list, elements: dates`| a list of event dates |
| `/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_date_daily`| `S1_Blocking_track_ERA5_daily.py`| `list, elements: dates`| a list of peaking dates |
| `/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_lat_daily`| `S1_Blocking_track_ERA5_daily.py`| `list, elements: lats`, 90~-90| a list of peaking lats |
| `Blockingday_1979_2021_Type{type_idx+1}_{rgname}_{ss}.npy`| `S2_Blocking_transfer2array.py`| `1d array`, 90~-90| a list of the day index of the blocking |
| `BlockingFlagmaskClusters_Type{type_idx+1}_{rgname}_{ss}.npy`| `S2_Blocking_transfer2array.py`| `(time, lat, lon), 0/1`, 90~-90| 3d array, blocked grid: 1, unblocked grid: 0; only the events within the target regions are remained |
| `BlockingFlagmaskClustersEventList_Type{type_idx+1}_{rgname}_{ss}`| `S2_Blocking_transfer2array.py`| `1d list`, 90~-90 | the event id of all the blocking event (corresponding to the order in the 'Blocking_peaking_date_daily' result file) |
| `BlockingClustersEventID_Type{type_idx+1}_{rgname}_{ss}.npy` |`S2_Blocking_transfer2array.py`| `(time, lat, lon), int`, 90~-90| 3d array, blocked grid: filled with the index of the event (index is the order in `BlockingFlagmaskClustersEventList_Type{type_idx+1}_{rgname}_{ss}`) |

