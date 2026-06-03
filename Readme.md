This project is licensed under the MIT License. See the LICENSE file for details.

> [!ABSTRACT] Project Overview
> This repository contains the technical workflow and implementation of the **Atmospheric Blocking as a Serendipitous Encounter Between Traveling Storms and Periodic Seeds**. 

> [!INFO] Revision Notes (June, 2026)
> This version includes the following updates: 
> 1. Added demo data (`./demo_data`): including intermediate results (e.g., blocking labels and event dates, tracks, BAM index, to facilitate code verification and workflow testing.
> 2. Added Tested Environment description. Added `./environment.yml` and `./cdoVersionInfo.txt` file to document the software environment and package dependencies required to run the analysis workflow.
> 3. Added Expected Runtime description.
> 4. LICENSE added 
> - Intermediate data files referenced hereafter are not included in this repository. They are provided only as examples to illustrate the purpose and expected outputs of different steps in the workflow.

> [!INFO] Revision Notes (April, 2026)
> This version includes the following updates:
> 2. Replaced JRA-55 with JRA-3Q dataset
> 3. Extended all datasets to 2025
> 4. Recomputed all diagnostics and figures accordingly

---
# Tested Environment
The code has been tested on the following environment:
- Operating System: Rocky Linux 8.10 (Green Obsidian)
- Computing Platform: Purdue RCAC Bell Cluster
- Hostname: bell-fe03.rcac.purdue.edu
- Python: 3.12.8
- CDO: 2.4.1
Core Python package dependencies are listed in environment.yml.
No non-standard hardware is required. The workflow can benefit from HPC resources due to data volume and computational cost.
## HPC Setups

> [!SETTINGS] System Configuration
> - **Compute Cluster**: All scripts and notebooks are executed on **Purdue RCAC Bell** ([Documentation](https://www.rcac.purdue.edu/compute/bell)).
> - **Job Submission**: Utilize `JupyterDebug.slurm` for interactive core allocation.
> - **Environment Setup**: 
> 	1. Run the slurm script to allocate resources.
> 	2. SSH to the specific compute node assigned.
> 	3. Select the Python interpreter/kernel as specified in the slurm output.
## Python Pathing
Ensure the project root is in your system path:
```bash
export PYTHONPATH=./Nature_Serendipity:$PYTHONPATH
```
In your Jupyter Notebooks, include the following:
```python
import sys
sys.path.insert(0, "./Nature_Serendipity")
```

## Expected Runtime
For the provided dataset, most scripts complete within a few minutes/hours on a standard desktop computer. The full analysis using the complete reanalysis datasets may require substantially longer runtimes and is recommended for execution on HPC systems.

---

# Data Inventory
### 1. Primary Datasets (raw data)

| Source     | Resolution    | Variable                                      | Access Link                                                                                                      | Description                                                           | Download process                                                                                                                                                                                                          |
| :--------- | :------------ | :-------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ERA5**   | 0.25° / F128  | Geopotential (z)                              | [Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)        | unit: m*2 s*-2; format: netcdf/grib                                   | use python script:<br>`./dataprep_ERA5/S0_downloadERA5Z500_F128.py` and `S0_downloadERA5Z500_originalGrid.py`                                                                                                             |
| **MERRA2** | 0.5° x 0.625° | Geopotential Height (H)                       | [NASA GES DISC](https://disc.gsfc.nasa.gov/datasets/M2I6NPANA_5.12.4/summary?keywords=MERRA-2%20inst6_3d_ana_Np) | unit: m; format: netcdf                                               | login → “Subset/Get Data” → select time → generate download links → .txt file (named e.g., `subset_M2I6NPANA_5.12.4_20260412_045649_.txt`) → change it as input in<br>`./dataprep_MERRA2/S0_downloadPick.sh`<br>→ run it. |
| **JRA3Q**  | ~1.25°        | Geopotential Height (isobaric analysis field) | [UCAR GDEX](https://gdex.ucar.edu/)                                                                              | unit: gpm; format: grib1; 0.01-1000hPa; var name: `hgt-pres-an-ll125` | data download from: [GDEX](https://gdex.ucar.edu/datasets/d640000/) (in Data Access → Globus Transfer)                                                                                                                    |

### 2. Processing Pipeline
#### ERA5
| Description                    | File Name (under `Data/processed`)                                                                                                                 | Details                                                                                                                | Generated With                                 |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------- |
| **S0: Download**               | `/Data/raw/ERA5_Z500_F128/ERA5_Z500_6hr_{year}.nc`<br>*(also copy to: `/scratch/bell/hu1029/LGHW/TRACK/ERA5_TRACK_inputdata_geopotential_yearly`)* | Geopotential, F128, original, 6-hourly, yearly file, Latitude decreasing (90 to -90)                                   | `dataprep_ERA5/S0_downloadERA5Z500_F128`       |
| **S1: Combine multiple files** | `ERA5_Z500_6hr_1979_2021_regulargrid_Float32.nc`                                                                                                   | Geopotential, 1440x721, lat decreasing 90 to -90, 0.25dg, 19790101-20211231, 6-hourly, varname: z, levels:1, at 500hPa | `cdo mergetime`                                |
| **S1: 1dg, for Blocking**      | `ERA5_Z500_6hr_1979_2025_1dg.nc`                                                                                                                   | Geopotential, 360x181, latitude increasing (-90 to 90), 0-359, 6-hourly                                                | `dataprep_ERA5/S1_Z500regrid.sh`               |
| **S1: F128, for TRACK**        | `ERA5_Z500_6hr_1979_2025_F128.nc`                                                                                                                  | Geopotential, F128, 6-hourly, Latitude increasing (-90 to 90), 0-359                                                   | `dataprep_ERA5/S1_combineNCbyTime_CDO_F128.sh` |
| **S2: F128 single-year files** | `/LGHW/TRACK/ERA5_TRACK_inputdata_geopotential_yearly`                                                                                             | Copy from `/Data/raw/ERA5_Z500_F128/ERA5_Z500_6hr_{year}.nc`                                                           | -                                              |
| **S3: Anomaly & Climatology**  | `ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc` / `..._F128.nc`<br>`ERA5_Z500climatology_monthly_1979_2021_1dg.nc` / `..._F128.nc`        | Geopotential height (m), 1dg and F128, lat increasing, 0-359; Anomaly and climatology                                  | `dataprep_ERA5/S3_Z500anomalyCal.py`           |
| **S4: Divide into years**      | `/LGHW/TRACK/ERA5_TRACK_inputdata_geopotentialAnomaly_yearly/ERA5_geopotentialAnomaly_6hr_{year}.nc`                                               | Geopotential, F128, lat decreasing                                                                                     | `dataprep_ERA5/S4_divideNCfiles.sh`            |

> [!INFO] Revision Notes (April, 2026)
> Extend to 2025: redownload data 2022-2025 (for original grid, use`S0_downloadERA5Z500_originalGrid.py`); regrid to 1dg and combine to the processed one (`SR_extend2025.sh`). For F128, use `SR_extend2025_F128` (invert first then combine to 1979-2025).
> 

#### MERRA2
| Description                                | file name (under **Data/processed**)                                                                                                                                                                                                               | details                                                                              | generated with                                                                                                |
| :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| S0: download and select vars and levels    | **MERRA2/Z500/MERRA2_100.inst6_3d_ana_Np.{yyyymmdd}_Z500.nc**                                                                                                                                                                                      | geopotential height (m), lon: -180 to 179.375 by 0.625, -90 to 90 by 0.5, 6-hourly   | `dataprep_MERRA2/S0_downloadPick.sh`                                                                          |
| S1: 1dg and F128 multi-year file           | **MERRA2_Z500_6hr_1980_2021_1dg.nc**<br>and<br>**MERRA2_Z500_6hr_1980_2021_F128.nc**                                                                                                                                                               | geopotential height (m), 360x181 and F128, latitude increasing (-90~90), 6-hourly    | `dataprep_MERRA2/S1_mergetime_1dg_F128.sh`                                                                    |
| S2: F128 single-year files, for TRACK      | **/scratch/bell/hu1029/LGHW/TRACK/MERRA2_TRACK_inputdata_geopotential_yearly**                                                                                                                                                                     | geopotential, F128, 6-hourly. Latitude increasing (-90~90), 0-359                    | `dataprep_MERRA2/S2_makesingleyearF128File`                                                                   |
| S3: Z500 anomaly and climatology           | **MERRA2_Z500anomaly_subtractseasonal_6hr_1980_2025_1dg.nc** and **MERRA2_Z500climatology_monthly_1980_2021_1dg.nc**<br><br>**MERRA2_Z500anomaly_subtractseasonal_6hr_1980_2025_F128.nc** and **MERRA2_Z500climatology_monthly_1980_2021_F128.nc** | geopotential height (m), 1dg and F128, lat increasing, 0-359 anomaly and climatology | `dataprep_MERRA2/S3_Z500anomalyCal.py`<br>also make sanity-check plots:<br>**ZanomClim12Months_1dg/F128.png** |
| S4: divide into single year                | **/LGHW/TRACK/MERRA2_TRACK_inputdata_geopotentialAnomaly_yearly/MERRA2_geopotentialAnomaly_6hr_{year}.nc**                                                                                                                                         | geopotential, F128, lat decreasing                                                   | `dataprep_MERRA2/S4_divideNCfiles.sh`                                                                         |
| S5: process U and V data for calculate EKE | **/scratch/bell/hu1029/Data/processed/MERRA2_UV_1dg/MERRA2_U/V_6hr_${y}_1dg.nc**                                                                                                                                                                   | m/s, at multiple levels, 1dg                                                         | `dataprep_MERRA2/S5_UV_RegridMerge.sh`                                                                        |

#### JRA3Q
| description                                | file name (under **Data/processed**)                                                                                                                                                                                                           | details                                                                              | generated with                                                                                               |
| :----------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| S1: 1dg and F128 multi-year file           | **JRA55_Z500_6hr_1980_2021_1dg.nc**<br>and<br>**JRA55_Z500_6hr_1980_2021_F128.nc**                                                                                                                                                             | geopotential height (m), 360x181 and F128, latitude increasing (-90~90), 6-hourly    | `dataprep_JRA55/S1_mergetime_1dg_F128.sh`                                                                    |
| S2: F128 single-year files, for TRACK      | **/scratch/bell/hu1029/LGHW/TRACK/MERRA2_TRACK_inputdata_geopotential_yearly**                                                                                                                                                                 | geopotential height (m), F128, 6-hourly. Latitude increasing (-90~90), 0-359         | `dataprep_JRA55/S2_makesingleyearF128File`                                                                   |
| S3: Z500 anomaly and climatology           | **JRA55_Z500anomaly_subtractseasonal_6hr_1980_2021_1dg.nc** and **JRA55_Z500climatology_monthly_1980_2021_1dg.nc**<br><br>**JRA55_Z500anomaly_subtractseasonal_6hr_1980_2021_F128.nc** and **JRA55_Z500climatology_monthly_1980_2021_F128.nc** | geopotential height (m), 1dg and F128, lat increasing, 0-359 anomaly and climatology | `dataprep_JRA55/S3_Z500anomalyCal.py`<br>also make sanity-check plots:<br>**ZanomClim12Months_1dg/F128.png** |
| S4: divide into single year                | **/LGHW/TRACK/JRA55_TRACK_inputdata_geopotentialAnomaly_yearly/JRA55_geopotentialAnomaly_6hr_{year}.nc**                                                                                                                                       | geopotential, F128, lat decreasing                                                   | `dataprep_JRA55/S4_divideNCfiles.sh`                                                                         |
| S5: process U and V data for calculate EKE | **/scratch/bell/hu1029/Data/processed/JRA55_UV_1dg/JRA55_U/V_6hr_${y}_1dg.nc**                                                                                                                                                                 | m/s, at multiple levels, 1dg                                                         | `dataprep_JRA55/S5_UV_RegridMerge.sh`                                                                        |


---

# Computational Modules

## TRACK (./TrackCodes)
> [!IMPORTANT]
> Input data: geopotential (not geopotential height), F128 (256latx512lon), 1979-2021, 6houly For the instructions for configuring and running TRACK, please refer to _**Instructions on running the TRACK program_Yanjun.docx**_
### Instructions & Setup

1. Download and set the TRACK package: `track-TRACK-1.5.4` (put it under `./LGHW/TRACK`)
2. Execution: Run the TRACK, based on the multiprocess scripts: `S1_CycloneTrack_geopotential_MultiTracks.sh` and `/S1_CycloneTrack_geopotential_MultiTracks_SH.sh`
	- **Cannot run simultaneously !!!
	- (see the configuration document `Instructions on running the TRACK program_Yanjun.docx` for detailed explanation of the scripts).
	- **outputs:** tracks for each year, stored in `.gz` files (e.g., `./LGHW/TRACK/TRACK_inputdata_geopotentialAnomaly/TRACKS_SH/ERA5_geopotentialAnomaly_6hr_F128_1979_zonefilt_T42_SH/ff_trs_neg.gz`)
3. TRACK results reading and orgnizing: `S2_readMultifilesTrack_NH.py` and `S2_readMultifilesTrack_SH.py`
    - **outputs:** `./LGHW/{trackType}Zanom_allyearTracks_{k}.pkl` (`{trackType}`=AC/CC, `{k}`=NH/SH)
4. Track trajectory point density: `S3_ERA5dipole_TrajectoryDensity_2Arrs.py`
    - **outputs:** 
	    - `./LGHW/{trackType}trackPoints_arrayF128_{k}.npy` and `./LGHW/{trackType}trackPoints_array1dg_{k}.npy` (Bool array, mask of track points).
	    - `./LGHW/{trackType}trackPoints_TrackIDarray_{k}_F128.npy` and `./LGHW/{trackType}trackPoints_TrackIDarray_{k}_1dg.npy` (Float, storing the track id at each grid point, Increasing Lat).
### Workflow

| description                                 | Inputs                                                                                                                                                                                           | Script (under ./TrackCodes/)                                                                       | Outputs                                                                                                                                                                                                                                                                                                                                                        |
| :------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S1: run the TRACK program                   | `/scratch/bell/hu1029/LGHW/TRACK/{dataset}_TRACK_inputdata_geopotential_yearly/{dataset}Z500_6hr{year}.nc`                                                                                       | `S1_CycloneTrack_geopotential_MultiTracks.sh` and `S1_CycloneTrack_geopotential_MultiTracks_SH.sh` | **TRACK/{dataset}_TRACK_inputdata_geopotential_yearly/TRACKS/{dataset}_Z500_6hr_zonefilt_T42/ff_trs_neg.gz or ff_trs_pos.gz** and **TRACK/{dataset}_TRACK_inputdata_geopotential_yearly/TRACKS_SH/{dataset}_Z500_6hr_zonefilt_T42/ff_trs_neg.gz or ff_trs_pos.gz**                                                                                             |
| S2: read from TRACK results, turn into .pkl | last outputs                                                                                                                                                                                     | `S2_readMultifilesTrack_NH.py` and `S2_readMultifilesTrack_SH.py`                                  | **{OUT_DIR}/{dtname}_CCZanom_allyearTracks_NH.pkl** and **{OUT_DIR}/{dtname}_CCZanom_allyearTracks_SH.pkl**                                                                                                                                                                                                                                                    |
| S3: transfer the Track to 3d-arrays         | last outputs. Time reference file: `/processed/{dtname}_Z500_6hr_1979_2021_1dg.nc`. Coordinates reference file: `/processed/{dtname}_Z500climatology_monthly_1979_2021_F128.nc` (lat increasing) | `S3_ERA5dipole_TrajectoryDensity_2Arrs.py`                                                         | **{OUT_DIR}/{dtname}{trackType}trackPoints_arrayF128/1dg_{k}.npy** : bool 3d array, 1/0 represents there are trackpoints or not. **{OUT_DIR}/{dtname}{trackType}trackPoints_TrackIDarray_{k}_F128/1dg.npy** : 3d array, each location store the track’s ID (0 for none). and sanity-check plots: `./TrackCodes/{dtname}_{trackType}pointFrequency{k}_F128.png` |
 
> [!HINT] 
> There are two sets of input files for running TRACK: **geopotential** and **geopotential anomaly**.
> Currently using: **geopotential anomaly**

---
## LWA Calculation (./LWACodes)

> [!NOTE]
> Code was provided by Zhaoyu, modified based on 6-hourly data, the original code is from /depot/wanglei/data/ERA5_LWA_Z500
### Instructions & Setup:

* **Inputs:** `{dtname}_Z500_6hr_1980_2021_1dg.nc` (1dg, lat increasing order)
* **Calculate LWA:** `main2.py` base on the function `LWA_f2.py` (calculation time: ~12hours for each dataset)
    * (Note: in the `LWA_f2.py`, the latitude of the input `.nc` file was forced to be decreasing (90~-90). In the `main2.py`, the results are converted to lat-increasing order.)
	* For `LWA_f2.py`: **It takes over 8 hours to calculate LWA.Need highmem (>256G mem) to run the script, otherwise may meet **OOM (out of memory)**
	* Use `mainParallel_ERA5.py` (and `submit_1_parallel_ERA5`) to do parallel computing. Change `dtname = "ERA5"` to change to a different dataset. The time running each is only ~5 mins.
* **Outputs:**
```python
np.save(f"{OUT_DIR}/{dtname}_LWA_td_{yearname}_6hr.npy", LWA_td)
np.save(f"{OUT_DIR}/{dtname}_LWA_td_A_{yearname}_6hr.npy", LWA_td_A)
np.save(f"{OUT_DIR}/{dtname}_LWA_td_C_{yearname}_6hr.npy", LWA_td_C)
np.save(f"{OUT_DIR}/{dtname}_LWA_lat_{yearname}_6hr.npy", lat)
np.save(f"{OUT_DIR}/{dtname}_LWA_lon_{yearname}_6hr.npy", lon)
```

---
## Blocking and Seeding (./BlockingSeedFinding)

> [!NOTE]
> The code for identifying blocking and seeding events, and selecting the target regions. NH and SH are separately calculated.
### Instructions & Setup:
1. **Blocking/Seeding tracking + peaking identification + diversity classifying: **
	- `S1_WatershedSeedBlocking_track_daily_NH` and `S1_WatershedSeedBlocking_track_daily_SH`
	- Method description
        1. For each grid point, determine whether its value exceeds the threshold. The threshold is defined as: For each day and longitude, the maximum LWA over latitude is first computed, forming a sample of size N_day*N_lon. The blocking threshold is then defined as the p-th percentile of this sample (50th for blocking, 40th for seeding). 
           Then, connected clusters are identified using `cv2.connectedComponentsWithStats` with `connectivity=4`, where each contiguous group of grid cells is assigned a unique label.
        2. Connect labels wrapping across the dateline (lon=0 and lon=360)
        3. Watershed split. Some clusters may contain multiple local maxima, indicating that several events have been merged into a single region. To separate them, local maxima of LWA are identified within each cluster and used as markers in a marker-controlled watershed applied to −LWA, splitting the cluster into subregions, each associated with a dominant peak.
	        - `MIN_DIST`: If two maxima are closer than MIN_DIST (5 degree), they are treated as the same peak.
	        - `DX_THRESH`: Peaks within DX_THRESH longitudinal distance (18*3 degree) are grouped into the same event.
	        - `TH_ABS`: Only peaks stronger than TH_ABS are considered (No requirement here).
        4. For each cluster, the longitudinal width and center (defined as the location of the local maximum LWA) are computed. Clusters are discarded if their latitude is below 30°, if their width is smaller than `BlockingLonWidth`/`SeedingLonWidth`, or if their longitudinal width exceeds 120°.
        5. Pairing events between consecutive days. For each day d, we pair each detected event with at most one event on day d+1 using a nearest-neighbor matching. We first compute a distance matrix between all event centers on the two days. Event pairs are then selected iteratively from the smallest remaining distance (greedy matching). Once a pair is accepted, the corresponding row and column are removed from further consideration.
	        - `lon_thresh` = 18, `lat_thresh` = 13.5 : Maximum allowed day-to-day longitudinal and latitudinal displacement between paired events.
        6. Event tracking across days. After establishing one-to-one event pairs between day d and d+1, we construct event tracks by following these links forward in time. For each untracked event on day d, tracking starts from its label and iteratively appends the paired event on the next day until no valid pair exists (or a stopping criterion is met). When stationary=True, we additionally require that each subsequent event remains within 1.5 times the day-to-day displacement thresholds relative to the initial event location. This filter was applied on blocking events only (parameter in BKSDIdentifyFun: stationary=True)
	        - `Duration`: minimum number of consecutive days required for a track to be retained as a blocking event. (SeedingDuration = 3; BlockingDuration = 5).
	        - `1.5*lon_thresh and 1.5*lat_thresh`: limiting drift relative to the *initial* event center
   
	- Key variables
		```python
		#%% Parameters ###
		DX_THRESH = int(18*3) 
		MIN_DIST = 5
		TH_ABS = None
		dlat = dlon = 1
		BlockingDuration = 5
		SeedingDuration = 3
		BlockingLonWidth = 15
		SeedingLonWidth = 10
		valueBlockingThresh = 50  # percentile
		valueSeedingThresh = 40    # percentile
		```

    - Inputs (./LGHW/): outputs from LWA calculation
    - Outputs:
		```python
		save_dict = { 
			f"{dtname}_SD_{Blocking/Seeding}_peaking_date_daily_{NH/SH}": Blocking_peaking_date, # 1d-list, each element represent the date of peaking of each event, single value
			f"{dtname}_SD_Blocking_peaking_lon_daily_NH": Blocking_peaking_lon, # 1d-list, each element represent the lon of peaking of each event, single value 
			f"{dtname}_SD_Blocking_peaking_lat_daily_NH": Blocking_peaking_lat, # 1d-list, each element represent the lat of peaking of each event, single value 
			f"{dtname}_SD_BlockingTotal_date_NH": BlockingTotal_date, # list[list], each sublist is a list of dates of each seeding/blocking event 
			f"{dtname}_SD_BlockingTotal_label_NH": BlockingTotal_label, #list[list of 2d-array], each sublist is a list of 2d bool masks of seeding/blocking locations (shape: 90,360; lat increasing) 
			f"{dtname}_SD_BlockingTypeI_NH": BlockingTypeI, # list, 1d, represent the type of each event, 1-ridge, 2-trough, 3-dipole 
			f"{dtname}_SD_Blocking_diversity_date_daily_NH": Blocking_diversity_date, # list[list of 3 types], [Ridge,Trough,Dipole]; the dates of each event 
			f"{dtname}_SD_Blocking_diversity_label_daily_NH": Blocking_diversity_label, # list[list of 3 types], [Ridge,Trough,Dipole]; each sublist is the 2d bool masks of seeding/blocking locations (shape: 90,360; lat increasing) 
			f"{dtname}_SD_Blocking_diversity_peaking_date_daily_NH": Blocking_diversity_peaking_date, # list[list of 3 types], [Ridge,Trough,Dipole]; list of the peaking dates of each event 
			f"{dtname}_SD_Blocking_diversity_peaking_lon_daily_NH": Blocking_diversity_peaking_lon, # list[list of 3 types], [Ridge,Trough,Dipole]; list of the peaking lat of each event 
			f"{dtname}_SD_Blocking_diversity_peaking_lat_daily_NH": Blocking_diversity_peaking_lat, # list[list of 3 types], [Ridge,Trough,Dipole]; list of the peaking lon of each event f"{dtname}_SD_Blocking_freq_NH": B_freq }
		```
		and a figure output: **{dtname}_SD_blocking/seedingFreq_daily_watershed_SH/NH.png’**   

2. **Blocking data organization, put into the 3D-array and plot: `S2_Blocking_transfer2array.py`**
    - Method description
		 **Regional and seasonal filtering of blocking/seeding events.**
		 For each dataset, blocking (or seeding) events are classified by type (ridge, trough, and dipole) and then filtered by geographic region and season. Three target regions are defined (Atlantic, North Pacific, and South Pacific), each specified by latitude–longitude bounds; for seeding, we expand the region 30 degrees westward, to include more potential seeding events that are close to the original region boundary. This is because the seeding events can be more flexible in location, and we want to make sure we don't miss those that are just outside the original boundary but still relevant for the blocking events in the target region.
		 An event is assigned to a season if any day within its lifetime falls within the target months. Spatial filtering is based on the event’s peak location (LWA maximum): events are retained only if their peak latitude and longitude lie within the specified regional bounds (accounting for periodic longitude across 0°/360°).
		 For the retained events, their daily footprints are mapped onto a three-dimensional grid (time × latitude × longitude) to construct a Boolean mask indicating the presence of an event. In addition, an event-ID array is generated that records the global event index at each grid cell and time step, and a list of event indices belonging to the target region–season–type combination is saved. 
    - Inputs:
        ```python
        with open(f"{OUT_DIR}/{dtname}_SD_{eve}_diversity_label_daily_{k}", "rb") as fp:
            Blocking_diversity_label = pickle.load(fp)   
        with open(f"{OUT_DIR}/{dtname}_SD_{eve}_diversity_date_daily_{k}", "rb") as fp:
            Blocking_diversity_date = pickle.load(fp)     
        with open(f"{OUT_DIR}/{dtname}_SD_{eve}_diversity_peaking_lon_daily_{k}", "rb") as fp:
            peakinglonList = pickle.load(fp)
        with open(f"{OUT_DIR}/{dtname}_SD_{eve}_diversity_peaking_lat_daily_{k}", "rb") as fp:
            peakinglatList = pickle.load(fp)
        ```

	 - Outputs:
        ```python
        # save the blocking array: 3d bool array, [time, lat, lon], bool, mask of block or not; if not: 0/False
        np.save(f"{OUT_DIR}/{dtname}_SD_{eve}FlagmaskClusters_Type{type_idx+1}_{rgname}_{ss}.npy", blocking_array)
        # save the blocking id list: 1d list, saving the target region’s blocking/seeding event global id
        with open(f"{OUT_DIR}/{dtname}_SD_{eve}FlagmaskClustersEventList_Type{type_idx+1}_{rgname}_{ss}", "wb") as fp:
            pickle.dump(ATLlist, fp)
        # save the id array: 3d array, [time, lat, lon], int, saving the blocking/seeding event’s global id in the target positions; position with no event: -1
        np.save(f"{OUT_DIR}/{dtname}_SD_{eve}ClustersEventID_Type{type_idx+1}_{rgname}_{ss}.npy", blockingID_array)
        ```
	     Also a figure output: **{dtname}*{eve}GlobalFrequency*{blkTypes[type_idx]}_{ss}.png**
	     
3. Blocking number in regions, seasons, types, datasets: `S3_getBlockingStatis.py`
	- Inputs: **{OUT_DIR}/{dtname}*SD*{eve}FlagmaskClustersEventList_Type{typeid}*{rgname}*{ss}**
	- Outputs: **{dtname}_totalNumber.txt** (the number of each types of blocking/seeding events)

---
## BAM (./BAM)
### Instructions & Setup:

1. **Calculate the EKE:** `S1_EKE_total.py`
    - **Input:** (`/depot/wanglei/data/ERA5_uvT/`): `u_component_of_wind_.nc` and `v_component_of_wind_.nc`
    - **Output:** (`/Data/processed/ERA5_EKE_total_1979_2021`): `{SH/NH}TROP{year}.nc`
    - run with: `submit_EKE.slurm`
2. **Get the BAM index:** `S2_SBAM_index_EKE.py`
    - **Output:** (`/scratch/bell/hu1029/LGHW/`): `{k}_BAM_index_total_no_leap.npy`
    - Details:
	    - Step1: get zonal-mean EKE (select 20-70N/S, 200-1000hPa; average along longitude)
	    - Step2: remove all 0229
	    - Step3: calculate daily climatology
	    - Step4: latitude area weighting and vertical mass weighting
	    - Step5: EOF
	    - Step6: PC1 power spectrum.
3. **Get the high and low BAM phase:** `S3_getBAMphase.py` 
    - *(Note: the BAM index has been interpolated to with 0229 since here)*
    - **Output:** `{k}_BAM_event_peak_list.pkl` and `{k}_BAM_event_low_list.pkl` 
    - **Description:** The date index of peak/low BAM state (only one day for each event).

---
## Eddy-Blocking Interactions (./EddyBlockingCodes)
### Instructions & Setup:

0. Get the reference lat/lon coordinate values for TRACK data: `S0_getTRACKrefLatLon.py`
1. Find the blocking-eddy interaction cases: `S1_blockingTrackInteraction.py`  
   (blocking daily data are transfered to 6-hourly by repeat 4 times per day; track points are transfer to 1dg resolution by finding the nearest grid point (findClosest); Only ‘through’ and ‘absorbed’ eddies are considered as interacting eddies.)
	- For parallel computing: use `S1_make_tasklist.py` to create `S1_tasklist.txt` as args inputs in `submit_S1.slurm`
	- Outputs (/scratch/bell/hu1029/LGHW/):
		- **BlockingEventPersistence_Type{typeid}*{rgname}*{ss}.npy**: blocking persistence for each event, 1-d array.
		- **BlockingType{typeid}*EventEddyNumber_1979_2021*{rgname}*{ss}*{cyc}.npy**: 1-d array, the number of interacting number for each blocking event. length = len of regional blocking event. 0 for no interaction
		- **TrackBlockingType{typeid}*Index_1979_2021*{rgname}*{ss}*{cyc}.npy**: 1-d array of the blockingid that each track is related to; length = track number. -1 for no interaction
		- **BlockingType{typeid}*ThroughTrack_1979_2021*{rgname}*{ss}*{cyc}.npy**: 1-d array, storing the ‘through’ track index (the global position)
		- **BlockingType{typeid}*AbsorbedTrack_1979_2021*{rgname}*{ss}*{cyc}.npy**: 1-d array, storing the ‘absorbed’ track index
		- **BlockingType{typeid}*EdgeTrack_1979_2021*{rgname}*{ss}*{cyc}.npy**: 1-d array, storing the ‘edge’ track index
		- **BlockingType{typeid}*InterType_1979_2021*{rgname}*{ss}*{cyc}.npy**: 1-d array of the interaction type that each track is related to; length = track number. ‘N’ for no interaction, ‘T’ for through, ‘A’ for absorbed, ‘E’ for edge.
		- **Interaction_summary.txt**: summary of length of each interaction type.
		- **BlkPersis_EddyNumber_Cor.txt**: summary of correlation between blocking persistence and eddy numbers
2. Find the first day center for each blocking event: `S2_00_getBlocking1stDayLoc.py`
	- **Outputs (/scratch/bell/hu1029/LGHW/):** **Blocking1stday{Date/Lat/Lon}List_blkType{typeid}*{rgname}*{ss}**  
     (the date, lat and lon values for each first day center)
3. Calculate the density of tracks that have interacted with blockings: `S2_01_InteractingTrajectoryDensity.py`
	- **Inputs (/scratch/bell/hu1029/LGHW/):** **TrackBlockingType{typeid}*Index_1979_2021*{rgname}*{ss}*{cyc}.npy** and **{cyc}Zanom_allyearTracks{HMi}.pkl**
	- **Outputs (/scratch/bell/hu1029/LGHW/):**
		   - **{cyc}trackInteracting_array_Type{typeid}*{rgname}*{ss}.npy** : 3d-array, bool, 1 for interacted track location, 0 for no interaction. Lat Increasing!
		- **{cyc}trackInteracting_idarr_Type{typeid}*{rgname}*{ss}.npy** : 3d-array, storing the eddy’s global id at each location (the id is directly extract from the .pkl). Lat Increasing!
4. Get the composite map of blocking events and related eddy track points: `S2_02_CenterComps_1stdayCenter.py`
	-  **Inputs (/scratch/bell/hu1029/LGHW/):** **{AC/CC}trackInteracting_array_Type{typeid}*{rgname}*{ss}.npy**, **Blocking1stday{Date/Lat/Lon}List_blkType{typeid}*{rgname}*{ss}**, **/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc** (Lat increasing)
	- **Outputs (/scratch/bell/hu1029/LGHW/):** **CenteredZ500_timewindow41_BlkType_Type{typeid}*{rgname}*{ss}.npy**, **CenteredAC_timewindow41_BlkType_Type{typeid}*{rgname}*{ss}.npy**, **CenteredCC_timewindow41_BlkType_Type{typeid}*{rgname}*{ss}.npy**. The centered slices of all blocking events and related AC/CC track points.  3d arr (event, relativeTime, relativeLat, relativeLon).
5. Get the enter time: `S3_EnterLeavingTime.py`
	 - **Inputs:**  **{AC/CC}Zanom_allyearTracks_{NH/SH}.pkl**, **TrackBlockingType{typeid}*Index_1979_2021*{rgname}*{ss}*{cyc}.npy**, **SD_BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}**, **Blocking1stdayDateList_blkType{typeid}*{rgname}*{ss}**
	- **Outputs:**
		- **EnterTimePointr2Blk1stDay_type{typeid}*{cyc}*{rgname}_{ss}.npy** (the eddies’ entry time, relative to blocking’s 1st day; length = len of interacting blk, -1 was skipped)
		- **LeaveTimePointr2Blk1stDay_type{typeid}*{cyc}*{rgname}_{ss}.npy**  (the eddies’ leave time, realtive to blocking’s 1st day; length = len of interacting blk, -1 was skipped)
		- **intopercentList_type{typeid}*{cyc}*{rgname}_{ss}.npy**  (the entrey time relative to the total blocking duration, percentage; length = len of interacting blk, -1 was skipped)
		- **leavepercentList_type{typeid}*{cyc}*{rgname}_{ss}.npy**  (the leave time relative to the total blocking duration, percentage; length = len of interacting blk, -1 was skipped)
			```python
			InteractingBlockID = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}*Index_1979_2021*{rgname}*{ss}*{cyc}.npy')
			for i, blockid in enumerate(InteractingBlockID):
			    if blockid >= 0:
			        intopercentList.append(intopercent)
			```

6.  Get LWA for each blocking event and each AC/CC eddy: `S4_getLWAforBlock` and `S4_getLWAforTrack`
	-  Inputs (`S4_getLWAforBlock` ): {dtname}_LWA_td_{yearname}_6hr.npy, {dtname}_SD_Blocking_diversity_label_daily_NH/SH and SD_Blocking_diversity_date_daily_NH/SH , {dtname}_SD_BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss} , {dtname}_BlockEventDailyLWAList_{yearname}_Type{typeid}_{rgname}_{ss}.pkl
	- Outputs (`S4_getLWAforBlock`): {dtname}_BlockEventAvgedLWA_Type{typeid}_{rgname}_{ss}.npy 
	- Inputs (`S4_getLWAforTrack`): {dtname}_LWA_td_{yearname}_6hr.npy, {dtname}_{cyc}Zanom_allyearTracks{HMi}.pkl
	- Outputs: {dtname}_{cyc}TrackLWA_{yearname}{HMi}.pkl

> [!note] Definition of Blocking LWA
> The local wave activity (LWA) is first computed at 6-hourly resolution. For each identified blocking event, a binary mask is constructed to represent the spatial extent of the blocking at each time step.
>
> The blocking LWA is defined as the area-integrated LWA within the blocking region:
>
> $$
> \mathrm{LWA}_{\text{block}}(t) = \sum_{\text{lat, lon}} \mathrm{LWA}(t, \text{lat}, \text{lon}) \cdot M(t, \text{lat}, \text{lon})
> $$
>
> where $M$ is the blocking mask (equal to 1 inside the blocking region and 0 elsewhere).
>
> This produces a time series of total wave activity associated with each blocking event. The overall intensity of a blocking event is then quantified by the temporal mean of $\mathrm{LWA}_{\text{block}}$ over its lifetime.
>
> This definition represents the total wave activity within the blocking region and therefore captures both the intensity and spatial extent of the blocking.

> [!note] Definition of Track LWA
> The local wave activity (LWA) is computed at 6-hourly resolution. For each track, the LWA associated with the track is evaluated along its trajectory.
>
> At each time step, the track position is mapped to the nearest grid point of the LWA field. A square window of ±5 grid points (i.e., an 11 × 11 grid-point neighborhood) centered at the track position is then defined, and the LWA within this window is summed:
>
> $$
> \mathrm{LWA}_{\text{track}}(t) = \sum_{i,j \in \Omega(t)} \mathrm{LWA}(t, i, j)
> $$
>
> where $\Omega(t)$ denotes the local spatial neighborhood around the track position.
>
> This produces a time series of LWA along each track. The overall LWA associated with a track is then quantified by the temporal mean over its lifetime.

7. Get the events when eddy enter between blocking events (at least 2 days left and at least 2 days after the blocking start): `S5_LWAvariation_composite.py`
	- Outputs: the average LWA series ({dtname}_MiddleEddiesTrackLWAseries_Type{typeid}_{rgname}_{ss}_{cyc}_composites.npy and {dtname}_MiddleEddiesBlkLWAseries_Type{typeid}_{rgname}_{ss}_{cyc}_composites.npy)
8. Get the track density for developed v.s. decayed seeds: `S6_SeedTrackDensity.py`
	- Track-density composite around seeding events. For each seeding event, the event center is defined on the **first day of the seeding event** as the grid point with the maximum daily LWA value within the first-day seeding mask. This center is used as a fixed reference point for the subsequent composite analysis. For both **developed** seeding events (those that later match a blocking event) and **decayed** seeding events (those that do not), a local box is extracted for each event from **day 0 to day 9** relative to the first seeding day. The extracted box spans: **±25 grid points in latitude** and **±30 grid points in longitude**. The same centered window is applied to both: daily mean **LWA** and daily **track-point mask**. 
	- The track data are originally 6-hourly. They are first converted to **daily occurrence masks** by taking the maximum over the four 6-hourly time steps within each day. Therefore, if a track point appears at least once within a given day at a grid point, that grid point is assigned a value of 1 for that day.
	- The composite track density is then calculated as follows: 1. **Sum over events** at each relative day and grid point. This gives the total number of events with a track point at that location. 2. **Divide by the number of events** in that category. This converts the total count into an **occurrence frequency** (or fractional density), i.e., the fraction of events with a track point at each relative location.
> [!note]
> 	Need to be run with `submit_run_seed_composite_array.sh` and `tasklist_seed.txt`
>

---
## Figure Generating (./FigurePlotting)

follow the .ipynb

