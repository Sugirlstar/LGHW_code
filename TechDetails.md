# technical details

---- TRACK ----
--- [Z500 anomaly (it's geopotential!, not divided by 9.8), F128 (256latx512lon), 1979-2021] ---
1. TRACK package used: 	track-TRACK-1.5.4 (under /scratch/bell/hu1029/LGHW/TRACK)
2. TRACK running script: /CycloneTrack_geopotentialAnom_MultiTracks.sh
3. TRACK input data: 
   3.1 downloadERA5Z500.py: download (output: /scratch/bell/hu1029/Data/raw/ERA5_Z500_F128/ERA5_Z500_6hr_{yr}.grb)
   3.2 S0_combineNCbyTime_CDO.sh: combine into one file (output: /scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_F128.nc)
   3.3 CCZanom_getZ500anomaly.py: calculate the Z500anomly (output: /scratch/bell/hu1029/Data/processed/ERA5_geopotential500_subtractseasonal_6hr_1979_2021.nc)
   3.4 CCZanom_divideNCfiles.py: divide into seprate years (output: /scratch/bell/hu1029/LGHW/TRACK/TRACK_inputdata_geopotentialAnomaly/ERA5_geopotentialAnomaly_6hr_F128_{year}.nc)
4. TRACK results loading: CCZanom_S3_readMultifilesTrack.py
5. Track trajectory point density: /GRL_code/TrackCodes/ERA5dipole_TrajectoryDensity.py
   outputs: 
   - /scratch/bell/hu1029/LGHW/CCtrackPoints_array.npy
   - ERA5dipole_CC_pointFrequency.png

---- LWA calculation ----
--- [code copy from /depot/wanglei/data/ERA5_LWA_Z500, put under /GRL_code/LWAcodes, recalculate on 6-hourly data]
1. Data preprocess
   1.1 Regrid the data to 1dg: Z500regrid.sh
   Input data: /scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_regulargrid_Float32.nc
   (1440x721, 90~-90, 0.5dg, 19790101-20211231, 6-hourly, varname: z, levels:1, at 500hPa)
   Output data: /scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc
   (360x181, -90~90, 1dg, 19790101-20211231, 6-hourly, varname: z, levels:1, at 500hPa)
2. Calculate LWA: /GRL_code/LWAcodes/main2.py (and function: LWA_f2.py)
3. Transfer the LWA_td, LWA_td_A and LWA_td_C npy files to float32 and plot the mean map: /GRL_code/LWAcodes/LWA_checkPlot.py
   (shape: 62824, 181, 360)

---- Blocking identification ----
# sperate the NH and SH from this point
1. Blocking track + peaking identification + diversity classifying: /GRL_code/BlockingCodes/S1_Blocking_track_ERA5_daily.py, S1_Blocking_track_ERA5_daily_SH.py
   [Note: the blocking identification is based on daily data (only 00:00 extracted), input data shape: (15706, 181, 360); lat is from 90 to 0, decreasing, in NH]
   outputs: /scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily, etc.
2. Blocking data organization, put into the 3D-array and plot: /GRL_code/BlockingCodes/S4_Blocking_transfer2array_ATLregion_newATLdefine.py
   outputs: 
   - /scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingFlagSector2maskClusters_1979_2021_Type{typeid}_newATLdefine.npy
   - ERA5dipole_Blk1_Frequency_Climatology_newATLdefine.png
   - ERA5dipole_Blk1_Frequency_Climatology_WithPeakingPoints_newATLdefine.png 



   

