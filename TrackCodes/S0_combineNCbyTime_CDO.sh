#!/bin/bash

module load cdo

cdo mergetime /scratch/bell/hu1029/Data/raw/ERA5_Z500_F128/*.nc /scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_F128.nc
