#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=mergetime_1dg_F128
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

# first, convert .grb to .nc, simply run: cdo -f nc -b F32 copy xxx.grb xxx.nc
INPUT_dir=/scratch/bell/hu1029/Data/raw/ERA5_Z500_F128
TEMP_dir=/scratch/bell/hu1029/Data/processed/temp_ERA5_Z500_F128
INPUT_fname=/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_F128.nc
OUTPUT_fname=/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2025_F128.nc

mkdir -p "$TEMP_dir"

# the 1979-2021 data have been inverted
# for the newly added data, invert first, then merge by time
for year in 2022 2023 2024 2025; do
  cdo -b F32 invertlat \
    "$INPUT_dir/ERA5_Z500_6hr_${year}.nc" \
    "$TEMP_dir/ERA5_Z500_6hr_${year}_inv.nc"
done

cdo -b F32 mergetime \
  "$INPUT_fname" \
  "$TEMP_dir/ERA5_Z500_6hr_2022_inv.nc" \
  "$TEMP_dir/ERA5_Z500_6hr_2023_inv.nc" \
  "$TEMP_dir/ERA5_Z500_6hr_2024_inv.nc" \
  "$TEMP_dir/ERA5_Z500_6hr_2025_inv.nc" \
  "$OUTPUT_fname"

cdo sinfo "$OUTPUT_fname"

# remove the temporary files
if [ -d "$TEMP_dir" ]; then
  rm -rf "$TEMP_dir"
fi