#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=era5_extend2025
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=6:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo


INPUT_DIR=/scratch/bell/hu1029/Data/raw/ERA5_Z500_originalGrid
INPUT_fname=/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc
OUTPUT_fname=/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2025_1dg.nc
temp_dir=/scratch/bell/hu1029/Data/processed/temp_ERA5_Z500
mkdir -p "$temp_dir"

# regrid the new year data to 1dg from original grid
for year in 2022 2023 2024 2025; do
  cdo remapbil,r360x181 "$INPUT_DIR/ERA5_Z500_6hr_${year}.nc" "$temp_dir/ERA5_Z500_6hr_${year}_1dg.nc"
done

# combine to the processed data ERA5_Z500_6hr_1979_2021_1dg.nc
cdo -b F32 mergetime "$INPUT_fname" "$temp_dir/ERA5_Z500_6hr_2022_1dg.nc" "$temp_dir/ERA5_Z500_6hr_2023_1dg.nc" "$temp_dir/ERA5_Z500_6hr_2024_1dg.nc" "$temp_dir/ERA5_Z500_6hr_2025_1dg.nc" "$OUTPUT_fname"

# remove the temporary files
if [ -d "$temp_dir" ]; then
  rm -rf "$temp_dir"
fi