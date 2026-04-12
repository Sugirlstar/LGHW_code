#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=makesingleyearF128
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

RAW_DIR=/scratch/bell/hu1029/Data/raw/JRA55/Z
OUT_DIR=/scratch/bell/hu1029/Data/processed/JRA55_UV_1dg
TMP_DIR=/scratch/bell/hu1029/tmp_jra55

mkdir -p "$OUT_DIR"

for dir in "$RAW_DIR"/*/; do

  y=$(basename "$dir")
  echo "Processing $y"

  rm -rf "$TMP_DIR"
  mkdir -p "$TMP_DIR"

  # merge then regrid to 1dg
  cdo -O -b F32 mergetime "$dir"/anl_p125.033_ugrd* "$TMP_DIR/U_merged.nc"
  cdo -O -b F32 mergetime "$dir"/anl_p125.034_vgrd* "$TMP_DIR/V_merged.nc"

  cdo -f nc -s -O -b F32 remapbil,r360x181 "$TMP_DIR/U_merged.nc" "$OUT_DIR/JRA55_U_6hr_${y}_1dg.nc"
  cdo -f nc -s -O -b F32 remapbil,r360x181 "$TMP_DIR/V_merged.nc" "$OUT_DIR/JRA55_V_6hr_${y}_1dg.nc"

done

