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
#SBATCH --time=18:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

RAW_DIR=/scratch/bell/hu1029/Data/raw/JRA3Q
OUT_DIR=/scratch/bell/hu1029/Data/processed/JRA3Q_UV_1dg
TMP_DIR=/scratch/bell/hu1029/tmp_jra3q

mkdir -p "$OUT_DIR"

for y in {1979..2025}; do
  echo "Processing $y"

  u_tmp="$TMP_DIR/U_${y}_merged.nc"
  v_tmp="$TMP_DIR/V_${y}_merged.nc"

  u_out="$OUT_DIR/JRA3Q_U_6hr_${y}_1dg.nc"
  v_out="$OUT_DIR/JRA3Q_V_6hr_${y}_1dg.nc"

  cdo -O -b F32 mergetime \
    "$RAW_DIR"/${y}??/jra3q.anl_p125.0_2_2.ugrd-pres-an* \
    "$u_tmp"

  cdo -O -b F32 mergetime \
    "$RAW_DIR"/${y}??/jra3q.anl_p125.0_2_3.vgrd-pres-an* \
    "$v_tmp"

  cdo -f nc -s -O -b F32 -remapbil,r360x181 -selname,ugrd-pres-an-ll125 "$u_tmp" "$u_out"
  cdo -f nc -s -O -b F32 -remapbil,r360x181 -selname,vgrd-pres-an-ll125 "$v_tmp" "$v_out"

  rm -f "$u_tmp" "$v_tmp"
done

