#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=remap_uv_month
#SBATCH --output=logs/remap_%A_%a.out
#SBATCH --error=logs/remap_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=66
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --array=9-12

set -euo pipefail

module load gcc/11.1.0
module load openmpi/4.1.6
module load cdo

mkdir -p logs

INPUT_DIR=/scratch/bell/hu1029/Data/raw/ERA5_UV_2425
TEMP_DIR=/scratch/bell/hu1029/Data/raw/ERA5_UV_2425/tmp_merge_1dg
mkdir -p "$TEMP_DIR"

var=v
year=2024

if [ "$var" = "u" ]; then
    prefix="u_component_of_wind"
else
    prefix="v_component_of_wind"
fi

month=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")

infile="${INPUT_DIR}/${prefix}_${year}_${month}.nc"
outfile="${TEMP_DIR}/${prefix}_${year}_${month}_1dg.nc"

if [ -f "$outfile" ]; then
    echo "Output already exists: $outfile"
    exit 0
fi

if [ ! -f "$infile" ]; then
    echo "Missing input file: $infile"
    exit 1
fi

echo "Remapping $infile -> $outfile"
cdo remapbil,r360x181 "$infile" "$outfile"

echo "Done."