#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=highmem
#SBATCH --job-name=parallel_job
#SBATCH --output=debugresult_%j.out
#SBATCH --error=debugerror_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=8:00:00

set -euo pipefail

module load gcc/11.1.0
module load openmpi/4.1.6
module load cdo

TEMP_DIR=/scratch/bell/hu1029/Data/raw/ERA5_UV_2425/tmp_merge_1dg
OUTPUT_DIR=/depot/wanglei/data/ERA5_uvT

var=v
year=2024

if [ "$var" = "u" ]; then
    prefix="u_component_of_wind"
else
    prefix="v_component_of_wind"
fi

temp_output="${TEMP_DIR}/${prefix}_${year}_1dg.nc"
final_output="${OUTPUT_DIR}/${prefix}_${year}_1dg.nc"

monthly_files=()
for month in $(seq -w 1 12); do
    f="${TEMP_DIR}/${prefix}_${year}_${month}_1dg.nc"
    if [ ! -f "$f" ]; then
        echo "Missing remapped monthly file: $f"
        exit 1
    fi
    monthly_files+=("$f")
done

cdo -b F32 mergetime "${monthly_files[@]}" "$temp_output"
mv "$temp_output" "$final_output"

echo "Done: $final_output"