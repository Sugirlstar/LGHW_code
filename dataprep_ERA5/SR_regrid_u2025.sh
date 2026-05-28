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
#SBATCH --time=24:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

INPUT_DIR=/scratch/bell/hu1029/Data/raw/ERA5_UV_2425
OUTPUT_DIR=/depot/wanglei/data/ERA5_uvT
TEMP_DIR=/scratch/bell/hu1029/Data/raw/ERA5_UV_2425/tmp_merge_1dg

mkdir -p "$TEMP_DIR"
mkdir -p "$OUTPUT_DIR"

for var in u; do
    if [ "$var" = "u" ]; then
        prefix="u_component_of_wind"
    else
        prefix="v_component_of_wind"
    fi

    for year in 2025; do
        final_output="${OUTPUT_DIR}/${prefix}_${year}_1dg.nc"
        temp_output="${TEMP_DIR}/${prefix}_${year}_1dg.nc"

        if [ -f "$final_output" ]; then
            echo "Output file $final_output already exists. Skipping ${prefix} for ${year}."
            continue
        fi

        echo "Processing ${prefix} for ${year}..."

        monthly_1dg_files=()

        for month in $(seq -w 1 12); do
            infile="${INPUT_DIR}/${prefix}_${year}_${month}.nc"
            outfile="${TEMP_DIR}/${prefix}_${year}_${month}_1dg.nc"

            # check if output file already exists
            if [ -f "$outfile" ]; then
                echo "  Output file $outfile already exists. Skipping remapping for ${infile}."
                monthly_1dg_files+=("$outfile")
                continue
            fi
            
            if [ ! -f "$infile" ]; then
                echo "Missing input file: $infile"
                exit 1
            fi

            echo "  Remapping ${infile}"
            cdo remapbil,r360x181 "$infile" "$outfile"

            monthly_1dg_files+=("$outfile")
        done

        echo "  Merging remapped monthly files..."
        cdo -b F32 mergetime "${monthly_1dg_files[@]}" "$temp_output"

        mv "$temp_output" "$final_output"

        echo "Done: $final_output"
    done
done

rm -rf "$TEMP_DIR"
echo "All done."