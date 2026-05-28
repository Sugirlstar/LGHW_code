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
#SBATCH --time=12:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo
module load parallel

RAW_DIR=/scratch/bell/hu1029/Data/raw/MERRA2/UV
OUT_DIR=/scratch/bell/hu1029/Data/processed/MERRA2_UV_1dg
TMP_DIR=/scratch/bell/hu1029/tmp_merra2

mkdir -p "$OUT_DIR"

# years
years=$(seq 2022 2025)

for y in $years; do

  echo "Processing y"
  mkdir -p "$TMP_DIR"

  final_u="$OUT_DIR/MERRA2_U_6hr_${y}_1dg.nc"
  final_v="$OUT_DIR/MERRA2_V_6hr_${y}_1dg.nc"

  # if existing final files are newer than raw files, skip processing
  if [[ -f "$final_u" && -f "$final_v" ]]; then
    echo "Final files for $y already exist. Skipping..."
    echo "---------------------------------------------"
    continue
  fi

  # build sorted daily file lists for this year
  mapfile -t UFILES < <(ls -1 "$RAW_DIR"/MERRA2_*.inst6_3d_ana_Np."$y"????_U.nc 2>/dev/null | sort)
  mapfile -t VFILES < <(ls -1 "$RAW_DIR"/MERRA2_*.inst6_3d_ana_Np."$y"????_V.nc 2>/dev/null | sort)
  echo "Found ${#UFILES[@]} U files and ${#VFILES[@]} V files for $y"
  echo "-------"

  # regrid to 1 degree (360x181) in parallel
  export TMP_DIR  # for parallel 

  process_pair () {
    local ufile="$1"
    local vfile="$2"

    local tmp_u="$TMP_DIR/U_$(basename "$ufile")"
    local tmp_v="$TMP_DIR/V_$(basename "$vfile")"

    echo "Processing $(basename "$ufile")"

    if [[ ! -f "$tmp_u" ]]; then
      cdo -s -O -b F32 remapbil,r360x181 "$ufile" "$tmp_u"
    else
      echo "  U exists, skip: $(basename "$tmp_u")"
    fi

    if [[ ! -f "$tmp_v" ]]; then
      cdo -s -O -b F32 remapbil,r360x181 "$vfile" "$tmp_v"
    else
      echo "  V exists, skip: $(basename "$tmp_v")"
    fi
  }
  export -f process_pair

  # parallel 64 jobs, or use SLURM_CPUS_PER_TASK if set (for better resource utilization if this script is run in a SLURM job with fewer CPUs)
  JOBS=${SLURM_CPUS_PER_TASK:-24}

  # feed UFILES and VFILES to parallel (must ensure both have the same number and order)
  parallel --link -j "$JOBS" --linebuffer process_pair ::: "${UFILES[@]}" ::: "${VFILES[@]}"

  echo "Finished regridding for $y. Now merging..."

  # merge yearly files into one file for this year
  cdo -s -O -b F32 mergetime "$TMP_DIR"/U_*.nc "$OUT_DIR/MERRA2_U_6hr_${y}_1dg.nc"
  cdo -s -O -b F32 mergetime "$TMP_DIR"/V_*.nc "$OUT_DIR/MERRA2_V_6hr_${y}_1dg.nc"
  echo "Merged U and V files for $y into single files"

  rm -rf "$TMP_DIR"

done


  # serial run
  # # regrid to 1 degree (360x181) and save as temporary files
  # for i in "${!UFILES[@]}"; do
  #   ufile="${UFILES[i]}"
  #   vfile="${VFILES[i]}"
  #   tmp_u="$TMP_DIR/U_$(basename "$ufile")"
  #   tmp_v="$TMP_DIR/V_$(basename "$vfile")"
  #   echo "Processing ${ufile##*/}"
  #   # ---- U ----
  #   if [[ -f "$tmp_u" ]]; then
  #     echo "  U already regridded. Skipping."
  #   else
  #     cdo -s -O -b F32 remapbil,r360x181 "$ufile" "$tmp_u"
  #   fi
  #   # ---- V ----
  #   if [[ -f "$tmp_v" ]]; then
  #     echo "  V already regridded. Skipping."
  #   else
  #     cdo -s -O -b F32 remapbil,r360x181 "$vfile" "$tmp_v"
  #   fi
  # done  
