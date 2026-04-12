#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=highmem
#SBATCH --job-name=download
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=06:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo
module load parallel

OUT_DIR=/scratch/bell/hu1029/Data/raw/MERRA2/UV
IN_DIR=/scratch/bell/hu1029/Data/raw/MERRA2/ALLVars

mkdir -p "$OUT_DIR"

JOBS=80   #${SLURM_CPUS_PER_TASK:-6}
echo "JOBS=$JOBS"

export IN_DIR OUT_DIR
export OMP_NUM_THREADS=1   # Disable OpenMP parallelism in CDO to avoid oversubscription

find "$IN_DIR" -type f -name "*.nc4" | \
parallel -j "$JOBS" --linebuffer '

  INFILE="{}"
  FNAME=$(basename "$INFILE")
  BASE=${FNAME%.nc4}
  OUTFILE_U="$OUT_DIR/${BASE}_U.nc"
  OUTFILE_V="$OUT_DIR/${BASE}_V.nc"

  if [ ! -s "$OUTFILE_U" ]; then
      cdo -b F32 -selname,U "$INFILE" "$OUTFILE_U" || exit 2
      cdo -b F32 -selname,V "$INFILE" "$OUTFILE_V" || exit 2
      echo "[DONE] $FNAME"
  else
      echo "[SKIP] $FNAME already processed"
  fi
'

echo "###########ALL DONE###########"