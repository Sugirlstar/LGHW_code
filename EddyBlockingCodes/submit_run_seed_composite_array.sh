#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=seed_comp
#SBATCH --output=logs/seed_comp_%A_%a.out
#SBATCH --error=logs/seed_comp_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=66
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --array=1-36%36

module load conda
export PYTHONPATH=/home/hu1029/Nature_Serendipity:$PYTHONPATH

mkdir -p logs

TASKFILE=tasklist_seed.txt

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${TASKFILE})

dtname=$(echo "$LINE" | awk '{print $1}')
kk=$(echo "$LINE"     | awk '{print $2}')
blktype=$(echo "$LINE"| awk '{print $3}')
trackType=$(echo "$LINE" | awk '{print $4}')

echo "Running task ${SLURM_ARRAY_TASK_ID}: dtname=${dtname}, kk=${kk}, blktype=${blktype}, trackType=${trackType}"

python S6_SeedTrackDensity.py "${dtname}" "${kk}" "${blktype}" "${trackType}"

