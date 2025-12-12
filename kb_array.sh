#!/bin/bash
#SBATCH --account=gfdl_o
#SBATCH --job-name=kb_array
#SBATCH --cluster=c5
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-12
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

# Absolute Python (avoid fragile 'conda activate' under srun)
ENV_PY="/gpfs/f5/gfdl_o/scratch/Xiaohui.Zhou/miniconda3/envs/analysis_PPAN/bin/python"

# Make threaded libs match your CPU reservation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

OUTDIR="/gpfs/f5/gfdl_o/scratch/Xiaohui.Zhou/Analysis/NNWave/Postprocess/GasFlux_SeaSaltEmission/KB_MONTHLY_NO_DASK"
mkdir -p "$OUTDIR"

MONTH=${SLURM_ARRAY_TASK_ID}
echo "Running month ${MONTH}"
"$ENV_PY" -u kb_one_month.py --month "$MONTH" --outdir "$OUTDIR"

# ENV_PY="/gpfs/f5/gfdl_o/scratch/Xiaohui.Zhou/miniconda3/envs/analysis_PPAN/bin/python"
# OUTDIR="/gpfs/f5/gfdl_o/scratch/Xiaohui.Zhou/Analysis/NNWave/Postprocess/GasFlux_SeaSaltEmission/KB_MONTHLY_STACKED"
# mkdir -p "$OUTDIR"
# $ENV_PY -u kb_mongthly_2019_2022.py  --outdir "$OUTDIR" --outfile kb_monthly_2019_2022.nc
