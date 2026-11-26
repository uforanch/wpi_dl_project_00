#!/bin/bash
#SBATCH --job-name=stab1_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
##SBATCH --partition=CHANGE_ME
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# STEPS 1+2: reproduce the current probe on the test bed, with detailed loss
# logs, so the collapses can be classified (diverged / never started / spiked).
#   sbatch --export=ALL,MODEL=videomae stab1_baseline.sbatch
#   sbatch --export=ALL,MODEL=crossmae stab1_baseline.sbatch
# Then: python probe_report.py dsert_stability/<model> --logs slurm-<jobid>.out

source "$SLURM_SUBMIT_DIR/stab_common.sh"
start_job

run_setting s1_baseline                      # lr 1e-3, no warmup, clip 10 (old defaults)
python probe_report.py "$SWEEP" --logs "slurm-$SLURM_JOB_ID.out"
