#!/bin/bash
#SBATCH --job-name=stab3_lr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
##SBATCH --partition=CHANGE_ME
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# STEP 3: lower learning rate (addresses divergence / spikes from big early
# updates to randomly initialised layers). Same test bed and seeds as step 1.

source "$SLURM_SUBMIT_DIR/stab_common.sh"
start_job

run_setting s3_lr3e-4 --lr 3e-4
run_setting s3_lr1e-4 --lr 1e-4
python probe_report.py "$SWEEP" --logs slurm-*.out
