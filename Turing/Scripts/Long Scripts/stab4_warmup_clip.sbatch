#!/bin/bash
#SBATCH --job-name=stab4_wc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
##SBATCH --partition=CHANGE_ME
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# STEP 4: warmup and tighter gradient clipping (needs the updated
# run_experiments.py with --warmup / --clip). Run at the original lr 1e-3 so the
# effect is separate from step 3; set LR4 to also test at the step-3 winner.

source "$SLURM_SUBMIT_DIR/stab_common.sh"
start_job
LR4="${LR4:-1e-3}"

run_setting "s4_warmup200_lr$LR4"       --lr "$LR4" --warmup 200
run_setting "s4_clip1_lr$LR4"           --lr "$LR4" --clip 1
run_setting "s4_warmup200_clip1_lr$LR4" --lr "$LR4" --warmup 200 --clip 1
python probe_report.py "$SWEEP" --logs slurm-*.out
