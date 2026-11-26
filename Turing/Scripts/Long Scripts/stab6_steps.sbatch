#!/bin/bash
#SBATCH --job-name=stab6_steps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
##SBATCH --partition=CHANGE_ME
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# STEP 6: training length, once runs are stable. Set BEST to the flags you
# chose from steps 3-5. Runs 10% and 100% so the report can show 100% gaining
# from more steps while 10% only memorises more.
#   sbatch --export=ALL,MODEL=crossmae,BEST="--lr 3e-4 --warmup 200 --clip 1" stab6_steps.sbatch

source "$SLURM_SUBMIT_DIR/stab_common.sh"
FRACS="${FRACS_S6:-0.1 1.0}"
start_job
BEST="${BEST:?set BEST to the flags chosen in steps 3-5}"

# shellcheck disable=SC2086
run_setting s6_best_1000 $BEST --steps 1000
# shellcheck disable=SC2086
run_setting s6_best_3000 $BEST --steps 3000
python probe_report.py "$SWEEP" --logs slurm-*.out
