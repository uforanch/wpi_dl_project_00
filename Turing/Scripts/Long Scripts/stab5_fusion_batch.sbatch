#!/bin/bash
#SBATCH --job-name=stab5_fb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
##SBATCH --partition=CHANGE_ME
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# STEP 5: fusion layers and BatchNorm. --train-fusion trains randomly
# initialised (a_imagenet) fusion layers through BatchNorm on batches of 8.
# Tests a bigger batch, and fusion frozen. STAB_FLAGS = best flags from steps
# 3-4 (default: none, i.e. the old settings), so only one thing changes.

source "$SLURM_SUBMIT_DIR/stab_common.sh"
start_job
STAB_FLAGS="${STAB_FLAGS:-}"

# shellcheck disable=SC2086
run_setting s5_batch16 $STAB_FLAGS --batch 16
SAVED_FUSION="$FUSION"; FUSION=""
# shellcheck disable=SC2086
run_setting s5_fusion_frozen $STAB_FLAGS
FUSION="$SAVED_FUSION"
python probe_report.py "$SWEEP" --logs slurm-*.out
