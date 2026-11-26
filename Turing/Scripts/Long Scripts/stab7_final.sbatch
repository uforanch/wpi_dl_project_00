#!/bin/bash
#SBATCH --job-name=stab7_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
##SBATCH --partition=CHANGE_ME
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# STEP 7: final comparison with the chosen settings. Both pretrainings probed on
# ONE cache with the SAME seeds, so they can be compared seed by seed.
# a_imagenet runs once (it doesn't depend on the checkpoints).
#   sbatch --export=ALL,MODEL=crossmae,BEST="--lr 3e-4 --warmup 200 --clip 1 --steps 3000" stab7_final.sbatch

source "$SLURM_SUBMIT_DIR/stab_common.sh"
SEEDS="${SEEDS_FINAL:-0 1 2 3 4}"
FRACS="${FRACS_FINAL:-0.1 1.0}"
start_job
BEST="${BEST:?set BEST to the flags chosen in steps 3-6}"
CKPT_5GB="${CKPT_5GB:-dsert_results/screen_$MODEL}"
CKPT_20GB="${CKPT_20GB:-dsert_results_step2_short/screen_$MODEL}"

EXPS="a_imagenet b_plain c_ego"; CKPT_FROM="$CKPT_5GB"
# shellcheck disable=SC2086
run_setting final_5gb $BEST
EXPS="b_plain c_ego"; CKPT_FROM="$CKPT_20GB"
# shellcheck disable=SC2086
run_setting final_20gb $BEST
python probe_report.py "$SWEEP/final_5gb" "$SWEEP/final_20gb" --logs slurm-*.out \
    --paired a_imagenet --md "$SWEEP/final_report.md"
