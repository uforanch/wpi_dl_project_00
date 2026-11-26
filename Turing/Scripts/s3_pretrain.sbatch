#!/bin/bash
#SBATCH --job-name=s3_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
##SBATCH --partition=CHANGE_ME
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Pretrain the experiments in PEXPS for one VARIANT, then check how well each
# checkpoint reconstructs the hidden modality (recon.csv + PNGs).
# Used by s3_submit.sh (STAGE=s3) and s4_submit.sh (STAGE=s4).
#   STAGE   = folder prefix (default s3) -> dsert_results/<STAGE>_<model>_<VARIANT>
#   PEXPS   = experiments to pretrain (default "b_plain c_ego")
#   VARIANT = folder suffix, e.g. full / enc0.1 / enc0
#   PFLAGS  = extra pretrain flags, e.g. "--save-at 250 500 1000" or "--encoder-lr-scale 0.1"
# Normally submitted by s3_submit.sh.

source "$SLURM_SUBMIT_DIR/stab_common.sh"
start_job
VARIANT="${VARIANT:?set VARIANT}"
PFLAGS="${PFLAGS:-}"
PSTEPS="${PSTEPS:-2000}"
STAGE="${STAGE:-s3}"
PEXPS="${PEXPS:-b_plain c_ego}"
OUT="dsert_results/${STAGE}_${MODEL}_${VARIANT}"
echo "pretrain $PEXPS -> $OUT  (steps $PSTEPS, flags: $PFLAGS)"

# shellcheck disable=SC2086
python run_experiments.py pretrain --model "$MODEL" --cache "$PROBE_CACHE" --out "$OUT" \
    --only $PEXPS --steps "$PSTEPS" --workers "$W" --device cuda $PFLAGS

for d in "$OUT" "$OUT"/at_step*; do
    [ -d "$d/ckpt" ] || continue
    python run_experiments.py recon --model "$MODEL" --cache "$PROBE_CACHE" --out "$d" \
        --max-frames 300 --n-vis 4 --device cuda
done
