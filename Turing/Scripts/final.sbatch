#!/bin/bash
#SBATCH --job-name=dsert_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
##SBATCH --partition=CHANGE_ME
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# One final run per model with the settings that worked best in the experiments:
#   pretraining : e2d_weighted (events visible -> reconstruct thermal, ego-weighted
#                 loss), encoders at 0.1x learning rate, 2000 steps
#   probe       : --lr 3e-4 --train-fusion --head-convs 1 --steps 1000 --max-val 500
#                 for e2d_weighted against a_imagenet and a_random_init,
#                 at 10/25/50/100 % labels, ONE seed
#   then recon (reconstruction check) and plot (label-efficiency figure)
# Usually submitted by final_submit.sh; by hand:
#   sbatch --export=ALL,MODEL=crossmae final.sbatch
#   sbatch --time=12:00:00 --export=ALL,MODEL=videomae final.sbatch
# Results: dsert_results/final_<model>/  (ckpt/, probe.csv, recon.csv,
#          label_efficiency.png, recon_e2d_weighted.png)
# Note: these settings were chosen on CrossMAE; VideoMAE gets the same ones,
# untuned. One seed means single numbers, not means: see the results doc for
# how much they vary.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # VideoMAE weights are cached
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

MODEL="${MODEL:-crossmae}"
CACHE="${CACHE:-dsert_frames_cache}"
SEED="${SEED:-0}"
OUT="dsert_results/final_${MODEL}"
COMMON=(--model "$MODEL" --cache "$CACHE" --out "$OUT" --device cuda --workers 0 --seed "$SEED")

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"
echo "final run: model $MODEL, cache $CACHE, seed $SEED -> $OUT"

echo "== pretrain"
python run_experiments.py pretrain "${COMMON[@]}" --only e2d_weighted \
    --steps 2000 --encoder-lr-scale 0.1

echo "== probe"
[ -f "$OUT/probe.csv" ] && mv "$OUT/probe.csv" "$OUT/probe_previous_$(date +%Y%m%d_%H%M%S).csv"
python run_experiments.py probe "${COMMON[@]}" \
    --only a_random_init a_imagenet e2d_weighted --fractions 0.1 0.25 0.5 1.0 \
    --lr 3e-4 --train-fusion --head-convs 1 --steps 1000 --max-val 500

echo "== recon"
python run_experiments.py recon "${COMMON[@]}" --only e2d_weighted --max-frames 300 --n-vis 4

echo "== plot"
python run_experiments.py plot --model "$MODEL" --out "$OUT"
echo
cat "$OUT/probe.csv"
