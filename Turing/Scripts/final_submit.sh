#!/bin/bash
# Submit the final run (final.sbatch) once per model. Run on the LOGIN node
# from the project folder:
#   bash final_submit.sh                   # crossmae and videomae
#   MODELS=crossmae bash final_submit.sh   # one model
set -euo pipefail
MODELS="${MODELS:-crossmae videomae}"
for f in final.sbatch run_experiments.py; do
    [ -f "$f" ] || { echo "missing $f in $(pwd)" >&2; exit 1; }
done
grep -q "def train_loader" run_experiments.py || { echo "run_experiments.py is an old version" >&2; exit 1; }
grep -q -- "--encoder-lr-scale" run_experiments.py || { echo "run_experiments.py lacks --encoder-lr-scale" >&2; exit 1; }
for M in $MODELS; do
    T=8:00:00
    [ "$M" = videomae ] && T=12:00:00          # VideoMAE trains more slowly
    if [ -f "dsert_results/final_$M/ckpt/e2d_weighted.pt" ]; then
        echo "note: dsert_results/final_$M already has a checkpoint; pretraining will be skipped"
    fi
    J=$(sbatch --parsable --time="$T" --export="ALL,MODEL=$M" final.sbatch)
    echo "$M: job $J -> dsert_results/final_$M/  (log slurm-$J.out)"
done
echo
echo "When done: dsert_results/final_<model>/label_efficiency.png and probe.csv"
