#!/bin/bash
# Stage s6: is the low-label advantage real? Adds probe seeds 3..9 for
# a_imagenet and e2d_weighted (s4 checkpoint, 0.1x encoders) at 10 % and 25 %
# labels, so each point has 10 seeds instead of 3. No pretraining.
# A probe seed also picks WHICH frames form the 10 % / 25 % subset, so this
# averages over both training noise and the choice of labelled frames.
# Run on the LOGIN node from the project folder:
#   bash s6_submit.sh
#   FRACS="0.1 0.25 0.5" bash s6_submit.sh                 # add 50 %
#   EXPS="a_imagenet e2d_weighted c_ego b_plain" CKPT=... bash s6_submit.sh
# One job per seed (2 experiments x 2 fractions = 4 probe runs), no
# data-loading workers (WORKERS=0), so the hang cannot happen.
# Probe results: dsert_stability/<model>/s6_lowlabel_seed<N>/
set -euo pipefail
MODEL="${MODEL:-crossmae}"
EXPS="${EXPS:-a_imagenet e2d_weighted}"
FRACS="${FRACS:-0.1 0.25}"
SEEDS="${SEEDS:-3 4 5 6 7 8 9}"
CKPT="${CKPT:-dsert_results/s4_${MODEL}_e2d_enc0.1}"
QTIME="${QTIME:-2:00:00}"
ST="dsert_stability/$MODEL"

for f in stab_quick.sbatch stab_common.sh probe_report.py plot_curve.py; do
    [ -f "$f" ] || { echo "missing $f in $(pwd)" >&2; exit 1; }
done
grep -q "def sign_p" probe_report.py || { echo "probe_report.py is an old version (no sign test)" >&2; exit 1; }
grep -q "baselines need no ckpt" stab_common.sh || { echo "stab_common.sh is an old version" >&2; exit 1; }
for e in $EXPS; do
    case "$e" in a_*) ;; *) [ -f "$CKPT/ckpt/$e.pt" ] || { echo "missing $CKPT/ckpt/$e.pt (set CKPT)" >&2; exit 1; } ;; esac
done

for S in $SEEDS; do
    J=$(sbatch --parsable --time="$QTIME" --cpus-per-task=4 --mem=16G \
        --export="ALL,MODEL=$MODEL,WORKERS=0,CKPT_FROM=$CKPT,TAG=s6_lowlabel_seed$S,EXPS=$EXPS,FRACS=$FRACS,SEEDS=$S,FLAGS=--lr 3e-4" \
        stab_quick.sbatch)
    echo "seed $S: probe job $J -> $ST/s6_lowlabel_seed$S"
done

cat <<MSG

Job IDs: sacct -u \$USER --starttime today --format=JobID,JobName%15,State,Elapsed
When everything has finished (seeds 0-2 come from the s5 curve runs):
  python probe_report.py $ST/s5_curve_base_seed* $ST/s5_curve_e2d_seed* $ST/s6_lowlabel_seed* \\
      --merge lowlabel --paired a_imagenet --md s6_lowlabel_$MODEL.md
  python plot_curve.py $ST/s5_curve_* $ST/s6_lowlabel_seed* --out s6_curve_$MODEL.png
In the report, look at e2d_weighted at 10 % and 25 %: 'pairs' should be about 10,
and 'p (mAP)' / 'p (Vehicle)' below 0.05 means the advantage is unlikely to be chance.
MSG
