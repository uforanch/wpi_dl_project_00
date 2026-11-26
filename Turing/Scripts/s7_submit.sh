#!/bin/bash
# Stage s7 (final): does the 10 %-label gain hold for the OTHER pretraining runs?
# Probes e2d_weighted from the two extra pretraining runs (dsert_results/
# s5_<model>_pseed1 and _pseed2, 0.1x encoders) at 10 % labels with probe seeds
# 0-9 -- the same seeds, and therefore the same labelled frames, as the
# a_imagenet runs in s5_curve_base / s6_lowlabel, so every run has a matched pair.
# Pretraining run 0 (s4 checkpoint) was already done in s5/s6 (7/7 pairs, p=0.016).
# No pretraining; no data-loading workers (WORKERS=0).
# Run on the LOGIN node from the project folder:
#   bash s7_submit.sh
#   FRACS="0.1 0.25" bash s7_submit.sh      # also 25 %
# Each pretraining run is split over two jobs (seeds 0-4 and 5-9). Folders:
#   dsert_stability/<model>/s7_pseed<P>_seed0   (probe seeds 0-4)
#   dsert_stability/<model>/s7_pseed<P>_seed5   (probe seeds 5-9)
# The report merges the two halves into one setting per pretraining run (s7_pseed<P>).
set -euo pipefail
MODEL="${MODEL:-crossmae}"
FRACS="${FRACS:-0.1}"
QTIME="${QTIME:-2:00:00}"
ST="dsert_stability/$MODEL"

for f in stab_quick.sbatch stab_common.sh probe_report.py; do
    [ -f "$f" ] || { echo "missing $f in $(pwd)" >&2; exit 1; }
done
grep -q "def sign_p" probe_report.py || { echo "probe_report.py is an old version (no sign test)" >&2; exit 1; }
for P in 1 2; do
    c="dsert_results/s5_${MODEL}_pseed$P/ckpt/e2d_weighted.pt"
    [ -f "$c" ] || { echo "missing $c" >&2; exit 1; }
done
for P in 1 2; do
    for HALF in "0 1 2 3 4" "5 6 7 8 9"; do
        FIRST="${HALF%% *}"
        TAG="s7_pseed${P}_seed$FIRST"
        [ -e "$ST/$TAG/probe.csv" ] && { echo "$ST/$TAG already has results, skipping (delete it to rerun)"; continue; }
        J=$(sbatch --parsable --time="$QTIME" --cpus-per-task=4 --mem=16G \
            --export="ALL,MODEL=$MODEL,WORKERS=0,CKPT_FROM=dsert_results/s5_${MODEL}_pseed$P,TAG=$TAG,EXPS=e2d_weighted,FRACS=$FRACS,SEEDS=$HALF,FLAGS=--lr 3e-4" \
            stab_quick.sbatch)
        echo "pretraining run $P, probe seeds $HALF: job $J -> $ST/$TAG"
    done
done

cat <<MSG

Job IDs: sacct -u \$USER --starttime today --format=JobID,JobName%15,State,Elapsed
When all 4 jobs have finished:
  python probe_report.py $ST/s7_pseed1_seed* $ST/s7_pseed2_seed* --paired a_imagenet \\
      --ref-from $ST/s5_curve_base_seed* $ST/s6_lowlabel_seed* --md s7_final_$MODEL.md
Read the paired table: one row per pretraining run (s7_pseed1, s7_pseed2), about
10 pairs each. Together with pretraining run 0 (s6 report: 7/7, p = 0.016) this
says in how many of 3 pretraining runs the 10 % gain holds.
MSG
