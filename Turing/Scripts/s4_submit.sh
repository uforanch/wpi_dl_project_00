#!/bin/bash
# Stage s4: mask direction x weighting (the professor's 2x2), at two encoder
# learning rates. Run on the LOGIN node from the project folder:
#   bash s4_submit.sh                  # CrossMAE
#   MODEL=videomae PTIME=6:00:00 QTIME=3:00:00 bash s4_submit.sh
#   DIRS=e2d bash s4_submit.sh         # only the events->thermal direction (half the jobs)
#   WORKERS=0 bash s4_submit.sh        # no data-loading workers (if hangs come back)
#
# For each direction (e2d: events visible -> reconstruct thermal;
#                     d2e: thermal visible -> reconstruct events)
# and each encoder learning rate (full = 1x, enc0.1 = 0.1x):
#   1 pretraining job : <dir>_plain + <dir>_weighted (ego weighting), 2000 steps, then recon
#   1 probe job       : starts when its pretraining succeeds; 100 % labels,
#                       seeds 0 1 2, --lr 3e-4 (the fixed probe)
# Folders:
#   checkpoints + recon : dsert_results/s4_<model>_<dir>_<full|enc0.1>/
#   probe results       : dsert_stability/<model>/s4_<dir>_<full|enc0.1>/
# The random-direction column of the 2x2 (b_plain / c_ego) is already in
# s3_step2000 (1x) and s3_enc0.1 (0.1x); a_imagenet is in fix_grid.
set -euo pipefail
MODEL="${MODEL:-crossmae}"
DIRS="${DIRS:-e2d d2e}"
PTIME="${PTIME:-4:00:00}"
QTIME="${QTIME:-2:00:00}"
WORKERS="${WORKERS:-}"
FIX="--lr 3e-4"
RES="dsert_results"
STAGE="s4"
EXTRA=""
[ -n "$WORKERS" ] && EXTRA=",WORKERS=$WORKERS"

for f in s3_pretrain.sbatch stab_quick.sbatch stab_common.sh probe_report.py; do
    [ -f "$f" ] || { echo "missing $f in $(pwd)" >&2; exit 1; }
done
grep -q "def train_loader" run_experiments.py || { echo "run_experiments.py is the old version (no hang fix)" >&2; exit 1; }
grep -q 'STAGE="${STAGE:-s3}"' s3_pretrain.sbatch || { echo "s3_pretrain.sbatch is the old version (no STAGE)" >&2; exit 1; }

declare -A LRFLAG=([full]="" [enc0.1]="--encoder-lr-scale 0.1")
for D in $DIRS; do
    for V in full enc0.1; do
        VARIANT="${D}_${V}"
        OUT="$RES/${STAGE}_${MODEL}_${VARIANT}"
        P=$(sbatch --parsable --time="$PTIME" \
            --export="ALL,MODEL=$MODEL,STAGE=$STAGE,VARIANT=$VARIANT,PEXPS=${D}_plain ${D}_weighted,PFLAGS=${LRFLAG[$V]}$EXTRA" \
            s3_pretrain.sbatch)
        Q=$(sbatch --parsable --time="$QTIME" --cpus-per-task=4 --mem=16G --dependency="afterok:$P" \
            --export="ALL,MODEL=$MODEL,CKPT_FROM=$OUT,TAG=${STAGE}_${VARIANT},EXPS=${D}_plain ${D}_weighted,FRACS=1.0,SEEDS=0 1 2,FLAGS=$FIX$EXTRA" \
            stab_quick.sbatch)
        echo "$VARIANT: pretrain job $P -> $OUT ; probe job $Q -> dsert_stability/$MODEL/${STAGE}_${VARIANT}"
    done
done

cat <<MSG

Job IDs above are also listed by:  sacct -u \$USER --starttime today --format=JobID,JobName%15,State,Elapsed
When everything has finished:
  python probe_report.py dsert_stability/$MODEL/${STAGE}_* dsert_stability/$MODEL/s3_step2000 \\
      dsert_stability/$MODEL/s3_enc0.1 dsert_stability/$MODEL/fix_grid_seed* \\
      --logs slurm-*.out --paired a_imagenet --md section4_$MODEL.md
  for d in $RES/${STAGE}_${MODEL}_*; do echo "== \$d"; cat "\$d/recon.csv"; done
Watch for hangs: if a job's slurm-<id>.out stops changing for 10+ minutes it will
now stop itself; resubmit that pair with WORKERS=0.
MSG
