#!/bin/bash
# Stage s5: firm up the direction result, and run the pieces of the professor's
# feedback that are still missing. Everything uses encoders at 0.1x learning
# rate (least harmful setting so far) and the fixed probe (--lr 3e-4, seeds 0 1 2).
# Run on the LOGIN node from the project folder:
#   bash s5_submit.sh                       # all three parts, CrossMAE
#   PARTS="seeds weights" bash s5_submit.sh # choose parts
#   WORKERS=0 bash s5_submit.sh             # no data-loading workers (if hangs return)
#
# PART seeds   : e2d_weighted, d2e_weighted, c_ego pretrained again with --seed 1
#                and --seed 2 (seed 0 already exists in s4 / s3), so each has 3
#                pretraining runs. Shows whether the direction ranking survives
#                pretraining-run noise.        2 pretrain + 2 probe jobs
# PART weights : the other weighting variants c_count, c_floor, c_recency
#                (random direction), to compare with b_plain / c_ego in s3_enc0.1.
#                                              1 pretrain + 1 probe job
# PART curve   : label-efficiency curve, 10/25/50/100 % labels, for
#                a_random_init, a_imagenet, b_plain, c_ego (s3_enc0.1 checkpoints)
#                and e2d_weighted (s4 checkpoint). No pretraining needed.
#                                              9 probe jobs
# Folders:
#   checkpoints : dsert_results/s5_<model>_pseed1, _pseed2, _weights
#   probes      : dsert_stability/<model>/s5_pseed1, s5_pseed2, s5_weights,
#                 s5_curve_{base,random,e2d}_seed{0,1,2}
set -euo pipefail
MODEL="${MODEL:-crossmae}"
PARTS="${PARTS:-seeds weights curve}"
PTIME="${PTIME:-6:00:00}"        # pretraining jobs train 3 experiments
QTIME="${QTIME:-3:00:00}"
WORKERS="${WORKERS:-}"
FIX="--lr 3e-4"
RES="dsert_results"
ST="dsert_stability/$MODEL"
STAGE="s5"
EXTRA=""
[ -n "$WORKERS" ] && EXTRA=",WORKERS=$WORKERS"

for f in s3_pretrain.sbatch stab_quick.sbatch stab_common.sh probe_report.py plot_curve.py; do
    [ -f "$f" ] || { echo "missing $f in $(pwd)" >&2; exit 1; }
done
grep -q "def train_loader" run_experiments.py || { echo "run_experiments.py is an old version" >&2; exit 1; }
grep -q 'STAGE="${STAGE:-s3}"' s3_pretrain.sbatch || { echo "s3_pretrain.sbatch is an old version" >&2; exit 1; }
grep -q "baselines need no ckpt" stab_common.sh || { echo "stab_common.sh is an old version" >&2; exit 1; }
grep -q -- "--ref-from" probe_report.py || { echo "probe_report.py is an old version" >&2; exit 1; }
need() { [ -f "$1/ckpt/$2.pt" ] || { echo "missing checkpoint $1/ckpt/$2.pt" >&2; exit 1; }; }

pretrain() {   # VARIANT PEXPS PFLAGS -> job id
    sbatch --parsable --time="$PTIME" \
        --export="ALL,MODEL=$MODEL,STAGE=$STAGE,VARIANT=$1,PEXPS=$2,PFLAGS=$3$EXTRA" s3_pretrain.sbatch
}
probe() {      # TAG CKPT_FROM EXPS FRACS SEEDS [AFTER_JOB] -> job id
    local dep=""
    [ -n "${6:-}" ] && dep="--dependency=afterok:$6"
    # shellcheck disable=SC2086
    sbatch --parsable --time="$QTIME" --cpus-per-task=4 --mem=16G $dep \
        --export="ALL,MODEL=$MODEL,CKPT_FROM=$2,TAG=$1,EXPS=$3,FRACS=$4,SEEDS=$5,FLAGS=$FIX$EXTRA" \
        stab_quick.sbatch
}

for PART in $PARTS; do
  case "$PART" in
  seeds)
    for PS in 1 2; do
        V="pseed$PS"
        P=$(pretrain "$V" "e2d_weighted d2e_weighted c_ego" "--encoder-lr-scale 0.1 --seed $PS")
        Q=$(probe "${STAGE}_$V" "$RES/${STAGE}_${MODEL}_$V" "e2d_weighted d2e_weighted c_ego" 1.0 "0 1 2" "$P")
        echo "seeds   pretrain seed $PS: pretrain $P, probe $Q -> $ST/${STAGE}_$V"
    done ;;
  weights)
    P=$(pretrain weights "c_count c_floor c_recency" "--encoder-lr-scale 0.1")
    Q=$(probe "${STAGE}_weights" "$RES/${STAGE}_${MODEL}_weights" "c_count c_floor c_recency" 1.0 "0 1 2" "$P")
    echo "weights : pretrain $P, probe $Q -> $ST/${STAGE}_weights" ;;
  curve)
    need "$RES/s3_${MODEL}_enc0.1" b_plain; need "$RES/s3_${MODEL}_enc0.1" c_ego
    need "$RES/s4_${MODEL}_e2d_enc0.1" e2d_weighted
    for S in 0 1 2; do
        a=$(probe "${STAGE}_curve_base_seed$S" none "a_random_init a_imagenet" "0.1 0.25 0.5 1.0" "$S")
        b=$(probe "${STAGE}_curve_random_seed$S" "$RES/s3_${MODEL}_enc0.1" "b_plain c_ego" "0.1 0.25 0.5 1.0" "$S")
        c=$(probe "${STAGE}_curve_e2d_seed$S" "$RES/s4_${MODEL}_e2d_enc0.1" "e2d_weighted" "0.1 0.25 0.5 1.0" "$S")
        echo "curve   seed $S: probe jobs $a $b $c"
    done ;;
  *) echo "unknown part $PART (use seeds, weights, curve)" >&2; exit 1 ;;
  esac
done

cat <<MSG

Job IDs: sacct -u \$USER --starttime today --format=JobID,JobName%15,State,Elapsed
When everything has finished:
  # 1. direction with 3 pretraining runs each (0.1x encoders)
  python probe_report.py $ST/${STAGE}_pseed1 $ST/${STAGE}_pseed2 $ST/s4_e2d_enc0.1 \\
      $ST/s4_d2e_enc0.1 $ST/s3_enc0.1 --merge enc0.1_all_pretrain_seeds \\
      --paired a_imagenet --ref-from $ST/fix_grid_seed* --md s5_direction_$MODEL.md
  # 2. weighting variants (0.1x encoders, one pretraining run each)
  python probe_report.py $ST/${STAGE}_weights $ST/s3_enc0.1 --merge weighting_enc0.1 \\
      --paired a_imagenet --ref-from $ST/fix_grid_seed* --md s5_weighting_$MODEL.md
  # 3. label-efficiency curve (table + figure)
  python probe_report.py $ST/${STAGE}_curve_* --merge curve --paired a_imagenet --md s5_curve_$MODEL.md
  python plot_curve.py $ST/${STAGE}_curve_* --out s5_curve_$MODEL.png
  # reconstruction of the new checkpoints
  for d in $RES/${STAGE}_${MODEL}_*; do echo "== \$d"; cat "\$d/recon.csv"; done
MSG
