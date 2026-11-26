# Shared settings for the probe-stability scripts (stab*.sbatch).
# Sourced, not run. Every variable can be overridden at submission, e.g.
#   sbatch --export=ALL,MODEL=videomae,SEEDS="0 1 2 3" stab1_baseline.sbatch

MODEL="${MODEL:-crossmae}"

# Cache the probe trains/evaluates on. The 5 GB cache is the default because it
# has more labelled frames (~3260 train) than dsert_cache_step2 (~1715), and it
# is where the 10/24 (VideoMAE) and 4/30 (CrossMAE) collapses were seen.
PROBE_CACHE="${PROBE_CACHE:-dsert_frames_cache}"

# Folder whose ckpt/ holds the pretrained backbones to probe.
#   5 GB pretraining : dsert_results/screen_$MODEL
#   20 GB pretraining: dsert_results_step2_short/screen_$MODEL
CKPT_FROM="${CKPT_FROM:-dsert_results/screen_$MODEL}"

# Every setting gets its own folder here (own probe.csv), with ckpt/ linked in.
SWEEP="${SWEEP:-dsert_stability/$MODEL}"

# Test bed (Step 1). Use seeds that collapsed before if you know them.
SEEDS="${SEEDS:-0 1 2}"
EXPS="${EXPS:-a_imagenet b_plain c_ego}"
FRACS="${FRACS:-1.0}"

# The screen set-up every earlier result used. --log-every 25 gives the log
# scanner (probe_report.py) enough points to tell how a run failed.
BASE="--steps 1000 --head-convs 1 --max-val 500 --log-every 25"
FUSION="--train-fusion"

W=$(( ${SLURM_CPUS_PER_TASK:-4} / 2 ))

# run_setting TAG [extra probe flags...]
# Runs every seed in $SEEDS on the test bed with $BASE $FUSION + extra flags.
# Later flags override earlier ones (e.g. "--steps 3000" beats BASE's 1000).
# A finished (tag, seed) leaves a .done file, so resubmitting after a timeout
# skips it instead of appending duplicate rows.
run_setting() {
    local tag="$1"; shift
    local dst="$SWEEP/$tag"
    mkdir -p "$dst"
    if [ -d "$CKPT_FROM/ckpt" ]; then
        ln -sfn "$(realpath "$CKPT_FROM/ckpt")" "$dst/ckpt"
    elif [ "$EXPS" != "a_imagenet" ]; then
        echo "ERROR: no checkpoints in $CKPT_FROM/ckpt (set CKPT_FROM)" >&2
        return 1
    fi
    for s in $SEEDS; do
        local done_flag="$dst/.done_${s}_$(echo "$EXPS $FRACS" | tr ' ' '_')"
        if [ -e "$done_flag" ]; then
            echo "### SETTING $dst SEED $s already done, skipping"
            continue
        fi
        echo "### SETTING $dst SEED $s FLAGS $BASE $FUSION $*"
        # shellcheck disable=SC2086
        python run_experiments.py probe --model "$MODEL" --cache "$PROBE_CACHE" \
            --out "$dst" --workers "$W" --device cuda \
            --only $EXPS --fractions $FRACS $BASE $FUSION --seed "$s" "$@"
        touch "$done_flag"
    done
}

start_job() {
    set -euo pipefail
    cd "$SLURM_SUBMIT_DIR"
    source .venv/bin/activate
    export PYTHONUNBUFFERED=1
    python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"
    echo "MODEL=$MODEL PROBE_CACHE=$PROBE_CACHE CKPT_FROM=$CKPT_FROM SWEEP=$SWEEP"
    echo "SEEDS=$SEEDS EXPS=$EXPS FRACS=$FRACS"
}
