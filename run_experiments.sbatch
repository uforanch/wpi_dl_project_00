#!/bin/bash
#SBATCH --job-name=dsert_mae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1                 # without this you get CPU only (slower than your laptop)
##SBATCH --partition=CHANGE_ME       # uncomment + set if your cluster needs a GPU partition
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Run `python run_experiments.py cache` on the LOGIN node first (compute nodes
# often have no internet), or copy your dsert_frames_cache/ folder over.

set -euo pipefail                     # stop at the first failing step
cd "$SLURM_SUBMIT_DIR"                # folder you ran sbatch from (project dir)

# --- environment: adjust to your cluster ---------------------------------
# module load python/3.11 cuda/12.1   # if your cluster uses modules
source .venv/bin/activate             # venv must contain a CUDA build of torch
export PYTHONUNBUFFERED=1             # log lines appear in slurm-*.out immediately

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"
nvidia-smi || true

W=$(( SLURM_CPUS_PER_TASK / 2 ))      # data-loading workers

python run_experiments.py inspect
python run_experiments.py cache
python run_experiments.py overlap
python run_experiments.py pretrain --workers "$W" --device cuda
python run_experiments.py probe    --workers "$W" --device cuda
python run_experiments.py plot