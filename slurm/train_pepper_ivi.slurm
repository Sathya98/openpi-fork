#!/bin/bash
#SBATCH --job-name=pi05_pepper
#SBATCH --partition=cees6000
#SBATCH --account=ceesusers
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

# IVI variant of train_drema.slurm: ~/openpi-fork, its uv venv, no network on the
# node (base params, tokenizer and norm stats are prepared on the login node),
# checkpoints on the node-local /nvmestore.
#
#   sbatch --job-name=<cfg> slurm/train_pepper_ivi.slurm <config_name> [exp_name]

set -euo pipefail
CONFIG=${1:?Usage: sbatch slurm/train_pepper_ivi.slurm <config_name> [exp_name]}
EXP_NAME=${2:-main}

REPO=$HOME/openpi-fork
cd "$REPO"
PY=$REPO/.venv/bin/python   # the venv python directly: activate did not put python on PATH in the batch shell
export HF_LEROBOT_HOME="$REPO/data"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=$REPO/.hf-home HF_DATASETS_CACHE=$REPO/.hf-home/datasets   # ~/.cache/huggingface is a dangling symlink to a node-local disk
mkdir -p "$HF_DATASETS_CACHE"   # missing dir makes lerobot fall back to the hub
export OPENPI_DATA_HOME=$HOME/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=${WANDB_MODE:-offline}   # no network on the node; sync from the login node later
for d in /nvmestore /ssdstore; do [ -d $d ] && CKPT_BASE=$d/$USER/openpi-checkpoints && break; done   # cees6000 nodes mount /nvmestore, cees nodes /ssdstore
mkdir -p "$CKPT_BASE"

echo "=== Config: $CONFIG | exp: $EXP_NAME | node: $(hostname) | $(date) ==="
nvidia-smi -L
[ -f "$REPO/assets/$CONFIG/"*"/norm_stats.json" ] || [ -d "$REPO/assets/$CONFIG" ] || { echo "ERROR: norm stats for $CONFIG not computed (run compute_norm_stats.py on the login node)"; exit 1; }

CKPT_DIR="$CKPT_BASE/$CONFIG/$EXP_NAME"
if [[ -d "$CKPT_DIR" && -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]]; then
  RESUME_FLAG="--resume"; echo "=== resuming $CKPT_DIR ==="
else
  RESUME_FLAG="--overwrite"; echo "=== fresh run -> $CKPT_DIR ==="
fi
"$PY" scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" --checkpoint-base-dir="$CKPT_BASE" $RESUME_FLAG "${@:3}"   # extra flags pass through, e.g. --fsdp_devices 2
echo "=== Done $(date). Checkpoints in $CKPT_DIR (node-local: $(hostname)) ==="
