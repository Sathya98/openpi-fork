#!/bin/bash
#SBATCH --job-name=pi05_pepper
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

# train_drema.slurm for the robot-stack pepper configs on Snellius as tnijdam: half an H100
# node (fsdp_devices=2 in the config), data / base weights / checkpoints on scratch.
# Not A100: Snellius's are 40 GB, and full FT runs out of memory there even sharded over two.
#
#   sbatch slurm/train_pepper_snellius.slurm <config_name> [exp_name]
#
# Resumes when a checkpoint for (config, exp_name) exists.

set -euo pipefail

CONFIG=${1:?Usage: sbatch slurm/train_pepper_snellius.slurm <config_name> [exp_name]}
EXP_NAME=${2:-main}

REPO=/projects/0/prjs1574/openpi-fork
SCRATCH=/scratch-shared/tnijdam/openpi
cd "$REPO"
mkdir -p slurm/logs
source .venv/bin/activate

export HF_LEROBOT_HOME=$SCRATCH/lerobot
export OPENPI_DATA_HOME=$SCRATCH/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_PROJECT=${WANDB_PROJECT:-openpi-pepper}

echo "=== $CONFIG | $EXP_NAME | $(hostname) ==="
nvidia-smi -L

echo "=== norm stats ==="
python scripts/compute_norm_stats.py --config-name "$CONFIG"

CKPT_DIR="$SCRATCH/checkpoints/$CONFIG/$EXP_NAME"
# A step dir, not just any file: a run that died before its first save leaves wandb_id.txt.
if compgen -G "$CKPT_DIR/[0-9]*" >/dev/null; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG="--overwrite"
fi

echo "=== train ($RESUME_FLAG) ==="
python scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" \
  --checkpoint-base-dir="$SCRATCH/checkpoints" $RESUME_FLAG

echo "=== done: $CKPT_DIR ==="
