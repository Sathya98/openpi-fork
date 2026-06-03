#!/bin/bash
#SBATCH --job-name=pi05_drema
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

# Unified training launcher for any DREMA config.
#
# Usage:
#   sbatch slurm/train_drema.slurm <config_name> [exp_name]
# e.g.
#   sbatch slurm/train_drema.slurm pi05_drema_lora_b32_15k
#   sbatch slurm/train_drema.slurm pi05_drema_full_b8_150k myrun
#
# Time is set to the 5-day max for the long b8_150k runs; shorter jobs finish early and
# release the node, so the high ceiling costs nothing but a little scheduling priority.
# Override per-job if you like: `sbatch --time=12:00:00 slurm/train_drema.slurm <cfg>`.
#
# Auto-resume: if a checkpoint already exists for (config, exp_name), the job resumes it;
# otherwise it starts fresh with --overwrite. This makes requeued/relaunched long jobs safe.

set -euo pipefail

CONFIG=${1:?Usage: sbatch slurm/train_drema.slurm <config_name> [exp_name]}
EXP_NAME=${2:-main}

REPO=/gpfs/work5/0/prjs2050/openpi-fork
cd "$REPO"
mkdir -p slurm/logs

# Use the existing environment (openpi + jax[cuda] + wandb installed).
source /gpfs/work5/0/prjs2050/pi_env/bin/activate

# LeRobotDataset resolves repo_id under HF_LEROBOT_HOME -> data/<repo_id>.
export HF_LEROBOT_HOME="$REPO/data"
# Let JAX use up to 90% of GPU memory (vs default 75%).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# W&B: requires WANDB_API_KEY in the environment, or a prior `wandb login` (~/.netrc).
export WANDB_PROJECT=${WANDB_PROJECT:-openpi-drema}
if [[ -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
  echo "ERROR: W&B logging is enabled but no WANDB_API_KEY and no ~/.netrc found." >&2
  echo "       Run 'wandb login' once on the login node, or export WANDB_API_KEY." >&2
  exit 1
fi

echo "=== Config: $CONFIG | exp: $EXP_NAME ==="
nvidia-smi -L

echo "=== Step 1/2: compute norm stats ==="
python scripts/compute_norm_stats.py --config-name "$CONFIG"

# Resume if a checkpoint dir already exists for this (config, exp), else start fresh.
CKPT_DIR="$REPO/checkpoints/$CONFIG/$EXP_NAME"
if [[ -d "$CKPT_DIR" && -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]]; then
  RESUME_FLAG="--resume"
  echo "=== Existing checkpoint found at $CKPT_DIR -> resuming ==="
else
  RESUME_FLAG="--overwrite"
  echo "=== No checkpoint at $CKPT_DIR -> starting fresh ==="
fi

echo "=== Step 2/2: train ==="
python scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" $RESUME_FLAG

echo "=== Done. Checkpoints in $CKPT_DIR ==="
