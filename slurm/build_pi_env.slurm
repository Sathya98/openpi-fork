#!/bin/bash
#SBATCH --job-name=build-pi-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
# Builds ../pi_env with scripts/build_pi_env.sh on a compute node.
# Launch from the openpi-fork repo root (sbatch or tools/hold/hold_run.sh).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "node: $(hostname)   start: $(date)"
nvidia-smi -L || true
scripts/build_pi_env.sh "$@"
echo "end: $(date)"
