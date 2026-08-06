#!/bin/bash
# Submit LIBERO evals for all four libero-trained DREMA checkpoints (latest step each).
# Usage:  bash slurm/submit_all_evals.sh [trials_per_task]
# Run from the repo root (/gpfs/work5/0/prjs2050/openpi-fork).
#
# Only the *_libero_* configs are evaluable in LIBERO; the drema-real checkpoints
# (pi05_drema_full/lora_*) are real-robot models with no sim benchmark.

set -euo pipefail

TRIALS=${1:-50}

CONFIGS=(
    pi05_drema_libero_full_b8_150k
    pi05_drema_libero_full_b32_15k
    pi05_drema_libero_lora_b8_150k
    pi05_drema_libero_lora_b32_15k
)

for cfg in "${CONFIGS[@]}"; do
    echo "Submitting eval for $cfg (trials=$TRIALS) ..."
    sbatch --job-name="eval_${cfg}" slurm/eval_libero.slurm "$cfg" "" "$TRIALS"
done

echo "Submitted ${#CONFIGS[@]} eval jobs. Check status with: squeue -u \$USER"
