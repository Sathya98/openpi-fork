#!/bin/bash
# Submit all eight DREMA training jobs via the unified launcher.
# Usage:  bash slurm/submit_all.sh
# Run from the repo root (/gpfs/work5/0/prjs2050/openpi-fork).

set -euo pipefail

CONFIGS=(
    # Real DREMA data (data/drema_real_data_20eps)
    pi05_drema_full_b8_150k
    # pi05_drema_full_b32_15k
    pi05_drema_lora_b8_150k
    pi05_drema_lora_b32_15k
    # LIBERO-object data (data/libero_object_original_filtered3ep_lrd)
    pi05_drema_libero_full_b8_150k
    pi05_drema_libero_full_b32_15k
    pi05_drema_libero_lora_b8_150k
    pi05_drema_libero_lora_b32_15k
)

for cfg in "${CONFIGS[@]}"; do
    # Short runs (b32, 15k steps) only need ~a day; long runs (b8, 150k steps) keep the
    # script's longer default. Conditioned on the config-name suffix.
    if [[ "$cfg" == *_15k ]]; then
        walltime=1-00:00:00
    else
        walltime=4-00:00:00
    fi
    echo "Submitting $cfg (time=$walltime) ..."
    sbatch --job-name="$cfg" --time="$walltime" slurm/train_drema.slurm "$cfg"
done

echo "Submitted ${#CONFIGS[@]} jobs. Check status with: squeue -u \$USER"
