#!/bin/bash
#SBATCH --job-name=smoke_libero_plus
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

# Smoke test for the LIBERO-plus eval stack: one policy server, then 2 tasks
# from EACH of the 7 perturbation categories (14 episodes total). Verifies
# perturbed env creation (new assets), EGL rendering, wand/ImageMagick noise,
# and the client-server loop before committing to full runs.
#
# Usage: sbatch slurm/smoke_libero_plus.slurm [config_name] [step]

set -euo pipefail

CONFIG=${1:-pi05_drema_libero_full_b32_15k}
STEP=${2:-}

REPO=/gpfs/work5/0/prjs2050/openpi-fork
PI_ENV=/gpfs/work5/0/prjs2050/pi_env
PLUS_ENV=/gpfs/work5/0/prjs2050/libero_plus_env
IMAGEMAGICK=/gpfs/work5/0/prjs2050/tools/imagemagick
cd "$REPO"
mkdir -p slurm/logs

if [[ -z "$STEP" ]]; then
  STEP=$(ls "checkpoints/$CONFIG/main" | grep -E '^[0-9]+$' | sort -n | tail -1)
fi
CKPT_DIR="$REPO/checkpoints/$CONFIG/main/$STEP"
RESULT_DIR="$REPO/results/libero_plus_eval/smoke-$CONFIG-$STEP"
mkdir -p "$RESULT_DIR"

PORT=$((8000 + SLURM_JOB_ID % 1000))
echo "=== LIBERO-plus SMOKE: $CONFIG @ $STEP | port=$PORT ==="
nvidia-smi -L

SERVER_LOG="$RESULT_DIR/server.log"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 "$PI_ENV/bin/python" scripts/serve_policy.py \
  --port "$PORT" \
  policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT_DIR" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "Waiting for policy server on port $PORT ..."
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: policy server died during startup; tail of $SERVER_LOG:" >&2
    tail -20 "$SERVER_LOG" >&2
    exit 1
  fi
  if curl -sf "http://localhost:$PORT/healthz" > /dev/null; then
    echo "Server ready after ~$((i * 5))s."
    break
  fi
  if [[ $i -eq 120 ]]; then
    echo "ERROR: server not ready after 10 min" >&2; tail -20 "$SERVER_LOG" >&2; exit 1
  fi
  sleep 5
done

FAILED=()
for PERTURBATION in camera robot language light background noise layout; do
  echo ""
  echo "--- smoke: $PERTURBATION (2 tasks) ---"
  OUT="$RESULT_DIR/$PERTURBATION"
  mkdir -p "$OUT"
  if LIBERO_CONFIG_PATH="$PLUS_ENV/.libero" \
     MAGICK_HOME="$IMAGEMAGICK" \
     LD_LIBRARY_PATH="$IMAGEMAGICK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
     MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PLUS_ENV/bin/python" examples/libero/main_plus.py \
       --args.host localhost \
       --args.port "$PORT" \
       --args.perturbation "$PERTURBATION" \
       --args.max-tasks 2 \
       --args.video-out-path "$OUT/videos" \
       2>&1 | tee "$OUT/eval.log"; then
    echo "OK: $PERTURBATION"
  else
    echo "FAILED: $PERTURBATION"
    FAILED+=("$PERTURBATION")
  fi
done

echo ""
echo "=== SMOKE SUMMARY ==="
for PERTURBATION in camera robot language light background noise layout; do
  RATE=$(grep -o '"success_rate": [0-9.]*' "$RESULT_DIR/$PERTURBATION/results.json" 2>/dev/null | tail -1 || echo "no results.json")
  echo "$PERTURBATION: $RATE"
done
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED categories: ${FAILED[*]}"
  exit 1
fi
echo "All 7 categories ran end-to-end."
