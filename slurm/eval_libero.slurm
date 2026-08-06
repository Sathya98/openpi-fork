#!/bin/bash
#SBATCH --job-name=eval_libero
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

# LIBERO benchmark evaluation: policy server (pi_env) + LIBERO client (libero_env)
# sharing one GPU. The server holds the pi0.5 model (~8GB bf16); the client only
# needs the GPU for EGL offscreen rendering.
#
# Usage:
#   sbatch slurm/eval_libero.slurm <config_name> [step] [trials_per_task]
# e.g.
#   sbatch slurm/eval_libero.slurm pi05_drema_libero_lora_b8_150k             # latest step, 50 trials
#   sbatch slurm/eval_libero.slurm pi05_drema_libero_full_b32_15k 7500 2     # smoke run

set -euo pipefail

CONFIG=${1:?Usage: sbatch slurm/eval_libero.slurm <config_name> [step] [trials_per_task]}
STEP=${2:-}
TRIALS=${3:-50}
TASK_SUITE=${TASK_SUITE:-libero_object}
REPLAN_STEPS=${REPLAN_STEPS:-8}

REPO=/gpfs/work5/0/prjs2050/openpi-fork
PI_ENV=/gpfs/work5/0/prjs2050/pi_env
LIBERO_ENV=/gpfs/work5/0/prjs2050/libero_env
cd "$REPO"
mkdir -p slurm/logs

# Default to the latest checkpoint step for this config.
if [[ -z "$STEP" ]]; then
  STEP=$(ls "checkpoints/$CONFIG/main" | grep -E '^[0-9]+$' | sort -n | tail -1)
fi
CKPT_DIR="$REPO/checkpoints/$CONFIG/main/$STEP"
if [[ ! -d "$CKPT_DIR/params" ]]; then
  echo "ERROR: no params/ under $CKPT_DIR" >&2
  exit 1
fi

RESULT_DIR="$REPO/results/libero_eval/$CONFIG-$STEP"
mkdir -p "$RESULT_DIR"

# Two eval jobs can share a node; derive a per-job port to avoid collisions.
PORT=$((8000 + SLURM_JOB_ID % 1000))

echo "=== Eval: $CONFIG @ step $STEP | suite=$TASK_SUITE trials=$TRIALS replan=$REPLAN_STEPS port=$PORT ==="
nvidia-smi -L

# --- 1. Policy server (background, pi_env) -----------------------------------
# Cap JAX preallocation so the EGL renderer has GPU memory headroom.
SERVER_LOG="$RESULT_DIR/server.log"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 "$PI_ENV/bin/python" scripts/serve_policy.py \
  --port "$PORT" \
  policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT_DIR" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# --- 2. Wait until the server is ready (checkpoint load + JIT warmup) --------
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
    echo "ERROR: server not ready after 10 min; tail of $SERVER_LOG:" >&2
    tail -20 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done

# --- 3. LIBERO client (foreground, libero_env) -------------------------------
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$LIBERO_ENV/bin/python" examples/libero/main.py \
  --args.host localhost \
  --args.port "$PORT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.num-trials-per-task "$TRIALS" \
  --args.replan-steps "$REPLAN_STEPS" \
  --args.video-out-path "$RESULT_DIR/videos" \
  2>&1 | tee "$RESULT_DIR/eval.log"

echo "=== Done. Summary: ==="
grep -E "Total success rate|Total episodes" "$RESULT_DIR/eval.log" || true
echo "Results in $RESULT_DIR"
