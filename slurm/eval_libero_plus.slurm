#!/bin/bash
#SBATCH --job-name=eval_libero_plus
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

# LIBERO-plus robustness evaluation: policy server (pi_env) + LIBERO-plus client
# (libero_plus_env) sharing one GPU. The perturbation dimension is selected per job.
#
# Usage:
#   sbatch slurm/eval_libero_plus.slurm <config_name> <perturbation> [step]
# Perturbations: camera | robot | language | light | background | noise | layout | all
# e.g.
#   sbatch slurm/eval_libero_plus.slurm pi05_drema_libero_lora_b8_150k light
#   sbatch slurm/eval_libero_plus.slurm pi05_drema_libero_full_b32_15k background 7500
#
# Episode counts per perturbation (libero_object, 1 trial/task): light=297,
# background=248, language=354, camera=396, robot=398, layout=403, noise=422,
# all=2518 (use a longer --time for "all").

set -euo pipefail

CONFIG=${1:?Usage: sbatch slurm/eval_libero_plus.slurm <config_name> <perturbation> [step]}
PERTURBATION=${2:?Missing perturbation: camera|robot|language|light|background|noise|layout|all}
STEP=${3:-}
TASK_SUITE=${TASK_SUITE:-libero_object}
REPLAN_STEPS=${REPLAN_STEPS:-8}

REPO=/gpfs/work5/0/prjs2050/openpi-fork
PI_ENV=/gpfs/work5/0/prjs2050/pi_env
PLUS_ENV=/gpfs/work5/0/prjs2050/libero_plus_env
IMAGEMAGICK=/gpfs/work5/0/prjs2050/tools/imagemagick
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

RESULT_DIR="$REPO/results/libero_plus_eval/$CONFIG-$STEP/$PERTURBATION"
mkdir -p "$RESULT_DIR"

# Two eval jobs can share a node; derive a per-job port to avoid collisions.
PORT=$((8000 + SLURM_JOB_ID % 1000))

echo "=== LIBERO-plus eval: $CONFIG @ step $STEP | suite=$TASK_SUITE perturbation=$PERTURBATION replan=$REPLAN_STEPS port=$PORT ==="
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

# --- 3. LIBERO-plus client (foreground, libero_plus_env) ---------------------
# LIBERO_CONFIG_PATH: this env's own config (must NOT read ~/.libero, which
# points at the original LIBERO). MAGICK_HOME: user-space ImageMagick for wand.
LIBERO_CONFIG_PATH="$PLUS_ENV/.libero" \
MAGICK_HOME="$IMAGEMAGICK" \
LD_LIBRARY_PATH="$IMAGEMAGICK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PLUS_ENV/bin/python" examples/libero/main_plus.py \
  --args.host localhost \
  --args.port "$PORT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.perturbation "$PERTURBATION" \
  --args.replan-steps "$REPLAN_STEPS" \
  --args.video-out-path "$RESULT_DIR/videos" \
  2>&1 | tee "$RESULT_DIR/eval.log"

echo "=== Done. Summary: ==="
grep -E "Total success rate|Total episodes" "$RESULT_DIR/eval.log" || true
echo "Results in $RESULT_DIR"
