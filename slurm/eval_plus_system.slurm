#!/bin/bash
#SBATCH --job-name=eval_plus
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.out

# One array task = one LIBERO-plus eval shard: policy server (pi_env) plus the
# shard's workers (libero_plus_env) sharing this node's GPU.
#
# Usage (normally through slurm/submit_plus_system.sh):
#   sbatch --array=0-31%8 slurm/eval_plus_system.slurm <job_map.txt>
#
# The job map lists one shard file per line; this task runs line
# $SLURM_ARRAY_TASK_ID (0-based). The map is the only source of truth for the
# array index -> shard mapping: never re-derive it from a directory listing, the
# same map may mix checkpoints and even eval names.
#
# gpu_mig cannot run this: EGL is unavailable on MIG slices.

set -euo pipefail

JOB_MAP=${1:?Usage: sbatch --array=0-<K-1>%<throttle> slurm/eval_plus_system.slurm <job_map.txt>}
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset; submit this script with --array}

REPO=/gpfs/work5/0/prjs2050/openpi-fork
PI_ENV=/gpfs/work5/0/prjs2050/pi_env
PLUS_ENV=/gpfs/work5/0/prjs2050/libero_plus_env
IMAGEMAGICK=/gpfs/work5/0/prjs2050/tools/imagemagick
cd "$REPO"
mkdir -p slurm/logs

# --- 1. Resolve this array index to its shard ---------------------------------
if [[ ! -f "$JOB_MAP" ]]; then
  echo "ERROR: job map not found: $JOB_MAP" >&2
  exit 1
fi
SHARD_FILE=$(awk -v line="$((TASK_ID + 1))" 'NR == line' "$JOB_MAP")
if [[ -z "$SHARD_FILE" ]]; then
  echo "ERROR: job map $JOB_MAP has no line $((TASK_ID + 1)) for array task $TASK_ID" >&2
  exit 1
fi
if [[ ! -f "$SHARD_FILE" ]]; then
  echo "ERROR: shard file not found: $SHARD_FILE (job map $JOB_MAP line $((TASK_ID + 1)))" >&2
  exit 1
fi

# Tab-separated: paths must survive the split even if one ever contains a space.
FIELDS=$(python3 -c '
import json, sys
spec = json.load(open(sys.argv[1]))
print("\t".join(str(spec[key]) for key in
                ("checkpoint_config", "checkpoint_dir", "num_workers", "result_dir")))
' "$SHARD_FILE")
IFS=$'\t' read -r CKPT_CONFIG CKPT_REL_DIR NUM_WORKERS RESULT_REL_DIR <<< "$FIELDS"
CKPT_DIR="$REPO/$CKPT_REL_DIR"
RESULT_DIR="$REPO/$RESULT_REL_DIR"
mkdir -p "$RESULT_DIR"

# Array elements each get their own job id, so this port is unique per task.
PORT=$((8000 + SLURM_JOB_ID % 1000))
TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK_ID}"

echo "=== LIBERO-plus shard eval: task $TASK_ID of $JOB_MAP ==="
echo "Shard file:  $SHARD_FILE"
echo "Checkpoint:  $CKPT_CONFIG @ $CKPT_DIR"
echo "Result dir:  $RESULT_DIR"
echo "Workers:     $NUM_WORKERS | port=$PORT"
nvidia-smi -L

# --- 2. Policy server (background, pi_env) ------------------------------------
# Cap JAX preallocation so the EGL renderers have GPU memory headroom
# (model is ~8 GB bf16; 0.4 x 40 GB leaves room for 15 workers' renderers).
SERVER_LOG="$RESULT_DIR/server_$TAG.log"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 "$PI_ENV/bin/python" scripts/serve_policy.py \
  --port "$PORT" \
  policy:checkpoint --policy.config "$CKPT_CONFIG" --policy.dir "$CKPT_DIR" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# --- 3. Wait until the server is ready (checkpoint load + JIT warmup) ---------
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

# --- 4. Shard workers (background, libero_plus_env) ---------------------------
# LIBERO_CONFIG_PATH: this env's own config (must NOT read ~/.libero, which
# points at the original LIBERO). MAGICK_HOME: user-space ImageMagick for wand.
# *_NUM_THREADS=1: a worker needs ~1 core, but its BLAS would spawn nproc threads and
# oversubscribe the node at 6+ workers.
WORKER_PIDS=()
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  printf -v WORKER_LOG "%s/worker%02d_%s.log" "$RESULT_DIR" "$worker" "$TAG"
  LIBERO_CONFIG_PATH="$PLUS_ENV/.libero" \
  MAGICK_HOME="$IMAGEMAGICK" \
  LD_LIBRARY_PATH="$IMAGEMAGICK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PLUS_ENV/bin/python" examples/libero/plus_system/run_shard.py \
    --shard-file "$SHARD_FILE" \
    --worker-id "$worker" \
    --num-workers "$NUM_WORKERS" \
    --host localhost \
    --port "$PORT" \
    --repo-root "$REPO" \
    > "$WORKER_LOG" 2>&1 &
  WORKER_PIDS+=($!)
  echo "Launched worker $worker (pid ${WORKER_PIDS[$worker]}) -> $WORKER_LOG"
done

# Wait for every worker before reporting: one worker aborting on its error
# circuit breaker must not cut its siblings off mid-episode.
STATUS=0
EXIT_CODES=""
for worker in "${!WORKER_PIDS[@]}"; do
  if wait "${WORKER_PIDS[$worker]}"; then
    CODE=0
  else
    CODE=$?
    STATUS=1
  fi
  EXIT_CODES+=" w$worker=$CODE"
done
echo "Worker exit codes:$EXIT_CODES"

# --- 5. Completion summary ----------------------------------------------------
RECORDED=$(python3 -c '
import pathlib, sys
print(sum(1 for path in pathlib.Path(sys.argv[1]).glob("episodes_*.jsonl")
          for line in path.open() if line.strip()))
' "$RESULT_DIR")
EXPECTED=$(python3 -c '
import json, sys
spec = json.load(open(sys.argv[1]))
print(len(spec["variants"]) * spec["trials_per_variant"])
' "$SHARD_FILE")
echo "=== Done. Episodes recorded: $RECORDED / $EXPECTED expected in $RESULT_DIR ==="
exit "$STATUS"
