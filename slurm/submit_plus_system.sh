#!/bin/bash
# Submit a LIBERO-plus eval config as one slurm array job (one task per shard).
#
# Usage:
#   bash slurm/submit_plus_system.sh <config.json> [--partition P] [--throttle N] [--time T] [--dry-run]
# e.g.
#   bash slurm/submit_plus_system.sh examples/libero/plus_system/configs/l123_sweep.json
#   bash slurm/submit_plus_system.sh examples/libero/plus_system/configs/calibration.json --dry-run
#
# The manifest is generated only if it does not exist yet, so resubmitting the
# same config resumes the sweep (the shard runner skips recorded episodes)
# instead of resharding it.

set -euo pipefail

CONFIG=${1:?Usage: bash slurm/submit_plus_system.sh <config.json> [--partition P] [--throttle N] [--time T] [--dry-run]}
shift

PARTITION=""
TIME_LIMIT=""
THROTTLE=8  # concurrent array tasks: the account's job budget
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION=${2:?--partition needs a value}; shift 2 ;;
    --throttle) THROTTLE=${2:?--throttle needs a value}; shift 2 ;;
    --time) TIME_LIMIT=${2:?--time needs a value}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
  esac
done

if ! [[ "$THROTTLE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --throttle must be a positive integer, got '$THROTTLE'" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 1
fi
CONFIG=$(readlink -f "$CONFIG")

REPO=/gpfs/work5/0/prjs2050/openpi-fork
cd "$REPO"

NAME=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$CONFIG")
OUT_DIR="$REPO/results/libero_plus_system/$NAME"
MANIFEST="$OUT_DIR/manifest.json"
JOB_MAP="$OUT_DIR/job_map.txt"

# --- 1. Manifest: generate once, then reuse (the resume path) -----------------
# Reusing a manifest that no longer matches the config would silently run the old
# plan, so the config's result-determining fields must still agree with it.
if [[ -f "$MANIFEST" ]]; then
  python3 - "$CONFIG" "$MANIFEST" <<'PY'
import json
import pathlib
import sys

config, manifest = (json.loads(pathlib.Path(argument).read_text()) for argument in sys.argv[1:3])
stored = manifest["config"]

differing = [
    field
    for field in ("task_suite", "perturbations", "difficulties", "base_tasks", "trials_per_variant",
                  "replan_steps", "seed", "max_steps")
    if config[field] != stored[field]
]
if "classification_json" in config and config["classification_json"] != stored["classification_json"]:
    differing.append("classification_json")
# Optional field: a manifest generated before it existed means "exclude nothing".
if sorted(config.get("exclude_noise_families", [])) != sorted(stored.get("exclude_noise_families", [])):
    differing.append("exclude_noise_families")
# The generator clamps num_shards to the number of selected variants.
if stored["num_shards"] != min(config["num_shards"], manifest["totals"]["num_variants"]):
    differing.append("num_shards")
# A null step in the config means "latest", which matches whatever step was resolved.
requested = [(entry["config"], entry["step"]) for entry in config["checkpoints"]]
resolved = [(entry["config"], entry["step"]) for entry in stored["checkpoints"]]
if len(requested) != len(resolved) or any(
    name != stored_name or (step is not None and step != stored_step)
    for (name, step), (stored_name, stored_step) in zip(requested, resolved)
):
    differing.append("checkpoints")

if differing:
    raise SystemExit(
        "ERROR: {} no longer matches the manifest; differing fields: {}\n"
        "Regenerate it with:\n"
        "  python3 examples/libero/plus_system/generate_manifest.py {} --force\n"
        "(add --discard-results if episodes were already collected under that name).".format(
            sys.argv[1], ", ".join(differing), sys.argv[1]
        )
    )
print("Using existing manifest {} (fingerprint {})".format(sys.argv[2], manifest["config_fingerprint"][:16]))
PY
else
  echo "Generating manifest for '$NAME' ..."
  python3 examples/libero/plus_system/generate_manifest.py "$CONFIG"
  echo "NOTE: generated a new manifest at $MANIFEST"
fi

# --- 2. Job map: array index -> shard file, in manifest order ----------------
COUNT=$(python3 - "$MANIFEST" "$JOB_MAP" <<'PY'
import json
import pathlib
import sys

manifest_path, map_path = (pathlib.Path(argument) for argument in sys.argv[1:3])
manifest = json.loads(manifest_path.read_text())
config = manifest["config"]
shard_dir = manifest_path.parent / "shards"

shard_files, missing = [], []
for checkpoint in config["checkpoints"]:
    tag = "{}-{}".format(checkpoint["config"], checkpoint["step"])
    for shard_index in range(config["num_shards"]):
        path = shard_dir / "{}_shard{:02d}.json".format(tag, shard_index)
        if not path.is_file():
            missing.append(str(path))
        shard_files.append(str(path))
if missing:
    raise SystemExit("Shard files the manifest calls for are missing:\n  " + "\n  ".join(missing))

map_path.write_text("\n".join(shard_files) + "\n")
print(len(shard_files))
PY
)
echo "Wrote job map $JOB_MAP ($COUNT shards)"

# --- 3. Submit ----------------------------------------------------------------
SBATCH_OPTS=()
if [[ -n "$PARTITION" ]]; then
  SBATCH_OPTS+=("--partition=$PARTITION")
fi
if [[ -n "$TIME_LIMIT" ]]; then
  SBATCH_OPTS+=("--time=$TIME_LIMIT")
fi
SBATCH_OPTS+=("--array=0-$((COUNT - 1))%$THROTTLE")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "--- job map ($COUNT lines) ---"
  head -1 "$JOB_MAP"
  if [[ "$COUNT" -gt 1 ]]; then
    echo "  ..."
    tail -1 "$JOB_MAP"
  fi
  echo "--- would run ---"
  echo "sbatch ${SBATCH_OPTS[*]} slurm/eval_plus_system.slurm $JOB_MAP"
  exit 0
fi

sbatch "${SBATCH_OPTS[@]}" slurm/eval_plus_system.slurm "$JOB_MAP"
echo "Submitted $COUNT shard tasks for '$NAME'. Check status with: squeue -u \$USER"
